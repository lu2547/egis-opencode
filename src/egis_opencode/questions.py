"""QuestionService — question 工具的挂起 / 应答 / 超时管理。

结构对齐 ``permissions.service.PermissionService``（同一套
PendingRequest + asyncio.Future + 超时清理模式），但语义独立：
权限是"能否执行"的三态审批，question 是"用户怎么说"的开放式
作答 —— 不合包，避免 permissions 语义被稀释。

生命周期：
1. question 工具 ``execute``：``create()`` 登记 pending（request_id +
   Future）→ 经 executor 注入的 event_handler 发 ``question_request``
   事件 → ``await wait_answer()``。
2. REST ``POST /api/coding/questions/{request_id}/respond``：用户作答
   落定 Future。
3. 超时 / run 结束（abort）→ 未作答清理，工具侧拿到
   ``answered=False``，由模型自行决策并说明假设。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from .config import settings


@dataclass(frozen=True)
class PendingQuestion:
    """一个挂起中的用户提问。"""

    request_id: str
    session_id: str
    run_id: str
    question: str
    #: 预设选项（前端按钮；空 = 自由文本作答）
    options: tuple[str, ...] = ()
    tool_call_id: str = ""
    created_at: float = field(default_factory=time.time)

    def to_payload(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "question": self.question,
            "options": list(self.options),
            "tool_call_id": self.tool_call_id,
        }


@dataclass(frozen=True)
class QuestionAnswer:
    """wait_answer 的结果（answered=False 时模型自决）。"""

    answered: bool
    answer: str = ""
    #: answered / timeout / discarded（与 SSE question_resolved.status 对齐，
    #: discarded 对内、unavailable 对外 —— 无通道在 wait 之前就返回）
    status: str = "timeout"
    request_id: str = ""


class UnknownQuestionRequest(KeyError):
    """应答的 request_id 不存在（已落定 / 超时清理 / 从未创建）。"""


class QuestionService:
    """进程级单例：管理所有挂起的用户提问。"""

    def __init__(self, timeout_seconds: int | None = None) -> None:
        self._timeout = (
            timeout_seconds if timeout_seconds is not None
            else settings.permission_timeout_seconds
        )
        self._pending: dict[str, tuple[PendingQuestion, asyncio.Future]] = {}

    # ── 工具侧 ─────────────────────────────────────────

    def create(
        self,
        *,
        session_id: str,
        run_id: str,
        question: str,
        options: tuple[str, ...] | list[str] = (),
        tool_call_id: str = "",
    ) -> PendingQuestion:
        """登记挂起提问（不阻塞）；随后调用方发事件再 ``await wait_answer``。"""
        request_id = uuid4().hex
        request = PendingQuestion(
            request_id=request_id,
            session_id=session_id,
            run_id=run_id,
            question=question,
            options=tuple(options),
            tool_call_id=tool_call_id,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[QuestionAnswer] = loop.create_future()
        self._pending[request_id] = (request, future)
        return request

    async def wait_answer(self, request: PendingQuestion) -> QuestionAnswer:
        """等待用户作答或超时（超时 → answered=False 并清理）。"""
        entry = self._pending.get(request.request_id)
        if entry is None or entry[0] is not request:
            return QuestionAnswer(
                answered=False, status="discarded",
                request_id=request.request_id,
            )
        _, future = entry
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request.request_id, None)
            return QuestionAnswer(
                answered=False, status="timeout",
                request_id=request.request_id,
            )

    # ── REST 侧 ────────────────────────────────────────

    def respond(self, request_id: str, answer: str) -> PendingQuestion:
        """落定挂起提问；返回原提问（供事件回执）。

        Raises:
            UnknownQuestionRequest: request_id 不存在。
            ValueError: answer 为空白。
        """
        answer = (answer or "").strip()
        if not answer:
            raise ValueError("question answer cannot be empty")
        entry = self._pending.pop(request_id, None)
        if entry is None:
            raise UnknownQuestionRequest(request_id)
        request, future = entry
        if not future.done():
            future.set_result(QuestionAnswer(
                answered=True, answer=answer, status="answered",
                request_id=request_id,
            ))
        return request

    def pending_for_session(self, session_id: str) -> list[PendingQuestion]:
        """某会话所有未落定提问（SSE 断线轮询兜底）。"""
        return [
            req
            for req, _ in self._pending.values()
            if req.session_id == session_id
        ]

    def discard_run(self, run_id: str) -> None:
        """run 结束/取消时清理该 run 下所有未落定提问（防泄漏）。"""
        for request_id in [
            rid
            for rid, (req, _) in self._pending.items()
            if req.run_id == run_id
        ]:
            entry = self._pending.pop(request_id, None)
            if entry and not entry[1].done():
                entry[1].set_result(QuestionAnswer(
                    answered=False, status="discarded",
                    request_id=request_id,
                ))


#: 进程级单例 — question 工具（创建请求）与 REST 应答端点共享。
question_service = QuestionService()


__all__ = [
    "PendingQuestion",
    "QuestionAnswer",
    "QuestionService",
    "UnknownQuestionRequest",
    "question_service",
]
