"""question 工具 — 运行中向用户提问（opencode ask 语义）。

与权限审批的区别：权限问"能否执行"（三态审批，guard 驱动），
question 问"用户怎么说"（开放式作答，模型主动调用）。

通道：executor 注入的 ``system:event_handler`` 发 ``question_request``
custom 事件 → REST ``POST /api/coding/questions/{request_id}/respond``
作答 → QuestionService 落定 Future → 工具返回答案文本。
session_id / run_id 从 ``temp:session_id`` / ``temp:run_id`` 读取
（chat 端点注入，temp: 前缀落盘剥离，与权限 bridge token 同模式）。

无交互通道（非流式 run / 无 handler）或超时：不报错，返回
"用户不在线/未答复"提示，由模型自行决策并说明假设 ——
question 是模型的能力扩展而非硬依赖。
"""

from __future__ import annotations

import logging
from typing import Any

from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ....events import QUESTION_REQUEST, QUESTION_RESOLVED
from ....questions import QuestionAnswer, question_service

from .base import CodingTool

logger = logging.getLogger(__name__)

#: 预设选项上限（卡片按钮空间）
_MAX_OPTIONS = 4

_UNAVAILABLE_PROMPT = (
    "用户当前不在此对话（无交互通道）。请基于已有信息自行决策，"
    "并在最终回复中明确说明你做了哪些假设。不要再次调用 question 工具。"
)

_TIMEOUT_PROMPT = (
    "用户未在时限内回答（超时）。请自行决策并在最终回复中说明假设；"
    "如果该问题对后续工作不是必需的，直接继续任务。"
)


class QuestionTool(CodingTool):
    """向用户提问并等待回答（预设选项或自由文本）。"""

    name = "question"
    description = (
        "向用户提出一个问题并等待回答。当任务存在必须由用户澄清的"
        "关键歧义（如目标不明确、多个方案取舍、需要业务决策）时使用；"
        "能通过读代码/文档自行解决的不要问。可选提供 2-4 个预设选项，"
        "用户也可能自由文本作答。用户不在线或超时你会得到提示，届时"
        "自行决策并说明假设。"
    )
    thinking_hint = "正在向用户提问，等待回答…"
    parameters = [
        ToolParameter(
            name="question", type="string",
            description="要问的问题（一句话，具体、可回答）",
            required=True,
        ),
        ToolParameter(
            name="options", type="array",
            description=(
                "2-4 个预设选项（用户点选；缺省则自由文本作答）。"
                "每个选项为简短字符串"
            ),
            required=False,
            items={"type": "string"},
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        question = str(args.get("question") or "").strip()
        if not question:
            return self._error(
                tool_call, "question 参数缺失或为空", context=context,
            )
        options = self._normalize_options(args.get("options"))
        if isinstance(options, str):  # 归一化错误信息
            return self._error(tool_call, options, context=context)

        ctx: dict[str, Any] = context or {}
        handler = ctx.get("system:event_handler")
        session_id = str(ctx.get("temp:session_id") or "")
        if handler is None or not session_id:
            # 无交互通道（非流式 run / 未经 chat 端点驱动）：
            # 不是错误 —— 让模型自决，不阻断任务
            return AgentToolResult.text_result(
                tool_call.id, _UNAVAILABLE_PROMPT,
                llm_digest="[tool:question] 用户不在线，模型自决。",
            )

        run_id = str(ctx.get("temp:run_id") or "")
        request = question_service.create(
            session_id=session_id,
            run_id=run_id,
            question=question,
            options=options,
            tool_call_id=tool_call.id,
        )
        handler.on_custom_event(
            QUESTION_REQUEST,
            {
                "request_id": request.request_id,
                "session_id": session_id,
                "run_id": run_id,
                "question": question,
                "options": list(options),
                "tool_call_id": tool_call.id,
            },
        )
        answer: QuestionAnswer | None = None
        try:
            answer = await question_service.wait_answer(request)
        finally:
            # 落定事件：卡片关闭（作答 / 超时都发；通道失败不影响结果）。
            # abort（CancelledError）时 answer 仍为 None —— 发 timeout 态关卡片
            try:
                handler.on_custom_event(
                    QUESTION_RESOLVED,
                    {
                        "request_id": request.request_id,
                        "status": (
                            "answered" if answer is not None and answer.answered
                            else "timeout"
                        ),
                        "answer": (
                            answer.answer
                            if answer is not None and answer.answered
                            else ""
                        ),
                        "tool_call_id": tool_call.id,
                    },
                )
            except Exception:  # noqa: BLE001 — 事件失败不影响工具结果
                logger.debug("question_resolved emit failed", exc_info=True)

        if answer.answered:
            return AgentToolResult.text_result(
                tool_call.id,
                f"用户回答：{answer.answer}",
                llm_digest=f"[tool:question] 用户已回答。",
            )
        return AgentToolResult.text_result(
            tool_call.id, _TIMEOUT_PROMPT,
            llm_digest="[tool:question] 用户未答复（超时），模型自决。",
        )

    @staticmethod
    def _normalize_options(raw: Any) -> list[str] | str:
        """规整预设选项；返回 list 或错误信息字符串。"""
        if raw is None:
            return []
        if not isinstance(raw, list):
            return "options 必须是字符串数组"
        options = [str(o).strip() for o in raw if str(o or "").strip()]
        if len(options) > _MAX_OPTIONS:
            return f"options 最多 {_MAX_OPTIONS} 个（收到 {len(options)}）"
        return options


__all__ = ["QuestionTool"]
