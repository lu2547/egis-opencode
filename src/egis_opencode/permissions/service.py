"""PermissionService — ask 状态权限请求的挂起 / 应答 / 超时管理。

生命周期：
1. ``ask()``：创建 pending 请求（request_id + asyncio.Future），
   等待 REST 应答或超时。调用方（guard）在 SSE 流上先发
   ``permission_request`` 事件再 await Future。
2. ``respond()``：REST 端点收到用户应答后按 request_id 落定 Future。
3. ``ask()`` 返回 ``PermissionResolution``；超时等价 reject。
4. ``always`` 应答产生的授权规则由调用方写入 run 级 approved 规则集
   （内存态，进程生命周期 —— 同 opencode InstanceState.approved 语义）。

pending 请求按 session_id 索引，支持断线后轮询 ``pending_for_session()`` 兜底。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from ..config import settings
from .rules import Rule


@dataclass(frozen=True)
class PendingRequest:
    """一个挂起中的权限请求。"""

    request_id: str
    session_id: str
    run_id: str
    permission: str
    pattern: str
    tool_name: str
    tool_call_id: str
    tool_args: dict
    #: 待审批的 pattern 全集（bash 复合命令 → 每条子命令文本；同 opencode patterns）
    patterns: tuple[str, ...] = ()
    #: "总是允许" 记忆的 pattern（bash → 命令前缀元数 + " *"；同 opencode always）
    always_patterns: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    def to_payload(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "permission": self.permission,
            "pattern": self.pattern,
            "patterns": list(self.patterns or (self.pattern,)),
            "always_patterns": list(self.always_patterns or (self.pattern,)),
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_args": self.tool_args,
        }


@dataclass(frozen=True)
class PermissionResolution:
    """ask 请求的最终结果（携带 always_patterns 供授权规则回填）。"""

    action: str  # once | always | reject | timeout
    request_id: str
    permission: str = ""
    pattern: str = ""
    always_patterns: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        return self.action in ("once", "always")

    @property
    def as_rules(self) -> list[Rule]:
        """``always`` 时产出授权规则（run 内后续同 pattern 调用免审批）。"""
        if self.action != "always" or not self.permission:
            return []
        patterns = self.always_patterns or (self.pattern,)
        return [
            Rule(permission=self.permission, pattern=pattern, action="allow")
            for pattern in patterns
        ]


class UnknownPermissionRequest(KeyError):
    """应答的 request_id 不存在（已落定 / 超时清理 / 从未创建）。"""


class PermissionService:
    """进程级单例：管理所有挂起的权限请求。"""

    def __init__(self, timeout_seconds: int | None = None) -> None:
        self._timeout = timeout_seconds or settings.permission_timeout_seconds
        self._pending: dict[str, tuple[PendingRequest, asyncio.Future]] = {}

    # ── guard 侧 ───────────────────────────────────────

    def create_request(
        self,
        *,
        session_id: str,
        run_id: str,
        permission: str,
        pattern: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict,
        patterns: tuple[str, ...] | list[str] = (),
        always_patterns: tuple[str, ...] | list[str] = (),
    ) -> PendingRequest:
        """登记挂起请求（不阻塞）；随后调用方应先发事件再 ``await resolve``。"""
        request_id = uuid4().hex
        request = PendingRequest(
            request_id=request_id,
            session_id=session_id,
            run_id=run_id,
            permission=permission,
            pattern=pattern,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            patterns=tuple(patterns),
            always_patterns=tuple(always_patterns),
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionResolution] = loop.create_future()
        self._pending[request_id] = (request, future)
        return request

    async def wait_resolution(
        self, request: PendingRequest
    ) -> PermissionResolution:
        """等待应答或超时（超时 → reject 语义并清理）。"""
        entry = self._pending.get(request.request_id)
        if entry is None or entry[0] is not request:
            return PermissionResolution(
                action="reject", request_id=request.request_id,
                permission=request.permission, pattern=request.pattern,
                always_patterns=request.always_patterns,
            )
        _, future = entry
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request.request_id, None)
            return PermissionResolution(
                action="timeout", request_id=request.request_id,
                permission=request.permission, pattern=request.pattern,
                always_patterns=request.always_patterns,
            )

    # ── REST 侧 ────────────────────────────────────────

    def respond(self, request_id: str, action: str) -> PendingRequest:
        """落定挂起请求；返回原请求（供事件回执）。

        Raises:
            UnknownPermissionRequest: request_id 不存在。
            ValueError: action 非法。
        """
        if action not in ("once", "always", "reject"):
            raise ValueError(f"invalid permission action: {action!r}")
        entry = self._pending.pop(request_id, None)
        if entry is None:
            raise UnknownPermissionRequest(request_id)
        request, future = entry
        if not future.done():
            future.set_result(
                PermissionResolution(
                    action=action,
                    request_id=request_id,
                    permission=request.permission,
                    pattern=request.pattern,
                    always_patterns=request.always_patterns,
                )
            )
        return request

    def pending_for_session(self, session_id: str) -> list[PendingRequest]:
        """某会话所有未落定请求（断线轮询兜底）。"""
        return [
            req
            for req, _ in self._pending.values()
            if req.session_id == session_id
        ]

    def get_request(self, request_id: str) -> PendingRequest | None:
        entry = self._pending.get(request_id)
        return entry[0] if entry else None

    def discard_run(self, run_id: str) -> None:
        """run 结束时清理该 run 下所有未落定请求（防泄漏）。"""
        for request_id in [
            rid
            for rid, (req, _) in self._pending.items()
            if req.run_id == run_id
        ]:
            entry = self._pending.pop(request_id, None)
            if entry and not entry[1].done():
                entry[1].set_result(
                    PermissionResolution(
                        action="reject",
                        request_id=request_id,
                        permission=entry[0].permission,
                        pattern=entry[0].pattern,
                        always_patterns=entry[0].always_patterns,
                    )
                )


#: 进程级单例 — guard（创建请求）与 REST 应答端点共享。
#: 模块级构造而非 agent 构造，保证两侧拿到同一实例。
permission_service = PermissionService()
