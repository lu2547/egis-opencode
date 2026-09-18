"""PermissionBridge — 权限守卫与 SSE 流之间的 custom 事件通道。

ark ``run_hooks`` 不把 handler 传给 hook（``BeforeToolCallback`` 协议
kwargs 仅 ``turn`` / ``tool_calls``），权限事件因此不能经 hook 参数直达
SSE 流。自建 chat 端点为每个流式 run 建立通道：

- ``input_context["temp:permission_bridge"]`` 只放 **session_id 字符串**
  （token）—— ark 会把 input_context 原样写进 user 消息 metadata 落盘
  （``base_agent._prepare_session``），不可序列化对象会炸掉 transcript；
- ``PermissionBridge`` 对象本身存本模块的进程级注册表，按 session_id
  索引（run_registry 保证同一 session 并发唯一，天然 1:1）；
- guard 经 ``ctx.input_context`` 取 token → 查注册表得 bridge → 发事件。

非流式请求（或未经本端点驱动的 run）没有注册项：guard 对 ask 一律自动
拒绝 —— 无交互通道即无授权路径。测试可直接把 ``PermissionBridge``
对象作为 token 值注入（``bridge_from_context`` 两种形态都认）。
"""

from __future__ import annotations

import asyncio
from typing import Any

#: input_context 中的通道键（temp: 前缀 → ark 落盘时自动剥离；
#: 值恒为 str token，保证 user 消息 metadata 可 JSON 序列化）
BRIDGE_KEY = "temp:permission_bridge"

#: session_id → 活跃 bridge（chat 端点 attach，run 结束 detach）
_bridges: dict[str, "PermissionBridge"] = {}


class PermissionBridge:
    """包装 SSE handler，供 guard 在 before_tool 阶段发射 custom 事件。

    兼任尾部事件的收尾信号：title 生成等 run 后异步事件经
    ``defer``/``settle`` 计数，chat 端点在关流前等 ``drained``（带上限），
    确保这类帧能落进 SSE 输出。
    """

    def __init__(self, handler: Any | None) -> None:
        self._handler = handler
        self._pending = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def handler(self) -> Any | None:
        """底层 AgentEventHandler（转交 ToolExecutor 注入工具执行事件）。"""
        return self._handler

    @property
    def available(self) -> bool:
        """是否存在可交互通道（None handler 的 bridge 等价无通道）。"""
        return self._handler is not None

    @property
    def drained(self) -> asyncio.Event:
        """所有 ``defer`` 登记的尾部事件均已落帧（初值即 set）。"""
        return self._drained

    def defer(self) -> None:
        """登记一个尚未落帧的尾部事件（如后台 title 生成）。"""
        self._pending += 1
        self._drained.clear()

    def settle(self) -> None:
        """一个尾部事件已落帧；计数归零时触发 ``drained``。"""
        self._pending = max(0, self._pending - 1)
        if self._pending == 0:
            self._drained.set()

    def emit(self, custom_type: str, custom_data: dict[str, Any]) -> None:
        """向 SSE 流发 custom 事件（handler 缺失时静默丢弃）。"""
        if self._handler is not None:
            self._handler.on_custom_event(custom_type, custom_data)


def attach_bridge(session_id: str, bridge: PermissionBridge) -> str:
    """登记会话的权限通道；返回应写入 input_context 的 str token。"""
    _bridges[session_id] = bridge
    return session_id


def detach_bridge(session_id: str) -> None:
    """run 结束时注销会话通道（幂等）。"""
    _bridges.pop(session_id, None)


def bridge_from_context(input_context: dict[str, Any] | None) -> PermissionBridge | None:
    """从 run 的 input_context 提取权限通道（未注入 / 已注销 → None）。

    token 值两种形态：生产路径为 session_id 字符串（查注册表）；
    测试可直接注入 ``PermissionBridge`` 对象。
    """
    if not input_context:
        return None
    raw = input_context.get(BRIDGE_KEY)
    if isinstance(raw, PermissionBridge):
        return raw
    if isinstance(raw, str) and raw:
        return _bridges.get(raw)
    return None


__all__ = [
    "BRIDGE_KEY",
    "PermissionBridge",
    "attach_bridge",
    "bridge_from_context",
    "detach_bridge",
]
