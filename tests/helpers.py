"""egis-opencode 测试公共件 — Mock LLM / SSE handler 录制器。

被 ``tests/conftest.py``（fixture 化）与各测试模块直接导入。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk


class RecordingHandler:
    """记录 on_custom_event 调用的 AgentEventHandler 假件。

    实现 AgentEventHandler 协议（executor 会调 on_tool_call_start 等）；
    custom 事件与工具启动事件被捕获供断言，其余 no-op。
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        #: (tool_call_id, name) — 验证 handler 是否被传给 ToolExecutor
        self.tool_starts: list[tuple[str, str]] = []

    def on_custom_event(self, custom_type: str, custom_data: dict[str, Any]) -> None:
        self.events.append((custom_type, custom_data))

    def of_type(self, custom_type: str) -> list[dict[str, Any]]:
        return [data for t, data in self.events if t == custom_type]

    # ── AgentEventHandler 协议其余方法 ─────────

    def on_step(self, text: str) -> None:  # noqa: D102
        pass

    def on_content_delta(self, delta: str, turn: int = 1) -> None:  # noqa: D102
        pass

    def on_tool_call_start(self, tool_call_id: str, name: str, args: dict[str, Any]) -> None:  # noqa: D102
        self.tool_starts.append((tool_call_id, name))

    def on_tool_call_result(self, tool_call_id: str, name: str, result: Any) -> None:  # noqa: D102
        pass

    def on_thinking_delta(self, delta: str, turn: int = 1) -> None:  # noqa: D102
        pass

    def on_ui_component(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass

    def on_citation(self, span: Any) -> None:  # noqa: D102
        pass

    def on_citation_list(self, citations: list[Any]) -> None:  # noqa: D102
        pass


class MockChatModel:
    """duck-typed BaseChatModel：ainvoke 依序返回脚本化响应。

    对齐 ark ``tests/unit/core/test_runner.MockChatModel``。
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.call_count = 0

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockChatModel":
        return self

    def model_copy(self, update: dict[str, Any] | None = None) -> "MockChatModel":
        return self

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        if self.call_count >= len(self.responses):
            raise RuntimeError("ainvoke called more times than responses provided")
        res = self.responses[self.call_count]
        self.call_count += 1
        return res

    async def astream(
        self, messages: list[Any], **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """流式形态：把脚本化响应转为携带 tool_call_chunks 的 AIMessageChunk。

        LLMCaller.call_streaming 聚合 chunk.tool_call_chunks（真实 provider
        形态）；裸 AIMessage 的 tool_calls 不会被识别。
        """
        if self.call_count >= len(self.responses):
            raise RuntimeError("astream called more times than responses provided")
        res = self.responses[self.call_count]
        self.call_count += 1

        tool_calls = getattr(res, "tool_calls", None) or []
        if tool_calls:
            chunk = AIMessageChunk(
                content=res.content or "",
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"], ensure_ascii=False),
                        "id": tc["id"],
                        "index": i,
                        "type": "tool_call_chunk",
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            )
            chunk.response_metadata = {"finish_reason": "tool_calls"}
        else:
            chunk = AIMessageChunk(content=res.content or "")
            chunk.response_metadata = {"finish_reason": "stop"}
        yield chunk


__all__ = ["MockChatModel", "RecordingHandler"]
