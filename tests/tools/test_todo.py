"""todo 工具测试 — 校验 / 全量替换语义 / todo_update 事件契约。"""

from __future__ import annotations

import pytest
from ark_agentic.core.types import ToolCall

from egis_opencode.agents.coding.tools.todo import TodoWriteTool
from egis_opencode.events import TODO_UPDATE

from tests.helpers import RecordingHandler

CTX = {"user:id": "alice"}


def _call(todos=None, **extra) -> ToolCall:
    return ToolCall.create("todo", {"todos": todos, **extra})


def _ctx(handler: RecordingHandler | None = None) -> dict:
    ctx = dict(CTX)
    if handler is not None:
        ctx["system:event_handler"] = handler
    return ctx


async def test_todo_update_emits_event(recorder):
    result = await TodoWriteTool().execute(
        _call([
            {"id": "1", "content": "调研现有实现", "status": "completed"},
            {"id": "2", "content": "编写测试", "status": "in_progress",
             "activeForm": "编写测试中"},
            {"id": "3", "content": "清理文档"},
        ]),
        _ctx(recorder),
    )

    assert result.is_error is False
    events = recorder.of_type(TODO_UPDATE)
    assert len(events) == 1
    todos = events[0]["todos"]
    assert len(todos) == 3
    assert todos[0]["id"] == "1"
    assert todos[0]["status"] == "completed"
    assert todos[1]["status"] == "in_progress"
    assert todos[1]["activeForm"] == "编写测试中"
    assert todos[2]["status"] == "pending"  # 缺省 pending
    assert todos[2]["activeForm"] == ""     # 缺省空


async def test_todo_full_replacement_semantics(recorder):
    """opencode 语义：每次全量替换（第二次调用后事件里只剩新列表）。"""
    tool = TodoWriteTool()
    await tool.execute(_call([{"id": "1", "content": "a"}]), _ctx(recorder))
    await tool.execute(_call([{"id": "2", "content": "b"}]), _ctx(recorder))

    events = recorder.of_type(TODO_UPDATE)
    assert len(events) == 2
    assert [t["id"] for t in events[0]["todos"]] == ["1"]
    assert [t["id"] for t in events[1]["todos"]] == ["2"]


async def test_todo_empty_list_clears(recorder):
    result = await TodoWriteTool().execute(_call([]), _ctx(recorder))
    assert result.is_error is False
    assert recorder.of_type(TODO_UPDATE)[0]["todos"] == []


async def test_todo_missing_todos_rejected():
    result = await TodoWriteTool().execute(
        ToolCall.create("todo", {}), _ctx(),
    )
    assert result.is_error is True


async def test_todo_invalid_status_rejected():
    result = await TodoWriteTool().execute(
        _call([{"id": "1", "content": "a", "status": "doing"}]), _ctx(),
    )
    assert result.is_error is True
    assert "status" in str(result.content)


async def test_todo_duplicate_id_rejected():
    result = await TodoWriteTool().execute(
        _call([
            {"id": "1", "content": "a"},
            {"id": "1", "content": "b"},
        ]),
        _ctx(),
    )
    assert result.is_error is True
    assert "重复" in str(result.content)


async def test_todo_missing_content_rejected():
    result = await TodoWriteTool().execute(
        _call([{"id": "1"}]), _ctx(),
    )
    assert result.is_error is True


async def test_todo_non_list_rejected():
    result = await TodoWriteTool().execute(
        ToolCall.create("todo", {"todos": "not-a-list"}), _ctx(),
    )
    assert result.is_error is True


async def test_todo_without_handler_silent():
    """无 event_handler（非流式）时静默成功，不炸。"""
    result = await TodoWriteTool().execute(
        _call([{"id": "1", "content": "a"}]), {},
    )
    assert result.is_error is False
