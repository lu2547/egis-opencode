"""task 工具测试 — SpawnSubtasksTool 薄包装（事件序列 / 只读白名单 / 取消兜底）。

真实子任务调度由 ark subtask 测试覆盖；此处验证 TaskTool 自身：
- started/finished/failed 事件序列与 digest 终态
- executor 超时 cancel 时补发 failed 事件（前端永久转圈的根因修复）
- 调用方传入的 tools 字段被只读白名单强制覆盖
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from ark_agentic.core.types import AgentToolResult, ToolCall

from egis_opencode.core.tools.task import (
    SUBTASK_ALLOWED_TOOLS,
    TaskTool,
    _extract_labels,
    _inject_readonly_tools,
)
from egis_opencode.events import SUBAGENT_PROGRESS, TOOL_DIGEST

from tests.helpers import RecordingHandler


class FakeInner:
    """duck-typed SpawnSubtasksTool：脚本化 execute 行为。"""

    def __init__(
        self,
        *,
        result: AgentToolResult | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._result = result
        self._raise = raise_exc
        self.calls: list[ToolCall] = []

    async def execute(self, tool_call: ToolCall, context: dict | None = None) -> AgentToolResult:
        self.calls.append(tool_call)
        if self._raise is not None:
            raise self._raise
        assert self._result is not None
        return self._result


def _call(tasks: list[dict[str, Any]]) -> ToolCall:
    return ToolCall.create("task", {"tasks": tasks})


def _ctx(handler: RecordingHandler) -> dict:
    return {"system:event_handler": handler}


def _subtasks_result(*statuses: str) -> AgentToolResult:
    """构造 SpawnSubtasksTool 形态的 JSON 结果（subtasks 摘要列表）。"""
    payload = {
        "subtasks": [
            {
                "label": f"t{i}",
                "status": status,
                "result": f"r{i}" if status == "completed" else "",
                "error": "" if status == "completed" else "boom",
            }
            for i, status in enumerate(statuses)
        ]
    }
    return AgentToolResult.json_result("c1", payload)


class TestCompletionEvents:
    """正常完成路径：逐子任务终态事件 + digest 汇总。"""

    @pytest.mark.asyncio
    async def test_all_completed_emits_finished_and_success_digest(self):
        handler = RecordingHandler()
        inner = FakeInner(result=_subtasks_result("completed", "completed"))
        tool = TaskTool(inner)  # type: ignore[arg-type]

        await tool.execute(
            _call([{"task": "a", "label": "t0"}, {"task": "b", "label": "t1"}]),
            _ctx(handler),
        )

        progress = handler.of_type(SUBAGENT_PROGRESS)
        # 2 started + 2 finished
        assert [p["status"] for p in progress] == [
            "started", "started", "finished", "finished",
        ]
        digest = handler.of_type(TOOL_DIGEST)
        assert digest[-1]["status"] == "success"
        assert digest[-1]["result_count"] == 2
        assert digest[-1]["note"] == "completed×2"

    @pytest.mark.asyncio
    async def test_partial_failure_emits_failed_and_error_digest(self):
        handler = RecordingHandler()
        inner = FakeInner(result=_subtasks_result("completed", "failed"))
        tool = TaskTool(inner)  # type: ignore[arg-type]

        await tool.execute(
            _call([{"task": "a", "label": "t0"}, {"task": "b", "label": "t1"}]),
            _ctx(handler),
        )

        progress = handler.of_type(SUBAGENT_PROGRESS)
        assert [p["status"] for p in progress] == [
            "started", "started", "finished", "failed",
        ]
        digest = handler.of_type(TOOL_DIGEST)
        assert digest[-1]["status"] == "error"


class TestCancelledGuard:
    """executor 超时 cancel：必须补发 failed 事件再 re-raise。

    历史缺陷：started 事件已发、execute 被 cancel 后完成事件永远
    缺席 —— 前端子任务卡片永久转圈。
    """

    @pytest.mark.asyncio
    async def test_cancelled_emits_failed_for_all_started_and_reraises(self):
        handler = RecordingHandler()
        inner = FakeInner(raise_exc=asyncio.CancelledError())
        tool = TaskTool(inner)  # type: ignore[arg-type]

        with pytest.raises(asyncio.CancelledError):
            await tool.execute(
                _call([
                    {"task": "a", "label": "d1"},
                    {"task": "b", "label": "d2"},
                ]),
                _ctx(handler),
            )

        progress = handler.of_type(SUBAGENT_PROGRESS)
        # 2 started + 2 补发 failed（取消语义保留：raise 出去）
        assert [p["status"] for p in progress] == [
            "started", "started", "failed", "failed",
        ]
        assert all("取消" in p["summary"] for p in progress[2:])
        digest = handler.of_type(TOOL_DIGEST)
        assert digest[-1]["status"] == "error"


class TestReadonlyWhitelist:
    """子任务工具白名单：调用方传入的 tools 一律被只读集合覆盖。"""

    def test_inject_overrides_caller_tools(self):
        rewritten = _inject_readonly_tools({
            "tasks": [
                {"task": "a", "tools": ["read", "bash", "write"]},
                {"task": "b"},
            ]
        })
        assert [t["tools"] for t in rewritten["tasks"]] == [
            list(SUBTASK_ALLOWED_TOOLS),
            list(SUBTASK_ALLOWED_TOOLS),
        ]

    def test_inject_keeps_other_fields(self):
        rewritten = _inject_readonly_tools(
            {"tasks": [{"task": "a", "label": "x"}]}
        )
        assert rewritten["tasks"][0]["task"] == "a"
        assert rewritten["tasks"][0]["label"] == "x"

    @pytest.mark.asyncio
    async def test_execute_forwards_rewritten_args(self):
        inner = FakeInner(result=_subtasks_result("completed"))
        tool = TaskTool(inner)  # type: ignore[arg-type]

        await tool.execute(
            _call([{"task": "a", "label": "t0", "tools": ["bash"]}]),
            _ctx(RecordingHandler()),
        )

        forwarded = inner.calls[0].arguments["tasks"][0]
        assert forwarded["tools"] == list(SUBTASK_ALLOWED_TOOLS)


class TestLabelExtraction:
    """label 提取：有 label 用 label，无 label 由 started 事件兜底 task-N。"""

    @pytest.mark.asyncio
    async def test_missing_label_falls_back_to_task_index(self):
        handler = RecordingHandler()
        inner = FakeInner(result=_subtasks_result("completed"))
        tool = TaskTool(inner)  # type: ignore[arg-type]

        await tool.execute(_call([{"task": "a"}]), _ctx(handler))

        progress = handler.of_type(SUBAGENT_PROGRESS)
        started = [p for p in progress if p["status"] == "started"]
        # _subtask_statuses 结果无 label → 完成事件按 labels 顺序兜底
        assert started[0]["scope_id"] == "task-1"
        assert started[0]["label"] == "子任务 1"

    def test_extract_labels(self):
        args = {"tasks": [{"task": "a", "label": " x "}, {"task": "b"}]}
        assert _extract_labels(args) == ["x", ""]

    def test_extract_labels_non_list(self):
        assert _extract_labels({"tasks": "oops"}) == []
        assert _extract_labels({}) == []
