"""task 工具 — SpawnSubtasksTool 的定制包装（name="task"）。

包装动机（零侵入 ark）：
1. 工具名对齐 opencode 语义（``task``），并面向 coding 场景重写 description。
2. ark 子 runner 经 ``_construct()`` 重建，**不复制 callbacks** —— 子 agent
   的工具执行不经过 PermissionGuard。为防 LLM 借子任务绕过写权限，
   本包装在转发前给每个子任务强制注入只读工具白名单
   （``tasks[].tools``，ark SpawnSubtasksTool 原生支持该字段）。
3. 补充 ``tool_digest(task)`` 与 ``subagent_progress`` custom 事件。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import replace
from typing import Any

from ark_agentic.core.subtask.tool import SpawnSubtasksTool
from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ....events import SUBAGENT_PROGRESS, SubagentProgressPayload
from .base import CodingTool

logger = logging.getLogger(__name__)

#: 子任务允许的工具白名单（只读 + todo 看板）；写操作由主任务在权限监督下完成
SUBTASK_ALLOWED_TOOLS: tuple[str, ...] = (
    "read", "glob", "grep", "list", "todo",
)


class TaskTool(CodingTool):
    """并行执行多个独立子任务（隔离会话、只读权限）。"""

    name = "task"
    description = (
        "并行执行多个相互独立的子任务并汇总结果（每个子任务在隔离"
        "会话中独立推理）。适用于多方向调研、批量信息收集等场景；"
        "子任务只能使用只读工具（read/glob/grep/list），"
        "写文件、执行命令等修改性操作请在主任务中完成。"
        "不要用于有先后依赖的任务。"
    )
    parameters = [
        ToolParameter(
            name="tasks", type="array",
            description="子任务列表（每项含 task 描述与可选 label 标识）",
            required=True,
            items={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "子任务完整描述（自包含，含目标与输出要求）",
                    },
                    "label": {
                        "type": "string",
                        "description": "标识标签（用于进度展示与结果对应）",
                    },
                },
                "required": ["task"],
            },
        ),
    ]

    def __init__(self, inner: SpawnSubtasksTool) -> None:
        self._inner = inner

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        labels = _extract_labels(tool_call.arguments or {})

        # 安全改写：强制只读白名单（覆盖调用方传入的 tools 字段）
        safe_call = replace(
            tool_call,
            arguments=_inject_readonly_tools(tool_call.arguments or {}),
        )

        for i, label in enumerate(labels):
            self._emit_progress(
                context, scope_id=label or f"task-{i + 1}",
                label=label or f"子任务 {i + 1}", status="started",
            )
        self._emit_digest(
            context,
            tool_name="task", tool_call_id=tool_call.id,
            display_type="task", status="running",
            title=f"task ×{len(labels)}",
            note="；".join(labels)[:200] if labels else "",
        )

        try:
            result = await self._inner.execute(safe_call, context)
        except asyncio.CancelledError:
            # executor 超时（tool_timeout）会 cancel 整个 execute 协程，
            # 正常的完成事件发不出去 —— started 事件已发，不补 failed
            # 的话前端子任务卡片永久转圈。补发后仍 re-raise 保持取消语义。
            for i, label in enumerate(labels):
                self._emit_progress(
                    context, scope_id=label or f"task-{i + 1}",
                    label=label or f"子任务 {i + 1}", status="failed",
                    summary="工具执行被取消/超时，子任务未完成",
                )
            self._emit_digest(
                context,
                tool_name="task", tool_call_id=tool_call.id,
                display_type="task", status="error",
                title=f"task ×{len(labels)}",
                note="工具执行被取消/超时，子任务未完成",
            )
            raise

        self._emit_completion(context, tool_call.id, result, labels)
        return result

    # ── 事件 ───────────────────────────────────────────

    def _emit_completion(
        self,
        context: dict[str, Any] | None,
        tool_call_id: str,
        result: AgentToolResult,
        labels: list[str],
    ) -> None:
        statuses = _subtask_statuses(result)
        ok = bool(statuses) and all(
            s.get("status") == "completed" for s in statuses
        )
        self._emit_digest(
            context,
            tool_name="task", tool_call_id=tool_call_id,
            display_type="task",
            status="success" if ok else "error",
            title=f"task ×{len(labels)}",
            result_count=len(statuses),
            note=_summarize_statuses(statuses),
        )
        for i, status in enumerate(statuses):
            label = str(status.get("label") or "") or (
                labels[i] if i < len(labels) else f"子任务 {i + 1}"
            )
            state = str(status.get("status") or "")
            if state == "completed":
                self._emit_progress(
                    context, scope_id=label, label=label,
                    status="finished",
                    summary=str(status.get("result") or "")[:200],
                )
            else:
                self._emit_progress(
                    context, scope_id=label, label=label,
                    status="failed",
                    summary=str(status.get("error") or state)[:200],
                )

    def _emit_progress(
        self,
        context: dict[str, Any] | None,
        *,
        scope_id: str,
        label: str,
        status: str,
        summary: str = "",
    ) -> None:
        if not context:
            return
        handler = context.get("system:event_handler")
        if handler is None:
            return
        try:
            handler.on_custom_event(
                SUBAGENT_PROGRESS,
                SubagentProgressPayload(
                    scope_id=scope_id, label=label,
                    status=status, summary=summary,  # type: ignore[arg-type]
                ).model_dump(),
            )
        except Exception:  # noqa: BLE001 — 事件失败不影响工具结果
            logger.debug("subagent_progress emit failed", exc_info=True)


def _inject_readonly_tools(args: dict[str, Any]) -> dict[str, Any]:
    """深拷贝参数并给每个子任务强制只读工具白名单。"""
    rewritten = copy.deepcopy(args)
    tasks = rewritten.get("tasks")
    if isinstance(tasks, list):
        rewritten["tasks"] = [
            {**t, "tools": list(SUBTASK_ALLOWED_TOOLS)}
            if isinstance(t, dict) else t
            for t in tasks
        ]
    return rewritten


def _extract_labels(args: dict[str, Any]) -> list[str]:
    tasks = args.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [
        str(t.get("label") or "").strip()
        for t in tasks if isinstance(t, dict)
    ]


def _subtask_statuses(result: AgentToolResult) -> list[dict[str, Any]]:
    """解析 SpawnSubtasksTool 结果中的 subtasks 摘要列表。

    content 形态有两种：dict（``json_result`` 构造，ark 现行返回）
    与 JSON 字符串（历史会话回放/其他构造路径），均需兼容。
    """
    content = result.content
    if isinstance(content, dict):
        payload: Any = content
    elif isinstance(content, str) and content:
        try:
            payload = json.loads(content)
        except ValueError:
            return []
    else:
        return []
    if not isinstance(payload, dict):
        return []
    subtasks = payload.get("subtasks")
    return [s for s in subtasks if isinstance(s, dict)] if isinstance(subtasks, list) else []


def _summarize_statuses(statuses: list[dict[str, Any]]) -> str:
    if not statuses:
        return "无子任务结果"
    counts: dict[str, int] = {}
    for s in statuses:
        key = str(s.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return "；".join(f"{k}×{v}" for k, v in counts.items())


__all__ = ["SUBTASK_ALLOWED_TOOLS", "TaskTool"]
