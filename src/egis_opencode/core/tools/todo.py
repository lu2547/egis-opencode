"""todo 工具 — 任务看板状态更新（opencode TodoWrite 语义）。

每次调用全量替换 todo 列表，经 executor 注入的 event_handler 发射
``todo_update`` custom 事件，前端渲染 Todo 看板。
"""

from __future__ import annotations

from typing import Any

from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ...events import TODO_UPDATE, TodoItem, todo_update_payload
from .base import CodingTool

#: 合法状态集
_VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

_TODO_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "条目唯一 ID（自选，保持稳定）"},
        "content": {"type": "string", "description": "任务内容（祈使句）"},
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "cancelled"],
            "description": "状态（默认 pending）",
        },
        "activeForm": {
            "type": "string",
            "description": "进行中时向用户展示的进行时描述",
        },
    },
    "required": ["id", "content"],
}


class TodoWriteTool(CodingTool):
    """更新任务看板（全量替换）。"""

    name = "todo"
    description = (
        "维护当前会话的任务清单（Todo 看板）。每次调用传入全量列表："
        "开始一个任务前把它的状态改为 in_progress，完成后改为 completed，"
        "放弃改为 cancelled。同一时间只应有一个 in_progress。"
    )
    parameters = [
        ToolParameter(
            name="todos", type="array",
            description="全量 todo 列表（替换现有列表；传空数组清空）",
            required=True,
            items=_TODO_ITEM_SCHEMA,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        raw_todos = args.get("todos")
        if raw_todos is None:
            return self._error(
                tool_call, "todos 参数缺失", context=context,
            )
        if not isinstance(raw_todos, list):
            return self._error(
                tool_call, "todos 必须是数组", context=context,
            )

        todos: list[TodoItem] = []
        seen_ids: set[str] = set()
        for i, raw in enumerate(raw_todos):
            if not isinstance(raw, dict):
                return self._error(
                    tool_call, f"todos[{i}] 必须是对象", context=context,
                )
            item_id = str(raw.get("id") or "").strip()
            content = str(raw.get("content") or "").strip()
            if not item_id or not content:
                return self._error(
                    tool_call,
                    f"todos[{i}] 缺少 id 或 content",
                    context=context,
                )
            if item_id in seen_ids:
                return self._error(
                    tool_call, f"todos id 重复: {item_id}", context=context,
                )
            seen_ids.add(item_id)

            status = str(raw.get("status") or "pending")
            if status not in _VALID_STATUSES:
                return self._error(
                    tool_call,
                    f"todos[{i}].status 非法: {status}（可选 "
                    f"{'/'.join(sorted(_VALID_STATUSES))}）",
                    context=context,
                )
            todos.append(
                TodoItem(
                    id=item_id,
                    content=content,
                    status=status,  # type: ignore[arg-type]
                    activeForm=str(raw.get("activeForm") or ""),
                )
            )

        self._emit_todo_update(context, todos)

        done = sum(1 for t in todos if t.status == "completed")
        summary = (
            f"任务清单已更新：{len(todos)} 项（{done} 完成）"
            if todos else "任务清单已清空"
        )
        return AgentToolResult.text_result(
            tool_call.id,
            summary,
            llm_digest=f"[tool:todo status=ok] {summary}。",
        )

    def _emit_todo_update(
        self, context: dict[str, Any] | None, todos: list[TodoItem],
    ) -> None:
        """经 executor 注入的 event_handler 发 todo_update 事件。"""
        if not context:
            return
        handler = context.get("system:event_handler")
        if handler is None:
            return
        try:
            handler.on_custom_event(TODO_UPDATE, todo_update_payload(todos))
        except Exception:  # noqa: BLE001 — 事件失败不影响工具结果
            pass


__all__ = ["TodoWriteTool"]
