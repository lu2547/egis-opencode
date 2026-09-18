"""egis-opencode 自定义 SSE 事件契约。

所有事件经 ``StreamEventBus.on_custom_event(custom_type, custom_data)`` 发出，
由 BareAGUIFormatter（protocol=agui）格式化为 ``event: custom`` SSE 帧，
前端 ``sse.ts`` 按 ``custom_type`` 分发。

custom_type 清单：
- ``permission_request``   权限审批请求（前端出审批卡片）
- ``permission_resolved``  权限请求落定（审批卡片关闭）
- ``question_request``     agent 向用户提问（前端出问答卡片；question 工具）
- ``question_resolved``    提问落定（问答卡片关闭）
- ``tool_digest``          工具过程摘要（工具卡片：file_edit/bash/search/todo…）
- ``todo_update``          Todo 看板状态
- ``subagent_progress``    子 agent 任务进度（配合 agent_scope 字段）
- ``title_generated``      会话标题已生成（侧栏刷新）

payload 均为纯 JSON dict（可序列化），契约字段见各 builder 函数。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── 契约常量 ────────────────────────────────────────────

PERMISSION_REQUEST = "permission_request"
PERMISSION_RESOLVED = "permission_resolved"
QUESTION_REQUEST = "question_request"
QUESTION_RESOLVED = "question_resolved"
TOOL_DIGEST = "tool_digest"
TODO_UPDATE = "todo_update"
SUBAGENT_PROGRESS = "subagent_progress"
TITLE_GENERATED = "title_generated"

#: tool_digest.display_type — 前端卡片形态
ToolDisplayType = Literal[
    "tool_progress",  # 通用过程（默认）
    "file_edit",      # 文件编辑（带 diff）
    "bash",           # shell 命令执行
    "search",         # glob/grep/list 检索
    "task",           # 子 agent 并行任务
]

#: tool_digest.status — 工具执行状态
ToolDigestStatus = Literal["pending", "running", "success", "error", "denied"]

#: todo 条目状态（对齐 opencode TodoWrite 语义）
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]

#: 权限应答动作
PermissionAction = Literal["once", "always", "reject"]


# ── 契约模型 ────────────────────────────────────────────


class TodoItem(BaseModel):
    """todo_update.todos 条目。"""

    id: str
    content: str
    status: TodoStatus = "pending"
    activeForm: str = ""


class PermissionRequestPayload(BaseModel):
    """permission_request 事件 payload。"""

    request_id: str
    session_id: str
    run_id: str = ""
    permission: str = Field(description="规则维度名，如 bash/edit/read")
    pattern: str = Field(default="*", description="匹配模式（工具相关）")
    tool_name: str
    tool_call_id: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class PermissionResolvedPayload(BaseModel):
    """permission_resolved 事件 payload。"""

    request_id: str
    action: PermissionAction
    tool_call_id: str = ""


class QuestionRequestPayload(BaseModel):
    """question_request 事件 payload（question 工具向用户提问）。"""

    request_id: str
    session_id: str
    run_id: str = ""
    question: str
    #: 预设选项（前端渲染按钮；空 = 自由文本作答）
    options: list[str] = Field(default_factory=list)
    tool_call_id: str = ""


class QuestionResolvedPayload(BaseModel):
    """question_resolved 事件 payload（问答卡片关闭）。"""

    request_id: str
    #: answered（用户作答）/ timeout（超时自决）/ unavailable（无交互通道）
    status: Literal["answered", "timeout", "unavailable"]
    answer: str = ""
    tool_call_id: str = ""


class ToolDigestPayload(BaseModel):
    """tool_digest 事件 payload。"""

    tool_name: str
    tool_call_id: str = ""
    display_type: ToolDisplayType = "tool_progress"
    status: ToolDigestStatus = "running"
    title: str = ""
    # file_edit
    path: str | None = None
    diff: str | None = None
    old_string_preview: str | None = None
    new_string_preview: str | None = None
    # bash
    command: str | None = None
    exit_code: int | None = None
    stdout_tail: str | None = None
    # search
    pattern: str | None = None
    result_count: int | None = None
    results_preview: list[str] | None = None
    # read 完整内容（对齐 opencode read.ts metadata.display：
    # 前端卡片可展示模型读到的全部内容，而非只有 preview 摘要）
    file_text: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    total_lines: int | None = None
    content_truncated: bool | None = None
    # 通用备注
    note: str | None = None


class SubagentProgressPayload(BaseModel):
    """subagent_progress 事件 payload。"""

    scope_id: str
    label: str
    status: Literal["started", "finished", "failed"]
    summary: str = ""


class TitleGeneratedPayload(BaseModel):
    """title_generated 事件 payload。"""

    session_id: str
    title: str


def todo_update_payload(todos: list[TodoItem]) -> dict[str, Any]:
    """构造 todo_update 事件的 custom_data。"""
    return {"todos": [t.model_dump() for t in todos]}


def tool_digest_payload(**kwargs: Any) -> dict[str, Any]:
    """构造 tool_digest 事件的 custom_data（字段校验由 ToolDigestPayload 承担）。"""
    return ToolDigestPayload(**kwargs).model_dump(exclude_none=True)


def diff_preview(text: str, max_lines: int = 6, max_line_chars: int = 200) -> str:
    """截断文本预览：限行数 + 限单行宽度，供 tool_digest 展示。"""
    if not text:
        return ""
    lines = text.splitlines()[:max_lines]
    truncated = [line[:max_line_chars] for line in lines]
    if len(text.splitlines()) > max_lines:
        truncated.append("…")
    return "\n".join(truncated)
