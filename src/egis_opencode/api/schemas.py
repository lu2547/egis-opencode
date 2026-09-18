"""egis-opencode API — pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/coding/chat 请求体。"""

    agent_id: str = Field(default="coding", description="coding | coding-plan")
    user_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, description="空则新建会话")
    message: str = Field(min_length=1)
    stream: bool = True
    protocol: Literal["agui"] = "agui"
    context: dict[str, Any] | None = None
    #: 工作目录绑定：None=沿用会话已存绑定；""=解绑（多租户模式）；
    #: "local:<绝对路径>"=锚定本地目录（校验后持久化到会话）
    workspace_root: str | None = None


class ChatResponse(BaseModel):
    """非流式 chat 响应。"""

    session_id: str
    message_id: str
    response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    turns: int = 0


class AbortRequest(BaseModel):
    """POST /api/coding/chat/abort 请求体。"""

    session_id: str = Field(min_length=1)


class RespondRequest(BaseModel):
    """POST /api/coding/permissions/{request_id}/respond 请求体。"""

    action: Literal["once", "always", "reject"]


class QuestionRespondRequest(BaseModel):
    """POST /api/coding/questions/{request_id}/respond 请求体。"""

    #: 自由文本或所选选项原文（不可空白）
    answer: str


class CloneRequest(BaseModel):
    """POST /api/coding/workspaces/clone 请求体。"""

    user_id: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    project_name: str | None = None
    branch: str | None = None


class WorkspaceBindRequest(BaseModel):
    """POST /api/coding/workspaces/bind 请求体（本地目录绑定/解绑）。"""

    user_id: str = Field(min_length=1)
    session_id: str | None = Field(default=None, description="空则仅校验不持久化")
    #: ""=解绑；"local:<绝对路径>"=绑定（后端校验）
    workspace_root: str = ""


class WorkspaceBindingResponse(BaseModel):
    """工作目录绑定信息（bind/binding/default 端点共用）。"""

    workspace_root: str = ""
    status: ProjectStatusResponse | None = None
    #: 当前 root 是否服务端 .env 默认（非用户显式绑定）——
    #: 前端据此区分“默认目录展示”与“显式绑定”：前者 chat 不携带
    #: workspace_root（后端每轮解析，.env 变更立即跟随）
    is_default: bool = False


class CommandInfo(BaseModel):
    """单个 slash 命令摘要。"""

    name: str
    description: str = ""


class FileNodeResponse(BaseModel):
    """文件树节点（path 相对工作目录根；目录 children=None 表示未加载）。"""

    name: str
    path: str
    type: Literal["dir", "file"]
    size: int = 0
    children: list[FileNodeResponse] | None = None


class FileTreeResponse(BaseModel):
    """GET /api/coding/workspaces/tree 响应（truncated = 触达节点上限）。"""

    nodes: list[FileNodeResponse] = Field(default_factory=list)
    truncated: bool = False


class FileContentResponse(BaseModel):
    """GET /api/coding/workspaces/file 响应（预览用，超限截断；二进制占位）。"""

    path: str
    content: str
    size: int
    truncated: bool
    binary: bool = False


class ProjectStatusResponse(BaseModel):
    """workspace 项目状态。"""

    name: str
    path: str
    is_git_repo: bool
    branch: str = ""
    head_short: str = ""
    dirty_files: int = 0
    untracked_files: int = 0


__all__ = [
    "AbortRequest",
    "ChatRequest",
    "ChatResponse",
    "CloneRequest",
    "CommandInfo",
    "FileContentResponse",
    "FileNodeResponse",
    "FileTreeResponse",
    "ProjectStatusResponse",
    "RespondRequest",
    "WorkspaceBindRequest",
    "WorkspaceBindingResponse",
]
