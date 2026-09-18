"""workspace 子包 — 多租户 workspace 路径守卫、项目管理与 slash 命令。"""

from .commands import CommandSpec, discover_commands, expand_command
from .paths import WorkspacePathError, WorkspacePaths, resolve_under_workspace, safe_segment
from .service import ProjectStatus, WorkspaceService

__all__ = [
    "CommandSpec",
    "ProjectStatus",
    "WorkspacePathError",
    "WorkspacePaths",
    "WorkspaceService",
    "discover_commands",
    "expand_command",
    "resolve_under_workspace",
    "safe_segment",
]
