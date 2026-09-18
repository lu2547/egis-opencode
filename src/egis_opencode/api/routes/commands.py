"""GET /api/coding/commands — 工作目录下的 slash 命令列表。

命令发现目录与 chat 展开一致：会话绑定目录（local:）优先，
否则用户 workspace 根（多租户模式）。前端 "/" 面板据此渲染。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ...workspace import discover_commands
from ...workspace.service import WorkspaceService
from ..schemas import CommandInfo

router = APIRouter()


@router.get("/commands")
async def list_commands(user_id: str, workspace_root: str = "") -> list[CommandInfo]:
    """slash 命令列表（.opencode/commands + .claude/commands）。"""
    root = _effective_root(user_id, workspace_root)
    commands = discover_commands(root) if root is not None else {}
    return [
        CommandInfo(name=spec.name, description=spec.description)
        for _, spec in sorted(commands.items())
    ]


def _effective_root(user_id: str, workspace_root: str) -> Path | None:
    """命令发现根：显式绑定串 > 会话已存绑定 > 用户 workspace 根。"""
    service = WorkspaceService()
    if workspace_root:
        try:
            return service.resolve_binding(workspace_root)
        except Exception:  # noqa: BLE001 — 无效绑定回落默认根
            pass
    return service.paths_for(user_id).ensure_user_root()


__all__ = ["list_commands", "router"]
