"""GET /api/coding/commands — slash 命令列表（agent 内置 + 工作目录）。

命令发现目录与 chat 展开一致：agent 内置 ``agents/<agent>/commands``
优先，其后会话绑定目录（local:）或用户 workspace 根（多租户模式）
下的 ``.opencode/commands`` / ``.claude/commands``。前端 "/" 面板据此渲染。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from ...workspace import discover_commands
from ...workspace.service import WorkspaceService
from ..deps import get_agent
from ..schemas import CommandInfo

router = APIRouter()


@router.get("/commands")
async def list_commands(
    user_id: str,
    workspace_root: str = "",
    agent_id: str = "",
    request: Request = None,  # type: ignore[assignment]
) -> list[CommandInfo]:
    """slash 命令列表：agent 内置命令 + 工作目录自定义命令。"""
    agent_command_dir: Path | None = None
    if agent_id:
        try:
            agent = get_agent(request, agent_id)
            agent_command_dir = getattr(agent, "command_dir", None)
        except Exception:  # noqa: BLE001 — 未知 agent 回落纯工作目录命令
            agent_command_dir = None
    root = _effective_root(user_id, workspace_root)
    commands = discover_commands(root, agent_command_dir) if root is not None else {}
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
