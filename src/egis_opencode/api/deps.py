"""FastAPI 依赖 — AgentRegistry / agent / 共享服务解析。

与 ark ``plugins/api/deps.py`` 同构：请求期从 ``app.state.ctx`` 解析，
不做模块级单例快照。
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from ark_agentic.core.protocol.app_context import AppContext
from ark_agentic.core.runtime.base_agent import BaseAgent
from ark_agentic.core.runtime.registry import AgentRegistry

from ..permissions.service import permission_service
from ..sessions.title import title_store
from ..workspace.service import WorkspaceService

#: coding 主 agent（build 模式）的 agent_id；会话列表等场景固定用它
CODING_AGENT_ID = "coding"


def get_ctx(request: Request) -> AppContext:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="AppContext not initialised")
    return ctx


def get_registry(request: Request) -> AgentRegistry:
    ctx = get_ctx(request)
    if ctx.agent_registry is None:
        raise HTTPException(status_code=503, detail="AgentRegistry not started")
    return ctx.agent_registry


def get_agent(request: Request, agent_id: str) -> BaseAgent:
    try:
        return get_registry(request).get(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")


def get_coding_agent(request: Request) -> BaseAgent:
    """会话/工作区等非 chat 场景统一走 build 主 agent。"""
    return get_agent(request, CODING_AGENT_ID)


def get_workspace_service() -> WorkspaceService:
    return WorkspaceService()


__all__ = [
    "CODING_AGENT_ID",
    "get_agent",
    "get_coding_agent",
    "get_ctx",
    "get_registry",
    "get_workspace_service",
    "permission_service",
    "title_store",
]
