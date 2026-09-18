"""CodingPlugin — egis-opencode 的 HTTP 装配插件（BasePlugin）。

职责：
1. ``install_routes``：挂载 /api/coding 前缀的全部路由（chat SSE + REST）。
2. ``start``：SandboxPlugin（先于本插件注册）产出的 SandboxManager 绑定到
   ``sandbox_binding``，供 BashTool 延迟取用；返回本插件自身供诊断。
3. ``stop``：解绑沙箱引用。
"""

from __future__ import annotations

import logging
from typing import Any

from ark_agentic.core.protocol.plugin import BasePlugin

from ..core.tools.bash import sandbox_binding
from . import chat
from .routes import commands, meta, permissions, questions, sessions, workspaces

logger = logging.getLogger(__name__)


class CodingPlugin(BasePlugin):
    """egis-opencode REST/SSE 入口。"""

    name = "coding"

    def is_enabled(self) -> bool:
        return True

    def install_routes(self, app: Any) -> None:
        from fastapi import APIRouter

        router = APIRouter(prefix="/api/coding")
        router.include_router(chat.router)
        router.include_router(sessions.router)
        router.include_router(permissions.router)
        router.include_router(questions.router)
        router.include_router(workspaces.router)
        router.include_router(commands.router)
        router.include_router(meta.router)
        app.include_router(router)
        logger.info("CodingPlugin routes mounted at /api/coding")

    async def start(self, ctx: Any) -> Any:
        manager = getattr(ctx, "sandbox", None)
        if manager is not None:
            sandbox_binding.bind(manager)
            logger.info("SandboxManager bound to coding bash tool")
        else:
            logger.warning(
                "SandboxManager not present on ctx — bash tool disabled "
                "(check ENABLE_SANDBOX and component order)"
            )
        return None

    async def stop(self) -> None:
        sandbox_binding.reset()


__all__ = ["CodingPlugin"]
