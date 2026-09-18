"""egis-opencode FastAPI 入口 — Bootstrap 装配薄壳。

组件顺序（关键）：SandboxPlugin 必须先于 CodingPlugin start，
CodingPlugin.start 才能把 SandboxManager 绑定给 bash 工具。
不挂 ark APIPlugin（自建 /api/coding chat 端点）。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# 项目根 .env（src/egis_opencode/app.py → 上溯两级）：无论从哪个 CWD
# 启动（uvicorn reload / IDE / systemd）都能命中；CWD 再兑底一次，
# 不覆盖已加载值。缺 .env 时所有 agent 会在 discovery 阶段构造失败被
# skip（LLM_PROVIDER 未设），服务起来但 /chat 会 404。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

from fastapi import FastAPI

from ark_agentic.core.protocol.bootstrap import Bootstrap
from ark_agentic.core.protocol.app_context import AppContext
from ark_agentic.core.storage.datasource import Datasource
from ark_agentic.plugins.mcp.plugin import MCPPlugin
from ark_agentic.plugins.sandbox.plugin import SandboxPlugin

from . import AGENTS_ROOT
from .api.plugin import CodingPlugin

logger = logging.getLogger(__name__)

_components = [
    MCPPlugin(),
    SandboxPlugin(),
    CodingPlugin(),
]
_bootstrap = Bootstrap(
    _components,
    datasource=Datasource.from_env(),
    agents_root=AGENTS_ROOT,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = AppContext()
    await _bootstrap.start(ctx)
    app.state.ctx = ctx
    try:
        yield
    finally:
        await _bootstrap.stop()


app = FastAPI(
    title="egis-opencode API",
    description="基于 ark 底座的 Coding Agent 平台",
    version="0.1.0",
    lifespan=lifespan,
)
_bootstrap.install_routes(app)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "egis_opencode.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "38083")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
