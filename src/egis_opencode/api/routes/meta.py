"""GET /api/coding/agents + /health — 模式元信息与健康检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ..deps import get_ctx

router = APIRouter()

_AGENT_META_FILE = Path(__file__).resolve().parents[2] / "agents" / "coding" / "agent.json"


@router.get("/agents")
async def agents(request: Request) -> dict[str, Any]:
    """build/plan 模式元信息（前端模式切换数据源）。"""
    ctx = get_ctx(request)
    registry = ctx.agent_registry
    meta: dict[str, Any] = {"modes": []}
    if _AGENT_META_FILE.is_file():
        try:
            meta = json.loads(_AGENT_META_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"modes": []}
    # 叠加 registry 里的实时注册状态
    for mode in meta.get("modes", []):
        agent_id = mode.get("agent_id", "")
        if registry is not None:
            mode["available"] = agent_id in registry.list_ids()
        else:
            mode["available"] = False
    return meta


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["router"]
