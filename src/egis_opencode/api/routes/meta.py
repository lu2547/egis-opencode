"""GET /api/coding/agents + /health — agent 模式元信息与健康检查。

聚合 ``agents/*/agent.json`` 的 modes（coding 的 build/plan 固定在前，
其余按目录字母序）—— 新增业务 agent 只需放一个 agent.json，前端
模式切换器即自动出现。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ..deps import get_ctx

logger = logging.getLogger(__name__)

router = APIRouter()

_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"

#: coding 是底座主入口，其 build/plan 排最前
_LEAD_AGENT = "coding"


@router.get("/agents")
async def agents(request: Request) -> dict[str, Any]:
    """各 agent 模式元信息聚合（前端模式切换数据源）。"""
    ctx = get_ctx(request)
    registry = ctx.agent_registry

    modes: list[dict[str, Any]] = []
    for agent_dir in _iter_agent_dirs():
        for mode in _load_modes(agent_dir):
            mode["available"] = (
                registry is not None
                and mode.get("agent_id", "") in registry.list_ids()
            )
            modes.append(mode)
    return {"modes": modes}


def _iter_agent_dirs() -> list[Path]:
    """agent 目录迭代：coding 在前（底座主入口），其余字母序。"""
    if not _AGENTS_DIR.is_dir():
        return []
    dirs = sorted(
        (d for d in _AGENTS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")),
        key=lambda d: d.name,
    )
    lead = _AGENTS_DIR / _LEAD_AGENT
    if lead in dirs:
        dirs.remove(lead)
        dirs.insert(0, lead)
    return dirs


def _load_modes(agent_dir: Path) -> list[dict[str, Any]]:
    meta_file = agent_dir / "agent.json"
    if not meta_file.is_file():
        return []
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("agent.json unreadable: %s", meta_file)
        return []
    modes = meta.get("modes")
    return modes if isinstance(modes, list) else []


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["router"]
