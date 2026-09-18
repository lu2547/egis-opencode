"""Datasource 配置契约 — DB_CONNECTION_STR 路由（PG 切换路径保障）。

egis-opencode 的 PG 切换完全依赖 ark ``Datasource.from_env()`` 的
前缀路由（app.py 构造 Bootstrap 时传入）。此测试锁住该契约：
ark 若改动路由行为，这里会先红，而不是生产环境静默回落 sqlite。
"""

from __future__ import annotations

import pytest
from ark_agentic.core.storage.datasource import Datasource


@pytest.mark.asyncio
async def test_from_env_routes_postgres(monkeypatch):
    """postgresql 前缀 → postgres 方言（engine 惰性创建，不实际连库）。"""
    monkeypatch.setenv(
        "DB_CONNECTION_STR",
        "postgresql+asyncpg://user:pw@localhost:5432/egis_opencode",
    )
    ds = Datasource.from_env()
    try:
        assert ds.dialect == "postgresql"
    finally:
        await ds.dispose()


@pytest.mark.asyncio
async def test_from_env_defaults_sqlite(monkeypatch):
    """未配置 → sqlite 方言（零配置默认，data/ark.db）。"""
    monkeypatch.delenv("DB_CONNECTION_STR", raising=False)
    ds = Datasource.from_env()
    try:
        assert ds.dialect == "sqlite"
        assert "ark.db" in str(ds.engine.url)
    finally:
        await ds.dispose()
