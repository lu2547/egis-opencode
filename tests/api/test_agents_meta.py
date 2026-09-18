"""GET /agents 聚合 + GET /commands?agent_id — 多 agent 元数据链路测试。

GET /agents 扫 agents/*/agent.json 文件系统聚合 modes（wiki-agent 的
单模式 + coding 的 build/plan）；GET /commands 带 agent_id 时合入该
agent 的内置 commands。
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ark_agentic.core.session import SessionManager

from egis_opencode.agents.wiki_agent import WikiAgent

from tests.helpers import MockChatModel


async def test_agents_endpoint_aggregates_all_agent_json(make_app):
    """modes 聚合 coding（build/plan 在前）+ 其余 agent 字母序（bid/wiki）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/api/coding/agents")
    assert resp.status_code == 200
    modes = resp.json()["modes"]
    # coding 底座在前，其余按 agent 目录字母序（bid_agent < wiki_agent）
    assert [m["mode"] for m in modes] == ["build", "plan", "bid", "wiki"]
    assert modes[0]["agent_id"] == "coding"
    assert modes[2]["agent_id"] == "bid"
    assert modes[3]["agent_id"] == "wiki"
    # registry 实时状态叠加：make_app 只注册 coding → 业务 agent 不可用
    assert modes[0]["available"] is True
    assert modes[2]["available"] is False
    assert modes[3]["available"] is False


async def test_agents_endpoint_wiki_available_when_registered(make_app, tmp_path: Path):
    """注册 WikiAgent 后 available 翻转 True。"""
    app, _ = make_app()
    wiki = WikiAgent._construct(
        llm=MockChatModel(responses=[]),
        session_manager=SessionManager(tmp_path / "sessions", agent_id="wiki"),
        agent_id="wiki",
    )
    app.state.ctx.agent_registry.register("wiki", wiki)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/api/coding/agents")
    modes = resp.json()["modes"]
    assert next(m for m in modes if m["mode"] == "wiki")["available"] is True


async def test_commands_with_agent_id_merges_builtin(make_app, tmp_path: Path):
    """带 agent_id=wiki：内置 ingest/lint/query 进面板；不带则无。"""
    app, _ = make_app()
    wiki = WikiAgent._construct(
        llm=MockChatModel(responses=[]),
        session_manager=SessionManager(tmp_path / "sessions", agent_id="wiki"),
        agent_id="wiki",
    )
    app.state.ctx.agent_registry.register("wiki", wiki)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with_builtin = await client.get(
            "/api/coding/commands",
            params={"user_id": "alice", "agent_id": "wiki"},
        )
        without_agent = await client.get(
            "/api/coding/commands",
            params={"user_id": "alice"},
        )
        unknown_agent = await client.get(
            "/api/coding/commands",
            params={"user_id": "alice", "agent_id": "nope"},
        )
    assert with_builtin.status_code == 200
    names = {c["name"] for c in with_builtin.json()}
    assert {"ingest", "query", "lint"} <= names
    # 不带 agent_id：纯 workspace 命令（测试 workspace 为空）
    assert without_agent.status_code == 200
    assert without_agent.json() == []
    # 未知 agent：回落纯 workspace 命令，不报错
    assert unknown_agent.status_code == 200
    assert unknown_agent.json() == []


async def test_commands_builtin_survives_binding(make_app, tmp_path: Path):
    """本地目录绑定时内置命令仍可见（平台命令不随工作目录变）。"""
    app, _ = make_app()
    wiki = WikiAgent._construct(
        llm=MockChatModel(responses=[]),
        session_manager=SessionManager(tmp_path / "sessions", agent_id="wiki"),
        agent_id="wiki",
    )
    app.state.ctx.agent_registry.register("wiki", wiki)

    local_dir = tmp_path / "some-project"
    local_dir.mkdir()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/commands",
            params={
                "user_id": "alice",
                "agent_id": "wiki",
                "workspace_root": f"local:{local_dir}",
            },
        )
    assert resp.status_code == 200
    assert {"ingest", "query", "lint"} <= {c["name"] for c in resp.json()}
