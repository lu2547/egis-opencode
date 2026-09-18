"""AGENTS.md 走 ark ``user:`` state → 系统提示 context 通道（3.2）。

注入点在 chat 端点：``input_context["user:agents_md"]`` → ark
``merge_input_context`` 合入 ``session.state`` → 每次 run
``_build_system_prompt`` 渲染进系统提示（等价 opencode instruction）。

关键语义：**每次 run 显式带键** —— merge 只 overwrite 不清除，
无文件/解绑时注入空串覆盖旧值，旧目录的规范不会残留。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from .conftest import chat_payload


async def _chat_non_stream(
    app, user_id: str, session_id: str | None, workspace_root: str | None,
) -> tuple[str, object]:
    """发一次非流式 chat，返回 (session_id, agent)。"""
    payload = chat_payload(
        "hi", user_id=user_id, session_id=session_id, stream=False,
    )
    if workspace_root is not None:
        payload["workspace_root"] = workspace_root
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/api/coding/chat", json=payload)
    assert resp.status_code == 200
    agent = app.state.ctx.agent_registry.get("coding")
    return resp.json()["session_id"], agent


async def _state_value(agent, session_id: str, user_id: str) -> object:
    entry = await agent.session_manager.load_session(session_id, user_id)
    return entry.state.get("user:agents_md")


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    d = tmp_path / "proj"
    d.mkdir()
    return d


async def test_agents_md_injected_into_session_state(
    make_app, project_dir, ws_root,
):
    """绑定本地目录后：AGENTS.md 全文进 session.state（渲染由 ark 完成）。"""
    (project_dir / "AGENTS.md").write_text(
        "# 项目规范\n- 一律用 uv 管理依赖", encoding="utf-8",
    )
    app, _ = make_app(responses=[AIMessage(content="ok")])

    session_id, agent = await _chat_non_stream(
        app, "alice", None, f"local:{project_dir}",
    )

    assert await _state_value(agent, session_id, "alice") == (
        "# 项目规范\n- 一律用 uv 管理依赖"
    )


async def test_claude_md_fallback(make_app, project_dir, ws_root):
    """无 AGENTS.md 时回落 CLAUDE.md（opencode 兼容形态）。"""
    (project_dir / "CLAUDE.md").write_text("CLAUDE spec", encoding="utf-8")
    app, _ = make_app(responses=[AIMessage(content="ok")])

    session_id, agent = await _chat_non_stream(
        app, "alice", None, f"local:{project_dir}",
    )

    assert await _state_value(agent, session_id, "alice") == "CLAUDE spec"


async def test_no_agents_md_uses_empty_string(make_app, project_dir, ws_root):
    """绑定目录但无规范文件：注入空串占位（键始终存在）。"""
    app, _ = make_app(responses=[AIMessage(content="ok")])

    session_id, agent = await _chat_non_stream(
        app, "alice", None, f"local:{project_dir}",
    )

    assert await _state_value(agent, session_id, "alice") == ""


async def test_unbind_overrides_stale_agents_md(
    make_app, project_dir, ws_root,
):
    """解绑后旧目录规范被空串覆盖，不残留到后续 run。"""
    (project_dir / "AGENTS.md").write_text("OLD SPEC", encoding="utf-8")
    app, _ = make_app(responses=[
        AIMessage(content="ok"), AIMessage(content="ok"),
    ])

    # 第一次：绑定有 AGENTS.md 的目录
    session_id, agent = await _chat_non_stream(
        app, "alice", None, f"local:{project_dir}",
    )
    assert await _state_value(agent, session_id, "alice") == "OLD SPEC"

    # 第二次：解绑（workspace_root=""）→ 空串覆盖旧值
    await _chat_non_stream(app, "alice", session_id, "")
    assert await _state_value(agent, session_id, "alice") == ""

    # 无绑定的新会话（workspace_root=None 沿用）：值仍为空串
    session_id2, _ = await _chat_non_stream(app, "alice", None, None)
    assert await _state_value(agent, session_id2, "alice") == ""


async def test_user_context_cannot_override_agents_md(
    make_app, project_dir, ws_root,
):
    """request.context 里的同名键不生效：文件是权威源（注入在后覆盖）。"""
    (project_dir / "AGENTS.md").write_text("FILE WINS", encoding="utf-8")
    app, _ = make_app(responses=[AIMessage(content="ok")])

    payload = chat_payload("hi", stream=False)
    payload["workspace_root"] = f"local:{project_dir}"
    payload["context"] = {"agents_md": "INJECTED VIA CONTEXT"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/api/coding/chat", json=payload)
    assert resp.status_code == 200
    agent = app.state.ctx.agent_registry.get("coding")
    assert await _state_value(
        agent, resp.json()["session_id"], "alice",
    ) == "FILE WINS"
