"""工作目录本地绑定 + slash 命令集成测试。

覆盖：bind/binding REST、chat 携带 workspace_root（工具锚定本地目录、
权限审批链路不变）、/ingest 展开进 user 消息、commands 列表端点、
.env 默认工作目录回落（前端不传 workspace_root 时的服务端兑底）。
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

from egis_opencode.config import settings
from egis_opencode.permissions.rules import Rule

from .conftest import chat_payload
from .test_chat_sse import _auto_responder, _consume_sse, _tool_call_message


@pytest.fixture
def local_wiki(tmp_path: Path) -> Path:
    """模拟 llm-wiki 本地目录：命令 + AGENTS.md + raw 素材。"""
    d = tmp_path / "llm-wiki"
    commands = d / ".opencode" / "commands"
    commands.mkdir(parents=True)
    (commands / "ingest.md").write_text(
        "---\ndescription: 编译 raw 到 wiki\n---\n\n"
        "# ingest 命令\n\n处理参数：$ARGUMENTS",
        encoding="utf-8",
    )
    (d / "AGENTS.md").write_text(
        "# 规范\n\n- 始终简体中文\n- raw/ 只读", encoding="utf-8",
    )
    (d / "raw").mkdir()
    (d / "raw" / "article.md").write_text("素材内容", encoding="utf-8")
    return d


def _binding(raw: str) -> str:
    return raw


# ── bind / binding REST ────────────────────────────────


async def test_bind_local_dir_returns_status(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/workspaces/bind",
            json={
                "user_id": "alice",
                "session_id": "sess-bind",
                "workspace_root": f"local:{local_wiki}",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace_root"] == f"local:{local_wiki}"
    assert body["status"]["name"] == "llm-wiki"
    assert body["status"]["is_git_repo"] is False


async def test_bind_rejects_missing_dir(make_app, tmp_path):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/workspaces/bind",
            json={
                "user_id": "alice",
                "workspace_root": f"local:{tmp_path / 'nope'}",
            },
        )
    assert resp.status_code == 400
    assert "目录不存在" in resp.json()["detail"]


async def test_binding_roundtrip_and_unbind(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/api/coding/workspaces/bind",
            json={
                "user_id": "alice",
                "session_id": "sess-rt",
                "workspace_root": f"local:{local_wiki}",
            },
        )
        resp = await client.get(
            "/api/coding/workspaces/binding",
            params={"session_id": "sess-rt"},
        )
        assert resp.json()["workspace_root"] == f"local:{local_wiki}"

        # 解绑
        await client.post(
            "/api/coding/workspaces/bind",
            json={
                "user_id": "alice",
                "session_id": "sess-rt",
                "workspace_root": "",
            },
        )
        resp = await client.get(
            "/api/coding/workspaces/binding",
            params={"session_id": "sess-rt"},
        )
        assert resp.json()["workspace_root"] == ""


# ── chat 锚定：工具落本地目录，权限链路不变 ───────────────


async def test_chat_local_binding_write_lands_in_local_dir(
    make_app, ws_user_root, local_wiki,
):
    app, _llm = make_app(
        responses=[
            _tool_call_message("write", path="wiki/new.md", content="新页面"),
            AIMessage(content="已写入。"),
        ],
        extra_rules=[Rule(permission="write", pattern="*", action="ask")],
    )
    payload = chat_payload("写 wiki/new.md", session_id="sess-local-write")
    payload["workspace_root"] = f"local:{local_wiki}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        events = await _consume_sse(
            client, payload, auto_respond="once",
        )

    # 文件落在本地目录（非多租户 workspace）
    assert (local_wiki / "wiki" / "new.md").read_text(encoding="utf-8") == "新页面"
    assert not (ws_user_root / "wiki").exists()
    # 权限审批链路照常（注入 write=ask 规则）
    requests = [
        e for e in events
        if e.get("type") == "custom" and e.get("custom_type") == "permission_request"
    ]
    assert requests


async def test_chat_invalid_binding_rejected(make_app, tmp_path):
    app, _ = make_app()
    payload = chat_payload("hello", session_id="sess-bad", stream=False)
    payload["workspace_root"] = f"local:{tmp_path / 'missing'}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/api/coding/chat", json=payload)
    assert resp.status_code == 400


async def test_chat_binding_persisted_for_next_request(
    make_app, local_wiki, ws_user_root,
):
    """首次带绑定，后续请求不带队列也沿用（store 持久化）。"""
    # responses 序列：[run1 回复, 标题生成, run2 read 调用, run2 回复]
    # （TitleGenerator 与主 LLM 共享 MockChatModel，标题会消费一条响应）
    app, llm = make_app(responses=[
        AIMessage(content="好的。"),
        AIMessage(content="读素材会话"),
        _tool_call_message("read", path="raw/article.md"),
        AIMessage(content="读到了。"),
    ])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # 第一次：绑定（非流式）
        payload = chat_payload("hi", session_id="sess-persist", stream=False)
        payload["workspace_root"] = f"local:{local_wiki}"
        resp = await client.post("/api/coding/chat", json=payload)
        assert resp.status_code == 200

        # 第二次：不携带 workspace_root，read 应命中本地目录文件
        resp = await client.post(
            "/api/coding/chat",
            json=chat_payload("读素材", session_id="sess-persist", stream=False),
        )
        assert resp.status_code == 200

        # read 工具结果落在会话消息里：命中本地目录的素材
        messages = await client.get(
            "/api/coding/sessions/sess-persist/messages",
            params={"user_id": "alice"},
        )
        tool_texts = [
            tr["content"]
            for m in messages.json()
            for tr in (m.get("tool_results") or [])
        ]
        assert any("素材内容" in str(t) for t in tool_texts), tool_texts


# ── slash 命令展开（chat 全链路）────────────────────


async def test_chat_slash_command_expanded_into_user_message(
    make_app, local_wiki,
):
    app, _ = make_app(responses=[AIMessage(content="开始处理。")])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        payload = chat_payload(
            "/ingest raw/article.md",
            session_id="sess-cmd", stream=False,
        )
        payload["workspace_root"] = f"local:{local_wiki}"
        resp = await client.post("/api/coding/chat", json=payload)
        assert resp.status_code == 200

        # user 消息 = 展开后的命令全文 + AGENTS.md 上下文
        messages = await client.get(
            f"/api/coding/sessions/sess-cmd/messages",
            params={"user_id": "alice"},
        )
        user_messages = [
            m for m in messages.json() if m["role"] == "user"
        ]
        assert "ingest 命令" in user_messages[0]["content"]
        assert "raw/article.md" in user_messages[0]["content"]
        assert "raw/ 只读" in user_messages[0]["content"]


async def test_chat_plain_message_not_expanded(make_app, local_wiki):
    app, _ = make_app(responses=[AIMessage(content="好的。")])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/chat",
            json=chat_payload("普通消息", session_id="sess-plain", stream=False),
        )
        assert resp.status_code == 200
        messages = await client.get(
            "/api/coding/sessions/sess-plain/messages",
            params={"user_id": "alice"},
        )
        assert messages.json()[0]["content"] == "普通消息"


# ── commands 列表端点 ──────────────────────────────────


async def test_commands_endpoint_lists_local_commands(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/commands",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
            },
        )
    assert resp.status_code == 200
    commands = resp.json()
    assert [c["name"] for c in commands] == ["ingest"]
    assert commands[0]["description"] == "编译 raw 到 wiki"


async def test_commands_endpoint_default_user_root(make_app, ws_user_root):
    """无绑定：回落用户 workspace 根（无命令目录 → 空列表）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/commands", params={"user_id": "alice"},
        )
    assert resp.status_code == 200
    assert resp.json() == []


# ── .env 默认工作目录（前端不传 workspace_root 的服务端兑底）──


@contextlib.contextmanager
def _default_workspace(mode: str, directory: str):
    """临时覆盖默认工作目录配置（frozen settings 整体换 __dict__）。"""
    old = dict(settings.__dict__)
    object.__setattr__(
        settings, "__dict__",
        dataclasses.replace(
            settings,
            default_workspace_mode=mode,
            default_workspace_dir=directory,
        ).__dict__,
    )
    try:
        yield
    finally:
        object.__setattr__(settings, "__dict__", old)


async def test_binding_falls_back_to_env_default(make_app, local_wiki):
    """无显式绑定的会话：binding 查询返回 .env 默认工作目录。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(local_wiki)):
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-no-binding"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace_root"] == f"local:{local_wiki}"
        assert body["status"]["name"] == "llm-wiki"
        # 未持久化：会话绑定存储仍为空（.env 变更重启后自动跟随）
        from egis_opencode.sessions.workspace_binding import workspace_binding_store
        assert not workspace_binding_store.get("sess-no-binding")


async def test_chat_without_workspace_root_uses_env_default(
    make_app, local_wiki, ws_user_root,
):
    """前端不传 workspace_root：chat 锚定 .env 默认工作目录。"""
    app, _llm = make_app(responses=[
        _tool_call_message("read", path="raw/article.md"),
        AIMessage(content="读到了。"),
        AIMessage(content="标题"),
    ])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(local_wiki)):
            resp = await client.post(
                "/api/coding/chat",
                json=chat_payload(
                    "读素材", session_id="sess-env-default", stream=False,
                ),
            )
            assert resp.status_code == 200
            messages = await client.get(
                "/api/coding/sessions/sess-env-default/messages",
                params={"user_id": "alice"},
            )
    # read 命中默认目录的素材（非多租户 workspace）
    tool_texts = [
        tr["content"]
        for m in messages.json()
        for tr in (m.get("tool_results") or [])
    ]
    assert any("素材内容" in str(t) for t in tool_texts), tool_texts


async def test_unbind_falls_back_to_env_default(make_app, local_wiki, tmp_path):
    """解绑 = 清除显式绑定，回落 .env 默认（与 binding 查询同源）。"""
    other = tmp_path / "other-proj"
    other.mkdir()
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(local_wiki)):
            # 显式绑定到另一个目录
            await client.post(
                "/api/coding/workspaces/bind",
                json={
                    "user_id": "alice",
                    "session_id": "sess-unbind",
                    "workspace_root": f"local:{other}",
                },
            )
            # 解绑：响应回落 .env 默认（而非空串）
            resp = await client.post(
                "/api/coding/workspaces/bind",
                json={
                    "user_id": "alice",
                    "session_id": "sess-unbind",
                    "workspace_root": "",
                },
            )
            assert resp.json()["workspace_root"] == f"local:{local_wiki}"
            # 后续 binding 查询同源：也返回默认目录
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-unbind"},
            )
            assert resp.json()["workspace_root"] == f"local:{local_wiki}"


async def test_invalid_default_dir_falls_back_to_multi(make_app, tmp_path):
    """默认目录不存在：静默回落多租户（binding 返回空串）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(tmp_path / "missing")):
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-bad-default"},
            )
        assert resp.status_code == 200
        assert resp.json()["workspace_root"] == ""


async def test_default_mode_multi_keeps_legacy_behavior(make_app):
    """mode=multi（默认）：无绑定空串多租户 —— 旧版行为不变。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("multi", ""):
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-multi"},
            )
        assert resp.status_code == 200
        assert resp.json()["workspace_root"] == ""


async def test_default_endpoint_returns_env_default(make_app, local_wiki):
    """GET /workspaces/default：直接返回 .env 默认（is_default=True）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(local_wiki)):
            resp = await client.get("/api/coding/workspaces/default")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace_root"] == f"local:{local_wiki}"
        assert body["is_default"] is True
        assert body["status"]["name"] == "llm-wiki"


async def test_default_endpoint_empty_when_not_configured(make_app):
    """未配置默认：空串 + is_default=False（前端回落多租户展示）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("multi", ""):
            resp = await client.get("/api/coding/workspaces/default")
        assert resp.status_code == 200
        assert resp.json() == {
            "workspace_root": "", "status": None, "is_default": False,
        }


async def test_binding_default_fallback_flagged(make_app, local_wiki):
    """binding 回落到默认时 is_default=True（区别于显式绑定）。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        with _default_workspace("local", str(local_wiki)):
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-flag"},
            )
            assert resp.json()["is_default"] is True
            # 显式绑定压过默认且 is_default=False
            await client.post(
                "/api/coding/workspaces/bind",
                json={
                    "user_id": "alice",
                    "session_id": "sess-flag",
                    "workspace_root": f"local:{local_wiki}",
                },
            )
            resp = await client.get(
                "/api/coding/workspaces/binding",
                params={"session_id": "sess-flag"},
            )
            assert resp.json()["is_default"] is False
