"""POST /api/coding/chat SSE 全链路集成测试。

脚本化 LLM（MockChatModel）驱动 ReAct：
permission_request SSE 帧 → REST respond → 工具真实执行 →
permission_resolved / tool_digest / run_finished。

应答通道说明：httpx ASGITransport 会缓冲整个 SSE 响应（帧在 run
结束后才可读），测试不能依赖 permission_request 帧触发应答 ——
改走 REST pending 轮询（产品为断线客户端设计的兑底通道），
与真实前端的应答路径等价。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

from egis_opencode.events import (
    PERMISSION_REQUEST,
    PERMISSION_RESOLVED,
    TITLE_GENERATED,
    TOOL_DIGEST,
    TODO_UPDATE,
)
from egis_opencode.permissions.rules import Rule

from .conftest import chat_payload

#: ask 流程测试的收紧规则（默认预设全放行）
WRITE_ASK = [Rule(permission="write", pattern="*", action="ask")]


def _tool_call_message(name: str, **args) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
    )


async def _auto_responder(
    client: httpx.AsyncClient,
    session_id: str,
    action: str,
) -> str | None:
    """轮询 pending 权限请求并应答首个，返回 request_id。"""
    for _ in range(600):
        resp = await client.get(
            "/api/coding/permissions/pending",
            params={"session_id": session_id},
        )
        pendings = resp.json()
        if pendings:
            request_id = pendings[0]["request_id"]
            await client.post(
                f"/api/coding/permissions/{request_id}/respond",
                json={"action": action},
            )
            return request_id
        await asyncio.sleep(0.01)
    return None


async def _consume_sse(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    auto_respond: str | None = None,
) -> list[dict[str, Any]]:
    """消费 SSE 流；auto_respond 时并发轮询应答首个权限请求。

    responder 必须在 stream 之前启动：ASGITransport 缓冲整个响应，
    stream 调用会阻塞到 run 结束，期间靠 responder 并发应答解锁。
    """
    events: list[dict[str, Any]] = []
    responder: asyncio.Task[str | None] | None = None
    if auto_respond is not None:
        responder = asyncio.create_task(
            _auto_responder(client, payload["session_id"], auto_respond),
        )
    try:
        async with client.stream("POST", "/api/coding/chat", json=payload) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[len("data: "):])
                except ValueError:
                    continue
                events.append(event)
    finally:
        if responder is not None:
            request_id = await responder
            assert request_id is not None, "permission request never appeared"
    return events


def _customs(events: list[dict], custom_type: str) -> list[dict]:
    return [
        e["custom_data"] for e in events
        if e.get("type") == "custom" and e.get("custom_type") == custom_type
    ]


# ── 全链路：ask → respond(once) → 工具执行 → run_finished ──


async def test_sse_permission_flow_write_once(make_app, ws_user_root):
    """write 工具（注入 ask 规则）→ 审批 → 文件落盘 → 流收尾。"""
    app, _llm = make_app(
        responses=[
            _tool_call_message("write", path="hello.txt", content="hello egis"),
            AIMessage(content="文件已创建。"),
        ],
        extra_rules=WRITE_ASK,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        events = await _consume_sse(
            client,
            chat_payload("创建 hello.txt", session_id="sess-write-once"),
            auto_respond="once",
        )

    # 权限请求事件（含完整审批信息）
    requests = _customs(events, PERMISSION_REQUEST)
    assert len(requests) == 1
    assert requests[0]["permission"] == "write"
    assert requests[0]["tool_name"] == "write"
    assert requests[0]["tool_args"] == {"path": "hello.txt", "content": "hello egis"}
    assert requests[0]["session_id"]  # 路由到正确会话

    # 落定事件
    resolved = _customs(events, PERMISSION_RESOLVED)
    assert resolved[0]["action"] == "once"

    # 工具真实执行
    assert (ws_user_root / "hello.txt").read_text(encoding="utf-8") == "hello egis"

    # 工具卡片事件（write 的 file_edit digest）
    digests = _customs(events, TOOL_DIGEST)
    file_edits = [d for d in digests if d["display_type"] == "file_edit"]
    assert any(d["status"] == "success" for d in file_edits)

    # run 终态
    assert any(e.get("type") == "run_finished" for e in events)
    # 会话 id 回传（新建会话）
    finished = next(e for e in events if e.get("type") == "run_finished")
    assert finished.get("session_id")


async def test_sse_permission_rejected_skips_tool(make_app, ws_user_root):
    """reject：工具不执行，LLM 收到 Permission denied 结果后收尾。"""
    app, _llm = make_app(
        responses=[
            _tool_call_message("write", path="evil.txt", content="x"),
            AIMessage(content="好的，我换个方式。"),
        ],
        extra_rules=WRITE_ASK,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        events = await _consume_sse(
            client,
            chat_payload("写文件", session_id="sess-reject-1"),
            auto_respond="reject",
        )

    assert not (ws_user_root / "evil.txt").exists()
    digests = _customs(events, TOOL_DIGEST)
    denied = [d for d in digests if d["status"] == "denied"]
    assert denied, "denied digest 应发出"
    assert any(e.get("type") == "run_finished" for e in events)


async def test_sse_allow_tool_without_permission_prompt(make_app, ws_user_root):
    """read（allow）不打扰用户：无 permission_request，直接执行。"""
    (ws_user_root / "note.md").write_text("# note\n")
    app, _llm = make_app(responses=[
        _tool_call_message("read", path="note.md"),
        AIMessage(content="读到了。"),
    ])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        events = await _consume_sse(client, chat_payload("读 note.md"))

    assert _customs(events, PERMISSION_REQUEST) == []
    assert any(e.get("type") == "run_finished" for e in events)
    # read 的 search digest
    assert any(
        d["tool_name"] == "read"
        for d in _customs(events, TOOL_DIGEST)
    )


async def test_sse_todo_update_event(make_app):
    """todo 工具 → todo_update 事件契约。"""
    app, _llm = make_app(responses=[
        _tool_call_message("todo", todos=[
            {"id": "1", "content": "步骤一", "status": "in_progress"},
        ]),
        AIMessage(content="开始。"),
    ])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        events = await _consume_sse(client, chat_payload("列任务"))

    updates = _customs(events, TODO_UPDATE)
    assert len(updates) == 1
    assert updates[0]["todos"][0]["content"] == "步骤一"
    assert updates[0]["todos"][0]["status"] == "in_progress"


async def test_sse_title_generated_event(make_app, titles_path):
    """after_agent 标题生成 → title_generated 事件 + 落盘。"""
    app, _llm = make_app(responses=[AIMessage(content="答案")])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        events = await _consume_sse(client, chat_payload("帮我规划重构"))

    titles = _customs(events, TITLE_GENERATED)
    assert len(titles) == 1
    assert titles[0]["session_id"]
    # title LLM 是同一个 MockChatModel（responses 只有 1 条 → 回落 user_input）
    assert titles[0]["title"]  # 回落标题非空


# ── 非流式 ─────────────────────────────────────────────


async def test_nonstream_chat_runs_agent(make_app, ws_user_root):
    """stream=False：JSON 响应，工具执行，response 为最终文本。"""
    (ws_user_root / "a.txt").write_text("content-123")
    app, _llm = make_app(responses=[
        _tool_call_message("read", path="a.txt"),
        AIMessage(content="文件内容是 content-123"),
    ])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/chat", json=chat_payload("读 a.txt", stream=False),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "文件内容是 content-123"
    assert body["session_id"]
    assert body["turns"] >= 2
    # tool_calls 为 run 累积语义（LoopStats.all_tool_calls）：含已执行的 read
    assert [tc["name"] for tc in body["tool_calls"]] == ["read"]
    # 非流式无交互通道：guard 对 allow 工具直接放行（read），无 pending 残留


# ── 错误路径 ───────────────────────────────────────────


async def test_unknown_agent_404(make_app):
    app, _llm = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/chat",
            json=chat_payload("hi", agent_id="no-such", stream=False),
        )
    assert resp.status_code == 404


async def test_concurrent_session_rejected_409(make_app):
    """同 session 并发：第二个请求 409（第一个挂起等待审批时）。"""
    app, llm = make_app(
        responses=[
            _tool_call_message("write", path="x.txt", content="x"),
            AIMessage(content="done"),
        ],
        extra_rules=WRITE_ASK,
    )
    session_id = "sess-concurrent-1"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # 第一个 run：SSE 消费中（遇到权限请求挂起）
        sse_task = asyncio.create_task(_consume_sse(
            client, chat_payload("写文件", session_id=session_id),
        ))
        # 等待 pending 权限请求出现
        rid = None
        for _ in range(300):
            resp = await client.get(
                "/api/coding/permissions/pending", params={"session_id": session_id},
            )
            pendings = resp.json()
            if pendings:
                rid = pendings[0]["request_id"]
                break
            await asyncio.sleep(0.01)
        assert rid is not None, "permission request should appear"

        # 第二个请求 → 409
        resp = await client.post(
            "/api/coding/chat",
            json=chat_payload("again", session_id=session_id, stream=False),
        )
        assert resp.status_code == 409

        # 落定第一个 run（reject）并消费完
        await client.post(
            f"/api/coding/permissions/{rid}/respond", json={"action": "reject"},
        )
        events = await sse_task
        assert any(e.get("type") == "run_finished" for e in events)


async def test_abort_cancels_running_chat(make_app):
    """POST /chat/abort 取消挂起审批中的 run → emit failed("run aborted")。"""
    app, llm = make_app(
        responses=[
            _tool_call_message("write", path="y.txt", content="y"),
            AIMessage(content="done"),
        ],
        extra_rules=WRITE_ASK,
    )
    session_id = "sess-abort-1"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        sse_task = asyncio.create_task(_consume_sse(
            client, chat_payload("写文件", session_id=session_id),
        ))
        rid = None
        for _ in range(300):
            resp = await client.get(
                "/api/coding/permissions/pending", params={"session_id": session_id},
            )
            if resp.json():
                rid = resp.json()[0]["request_id"]
                break
            await asyncio.sleep(0.01)
        assert rid is not None

        resp = await client.post(
            "/api/coding/chat/abort", json={"session_id": session_id},
        )
        assert resp.status_code == 200
        assert resp.json()["aborted"] is True

        events = await sse_task
        assert any(e.get("type") == "run_error" for e in events)


async def test_abort_idle_session_returns_false(make_app):
    app, _llm = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/api/coding/chat/abort", json={"session_id": "idle-session"},
        )
    assert resp.status_code == 200
    assert resp.json()["aborted"] is False
