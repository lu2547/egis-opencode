"""GET/DELETE /api/coding/sessions — 会话列表 / 历史 / 删除。

列表 = ark ``SessionManager.list_user_session_metas`` + egis 自有标题存储
合并；历史消息渲染为前端可直接回放的 UI 消息结构。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ...permissions.service import permission_service
from ...sessions.title import title_store
from ...sessions.workspace_binding import workspace_binding_store
from ..deps import get_coding_agent

router = APIRouter()


@router.get("/sessions")
async def list_sessions(
    request: Request, user_id: str, limit: int = 100,
) -> list[dict[str, Any]]:
    """会话列表（按更新时间倒序，title 缺省回落首条用户消息截断）。"""
    agent = get_coding_agent(request)
    metas = await agent.session_manager.list_user_session_metas(user_id)
    titles = title_store.titles()
    sessions = []
    for meta in sorted(metas, key=lambda m: m.updated_at, reverse=True)[:limit]:
        sessions.append(
            {
                "session_id": meta.session_id,
                "title": titles.get(meta.session_id, ""),
                "updated_at": meta.updated_at,
                "model": meta.model,
                "total_tokens": meta.total_tokens,
            }
        )
    return sessions


@router.get("/sessions/{session_id}/messages")
async def session_messages(
    session_id: str, request: Request, user_id: str,
) -> list[dict[str, Any]]:
    """历史消息（user/assistant + tool_calls/tool_results 结构化回放）。"""
    agent = get_coding_agent(request)
    messages = await agent.session_manager.load_session_messages(
        session_id, user_id,
    )
    rendered: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {
            "role": msg.role.value,
            "content": msg.content or "",
        }
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        if msg.tool_results:
            entry["tool_results"] = [
                {
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                    "llm_digest": tr.llm_digest,
                }
                for tr in msg.tool_results
            ]
        if msg.thinking:
            entry["thinking"] = msg.thinking
        rendered.append(entry)
    return rendered


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, request: Request, user_id: str,
) -> dict[str, Any]:
    """删除会话（ark session + 标题 + 挂起权限请求清理）。"""
    agent = get_coding_agent(request)
    deleted = await agent.session_manager.delete_session(session_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    title_store.delete(session_id)
    workspace_binding_store.delete(session_id)
    for pending in permission_service.pending_for_session(session_id):
        try:
            permission_service.respond(pending.request_id, "reject")
        except Exception:  # noqa: BLE001 — 清理 best-effort
            pass
    return {"session_id": session_id, "deleted": True}


__all__ = ["router"]
