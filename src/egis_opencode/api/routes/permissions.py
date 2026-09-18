"""GET /api/coding/permissions/* — 权限请求应答端点。

SSE 流上 guard 发出 ``permission_request`` 事件后 await Future；
本端点收到用户应答即落定 Future（guard 侧继续/短路工具执行）。
``GET /pending`` 供前端断线重连后轮询兜底。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...permissions.service import (
    UnknownPermissionRequest,
    permission_service,
)
from ..schemas import RespondRequest

router = APIRouter()


@router.get("/permissions/pending")
async def pending_permissions(session_id: str) -> list[dict[str, Any]]:
    """某会话所有未落定的权限请求（SSE 断线时的轮询兜底）。"""
    return [
        req.to_payload()
        for req in permission_service.pending_for_session(session_id)
    ]


@router.post("/permissions/{request_id}/respond")
async def respond_permission(
    request_id: str, body: RespondRequest,
) -> dict[str, Any]:
    """用户应答：once（允许一次）/ always（本次 run 内总是允许）/ reject。"""
    try:
        request = permission_service.respond(request_id, body.action)
    except UnknownPermissionRequest:
        raise HTTPException(
            status_code=404,
            detail=f"Permission request not found or already resolved: {request_id}",
        )
    return {
        "request": request.to_payload(),
        "action": body.action,
    }


__all__ = ["router"]
