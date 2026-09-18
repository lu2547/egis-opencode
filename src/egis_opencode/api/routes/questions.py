"""GET/POST /api/coding/questions/* — question 工具应答端点。

question 工具经 executor 注入的 event_handler 发出 ``question_request``
事件后 await Future；本端点收到用户作答即落定 Future（工具侧拿到答案
继续 run）。``GET /pending`` 供前端断线重连后轮询兜底。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...questions import UnknownQuestionRequest, question_service
from ..schemas import QuestionRespondRequest

router = APIRouter()


@router.get("/questions/pending")
async def pending_questions(session_id: str) -> list[dict[str, Any]]:
    """某会话所有未落定的用户提问（SSE 断线时的轮询兜底）。"""
    return [
        req.to_payload()
        for req in question_service.pending_for_session(session_id)
    ]


@router.post("/questions/{request_id}/respond")
async def respond_question(
    request_id: str, body: QuestionRespondRequest,
) -> dict[str, Any]:
    """用户作答：answer 为自由文本或所选选项原文。"""
    try:
        request = question_service.respond(request_id, body.answer)
    except UnknownQuestionRequest:
        raise HTTPException(
            status_code=404,
            detail=f"Question request not found or already resolved: {request_id}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "request": request.to_payload(),
        "answer": body.answer,
    }


__all__ = ["router"]
