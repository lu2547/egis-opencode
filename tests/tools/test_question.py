"""question 工具测试 — 交互全链路 / 无通道自决 / 超时 / 参数校验。

工具级直测：context 注入 RecordingHandler（executor 的
``system:event_handler`` 形态）+ ``temp:session_id`` / ``temp:run_id``
（chat 端点注入形态），question_service 用独立实例（monkeypatch 替换
工具模块绑定的单例，避免进程级状态串扰）。
"""

from __future__ import annotations

import asyncio

import pytest
from ark_agentic.core.types import ToolCall

from egis_opencode.agents.coding.tools.question import QuestionTool
from egis_opencode.events import QUESTION_REQUEST, QUESTION_RESOLVED
from egis_opencode.questions import (
    QuestionService,
    UnknownQuestionRequest,
    question_service,
)

from tests.helpers import RecordingHandler


@pytest.fixture
def svc(monkeypatch: pytest.MonkeyPatch) -> QuestionService:
    """独立短超时实例替换工具模块绑定的进程级单例。"""
    service = QuestionService(timeout_seconds=2)
    monkeypatch.setattr(
        "egis_opencode.agents.coding.tools.question.question_service", service,
    )
    return service


def _tc(**arguments) -> ToolCall:
    return ToolCall.create("question", arguments)


def _ctx(recorder: RecordingHandler, *, with_channel: bool = True):
    ctx: dict = {}
    if with_channel:
        ctx["system:event_handler"] = recorder
        ctx["temp:session_id"] = "sess-q"
        ctx["temp:run_id"] = "run-q"
    return ctx


async def _respond_when_pending(svc: QuestionService, answer: str) -> str:
    """轮询等 pending 提问出现后作答，返回 request_id。"""
    for _ in range(300):
        pendings = svc.pending_for_session("sess-q")
        if pendings:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("question request never appeared")
    request_id = pendings[0].request_id
    svc.respond(request_id, answer)
    return request_id


async def test_question_answer_roundtrip(svc, recorder):
    """全链路：question_request 事件 → REST 作答 → 答案进工具结果 + resolved。"""
    task = asyncio.create_task(QuestionTool().execute(
        _tc(question="用 uv 还是 pip？", options=["uv", "pip"]),
        _ctx(recorder),
    ))
    request_id = await _respond_when_pending(svc, "用 uv")
    result = await task

    assert result.is_error is False
    assert "用 uv" in str(result.content)

    request = recorder.of_type(QUESTION_REQUEST)[0]
    assert request["request_id"] == request_id
    assert request["question"] == "用 uv 还是 pip？"
    assert request["options"] == ["uv", "pip"]
    assert request["session_id"] == "sess-q"

    resolved = recorder.of_type(QUESTION_RESOLVED)[0]
    assert resolved["request_id"] == request_id
    assert resolved["status"] == "answered"
    assert resolved["answer"] == "用 uv"
    # 作答后 pending 清空
    assert svc.pending_for_session("sess-q") == []


async def test_no_channel_returns_self_decision(svc, recorder):
    """无交互通道（非流式/无 handler）：不报错，提示模型自决。"""
    result = await QuestionTool().execute(
        _tc(question="怎么处理？"), _ctx(recorder, with_channel=False),
    )

    assert result.is_error is False
    assert "自行决策" in str(result.content)
    assert recorder.of_type(QUESTION_REQUEST) == []
    assert svc.pending_for_session("sess-q") == []


async def test_timeout_returns_self_decision(svc, recorder):
    """超时：answered=False，工具返回自决提示并发 timeout 态 resolved。"""
    svc._timeout = 0  # 立即超时
    result = await QuestionTool().execute(
        _tc(question="在吗？"), _ctx(recorder),
    )

    assert result.is_error is False
    assert "未在时限内" in str(result.content)
    resolved = recorder.of_type(QUESTION_RESOLVED)[0]
    assert resolved["status"] == "timeout"
    assert svc.pending_for_session("sess-q") == []


async def test_discard_run_settles_pending(svc):
    """run 结束/abort：discard_run 落定 Future（防泄漏）。"""
    request = svc.create(
        session_id="sess-q", run_id="run-x",
        question="会被丢弃吗？", tool_call_id="t1",
    )
    svc.discard_run("run-x")
    answer = await svc.wait_answer(request)
    assert answer.answered is False
    assert answer.status == "discarded"
    with pytest.raises(UnknownQuestionRequest):
        svc.respond(request.request_id, "too late")


async def test_respond_validates_and_unknown(svc):
    request = svc.create(
        session_id="s", run_id="r", question="q", tool_call_id="t",
    )
    with pytest.raises(ValueError):
        svc.respond(request.request_id, "   ")
    with pytest.raises(UnknownQuestionRequest):
        svc.respond("no-such-id", "x")
    assert svc.respond(request.request_id, "ok") is request


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"question": ""}, "question 参数缺失"),
        ({"question": "  "}, "question 参数缺失"),
        ({"question": "q", "options": "uv"}, "options 必须是字符串数组"),
        (
            {"question": "q", "options": ["a", "b", "c", "d", "e"]},
            "options 最多 4 个",
        ),
    ],
)
async def test_argument_validation(svc, recorder, arguments, fragment):
    """参数校验失败 → error result（不发事件、不挂 pending）。"""
    result = await QuestionTool().execute(_tc(**arguments), _ctx(recorder))

    assert result.is_error is True
    assert fragment in str(result.content)
    assert recorder.of_type(QUESTION_REQUEST) == []
    assert svc.pending_for_session("sess-q") == []


async def test_options_normalized(svc, recorder):
    """空白选项被剔除；无 options 字段等价自由文本（空列表）。"""
    task = asyncio.create_task(QuestionTool().execute(
        _tc(question="选哪个？", options=[" uv ", "", None]),
        _ctx(recorder),
    ))
    await _respond_when_pending(svc, "uv")
    await task

    request = recorder.of_type(QUESTION_REQUEST)[0]
    assert request["options"] == ["uv"]


async def test_default_singleton_wired():
    """工具模块默认绑定进程级单例（生产路径）。"""
    from egis_opencode.agents.coding.tools import question as question_module

    assert question_module.question_service is question_service
