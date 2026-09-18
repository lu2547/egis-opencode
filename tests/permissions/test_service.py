"""PermissionService 测试 — 挂起 / 应答 / 超时 / run 清理。"""

from __future__ import annotations

import asyncio

import pytest

from egis_opencode.permissions.rules import Rule
from egis_opencode.permissions.service import (
    PermissionService,
    UnknownPermissionRequest,
)


def _make_service(timeout: float = 2.0) -> PermissionService:
    return PermissionService(timeout_seconds=timeout)


def _create(service: PermissionService, **overrides):
    kwargs = dict(
        session_id="sess-1",
        run_id="run-1",
        permission="write",
        pattern="/ws/alice/x.py",
        tool_name="write",
        tool_call_id="tc-1",
        tool_args={"path": "x.py"},
    )
    kwargs.update(overrides)
    return service.create_request(**kwargs)


async def test_respond_once_approves():
    service = _make_service()
    request = _create(service)

    async def _waiter():
        return await service.wait_resolution(request)

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0.01)
    returned = service.respond(request.request_id, "once")
    resolution = await task

    assert returned.request_id == request.request_id
    assert resolution.approved is True
    assert resolution.action == "once"
    assert resolution.as_rules == []  # only always produces rules
    # 落定后 pending 清空
    assert service.pending_for_session("sess-1") == []


async def test_respond_always_produces_rules():
    """always：always_patterns 逐个回填 allow 规则（同 opencode approved.push）。"""
    service = _make_service()
    request = _create(
        service,
        patterns=("git checkout main",),
        always_patterns=("git checkout *",),
    )

    task = asyncio.create_task(service.wait_resolution(request))
    await asyncio.sleep(0.01)
    service.respond(request.request_id, "always")
    resolution = await task

    assert resolution.action == "always"
    assert resolution.as_rules == [
        Rule(permission="write", pattern="git checkout *", action="allow"),
    ]


async def test_respond_always_without_patterns_falls_back_to_pattern():
    """未显式传 always_patterns 时回落单 pattern（非 bash 工具路径）。"""
    service = _make_service()
    request = _create(service)

    task = asyncio.create_task(service.wait_resolution(request))
    await asyncio.sleep(0.01)
    service.respond(request.request_id, "always")
    resolution = await task

    assert resolution.as_rules == [
        Rule(permission="write", pattern="/ws/alice/x.py", action="allow"),
    ]


async def test_respond_reject():
    service = _make_service()
    request = _create(service)

    task = asyncio.create_task(service.wait_resolution(request))
    await asyncio.sleep(0.01)
    service.respond(request.request_id, "reject")
    resolution = await task

    assert resolution.approved is False
    assert resolution.action == "reject"


async def test_timeout_rejects_and_cleans_up():
    service = _make_service(timeout=0.05)
    request = _create(service)

    resolution = await service.wait_resolution(request)

    assert resolution.action == "timeout"
    assert resolution.approved is False
    # 超时后 pending 已清理，重复应答报未知
    with pytest.raises(UnknownPermissionRequest):
        service.respond(request.request_id, "once")


async def test_discard_run_rejects_pending_of_run():
    service = _make_service()
    r1 = _create(service, run_id="run-1")
    r2 = _create(
        service, run_id="run-2",
        tool_call_id="tc-2", tool_args={},
    )

    task1 = asyncio.create_task(service.wait_resolution(r1))
    task2 = asyncio.create_task(service.wait_resolution(r2))
    await asyncio.sleep(0.01)

    service.discard_run("run-1")

    res1 = await task1
    assert res1.action == "reject"
    # run-2 不受影响，仍可正常应答
    service.respond(r2.request_id, "once")
    res2 = await task2
    assert res2.action == "once"


async def test_pending_for_session_and_get_request():
    service = _make_service()
    request = _create(service, session_id="sess-a")
    _create(
        service, session_id="sess-b", tool_call_id="tc-b", tool_args={},
    )

    pendings = service.pending_for_session("sess-a")
    assert [p.request_id for p in pendings] == [request.request_id]
    assert service.get_request(request.request_id) is request

    payload = pendings[0].to_payload()
    assert payload["permission"] == "write"
    assert payload["tool_name"] == "write"
    assert payload["session_id"] == "sess-a"
    assert payload["tool_args"] == {"path": "x.py"}
    # patterns / always_patterns 缺省回落单 pattern（前端展示兼容）
    assert payload["patterns"] == ["/ws/alice/x.py"]
    assert payload["always_patterns"] == ["/ws/alice/x.py"]


def test_respond_unknown_request_id():
    service = _make_service()
    with pytest.raises(UnknownPermissionRequest):
        service.respond("no-such-id", "once")


async def test_respond_invalid_action():
    service = _make_service()
    request = _create(service)
    with pytest.raises(ValueError, match="invalid permission action"):
        service.respond(request.request_id, "explode")
    # 非法 action 不影响 pending 状态
    assert service.get_request(request.request_id) is request


async def test_wait_resolution_on_missing_entry_rejects():
    """请求已不存在（如被 discard）时 wait 直接按拒绝返回。"""
    service = _make_service()
    request = _create(service)
    service.discard_run("run-1")  # 内部落定 reject 并清理

    resolution = await service.wait_resolution(request)
    assert resolution.action == "reject"
