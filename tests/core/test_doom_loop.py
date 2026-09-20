"""DoomLoopGuard 测试 — 跨轮重复调用检测（对齐 opencode DOOM_LOOP_THRESHOLD）。

覆盖：3 连相同触发拦截（error 喂回）、变参数不触发、5 次熔断（STOP）、
混合轮代执行、discard_run 清理。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from ark_agentic.core.runtime.callbacks import HookAction
from ark_agentic.core.types import AgentToolResult, ToolCall, ToolLoopAction

from egis_opencode.core.doom_loop import (
    DOOM_LOOP_MAX_HITS,
    DOOM_LOOP_THRESHOLD,
    DoomLoopGuard,
)


def _ctx(run_id: str = "r1") -> Any:
    return SimpleNamespace(
        run_id=run_id,
        input_context={},
        session=SimpleNamespace(
            session_id="session-12345678", current_active_skill_id=None,
        ),
    )


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall.create(name, arguments)


class _FakeExecutor:
    """代执行桩：记录收到的 calls，返回固定成功结果。"""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, calls, context, handler=None):
        self.executed.extend(tc.name for tc in calls)
        return [
            AgentToolResult.text_result(tc.id, f"executed {tc.name}")
            for tc in calls
        ]


def _agent(executor: _FakeExecutor | None = None) -> Any:
    return SimpleNamespace(_tool_executor=executor, tool_registry=None)


async def test_three_identical_calls_intercepted():
    """连续 3 轮完全相同调用 → 第 3 轮 OVERRIDE，error 喂回。"""
    guard = DoomLoopGuard(agent=_agent())
    ctx = _ctx()
    calls = [_call("write", path="a.py", content="x")]

    r1 = await guard(ctx, turn=1, tool_calls=calls)
    r2 = await guard(ctx, turn=2, tool_calls=calls)
    assert r1 is None and r2 is None, "前两轮不应触发"

    r3 = await guard(ctx, turn=3, tool_calls=calls)
    assert r3 is not None and r3.action == HookAction.OVERRIDE
    assert len(r3.tool_results) == 1
    result = r3.tool_results[0]
    assert result.is_error, "1~4 次命中应喂回 error 促自纠"
    assert "Doom loop" in str(result.content)
    assert result.loop_action == ToolLoopAction.CONTINUE


async def test_changed_arguments_no_trigger():
    """参数变化（如 offset 递增）不触发 —— 合法的连续同类调用。"""
    guard = DoomLoopGuard(agent=_agent())
    ctx = _ctx()

    for turn, offset in enumerate((1, 50, 100), start=1):
        r = await guard(
            ctx, turn=turn,
            tool_calls=[_call("read", path="a.py", offset=offset)],
        )
        assert r is None, f"turn={turn} 参数不同，不应触发"


async def test_fifth_hit_stops_run():
    """累计第 5 次命中 → STOP（首次命中在 turn3，第 5 次在 turn7）。"""
    executor = _FakeExecutor()
    guard = DoomLoopGuard(agent=_agent(executor))
    ctx = _ctx()
    calls = [_call("write", path="a.py", content="x")]

    r: Any = None
    for turn in range(1, DOOM_LOOP_MAX_HITS + DOOM_LOOP_THRESHOLD):
        r = await guard(ctx, turn=turn, tool_calls=calls)
        if turn < DOOM_LOOP_THRESHOLD:
            assert r is None
        elif turn < DOOM_LOOP_MAX_HITS + DOOM_LOOP_THRESHOLD - 1:
            # turn 3..6：命中但未到熔断阈值 → error 喂回
            assert r is not None and r.tool_results[0].is_error
    # 第 5 次命中（turn 7）熔断 STOP
    assert r is not None
    result = r.tool_results[0]
    assert result.loop_action == ToolLoopAction.STOP
    assert not result.is_error, "STOP content 作为终止兕底答复，须非 error"
    assert "死循环" in str(result.content)
    assert executor.executed == [], "doomed 轮不应有代执行"


async def test_mixed_batch_partial_execution():
    """混合轮：重复的 call 拦截，非重复的照常代执行（opencode 语义）。

    注：窗口是 call 级连续性 —— 前驱被变参调用打断则不命中
    （与 opencode parts 流一致），故构造重复尾 + 新调用头部。
    """
    executor = _FakeExecutor()
    guard = DoomLoopGuard(agent=_agent(executor))
    ctx = _ctx()
    # 前两轮：单 write A
    calls = [_call("write", path="a.py", content="x")]
    assert await guard(ctx, turn=1, tool_calls=calls) is None
    assert await guard(ctx, turn=2, tool_calls=calls) is None

    # 第三轮：[write A（三连）, read NEW（首次）] → write 拦截、read 代执行
    mixed = [calls[0], _call("read", path="b.md")]
    r = await guard(ctx, turn=3, tool_calls=mixed)
    assert r is not None and r.action == HookAction.OVERRIDE
    by_id = {res.tool_call_id: res for res in r.tool_results}
    write_res = by_id[mixed[0].id]
    read_res = by_id[mixed[1].id]
    assert write_res.is_error, "三连相同的 write 应被拦截"
    assert not read_res.is_error, "首次的 read 应正常执行"
    assert executor.executed == ["read"], "仅非重复调用走代执行"


async def test_interrupted_streak_no_trigger():
    """变参调用打断连续窗口：read 换参数后，write 的三连被重置。"""
    guard = DoomLoopGuard(agent=_agent())
    ctx = _ctx()
    batch = [_call("read", path="x.md"), _call("write", path="a.py", content="x")]
    await guard(ctx, turn=1, tool_calls=batch)
    await guard(ctx, turn=2, tool_calls=batch)
    # 第三轮 read 变参 → 窗口序列 [rx, wA, rx, wA, r10, wA]，
    # write 的前驱是 r10（非 wA）→ 连续性被打断，不触发
    mixed = [_call("read", path="x.md", offset=10), _call("write", path="a.py", content="x")]
    r = await guard(ctx, turn=3, tool_calls=mixed)
    assert r is None, "变参打断连续窗口后不应触发（与 opencode 语义一致）"


async def test_discard_run_resets_history():
    """discard_run 后同 run_id 历史/计数清零（after_agent 清理路径）。"""
    guard = DoomLoopGuard(agent=_agent())
    ctx = _ctx()
    calls = [_call("write", path="a.py", content="x")]
    await guard(ctx, turn=1, tool_calls=calls)
    await guard(ctx, turn=2, tool_calls=calls)
    await guard(ctx, turn=3, tool_calls=calls)  # 命中 1 次

    guard.discard_run(ctx.run_id)
    r = await guard(ctx, turn=4, tool_calls=calls)
    assert r is None, "清理后重新计数，单次调用不触发"


async def test_per_run_isolation():
    """不同 run_id 互不串扰（agent 是进程级单例）。"""
    guard = DoomLoopGuard(agent=_agent())
    calls = [_call("write", path="a.py", content="x")]
    await guard(_ctx("run-a"), turn=1, tool_calls=calls)
    await guard(_ctx("run-a"), turn=2, tool_calls=calls)
    # run-b 独立计数：首个三连在第 3 次才触发
    r = await guard(_ctx("run-b"), turn=1, tool_calls=calls)
    assert r is None
