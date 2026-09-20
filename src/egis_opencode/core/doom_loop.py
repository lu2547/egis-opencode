"""DoomLoopGuard — 跨轮重复调用检测（对齐 opencode DOOM_LOOP_THRESHOLD=3）。

opencode processor.ts 的语义：每个 tool-call 到来时取最近 3 个 tool
parts，同名工具 + input JSON 逐字节相同 → ``permission.ask(doom_loop)``
交用户裁决（放行或断环）。egis 环境 ``silent_allow=true``，权限裁决
等于自动放行 → 改为两级处置：

- 命中 1~4 次：error 结果喂回（递进警告）促模型自纠；
- 累计第 5 次：``ToolLoopAction.STOP`` 终止 run（ark 原生循环控制
  信号，前端走 ``tool_stopped`` outcome）。

判定粒度是 **call 级**（非轮级）：扁平历史跨 LLM 轮追加每个
``(name, canonical arguments)``，与最近前驱构成 3 连相同才命中 ——
混合轮（部分重复）时非重复调用仍照常执行（复刻 PermissionGuard 的
OVERRIDE + 代执行模式），与 opencode parts 流语义一致。

背景：qwen3.5 在低温采样下的 turn 级退化（每轮复读同一 write 调用，
finish_reason=stop 但循环靠 tool call 驱动继续），上下文自我强化
（历史里 N 个相同调用 → 模型模仿第 N+1 个），无护栏时 200 轮内
无限复读。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ark_agentic.core.runtime.callbacks import CallbackResult, HookAction
from ark_agentic.core.types import AgentToolResult, ToolLoopAction

from ..config import settings

logger = logging.getLogger(__name__)

#: 连续相同调用判定窗口（默认对齐 opencode DOOM_LOOP_THRESHOLD = 3；
#: env CODING_DOOM_LOOP_THRESHOLD 可调，进程重启后生效）
DOOM_LOOP_THRESHOLD = settings.doom_loop_threshold

#: 命中累计熔断阈值：第 N 次仍重复 → STOP 终止 run
#: （默认 5；env CODING_DOOM_LOOP_MAX_HITS 可调，进程重启后生效）
DOOM_LOOP_MAX_HITS = settings.doom_loop_max_hits

#: run 观察态容量兜底（正常路径 after_agent 清理；防异常泄漏）
_RUNS_HARD_CAP = 512


@dataclass
class _RunWatch:
    """单 run 的调用历史与命中计数。"""

    history: list[tuple[str, str]] = field(default_factory=list)
    doom_hits: int = 0


def _canon_args(arguments: Any) -> str:
    """工具参数规范化键（dict 键序无关，可比对）。"""
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(arguments)


class DoomLoopGuard:
    """before_tool hook：跨轮重复调用检测 + 拦截。"""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._runs: dict[str, _RunWatch] = {}

    async def __call__(
        self,
        ctx: Any,
        *,
        turn: int,
        tool_calls: list,
        **kwargs: Any,
    ) -> CallbackResult | None:
        if not tool_calls:
            return None

        watch = self._runs.setdefault(ctx.run_id, _RunWatch())
        if len(self._runs) > _RUNS_HARD_CAP:
            # 异常残留兜底：丢最旧 run 的观察态（正常路径由 after_agent 清理）
            oldest = next(iter(self._runs))
            self._runs.pop(oldest, None)

        # call 级判定：前驱 = 已落历史 + 本轮已判定的前序 calls
        turn_keys: list[tuple[str, str]] = []
        doomed: list[int] = []
        for i, tc in enumerate(tool_calls):
            key = (tc.name, _canon_args(tc.arguments))
            prior = (watch.history + turn_keys)[-(DOOM_LOOP_THRESHOLD - 1):]
            if (
                len(prior) == DOOM_LOOP_THRESHOLD - 1
                and all(p == key for p in prior)
            ):
                doomed.append(i)
            turn_keys.append(key)
        watch.history.extend(turn_keys)

        if not doomed:
            return None

        watch.doom_hits += len(doomed)
        sid_short = ctx.session.session_id[:8]
        logger.warning(
            "[DOOM_LOOP] session=%s turn=%d hits=%d tools=%s —— 连续 %d 次"
            "完全相同调用被拦截",
            sid_short, turn, watch.doom_hits,
            [tool_calls[i].name for i in doomed], DOOM_LOOP_THRESHOLD,
        )

        results: list[AgentToolResult | None] = [None] * len(tool_calls)
        stop = watch.doom_hits >= DOOM_LOOP_MAX_HITS
        for i in doomed:
            results[i] = (
                self._stop_result(tool_calls[i])
                if stop else self._warn_result(tool_calls[i], watch.doom_hits)
            )

        # 混合轮：非重复调用照常执行（保 ark 全部执行语义）
        allowed = [tc for i, tc in enumerate(tool_calls) if i not in doomed]
        slots = [i for i in range(len(tool_calls)) if i not in doomed]
        if allowed:
            executed = await self._execute_allowed(ctx, allowed)
            for slot, result in zip(slots, executed):
                results[slot] = result

        return CallbackResult(
            action=HookAction.OVERRIDE,
            tool_results=[r for r in results if r is not None],
        )

    def discard_run(self, run_id: str) -> None:
        """after_agent：run 结束清理观察态（agent 是进程级单例）。"""
        self._runs.pop(run_id, None)

    # ── 结果构造 ───────────────────────────────────────

    def _warn_result(self, tc: Any, hits: int) -> AgentToolResult:
        """1~4 次命中：error 喂回促自纠（递进语气）。"""
        return AgentToolResult.error_result(
            tc.id,
            f"Doom loop detected: {tc.name} 已连续 {DOOM_LOOP_THRESHOLD} 次"
            f"以完全相同的参数调用（累计第 {hits} 次警告），本次调用已拦截、"
            "未执行。请勿重复相同调用：若任务已完成，直接输出最终答复；"
            "若未完成，请改变参数、更换工具，或说明当前遇到的障碍。",
            tool_name=tc.name,
            llm_digest=(
                f"[tool:{tc.name} status=error] 重复调用被拦截"
                f"（连续{DOOM_LOOP_THRESHOLD}次相同，累计{hits}次）。"
            ),
        )

    def _stop_result(self, tc: Any) -> AgentToolResult:
        """第 5 次命中：STOP 终止 run。

        content 面向最终用户（_detect_stop_response 收集非 error 的
        文本作为 run 终止时的兜底答复），loop_action=STOP 是 ark 的
        原生循环终止信号。
        """
        return AgentToolResult.text_result(
            tc.id,
            "已终止本次运行：检测到同一工具以完全相同的参数连续重复"
            f"调用 {DOOM_LOOP_MAX_HITS} 次（疑似死循环）。可发送“继续”"
            "并明确下一步指令，或调整任务要求后重新开始。",
            loop_action=ToolLoopAction.STOP,
        )

    # ── 代执行 ─────────────────────────────────────────

    async def _execute_allowed(self, ctx: Any, calls: list) -> list:
        """混合批中非重复调用走 agent 的 ToolExecutor（对齐 PermissionGuard）。"""
        executor = getattr(self._agent, "_tool_executor", None)
        context_keys = {
            k: v for k, v in (ctx.input_context or {}).items()
            if k.startswith("user:") or k == "workspace:root"
        }
        exec_context = {
            **context_keys,
            "session_id": ctx.session.session_id,
            "_active_skill_id": getattr(
                ctx.session, "current_active_skill_id", None,
            ),
            "system:run_id": ctx.run_id,
        }
        if executor is not None:
            return await executor.execute(calls, exec_context)
        results = []
        for tc in calls:
            tool = self._agent.tool_registry.get(tc.name)
            if tool is None:
                results.append(
                    AgentToolResult.error_result(tc.id, f"Tool not found: {tc.name}")
                )
            else:
                results.append(await tool.execute(tc, exec_context))
        return results


__all__ = ["DOOM_LOOP_THRESHOLD", "DOOM_LOOP_MAX_HITS", "DoomLoopGuard"]
