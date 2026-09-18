"""PermissionGuard — before_tool 权限守卫。

挂载方式：``CodingAgent.build_callbacks()`` 返回
``RunnerCallbacks(before_tool=guard)``，ark runner 在每个 ReAct 轮
工具批执行前调用本守卫（零侵入：不改 ark 代码）。

行为：
- 批内全部 allow → 返回 ``None``（PASS，原路径执行）
- deny → OVERRIDE 返回 error tool_result（LLM 可见 "Permission denied"）
- ask → 经 ``input_context["temp:permission_bridge"]`` 通道发
  ``permission_request`` custom 事件 → 等待 REST 应答（超时=拒绝）；
  批准的调用经 agent 的 ToolExecutor 原路径执行
  （保留超时 / 事件派发 / 错误降级语义），拒绝的返回 error result

事件通道说明：ark ``run_hooks`` 不把 handler 传给 hook，guard 经
ctx.input_context 里的 ``PermissionBridge``（自建 chat 端点注入）发射
SSE 事件；无 bridge（非流式 / 未注入）→ ask 自动拒绝。

run 内 "always" 应答产生的授权规则缓存在守卫内（run_id 维度，内存态
同 opencode approved 数组 —— 进程生命周期，重启后重新询问），
run 结束后由 ``discard_run()`` 清理。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ark_agentic.core.runtime.callbacks import CallbackResult, HookAction
from ark_agentic.core.types import AgentToolResult

from ..config import settings
from ..events import (
    PERMISSION_REQUEST,
    PERMISSION_RESOLVED,
    TOOL_DIGEST,
    tool_digest_payload,
)
from .bridge import PermissionBridge, bridge_from_context
from .rules import (
    PIP_VIOLATION_GUIDE,
    Rule,
    Ruleset,
    bash_always_patterns,
    bash_patterns,
    evaluate,
    find_pip_violation,
    pattern_for,
)
from .service import PermissionResolution, PermissionService

if TYPE_CHECKING:
    from ark_agentic.core.runtime.callbacks import CallbackContext
    from ark_agentic.core.types import ToolCall

logger = logging.getLogger(__name__)

#: ark 框架自带工具（技能加载/记忆/引用渲染）— 框架机制，不属于用户操作审批范围
FRAMEWORK_TOOLS: frozenset[str] = frozenset(
    {"read_skill", "read_reference", "memory_write", "render_a2ui"}
)

#: 工具名 → 权限维度名的映射（bash 沙箱工具名可在 sandbox.json 自定义）
_TOOL_PERMISSION_ALIASES: dict[str, str] = {
    "sandbox_exec": "bash",
}


def _permission_dimension(tool_name: str) -> str:
    return _TOOL_PERMISSION_ALIASES.get(tool_name, tool_name)


class PermissionGuard:
    """before_tool 权限守卫（实现 ark BeforeToolCallback 协议形态）。"""

    def __init__(
        self,
        *,
        agent: Any,
        service: PermissionService,
        base_ruleset: Ruleset,
        silent_allow: bool | None = None,
    ) -> None:
        self._agent = agent
        self._service = service
        self._base_ruleset = base_ruleset
        #: 静默放行模式（None = 跟随 settings.permission_silent_allow）。
        #: 开启后 ask 规则视同 allow（不弹审批卡片）；deny 硬禁令不受影响
        self._silent_allow = (
            silent_allow if silent_allow is not None
            else settings.permission_silent_allow
        )
        #: run_id → 本 run 内 "always" 授权的追加规则
        self._approved: dict[str, Ruleset] = {}

    # ── BeforeToolCallback 协议入口 ─────────────────────

    async def __call__(
        self,
        ctx: "CallbackContext",
        *,
        turn: int,
        tool_calls: list["ToolCall"],
        **kwargs: Any,
    ) -> CallbackResult | None:
        if not tool_calls:
            return None

        bridge = bridge_from_context(ctx.input_context)
        decisions = [self._evaluate(ctx, tc) for tc in tool_calls]
        if all(rule.action == "allow" for rule in decisions):
            return None  # 全放行 → 走 ark 原执行路径

        results: list[AgentToolResult | None] = [None] * len(tool_calls)
        allowed_calls: list[ToolCall] = []
        allowed_slots: list[int] = []

        for i, (tc, rule) in enumerate(zip(tool_calls, decisions)):
            if rule.action == "allow":
                allowed_calls.append(tc)
                allowed_slots.append(i)
                continue
            if rule.action == "deny":
                results[i] = self._deny_result(tc, rule)
                self._emit_denied_digest(bridge, tc)
                continue
            # ask
            resolution = await self._ask(ctx, tc, rule, bridge)
            if resolution.approved:
                allowed_calls.append(tc)
                allowed_slots.append(i)
            else:
                if resolution.action == "timeout":
                    reason = "等待用户确认超时"
                elif resolution.action == "reject" and resolution.request_id:
                    reason = "用户拒绝"
                else:
                    reason = "无交互通道，自动拒绝"
                results[i] = self._deny_result(tc, rule, reason=reason)
                self._emit_denied_digest(bridge, tc, note=reason)

        if allowed_calls:
            executed = await self._execute_allowed(
                ctx, allowed_calls, bridge,
            )
            for slot, result in zip(allowed_slots, executed):
                results[slot] = result

        return CallbackResult(
            action=HookAction.OVERRIDE,
            tool_results=[r for r in results if r is not None],
        )

    # ── 评估 ───────────────────────────────────────────

    def _evaluate(self, ctx: "CallbackContext", tc: "ToolCall") -> Rule:
        if tc.name in FRAMEWORK_TOOLS:
            return Rule(permission=tc.name, pattern="*", action="allow")
        dimension = _permission_dimension(tc.name)
        ruleset = list(self._base_ruleset) + self._approved.get(ctx.run_id, [])
        if dimension == "bash":
            # 裸 pip 硬禁令：先于一切规则（含 silent / always 记忆），
            # 不可审批绕过 —— 本地直连继承服务进程环境，裸 pip 会把包
            # 装进平台自身的 Python 环境（沙箱同样禁止）
            command = str((tc.arguments or {}).get("command") or "").strip()
            violation = find_pip_violation(command)
            if violation is not None:
                return Rule(
                    permission="bash", pattern="*", action="deny",
                    reason=f"禁止裸 pip 安装/卸载（命中：{violation}）", 
                )
            # 逐子命令评估（同 opencode Permission.ask 循环）：
            # 任一 deny 短路；全 allow 放行；否则 ask（首个 ask 的子命令文本）
            ask_pattern = ""
            for segment in bash_patterns(command):
                rule = evaluate("bash", segment, ruleset)
                if rule.action == "deny":
                    return rule
                if rule.action != "allow" and not ask_pattern:
                    ask_pattern = segment
            if ask_pattern and not self._silent_allow:
                return Rule(permission="bash", pattern=ask_pattern, action="ask")
            return Rule(permission="bash", pattern="*", action="allow")
        pattern = pattern_for(tc.name, tc.arguments or {})
        rule = evaluate(dimension, pattern, ruleset)
        if rule.action == "ask" and self._silent_allow:
            # 静默放行：ask 视同 allow（deny 硬禁令不受影响）
            return Rule(permission=dimension, pattern="*", action="allow")
        return rule

    # ── ask 流程 ───────────────────────────────────────

    async def _ask(
        self,
        ctx: "CallbackContext",
        tc: "ToolCall",
        rule: Rule,
        bridge: PermissionBridge | None,
    ) -> Any:
        dimension = _permission_dimension(tc.name)
        if dimension == "bash":
            # patterns = 每条子命令完整文本；always = 前缀元数 + " *"
            # （同 opencode shell.ts ask 的 patterns / always 两字段）
            command = str((tc.arguments or {}).get("command") or "").strip()
            patterns = bash_patterns(command) or [rule.pattern]
            always_patterns = bash_always_patterns(command) or [rule.pattern]
        else:
            patterns = [
                pattern_for(tc.name, tc.arguments or {}),
            ]
            always_patterns = list(patterns)

        if bridge is None or not bridge.available:
            # 非流式通道无法交互审批 → 自动拒绝
            return PermissionResolution(
                action="reject", request_id="",
            )
        request = self._service.create_request(
            session_id=ctx.session.session_id,
            run_id=ctx.run_id,
            permission=dimension,
            pattern=rule.pattern,
            tool_name=tc.name,
            tool_call_id=tc.id,
            tool_args=tc.arguments or {},
            patterns=patterns,
            always_patterns=always_patterns,
        )
        bridge.emit(
            PERMISSION_REQUEST, request.to_payload(),
        )
        try:
            resolution = await self._service.wait_resolution(request)
        finally:
            bridge.emit(
                PERMISSION_RESOLVED,
                {
                    "request_id": request.request_id,
                    "action": resolution.action,
                    "tool_call_id": tc.id,
                },
            )
        if resolution.action == "always":
            # 内存态授权（run 级缓存，同 opencode approved.push）
            self._approved.setdefault(ctx.run_id, []).extend(
                resolution.as_rules
            )
        return resolution

    # ── 执行与结果 ─────────────────────────────────────

    async def _execute_allowed(
        self,
        ctx: "CallbackContext",
        calls: list["ToolCall"],
        bridge: PermissionBridge | None,
    ) -> list[AgentToolResult]:
        """混合批中放行调用走 agent 的 ToolExecutor（保留全部执行语义）。"""
        executor = getattr(self._agent, "_tool_executor", None)
        # user:* 键透传（文件工具按 user:id 定位多租户 workspace 目录）；
        # ark 原路径的 executor context 含 session.state（经 merge_input_context
        # 合入 user:* 键），before_tool hook 拿不到 state → 从 ctx.input_context 补齐。
        # workspace:root 是会话工作目录锚定（本地目录模式），必须一并透传，
        # 否则守卫代执行的调用会回落多租户目录。
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
        handler = bridge.handler if bridge is not None else None
        if executor is not None:
            return await executor.execute(calls, exec_context, handler=handler)
        # 兜底：直接经注册表逐个执行（不应出现在正常运行时）
        results = []
        for tc in calls:
            tool = self._agent.tool_registry.get(tc.name)
            if tool is None:
                results.append(
                    AgentToolResult.error_result(
                        tc.id, f"Tool not found: {tc.name}",
                    )
                )
            else:
                results.append(await tool.execute(tc, exec_context))
        return results

    def _deny_result(
        self, tc: "ToolCall", rule: Rule, *, reason: str = "规则拒绝",
    ) -> AgentToolResult:
        detail = (
            f"Permission denied ({rule.permission}:{rule.pattern}) — {reason}."
        )
        if rule.reason:
            # 规则自带说明（如 pip 硬禁令）→ 拼进 LLM 可见详情，
            # 引导模型改用 venv + uv 路径而非直接放弃
            detail = f"{detail}\n{rule.reason}。{PIP_VIOLATION_GUIDE}"
        else:
            detail = (
                f"{detail}"
                " 如需该操作请说明理由并请用户在界面上调整权限或改用其他方式完成任务。"
            )
        return AgentToolResult.error_result(
            tc.id,
            detail,
            tool_name=tc.name,
            llm_digest=f"[tool:{tc.name} status=error] 权限拒绝({reason})，未执行该操作。",
        )

    # ── 事件 ───────────────────────────────────────────

    def _emit_denied_digest(
        self, bridge: PermissionBridge | None, tc: "ToolCall", *, note: str = "规则拒绝",
    ) -> None:
        if bridge is None or not bridge.available:
            return
        bridge.emit(
            TOOL_DIGEST,
            tool_digest_payload(
                tool_name=tc.name,
                tool_call_id=tc.id,
                display_type="tool_progress",
                status="denied",
                title=note,
                note=f"{tc.name} 被权限规则拒绝",
            ),
        )

    # ── 生命周期 ───────────────────────────────────────

    def discard_run(self, run_id: str) -> None:
        """run 结束时清理授权缓存与挂起请求。"""
        self._approved.pop(run_id, None)
        self._service.discard_run(run_id)
