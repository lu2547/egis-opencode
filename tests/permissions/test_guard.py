"""PermissionGuard 三态测试 — allow PASS / deny 短路 / ask 全链路（once/always/reject/timeout）。

预设已对齐 opencode defaults（默认全放行，仅 ``*.env`` 读取 ask），
因此 ask/deny 场景用注入 ruleset（``ASK_WRITE`` / ``ASK_DOCKER``）驱动；
默认预设行为（bash 不问、.env 询问）单独断言。

guard 挂在 before_tool：工具执行走 agent 的 ToolExecutor
（``BaseAgent._construct`` 构造，对齐 ark test_runner 模式）。

事件通道：ark ``run_hooks`` 只传 ``turn``/``tool_calls`` 给 hook（不传
handler），guard 经 ``input_context["temp:permission_bridge"]`` 发事件 ——
测试用 ``_bridge_ctx`` 复刻 chat 端点的 bridge 注入。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from ark_agentic.core.runtime.callbacks import CallbackContext, HookAction
from ark_agentic.core.session import SessionManager
from ark_agentic.core.tools.base import AgentTool, ToolParameter
from ark_agentic.core.types import AgentToolResult, SessionEntry, ToolCall

from egis_opencode.agents.coding.agent import CodingAgent
from egis_opencode.events import (
    PERMISSION_REQUEST,
    PERMISSION_RESOLVED,
    TOOL_DIGEST,
)
from egis_opencode.permissions.bridge import BRIDGE_KEY, PermissionBridge
from egis_opencode.permissions.guard import PermissionGuard
from egis_opencode.permissions.presets import build_ruleset, plan_ruleset
from egis_opencode.permissions.rules import Rule
from egis_opencode.permissions.service import PermissionService

from tests.helpers import MockChatModel, RecordingHandler


# ── 测试工具 ───────────────────────────────────────────


class EchoTool(AgentTool):
    """echo 工具：把 args 原样返回（验证真实执行路径）。"""

    name = "echo"
    description = "echo arguments back"
    parameters = [
        ToolParameter(name="text", type="string", description="echo text"),
    ]

    def __init__(self) -> None:
        self.executed: list[dict] = []

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        self.executed.append(dict(tool_call.arguments or {}))
        return AgentToolResult.text_result(
            tool_call.id, f"echo: {dict(tool_call.arguments or {})}",
        )


class TaggedTool(EchoTool):
    """按 name 实例化的多形态 echo（注册多个不同维度名）。"""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


#: ask 场景驱动规则集（预设默认放行，测试注入收紧规则）
ASK_WRITE = build_ruleset() + [
    Rule(permission="write", pattern="*", action="ask"),
]
#: defaults + docker 命令 ask（bash 复合/arity 记忆场景）
ASK_DOCKER = build_ruleset() + [
    Rule(permission="bash", pattern="docker *", action="ask"),
]


@pytest.fixture
def agent(tmp_path: Path) -> CodingAgent:
    registry_tool = TaggedTool("echo")
    agent = CodingAgent._construct(
        llm=MockChatModel(),
        session_manager=SessionManager(tmp_path / "sessions", agent_id="coding"),
        agent_id="coding",
    )
    agent.tool_registry.register(TaggedTool("read"))
    agent.tool_registry.register(TaggedTool("write"))
    agent.tool_registry.register(TaggedTool("bash"))
    agent.tool_registry.register(registry_tool)
    agent._echo_tool = registry_tool  # type: ignore[attr-defined]
    return agent


@pytest.fixture
def ctx() -> CallbackContext:
    return CallbackContext(
        run_id="run-1",
        user_input="do something",
        input_context={},
        session=SessionEntry(session_id="sess-1", user_id="alice"),
    )


def _bridge_ctx(ctx: CallbackContext, recorder: RecordingHandler) -> CallbackContext:
    """注入权限事件通道（复刻 chat 端点的 temp:permission_bridge）。"""
    ctx.input_context[BRIDGE_KEY] = PermissionBridge(recorder)
    return ctx


def _guard(agent, service, ruleset=None, silent_allow=None) -> PermissionGuard:
    return PermissionGuard(
        agent=agent,
        service=service,
        base_ruleset=ruleset if ruleset is not None else build_ruleset(),
        silent_allow=silent_allow,
    )


def _tc(name: str, **arguments) -> ToolCall:
    return ToolCall.create(name, arguments)


async def _respond_when_pending(service: PermissionService, session_id: str, action: str) -> str:
    """轮询等 pending 请求出现后应答，返回 request_id。"""
    for _ in range(500):
        pendings = service.pending_for_session(session_id)
        if pendings:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("permission request never appeared")
    request_id = pendings[0].request_id
    service.respond(request_id, action)
    return request_id


# ── allow：默认预设全放行（对齐 opencode defaults）─────


async def test_default_preset_allows_without_ask(agent, ctx, recorder):
    """默认预设：bash（含 rm/docker 任意命令）/ write / read 均不问。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)
    _bridge_ctx(ctx, recorder)

    for tc in [
        _tc("bash", command="git status"),
        _tc("bash", command="rm -rf /tmp/x"),
        _tc("bash", command="docker ps"),
        _tc("write", path="a.py", content="x"),
    ]:
        result = await guard(ctx, turn=1, tool_calls=[tc])
        assert result is None, f"默认放行: {tc.name}"
    assert recorder.of_type(PERMISSION_REQUEST) == []


async def test_env_read_asks_by_default(agent, ctx, recorder):
    """默认预设唯一 ask 项：读 .env 敏感凭证文件。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("read", path="/ws/alice/.env")],
    ))
    request_id = await _respond_when_pending(service, "sess-1", "once")
    result = await task

    request = recorder.of_type(PERMISSION_REQUEST)[0]
    assert request["request_id"] == request_id
    assert request["permission"] == "read"
    assert result is not None  # ask → OVERRIDE 代执行
    assert result.tool_results[0].is_error is False


async def test_framework_tools_always_allowed(agent, ctx, recorder):
    """ark 框架机制工具（read_skill 等）不经用户审批。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("read_skill", skill_id="git-workflow")],
    )

    assert result is None


# ── deny：OVERRIDE + error tool_result + denied digest ──


async def test_deny_short_circuits(agent, ctx, recorder):
    """plan 预设 bash deny：不执行工具，返回 error result + denied digest。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=plan_ruleset())
    bash_tool = agent.tool_registry.get("bash")
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="ls -la")],
    )

    assert result is not None
    assert result.action == HookAction.OVERRIDE
    assert result.tool_results is not None
    denied = result.tool_results[0]
    assert denied.is_error is True
    assert "Permission denied" in str(denied.content)
    assert bash_tool.executed == []
    # denied digest 事件
    digests = recorder.of_type(TOOL_DIGEST)
    assert len(digests) == 1
    assert digests[0]["status"] == "denied"
    assert digests[0]["tool_name"] == "bash"


async def test_bash_deny_segment_short_circuits_chain(agent, ctx, recorder):
    """复合命令任一子命令 deny → 整条拒绝（同 opencode DeniedError 短路）。"""
    ruleset = build_ruleset() + [
        Rule(permission="bash", pattern="curl *", action="deny"),
    ]
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ruleset)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="ls && curl http://x")],
    )

    assert result is not None
    assert result.tool_results[0].is_error is True


async def test_plan_mode_denies_write(agent, ctx, recorder):
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=plan_ruleset())
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    )

    assert result is not None
    assert result.action == HookAction.OVERRIDE
    assert result.tool_results[0].is_error is True


# ── ask：审批全链路 ────────────────────────────────────


async def test_ask_once_executes_tool_via_executor(agent, ctx, recorder):
    """ask + once：工具经 agent ToolExecutor 真实执行，结果进 OVERRIDE。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    write_tool = agent.tool_registry.get("write")
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="hello")],
    ))
    request_id = await _respond_when_pending(service, "sess-1", "once")
    result = await task

    # permission_request 事件已发（含完整 payload）
    requests = recorder.of_type(PERMISSION_REQUEST)
    assert len(requests) == 1
    assert requests[0]["request_id"] == request_id
    assert requests[0]["permission"] == "write"
    assert requests[0]["tool_name"] == "write"
    assert requests[0]["tool_args"] == {"path": "a.py", "content": "hello"}
    # 非 bash 工具：patterns 回落单 pattern
    assert requests[0]["patterns"] == ["a.py"]
    assert requests[0]["always_patterns"] == ["a.py"]

    # permission_resolved 事件
    resolved = recorder.of_type(PERMISSION_RESOLVED)
    assert resolved[0]["action"] == "once"
    assert resolved[0]["request_id"] == request_id

    # 工具被真实执行，结果作为 OVERRIDE tool_results
    assert write_tool.executed == [{"path": "a.py", "content": "hello"}]
    assert result is not None
    assert result.action == HookAction.OVERRIDE
    assert result.tool_results[0].is_error is False
    assert "'content': 'hello'" in str(result.tool_results[0].content)


async def test_ask_reject_returns_error(agent, ctx, recorder):
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    write_tool = agent.tool_registry.get("write")
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    ))
    await _respond_when_pending(service, "sess-1", "reject")
    result = await task

    assert write_tool.executed == []
    assert result is not None
    assert result.tool_results[0].is_error is True
    assert "用户拒绝" in str(result.tool_results[0].content)


async def test_ask_timeout_rejects(agent, ctx, recorder):
    service = PermissionService(timeout_seconds=0.05)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    )

    assert result is not None
    assert result.tool_results[0].is_error is True
    assert "超时" in str(result.tool_results[0].content)
    # resolved 事件携带 timeout 动作
    assert recorder.of_type(PERMISSION_RESOLVED)[0]["action"] == "timeout"


async def test_ask_without_bridge_auto_rejects(agent, ctx):
    """无事件通道（非流式 / 未注入 bridge）：ask 自动拒绝（不留 pending）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    )

    assert result is not None
    assert result.tool_results[0].is_error is True
    assert "无交互通道" in str(result.tool_results[0].content)
    assert service.pending_for_session("sess-1") == []


async def test_ask_always_caches_rule_for_run(agent, ctx, recorder):
    """always：本 run 内同 pattern 后续调用直接放行（PASS）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="first")],
    ))
    await _respond_when_pending(service, "sess-1", "always")
    result = await task
    assert result is not None
    assert result.tool_results[0].is_error is False

    # 第二次同 pattern → approved 规则 allow → 全批 allow → None（PASS）
    result2 = await guard(
        ctx, turn=2,
        tool_calls=[_tc("write", path="a.py", content="second")],
    )
    assert result2 is None

    # discard_run 后缓存失效，重新 ask
    guard.discard_run("run-1")
    task3 = asyncio.create_task(guard(
        ctx, turn=3,
        tool_calls=[_tc("write", path="a.py", content="third")],
    ))
    await _respond_when_pending(service, "sess-1", "reject")
    result3 = await task3
    assert result3 is not None
    assert result3.tool_results[0].is_error is True


async def test_mixed_batch_allow_and_ask(agent, ctx, recorder):
    """混合批：allow 项与批准项都经 executor 执行，结果按位对齐。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    read_tool = agent.tool_registry.get("read")
    write_tool = agent.tool_registry.get("write")
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[
            _tc("read", path="a.py"),
            _tc("write", path="b.py", content="x"),
        ],
    ))
    await _respond_when_pending(service, "sess-1", "once")
    result = await task

    assert read_tool.executed == [{"path": "a.py"}]
    assert write_tool.executed == [{"path": "b.py", "content": "x"}]
    assert result is not None
    assert len(result.tool_results) == 2
    assert all(r.is_error is False for r in result.tool_results)


# ── bash：pattern 语义（子命令完整文本 + arity 前缀记忆）──


async def test_bash_ask_pattern_is_full_segment_text(agent, ctx, recorder):
    """ask 的 pattern = 首个 ask 子命令的完整文本（非首词，同 opencode）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_DOCKER)
    _bridge_ctx(ctx, recorder)

    # git → defaults allow
    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="git status")],
    )
    assert result is None

    # docker → ask，pattern 为子命令全文
    task = asyncio.create_task(guard(
        ctx, turn=2,
        tool_calls=[_tc("bash", command="docker ps -a")],
    ))
    request_id = await _respond_when_pending(service, "sess-1", "reject")
    result = await task

    request = recorder.of_type(PERMISSION_REQUEST)[0]
    assert request["request_id"] == request_id
    assert request["permission"] == "bash"
    assert request["pattern"] == "docker ps -a"
    assert request["patterns"] == ["docker ps -a"]
    # docker arity=2 → 记忆前缀是 "docker ps *"（同 opencode BashArity）
    assert request["always_patterns"] == ["docker ps *"]
    assert result is not None
    assert result.tool_results[0].is_error is True


async def test_bash_chain_asks_with_all_segment_patterns(agent, ctx, recorder):
    """复合命令 ask：patterns 带全部子命令文本，pattern 为首个 ask 段。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_DOCKER)
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="ls raw && docker ps")],
    ))
    await _respond_when_pending(service, "sess-1", "reject")
    result = await task

    request = recorder.of_type(PERMISSION_REQUEST)[0]
    assert request["pattern"] == "docker ps"
    assert request["patterns"] == ["ls raw", "docker ps"]
    assert request["always_patterns"] == ["ls *", "docker ps *"]
    assert result is not None
    assert result.tool_results[0].is_error is True


async def test_bash_command_substitution_stays_in_pattern(agent, ctx, recorder):
    """命令替换留在 pattern 文本内：审批展示真实完整命令。"""
    ruleset = build_ruleset() + [
        Rule(permission="bash", pattern="echo *", action="ask"),
    ]
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ruleset)
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="echo $(rm -rf /tmp/x)")],
    ))
    await _respond_when_pending(service, "sess-1", "reject")
    result = await task

    request = recorder.of_type(PERMISSION_REQUEST)[0]
    assert request["pattern"] == "echo $(rm -rf /tmp/x)"
    assert result is not None
    assert result.tool_results[0].is_error is True


async def test_bash_always_caches_arity_prefix(agent, ctx, recorder):
    """always 记忆 arity 前缀（docker ps *）：同前缀新命令免审。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_DOCKER)
    _bridge_ctx(ctx, recorder)

    # docker ps → always，记忆 "docker ps *"（docker arity=2）
    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="docker ps")],
    ))
    await _respond_when_pending(service, "sess-1", "always")
    result = await task
    assert result is not None
    assert result.tool_results[0].is_error is False

    # 同前缀新命令 → approved "docker ps *" allow → PASS
    result2 = await guard(
        ctx, turn=2,
        tool_calls=[_tc("bash", command="docker ps -a")],
    )
    assert result2 is None

    # 不同子命令（docker images）不命中 "docker ps *" → 重新 ask
    task3 = asyncio.create_task(guard(
        ctx, turn=3,
        tool_calls=[_tc("bash", command="docker images")],
    ))
    await _respond_when_pending(service, "sess-1", "reject")
    result3 = await task3
    assert result3 is not None
    assert result3.tool_results[0].is_error is True


async def test_sandbox_exec_alias_maps_to_bash(agent, ctx, recorder):
    """sandbox_exec 工具名映射到 bash 权限维度。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("sandbox_exec", command="ls -la")],
    )
    assert result is None  # bash 默认 allow


async def test_bash_readonly_chain_passes_without_ask(agent, ctx, recorder):
    """默认预设下复合命令（用户截图 echo && ls 场景）整链放行。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)
    _bridge_ctx(ctx, recorder)

    for command in [
        'echo "Files in raw/docs:" && ls raw/docs/',
        "find . -type d | head -30",
        "ls -la | wc -l",
        "cat a.py; echo done",
        "pytest -q 2>&1 | tail -5",
    ]:
        result = await guard(
            ctx, turn=1,
            tool_calls=[_tc("bash", command=command)],
        )
        assert result is None, f"应免审批: {command}"
    assert recorder.of_type(PERMISSION_REQUEST) == []


async def test_discard_run_cleans_service_pending():
    """guard.discard_run 同步清理 service 中该 run 的挂起请求。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(object(), service)  # agent 不参与该路径
    service.create_request(
        session_id="s", run_id="run-x", permission="write",
        pattern="x", tool_name="write", tool_call_id="t", tool_args={},
    )
    guard.discard_run("run-x")
    assert service.pending_for_session("s") == []


# ── bridge 通道本身 ────────────────────────────────────


async def test_bridge_handler_feeds_executor_events(agent, ctx, recorder):
    """批准执行时 bridge.handler 传给 executor → 工具启动事件发出。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE)
    write_tool = agent.tool_registry.get("write")
    _bridge_ctx(ctx, recorder)

    task = asyncio.create_task(guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    ))
    await _respond_when_pending(service, "sess-1", "once")
    result = await task

    assert write_tool.executed == [{"path": "a.py", "content": "x"}]
    assert result is not None
    assert result.tool_results[0].is_error is False
    # executor 收到 handler：工具启动事件已发
    assert [name for _id, name in recorder.tool_starts] == ["write"]


# ── 静默放行模式（silent_allow）──────────────────────


async def test_silent_allow_passes_ask_rules_without_card(agent, ctx, recorder):
    """静默模式下 ask 规则视同 allow：不弹卡片、直接放行走真实执行。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_WRITE, silent_allow=True)
    _bridge_ctx(ctx, recorder)

    for tc in [
        _tc("write", path="a.py", content="x"),   # ASK_WRITE ask 项
        _tc("read", path="/ws/alice/.env"),          # defaults 唯一 ask 项
    ]:
        result = await guard(ctx, turn=1, tool_calls=[tc])
        assert result is None, f"静默放行: {tc.name}"
    assert recorder.of_type(PERMISSION_REQUEST) == []


async def test_silent_allow_passes_ask_bash_segments(agent, ctx, recorder):
    """静默模式 bash 复合命令：ask 子命令不再短路，整条放行。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=ASK_DOCKER, silent_allow=True)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="docker ps && ls -la")],
    )

    assert result is None
    assert recorder.of_type(PERMISSION_REQUEST) == []


async def test_silent_allow_still_denies_hard_rules(agent, ctx, recorder):
    """静默只豁免 ask：deny 硬禁令原样拒绝（plan 只读 / curl deny）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, ruleset=plan_ruleset(), silent_allow=True)
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("write", path="a.py", content="x")],
    )

    assert result is not None
    assert result.tool_results[0].is_error is True
    assert recorder.of_type(PERMISSION_REQUEST) == []


# ── 裸 pip 硬禁令（高于 silent / always / 审批）──────


async def test_pip_install_denied_even_full_allow(agent, ctx, recorder):
    """默认全放行规则下，裸 pip 安装仍被拒：deny 详情带 venv+uv 指引。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service)  # build_ruleset：全 allow
    bash_tool = agent.tool_registry.get("bash")
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="pip install requests")],
    )

    assert result is not None
    denied = result.tool_results[0]
    assert denied.is_error is True
    assert "uv venv" in str(denied.content)      # venv 建法指引
    assert "uv pip install" in str(denied.content)
    assert bash_tool.executed == []
    # pip 拒绝不弹审批卡片（不可审批绕过）
    assert recorder.of_type(PERMISSION_REQUEST) == []


async def test_pip_install_denied_in_silent_mode(agent, ctx, recorder):
    """静默模式也拦不住裸 pip：硬禁令高于 silent flag。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, silent_allow=True)
    _bridge_ctx(ctx, recorder)

    for command in [
        "pip install requests",
        "sudo pip install requests",
        "PIP_INDEX_URL=http://x pip install requests",
        "make build && pip install requests",
        "python3 -m pip install requests",
    ]:
        result = await guard(
            ctx, turn=1,
            tool_calls=[_tc("bash", command=command)],
        )
        assert result is not None, f"硬禁令应拦: {command}"
        assert result.tool_results[0].is_error is True


async def test_pip_hard_deny_beats_approved_always(agent, ctx, recorder):
    """run 级 always 记忆也绕不过 pip 硬禁令（评估先于 approved 规则）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(
        agent, service,
        ruleset=build_ruleset() + [
            Rule(permission="bash", pattern="pip *", action="allow"),
        ],
    )
    _bridge_ctx(ctx, recorder)

    result = await guard(
        ctx, turn=1,
        tool_calls=[_tc("bash", command="pip install requests")],
    )

    assert result is not None
    assert result.tool_results[0].is_error is True


async def test_uv_pip_still_allowed(agent, ctx, recorder):
    """硬禁令只拦裸 pip：uv 形态的依赖安装照常放行（正常工具链）。"""
    service = PermissionService(timeout_seconds=2)
    guard = _guard(agent, service, silent_allow=False)
    _bridge_ctx(ctx, recorder)

    for command in [
        "uv venv .venv",
        "uv pip install --python .venv requests",
        "uv add requests",
        "pip list",
    ]:
        result = await guard(
            ctx, turn=1,
            tool_calls=[_tc("bash", command=command)],
        )
        assert result is None, f"uv/只读放行: {command}"
