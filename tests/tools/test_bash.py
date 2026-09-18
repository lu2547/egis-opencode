"""bash 工具测试 — SandboxManager 薄适配（FakeManager + 延迟绑定生命周期）。

真实沙箱执行（process/docker 后端）由 ark sandbox 测试覆盖；此处验证
BashTool 自身：绑定缺失降级 / payload 解析 / digest 事件 / stdout 截断。
"""

from __future__ import annotations

from typing import Any

import pytest
from ark_agentic.core.types import AgentToolResult, ToolCall

from egis_opencode.agents.coding.tools.bash import (
    BashTool,
    SandboxBinding,
    sandbox_binding,
)
from egis_opencode.events import TOOL_DIGEST

from tests.helpers import RecordingHandler

CTX = {"user:id": "alice"}


def _call(**arguments) -> ToolCall:
    return ToolCall.create("bash", arguments)


def _ctx(handler: RecordingHandler | None = None) -> dict:
    ctx = dict(CTX)
    if handler is not None:
        ctx["system:event_handler"] = handler
    return ctx


class FakeSandboxManager:
    """duck-typed SandboxManager：脚本化 shell 执行结果。"""

    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        raise_exc: Exception | None = None,
    ) -> None:
        self._exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr
        self._timed_out = timed_out
        self._raise = raise_exc
        self.calls: list[tuple[str, ToolCall, dict[str, Any]]] = []

    async def execute_shell_tool(
        self, agent_id: str, tool_call: ToolCall, context: dict[str, Any],
    ) -> AgentToolResult:
        self.calls.append((agent_id, tool_call, dict(context)))
        if self._raise is not None:
            raise self._raise
        return AgentToolResult.json_result(
            tool_call.id,
            {
                "exit_code": self._exit_code,
                "stdout": self._stdout,
                "stderr": self._stderr,
                "timed_out": self._timed_out,
            },
        )


@pytest.fixture
def tool() -> BashTool:
    return BashTool(agent_id="coding")


@pytest.fixture(autouse=True)
def _clean_binding():
    """每个用例前后清空全局绑定（测试不互相污染）。"""
    sandbox_binding.reset()
    yield
    sandbox_binding.reset()


# ── 未绑定 ─────────────────────────────────────────────


async def test_unbound_sandbox_returns_error(tool):
    result = await tool.execute(_call(command="ls"), _ctx())
    assert result.is_error is True
    assert "沙箱未启用" in str(result.content)


# ── 正常执行 ───────────────────────────────────────────


async def test_success_passes_sandbox_result_and_digest(tool, recorder):
    manager = FakeSandboxManager(stdout="file1\nfile2\n")
    sandbox_binding.bind(manager)

    result = await tool.execute(
        _call(command="ls"), _ctx(recorder),
    )

    # 结果原样透传（sandbox 的 json_result）
    assert result.is_error is False
    assert result.content == {
        "exit_code": 0, "stdout": "file1\nfile2\n",
        "stderr": "", "timed_out": False,
    }
    # agent_id / context 透传给 manager
    assert manager.calls[0][0] == "coding"
    # digest：running + success 两帧
    digests = recorder.of_type(TOOL_DIGEST)
    assert [d["status"] for d in digests] == ["running", "success"]
    final = digests[1]
    assert final["display_type"] == "bash"
    assert final["exit_code"] == 0
    assert final["stdout_tail"] == "file1\nfile2\n"
    assert final["command"] == "ls"


async def test_nonzero_exit_marks_error_digest(tool, recorder):
    sandbox_binding.bind(
        FakeSandboxManager(exit_code=1, stdout="", stderr="boom"),
    )
    result = await tool.execute(_call(command="make build"), _ctx(recorder))

    assert result.is_error is False  # 工具本身没失败，命令失败信息在 payload 里
    digests = recorder.of_type(TOOL_DIGEST)
    assert digests[-1]["status"] == "error"
    assert digests[-1]["exit_code"] == 1


async def test_sandbox_exception_degrades_to_error(tool, recorder):
    sandbox_binding.bind(
        FakeSandboxManager(raise_exc=RuntimeError("backend down")),
    )
    result = await tool.execute(_call(command="ls"), _ctx(recorder))

    assert result.is_error is True
    assert "沙箱执行异常" in str(result.content)
    digests = recorder.of_type(TOOL_DIGEST)
    assert digests[-1]["status"] == "error"
    assert "backend down" in digests[-1]["note"]


# ── 参数校验 ───────────────────────────────────────────


async def test_missing_command_rejected(tool):
    sandbox_binding.bind(FakeSandboxManager())
    result = await tool.execute(_call(), _ctx())
    assert result.is_error is True
    assert "command" in str(result.content)


async def test_stdout_tail_truncated_in_digest(tool, recorder):
    # 超过 2000 字符的输出取尾部（… + 最后 2000 字符）
    long_output = "x" * 3000
    sandbox_binding.bind(FakeSandboxManager(stdout=long_output))
    await tool.execute(_call(command="cat big"), _ctx(recorder))

    tail = recorder.of_type(TOOL_DIGEST)[-1]["stdout_tail"]
    assert len(tail) == 2001  # … + 2000 字符
    assert tail.startswith("…")


# ── 本地目录模式（anchored 直连执行）───────────────


async def test_local_mode_runs_in_anchored_dir(tool, recorder, tmp_path):
    """workspace:root 锚定后绕过沙箱，subprocess 在锁定目录执行。"""
    result = await tool.execute(
        _call(command="pwd"),
        {**_ctx(recorder), "workspace:root": str(tmp_path)},
    )

    assert result.is_error is False
    assert result.content["exit_code"] == 0
    assert str(tmp_path) in result.content["stdout"]
    digests = recorder.of_type(TOOL_DIGEST)
    assert [d["status"] for d in digests] == ["running", "success"]
    assert digests[1]["display_type"] == "bash"


async def test_local_mode_workdir_resolved(tool, tmp_path):
    (tmp_path / "sub").mkdir()
    result = await tool.execute(
        _call(command="pwd", workdir="sub"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    assert result.content["exit_code"] == 0
    assert str(tmp_path / "sub") in result.content["stdout"]


async def test_local_mode_workdir_escape_rejected(tool, tmp_path):
    result = await tool.execute(
        _call(command="pwd", workdir="../outside"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    assert result.is_error is True
    assert "越出 workspace" in str(result.content)


async def test_local_mode_nonzero_exit_is_error_digest(tool, recorder, tmp_path):
    result = await tool.execute(
        _call(command="exit 3"),
        {**_ctx(recorder), "workspace:root": str(tmp_path)},
    )
    assert result.content["exit_code"] == 3
    assert recorder.of_type(TOOL_DIGEST)[-1]["status"] == "error"


async def test_local_mode_timeout_kills_process(tool, tmp_path):
    result = await tool.execute(
        _call(command="sleep 30", timeout=0.2),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    assert result.content["timed_out"] is True
    assert result.content["exit_code"] == -1


async def test_local_mode_bad_workdir_errors(tool, tmp_path):
    result = await tool.execute(
        _call(command="pwd", workdir="no-such-dir"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    # workdir 解析合法（不存在但未越界）→ subprocess cwd 失败 → 工具错误
    assert result.is_error is True


async def test_local_mode_ignores_sandbox_binding(tool, tmp_path):
    """锚定模式下即使沙箱已绑定也走本地分支（manager 不被调用）。"""
    manager = FakeSandboxManager(stdout="SHOULD NOT APPEAR")
    sandbox_binding.bind(manager)
    result = await tool.execute(
        _call(command="echo local"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    assert result.content["stdout"].strip() == "local"
    assert manager.calls == []


async def test_local_mode_output_truncated(tool, tmp_path):
    """大输出双层截断（对齐 opencode shell tail + Truncate.output）：
    字节层尾部保留（错误信息在末尾）→ 语义层 2000 行/50KB 截断，
    全文落盘 .truncation/ 并附续读指引。"""
    result = await tool.execute(
        _call(command="seq 1 100000"),  # ≈688KB、100000 行
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    stdout = result.content["stdout"]
    # 语义层截断后：preview + 截断量提示 + 续读指引，总量可控
    assert len(stdout) < 60_000
    assert "truncated" in stdout
    assert "Full output saved to" in stdout
    # 尾部优先：最后一段输出可见（错误信息通常在末尾）
    assert "100000" in stdout
    # 全文落盘（可用 read/grep 续读）
    saved = list((tmp_path / ".truncation").glob("tool_*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8").endswith("100000\n")


async def test_local_mode_small_output_untouched(tool, tmp_path):
    """小输出（50KB/2000 行内）完整保留，不截断不落盘。"""
    result = await tool.execute(
        _call(command="printf 'x%.0s' $(seq 1 30000)"),  # 单行 30000 字符
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    assert result.content["stdout"] == "x" * 30000
    assert not (tmp_path / ".truncation").exists()


# ── SandboxBinding 生命周期 ────────────────────────────


def test_binding_bind_and_reset():
    binding = SandboxBinding()
    assert binding.manager is None
    manager = FakeSandboxManager()
    binding.bind(manager)
    assert binding.manager is manager
    binding.reset()
    assert binding.manager is None


# ── 裸 pip 守卫（本地直连继承服务进程环境，必须入口拦截）───────


class TestBarePipGuard:
    """拦截所有裸 pip 安装形态；uv/只读子命令放行。"""

    @pytest.mark.parametrize(
        "command",
        [
            "pip install requests",
            "pip3 install requests",
            "pip install -r requirements.txt",
            "pip uninstall requests",
            "python -m pip install requests",
            "python3 -m pip install requests -q",
            "python3.12 -m pip install requests",
            "sudo pip install requests",
            "env pip install requests",
            "PIP_INDEX_URL=http://x pip install requests",
            "cd proj && pip install requests",
            "make build || pip install requests",
            "echo hi; pip install requests",
            "cat setup.py | pip install -e .",
        ],
    )
    def test_bare_pip_blocked(self, command):
        from egis_opencode.agents.coding.tools.bash import _find_bare_pip_install

        assert _find_bare_pip_install(command) is not None

    @pytest.mark.parametrize(
        "command",
        [
            "uv pip install requests",
            "uv add requests",
            "uv venv .venv",
            "uv pip install --python .venv requests",
            "uv run pytest",
            "pip list",
            "pip show requests",
            "pip --version",
            "uv run pip list",  # uv 管理的项目内只读检查
            "echo install",
            "grep pip install README.md",
        ],
    )
    def test_safe_commands_pass(self, command):
        from egis_opencode.agents.coding.tools.bash import _find_bare_pip_install

        assert _find_bare_pip_install(command) is None

    async def test_execute_blocks_before_reaching_sandbox(self, tool, recorder):
        """拦截发生在入口：沙箱管理器不应收到调用，错误带 uv 指引 + digest。"""
        manager = FakeSandboxManager(stdout="should not run")
        sandbox_binding.bind(manager)

        result = await tool.execute(
            _call(command="pip install requests"), _ctx(recorder),
        )

        assert result.is_error is True
        assert "uv venv" in str(result.content)
        assert "uv pip install" in str(result.content)
        assert manager.calls == [], "裸 pip 必须在到达执行后端前被拦截"

        digests = recorder.of_type(TOOL_DIGEST)
        assert digests and digests[-1]["status"] == "error"
        assert "uv" in digests[-1]["note"]

    async def test_execute_allows_uv_pip(self, tool, recorder):
        """uv pip install 放行并正常到达沙箱执行。"""
        manager = FakeSandboxManager(stdout="installed")
        sandbox_binding.bind(manager)

        result = await tool.execute(
            _call(command="uv pip install --python .venv requests"),
            _ctx(recorder),
        )

        assert result.is_error is False
        assert len(manager.calls) == 1
