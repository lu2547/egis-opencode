"""bash 工具 — SandboxManager 薄适配 + 本地目录直连执行。

两种执行形态（context["workspace:root"] 决定）：
- 多租户（默认）：``BashTool.execute`` → ``SandboxManager.execute_shell_tool``
  （process/docker/opensandbox 后端、workspace 守卫、输出截断均由 ark
  sandbox 承担）
- 锚定（本地/项目目录会话）：在 egis 层用 subprocess 直连执行，
  cwd 锚在锁定目录，路径守卫/超时/输出截断与 sandbox 对齐
  （ark sandbox root 读的是配置文件，无按调用覆盖通道，且 ark 禁改）

两种形态都补充 ``tool_digest`` custom 事件与 LLM digest。

绑定时机：agent 构造早于 SandboxPlugin.start()，故 ``SandboxBinding``
为进程级 holder，由 ``CodingPlugin.start()``（排在 SandboxPlugin 之后）
注入 manager —— 延迟绑定（Late Binding）模式。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ....config import settings
from ....events import TOOL_DIGEST, tool_digest_payload
from ....permissions.rules import PIP_VIOLATION_GUIDE, find_pip_violation
from ....workspace import WorkspacePathError
from .base import CodingTool, _anchored_root_from_context
from .truncate import truncate_output

if TYPE_CHECKING:
    from ark_agentic.plugins.sandbox.manager import SandboxManager
    from ark_agentic.core.types import ToolCall

logger = logging.getLogger(__name__)

#: stdout 摘要展示长度（完整输出仍进 LLM 结果，由双层截断收口：
#: bash_max_output_bytes 字节层尾部保留 + truncate_output 语义层截断）
_STDOUT_TAIL_CHARS = 2000

#: 裸 pip 检测与拒绝文案的单一权威源在 permissions/rules.py
#: （guard 硬禁令与本工具入口拦截共用）；旧名保留作兼容导出
_find_bare_pip_install = find_pip_violation
_PIP_DENIED_GUIDE = PIP_VIOLATION_GUIDE


class SandboxBinding:
    """进程级 SandboxManager holder（CodingPlugin 启动时注入）。"""

    def __init__(self) -> None:
        self._manager: "SandboxManager | None" = None

    def bind(self, manager: "SandboxManager") -> None:
        self._manager = manager

    @property
    def manager(self) -> "SandboxManager | None":
        return self._manager

    def reset(self) -> None:
        self._manager = None


#: 进程级单例（CodingPlugin 与 BashTool 共享）
sandbox_binding = SandboxBinding()


class BashTool(CodingTool):
    """在沙箱 workspace 内执行 shell 命令。"""

    name = "bash"
    description = (
        "在沙箱工作区内执行 shell 命令（工作目录默认为用户 workspace 根，"
        "可用 workdir 指定子目录）。命令与路径必须留在工作区内；"
        "长时间命令用 timeout 参数控制（秒）。"
        "查看目录/文件/检索内容优先用 list、read、glob、grep 工具"
        "（结构化输出更可靠）；bash 用于真正需要 shell 的场景"
        "（构建、测试、安装等）。"
    )
    parameters = [
        ToolParameter(
            name="command", type="string",
            description="要执行的 shell 命令",
            required=True,
        ),
        ToolParameter(
            name="timeout", type="number",
            description="超时秒数（可选，默认由服务端配置）",
            required=False,
        ),
        ToolParameter(
            name="workdir", type="string",
            description="工作目录（相对用户 workspace，可选）",
            required=False,
        ),
    ]

    def __init__(self, *, agent_id: str) -> None:
        self._agent_id = agent_id

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        # 裸 pip 守卫在分流前统一拦截（本地直连继承服务进程环境，沙箱
        # 同样不允许 —— 依赖必须由 uv 装进工作区 venv，见指引）
        command = str((tool_call.arguments or {}).get("command") or "")
        bare_pip = _find_bare_pip_install(command)
        if bare_pip is not None:
            return self._error(
                tool_call,
                f"拒绝执行「{bare_pip}」。{_PIP_DENIED_GUIDE}",
                digest="[tool:bash status=error] 裸 pip 被拒绝，请先建 venv 并改用 uv。",
                context=context,
            )

        anchored = _anchored_root_from_context(context)
        if anchored is not None:
            return await self._execute_local(tool_call, context, anchored)
        return await self._execute_sandbox(tool_call, context)

    async def _execute_local(
        self, tool_call, context, anchored: Path,
    ) -> AgentToolResult:
        """本地目录模式：subprocess 直连执行（cwd 锚在锁定目录）。"""
        args = tool_call.arguments or {}
        command = str(args.get("command") or "").strip()
        if not command:
            return self._error(
                tool_call, "command 参数缺失", context=context,
            )

        try:
            workdir = self._resolve_workdir(context, anchored, str(args.get("workdir") or ""))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)

        timeout = float(args.get("timeout") or settings.bash_timeout_seconds)
        self._emit_digest(
            context,
            tool_name="bash", tool_call_id=tool_call.id,
            display_type="bash", status="running",
            title=command[:120],
            command=command,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", "-c", command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return self._bash_error(
                tool_call, context, command, f"无法启动 shell: {exc}",
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            stdout, stderr = b"", b""
            timed_out = True

        limit = settings.bash_max_output_bytes
        # 对齐 opencode shell tail 策略：字节层保尾部（错误信息集中在
        # 输出末尾）；语义层（2000 行 / 50KB）再由 truncate_output 收口，
        # 超限全文落盘 .truncation/ 并引导 grep/read 分段处理
        stdout_text = _clip_bash_output(
            (stdout or b"")[-limit:].decode("utf-8", "replace"), anchored,
        )
        stderr_text = _clip_bash_output(
            (stderr or b"")[-limit:].decode("utf-8", "replace"), anchored,
        )
        payload = {
            "exit_code": proc.returncode if not timed_out else -1,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "timed_out": timed_out,
        }
        self._emit_digest(
            context,
            tool_name="bash", tool_call_id=tool_call.id,
            display_type="bash",
            status="error" if timed_out or payload["exit_code"] != 0 else "success",
            title=command[:120],
            command=command,
            exit_code=payload["exit_code"],
            stdout_tail=_tail(payload["stdout"]),
            note=("命令超时被终止" if timed_out else None),
        )
        # ark 语义：tool 消息发给 LLM 的 content 就是 llm_digest（显式 digest
        # 优先于 content）。bash 的 content 是 JSON payload（stdout 转义不可读），
        # 故 digest 显式携带可读全文：exit_code + stdout + stderr（均已过
        # truncate_output 语义收口，≤50KB + 落盘续读指引）
        if timed_out:
            digest = f"[tool:bash status=error] 命令超时（>{int(timeout)}s）。"
        else:
            digest = f"exit_code={payload['exit_code']}"
            if payload["stdout"].strip():
                digest += f"\n{payload['stdout'].strip()}"
            if payload["stderr"].strip():
                digest += f"\nstderr:\n{payload['stderr'].strip()}"
        return AgentToolResult.json_result(
            tool_call.id, payload,
            llm_digest=digest,
        )

    def _resolve_workdir(
        self, context, anchored: Path, raw_workdir: str,
    ) -> Path:
        """解析 workdir（空 → 锁定目录根；必须落在锁定目录内）。"""
        if not raw_workdir.strip():
            return anchored
        return self._resolve(context, raw_workdir)

    def _bash_error(
        self, tool_call, context, command: str, message: str,
    ) -> AgentToolResult:
        self._emit_digest(
            context,
            tool_name="bash", tool_call_id=tool_call.id,
            display_type="bash", status="error",
            title=command[:120], command=command,
            note=message,
        )
        # 已发过详细 bash digest（display_type/command/note），不再走
        # 通用 error digest —— 后到的通用帧会把前端卡片形态覆盖退化
        return self._error(tool_call, message)

    async def _execute_sandbox(
        self, tool_call, context,
    ) -> AgentToolResult:
        manager = sandbox_binding.manager
        if manager is None:
            return self._error(
                tool_call,
                "沙箱未启用（ENABLE_SANDBOX 或 sandbox.json 缺失），"
                "bash 工具不可用",
                digest="[tool:bash status=error] 沙箱未启用。",
                context=context,
            )

        args = tool_call.arguments or {}
        command = str(args.get("command") or "").strip()
        if not command:
            return self._error(
                tool_call, "command 参数缺失", context=context,
            )

        self._emit_digest(
            context,
            tool_name="bash", tool_call_id=tool_call.id,
            display_type="bash", status="running",
            title=command[:120],
            command=command,
        )

        try:
            result = await manager.execute_shell_tool(
                self._agent_id, tool_call, dict(context or {}),
            )
        except Exception as exc:  # noqa: BLE001 — 沙箱异常降级为工具错误
            logger.warning("bash sandbox execution failed: %s", exc, exc_info=True)
            return self._bash_error(
                tool_call, context, command, f"沙箱执行异常: {exc}",
            )

        payload = _parse_shell_payload(result)
        self._emit_digest(
            context,
            tool_name="bash", tool_call_id=tool_call.id,
            display_type="bash",
            status="success" if payload.get("exit_code") == 0 else "error",
            title=command[:120],
            command=command,
            exit_code=payload.get("exit_code"),
            stdout_tail=_tail(payload.get("stdout", "")),
        )
        return result


def _clip_bash_output(text: str, workspace_root: Path) -> str:
    """bash 输出语义层收口（对齐 opencode Truncate.output，tail 方向）。

    未超限原样返回；超限（行数/字节）时全文落盘 ``<root>/.truncation/``
    并拼接续读指引。bash 工具仅在 build agent（有 task 工具）下挂载，
    故 hint 走 Task 委派分支。
    """
    clipped = truncate_output(
        text, workspace_root=workspace_root,
        direction="tail", has_task=True,
    )
    return clipped.content


def _parse_shell_payload(result: AgentToolResult) -> dict[str, Any]:
    """从 sandbox shell 结果中解析 {exit_code, stdout, stderr, timed_out}。

    ``execute_shell_tool`` 返回 ``json_result``（content 为 dict）；
    兼容 str 形态（旧序列化/测试假件）。
    """
    content = result.content
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content:
        try:
            payload = json.loads(content)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _tail(text: str, limit: int = _STDOUT_TAIL_CHARS) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


__all__ = [
    "BashTool", "SandboxBinding", "sandbox_binding",
    "TOOL_DIGEST", "tool_digest_payload", "_find_bare_pip_install",
]
