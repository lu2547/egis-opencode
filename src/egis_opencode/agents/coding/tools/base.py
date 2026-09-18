"""coding 工具集 — opencode 风格的文件与代码操作工具。

所有工具：
- ``visibility = "always"``：始终出现在 LLM 工具列表
- 路径参数经 ``WorkspacePaths`` 守卫解析（拒绝越界）
- 通过 ``ctx["system:event_handler"]`` 发射 ``tool_digest`` custom 事件
- 用户标识从 context 的 ``user:id`` / ``user_id`` 读取（与 ark/sandbox 约定一致）

工作目录两种形态（context["workspace:root"] 锚定覆盖）：
- 未设置（多租户）：``<workspace_root>/<user_id>``
- 已设置（本地/项目目录会话）：该目录本身即为工作区根
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ark_agentic.core.tools.base import AgentTool
from ark_agentic.core.types import AgentToolResult

from ....config import settings
from ....events import TOOL_DIGEST, tool_digest_payload
from ....workspace import WorkspacePathError, WorkspacePaths

if TYPE_CHECKING:
    from ark_agentic.core.types import ToolCall

logger = logging.getLogger(__name__)


def _user_id_from_context(context: dict[str, Any] | None) -> str:
    if not context:
        return "default"
    for key in ("user:id", "user_id", "user.id"):
        value = context.get(key)
        if value:
            return str(value)
    return "default"


def _anchored_root_from_context(context: dict[str, Any] | None) -> Path | None:
    """会话锁定的工作目录（chat 端点校验后注入的绝对路径）。"""
    value = (context or {}).get("workspace:root")
    if not value:
        return None
    try:
        root = Path(str(value)).resolve(strict=False)
    except OSError:
        return None
    return root if root.is_dir() else None


class CodingTool(AgentTool):
    """coding 工具基类 — workspace 解析 + tool_digest 事件发射。"""

    visibility = "always"
    group = "coding"

    # 子类在 __init__ 中赋值
    name = "coding_tool"
    description = "coding tool base"

    def _workspace(self, context: dict[str, Any] | None) -> WorkspacePaths:
        anchored = _anchored_root_from_context(context)
        if anchored is not None:
            return WorkspacePaths.anchored(anchored)
        return WorkspacePaths(
            root=settings.workspace_root_resolved,
            user_id=_user_id_from_context(context),
        )

    def _emit_digest(
        self, context: dict[str, Any] | None, **kwargs: Any,
    ) -> None:
        """经 executor 注入的 event_handler 发 tool_digest 事件（无 handler 时静默）。"""
        if not context:
            return
        handler = context.get("system:event_handler")
        if handler is None:
            return
        try:
            handler.on_custom_event(
                TOOL_DIGEST, tool_digest_payload(**kwargs),
            )
        except Exception:  # noqa: BLE001 — 事件失败不影响工具结果
            logger.debug("tool_digest emit failed", exc_info=True)

    def _error(
        self, tool_call: "ToolCall", message: str, *,
        digest: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentToolResult:
        """构造错误结果；提供 ``context`` 时同步发 error digest 事件。

        错误也必须在 UI 可见：``tool_call_result`` 帧只携带裸 content，
        前端无法区分成败，卡片状态与错误文案完全依赖 digest 事件 ——
        否则错误卡片会被前端兑底标成“成功（无详情）”，错误信息丢失。
        """
        if context is not None:
            self._emit_digest(
                context,
                tool_name=self.name, tool_call_id=tool_call.id,
                status="error", title=self.name, note=message,
            )
        return AgentToolResult.error_result(
            tool_call.id, message,
            tool_name=self.name, llm_digest=digest,
        )

    def _path_error_result(
        self, tool_call: "ToolCall", exc: WorkspacePathError,
    ) -> AgentToolResult:
        return self._error(
            tool_call,
            f"路径不合法或越出 workspace：{exc}",
            digest=f"[tool:{self.name} status=error] 路径被拒绝（越出 workspace）。",
        )

    def _resolve(
        self, context: dict[str, Any] | None, raw_path: str,
    ) -> Path:
        return self._workspace(context).resolve(raw_path)

    def _display(
        self, context: dict[str, Any] | None, path: Path,
    ) -> str:
        """相对当前工作区根的展示路径。"""
        try:
            return str(path.relative_to(self._workspace(context).user_root))
        except ValueError:
            return str(path)
