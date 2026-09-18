"""CodingAgent — opencode 风格的编码智能体（build / plan 双模式）。

模式即权限预设（策略模式，规则对齐 opencode agent.ts defaults ——
默认全放行，仅敏感项如读 .env 询问）：
- ``CodingAgent``（agent_id="coding"，build）：全量工具，默认放行。
- ``CodingPlanAgent``（agent_id="coding-plan"，plan）：只读工具集，
  写操作连工具都不注册（子 agent 继承该 registry，天然只读）。

两个类经 ark discovery（``agent_id`` 在自身 ``__dict__``）自动注册。
``enable_subtasks`` 保持 False —— task 工具由 ``create_coding_tools``
自行构造 SpawnSubtasksTool 定制注入，避免 ark 重复注册。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from ark_agentic import BaseAgent
from ark_agentic.core.runtime.callbacks import RunnerCallbacks
from ark_agentic.core.runtime._runner_types import RunnerConfig

from ...permissions import PermissionGuard, ruleset_for_mode
from ...permissions.service import permission_service
from ...sessions.title import TitleGenerator, title_store
from .tools import CodingMode, create_coding_tools

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class CodingAgent(BaseAgent):
    """编码智能体（build 模式，默认）。"""

    agent_id = "coding"
    agent_name = "编码智能体"
    agent_description = (
        "在沙箱工作区内读代码、改代码、执行命令的编码智能体。"
    )
    system_protocol = _load_prompt("system.md")

    #: 权限模式（build/plan）→ 工具集与权限预设
    mode: ClassVar[CodingMode] = "build"
    #: ReAct 循环上限。opencode 主循环无轮数限制（模型自然结束/用户 abort），
    #: 此处仅作失控安全阀 —— 取值需覆盖长任务的探索+执行全程
    #: （实测中型任务纯探索期即可耗 60+ 轮），不应成为正常任务的边界。
    max_turns: ClassVar[int] = 200
    enable_subtasks: ClassVar[bool] = False

    def build_tools(self):
        return create_coding_tools(self, mode=self.mode)

    def build_runner_config(self) -> RunnerConfig:
        """执行钮子定制：工具层超时对齐长耗时工具。

        框架默认 ``tool_timeout=30`` 会把 task（并行读多个 docx）整体
        cancel 掉 —— 且取消发生在事件发完后，前端子任务卡片永久转圈。
        600s 覆盖并行子任务全程；bash 自身另有 timeout 参数控制长命令，
        此处仅作外层安全阀。
        """
        config = super().build_runner_config()
        return RunnerConfig(
            **{
                **config.__dict__,
                "tool_timeout": 600.0,
            }
        )

    def build_callbacks(self) -> RunnerCallbacks | None:
        guard = PermissionGuard(
            agent=self,
            service=permission_service,
            base_ruleset=ruleset_for_mode(self.mode),
        )

        async def _cleanup_run(ctx: Any, **kwargs: Any) -> None:
            """after_agent：run 结束清理授权缓存与挂起请求。"""
            guard.discard_run(ctx.run_id)

        return RunnerCallbacks(
            before_tool=[guard],
            after_agent=[_cleanup_run, TitleGenerator(title_store)],
        )


class CodingPlanAgent(CodingAgent):
    """规划智能体（plan 模式，只读）。"""

    agent_id = "coding-plan"
    agent_name = "规划智能体（只读）"
    agent_description = (
        "只读的规划智能体：检索与分析代码、产出技术方案，"
        "不做任何修改性操作。"
    )
    system_protocol = _load_prompt("plan.md")
    mode: ClassVar[CodingMode] = "plan"


__all__ = ["CodingAgent", "CodingPlanAgent"]
