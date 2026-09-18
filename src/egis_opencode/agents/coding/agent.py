"""CodingAgent — coding 底座上的编码智能体薄壳（build / plan 双模式）。

工具集 / 权限 / run 钮子 / skills 挂载全部由 ``core.CodingBaseAgent``
提供；本模块只声明身份与模式差异：
- ``CodingAgent``（agent_id="coding"，build）：全量工具，默认放行。
- ``CodingPlanAgent``（agent_id="coding-plan"，plan）：只读工具集，
  写操作连工具都不注册（子 agent 继承该 registry，天然只读）。

两个类经 ark discovery（``agent_id`` 在自身 ``__dict__``）自动注册。
"""

from __future__ import annotations

from typing import ClassVar

from ...core import CodingBaseAgent, base_prompt
from ...core.tools import CodingMode


class CodingAgent(CodingBaseAgent):
    """编码智能体（build 模式，默认）。"""

    agent_id = "coding"
    agent_name = "编码智能体"
    agent_description = (
        "在沙箱工作区内读代码、改代码、执行命令的编码智能体。"
    )
    system_protocol = base_prompt("system.md")

    #: 权限模式（build/plan）→ 工具集与权限预设
    mode: ClassVar[CodingMode] = "build"


class CodingPlanAgent(CodingAgent):
    """规划智能体（plan 模式，只读）。"""

    agent_id = "coding-plan"
    agent_name = "规划智能体（只读）"
    agent_description = (
        "只读的规划智能体：检索与分析代码、产出技术方案，"
        "不做任何修改性操作。"
    )
    system_protocol = base_prompt("plan.md")
    mode: ClassVar[CodingMode] = "plan"


__all__ = ["CodingAgent", "CodingPlanAgent"]
