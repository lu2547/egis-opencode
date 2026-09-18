"""coding 工具集工厂 — 按模式（build/plan）组装工具。

- 公共：读 + 检索 + todo + question + task
- build：另加写（write/edit）+ bash
- plan：只读子集

task 工具自行构造 ark ``SpawnSubtasksTool``（``enable_subtasks`` 保持
False，避免 ark 重复注册默认 ``spawn_subtasks``）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ark_agentic.core.subtask.tool import SpawnSubtasksTool, SubtaskConfig
from ark_agentic.core.tools.base import AgentTool

from .bash import BashTool
from .files import EditTool, ListTool, ReadTool, WriteTool
from .question import QuestionTool
from .search import GlobTool, GrepTool
from .task import TaskTool
from .todo import TodoWriteTool

if TYPE_CHECKING:
    from ark_agentic.core.runtime.base_agent import BaseAgent

CodingMode = Literal["build", "plan"]


def create_coding_tools(
    agent: "BaseAgent", *, mode: CodingMode = "build",
) -> list[AgentTool]:
    """按模式组装 coding 工具集。

    Args:
        agent: 宿主 CodingAgent 实例（提供 agent_id / llm / session_manager）。
        mode: ``build`` 全量；``plan`` 只读子集。
    """
    tools: list[AgentTool] = [
        ReadTool(), GlobTool(), GrepTool(), ListTool(),
        TodoWriteTool(),
        QuestionTool(),
        TaskTool(
            SpawnSubtasksTool(
                runner=agent,
                session_manager=agent.session_manager,
                # 子任务内层超时与 executor 外层（agent.py tool_timeout=600）
                # 对齐：默认 300s 会截断“并行读多个大文档”类调研任务；
                # max_concurrent 提到 6 覆盖典型“每个文件一个子任务”场景
                # （默认 4 会排队，不失败但拉长总时长）
                config=SubtaskConfig(timeout_seconds=600.0, max_concurrent=6),
            )
        ),
    ]
    if mode == "build":
        tools.extend([WriteTool(), EditTool(), BashTool(agent_id=agent.agent_id)])
    return tools


__all__ = ["CodingMode", "create_coding_tools"]
