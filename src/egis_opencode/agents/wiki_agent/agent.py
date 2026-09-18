"""WikiAgent — coding 底座上的 LLM Wiki 维护智能体。

工具集 / 权限 / skills 挂载由 ``core.CodingBaseAgent`` 提供；本 agent
差异仅在：
- 角色 prompt（wiki 维护者，见 prompts/system.md）
- 内置命令 ``commands/``：/ingest /query /lint（源自 llm-wiki 项目
  的 .claude/skills/*/SKILL.md，提炼为 commands 形态）
- workspace 里用户自带的 skills（.opencode/skills 等）经 ark
  SkillLoader 动态挂载，模型可 read_skill 按需加载
"""

from __future__ import annotations

from pathlib import Path

from ...core import CodingBaseAgent

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class WikiAgent(CodingBaseAgent):
    """LLM Wiki 知识库维护智能体（摄取 / 检索 / 巡检）。"""

    agent_id = "wiki"
    agent_name = "Wiki 知识库智能体"
    agent_description = (
        "维护 LLM Wiki 知识库：摄取 raw/ 资料、检索回答、"
        "健康度巡检。"
    )
    system_protocol = (_PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


__all__ = ["WikiAgent"]
