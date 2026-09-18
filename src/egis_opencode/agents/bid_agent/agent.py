"""BidAgent — coding 底座上的标书框架生成智能体。

工具集 / 权限 / skills 挂载由 ``core.CodingBaseAgent`` 提供；本 agent
差异仅在角色 prompt（标书框架生成，见 prompts/system.md）。
专属 commands / skills / tools 暂未挂载，后续按需补充：
- commands：``commands/`` 目录放 *.md 即被双层发现自动收录
- skills：``skills/`` 目录 + workspace skills 经 ark SkillLoader 挂载
"""

from __future__ import annotations

from pathlib import Path

from ...core import CodingBaseAgent

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class BidAgent(CodingBaseAgent):
    """标书框架生成智能体（解析需求 → 产出标书文档框架）。"""

    agent_id = "bid"
    agent_name = "标书框架生成智能体"
    agent_description = (
        "根据招标文件与项目需求生成标书文档框架："
        "章节结构、内容要点与撰写分工建议。"
    )
    system_protocol = (_PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


__all__ = ["BidAgent"]
