"""BidAgent 测试 — coding 底座上的标书框架生成 agent（空壳形态）。

专属 commands / skills / tools 均未挂载：验证空壳下 command_dir 回落、
ark discovery 注册条件（不实例化，避免 LLM/DB 副作用）、底座能力
全继承。后续补充 commands/skills 后在此扩展。
"""

from __future__ import annotations

from pathlib import Path

from ark_agentic import BaseAgent

from egis_opencode.agents.bid_agent import BidAgent
from egis_opencode.workspace.commands import discover_commands


def test_bid_agent_identity():
    assert BidAgent.agent_id == "bid"
    assert "标书" in BidAgent.agent_name
    # 底座能力全继承（不声明专属 max_turns 等即用 CodingBaseAgent 默认）
    assert BidAgent.mode == "build"
    assert BidAgent.max_turns == 200
    assert "标书" in BidAgent.system_protocol


def test_bid_agent_discovery_conditions():
    """ark discovery 的全部注册判定（discover_agents 真实走 cls()，
    测试里验证注册条件本身而非跑完整 discovery）：
    BaseAgent 子类 + 自身 __dict__ 声明 agent_id + 模块在扫描包内
    （__module__ 前缀同时排除 __init__.py 的 re-export 重复）。
    """
    assert issubclass(BidAgent, BaseAgent)
    assert "agent_id" in BidAgent.__dict__
    assert BidAgent.__module__.startswith("egis_opencode.agents.")


def test_bid_agent_no_builtin_commands(tmp_path: Path):
    """空壳形态：无 commands/ 目录时内置命令为空（双层发现安全回落）。"""
    # skills_dir 是实例 property（类访问拿 property 对象）——按约定拼路径
    import egis_opencode.agents.bid_agent as pkg

    agent_dir = Path(pkg.__file__).parent
    assert not (agent_dir / "skills").is_dir()  # 无 agent 私有 skills
    assert not (agent_dir / "commands").is_dir()
    # command_dir 不存在 → discover 只扫 workspace（空目录 → 无命令）
    commands = discover_commands(tmp_path, agent_dir / "commands")
    assert commands == {}
