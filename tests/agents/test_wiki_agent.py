"""WikiAgent 测试 — coding 底座上的第一个业务 agent。

覆盖：ark discovery 注册（连字符目录名）、command_dir 内置命令、
reload_workspace_skills 挂 workspace skills（含一层项目子目录形态）。
不实例化 LLM（_construct 注入 MockChatModel），与 tests/api/conftest
同款装配模式。
"""

from __future__ import annotations

from pathlib import Path

from ark_agentic.core.session import SessionManager
from ark_agentic.core.skills.base import SkillConfig
from ark_agentic.core.skills.loader import SkillLoader
from ark_agentic.core.types import SkillLoadMode

from egis_opencode.agents.wiki_agent import WikiAgent
from egis_opencode.workspace.commands import discover_commands

from tests.helpers import MockChatModel


def _make_agent(tmp_path: Path) -> WikiAgent:
    """绕过 LLM 配置构造（agent 身份/命令/ skills 与 LLM 无关）。"""
    return WikiAgent._construct(
        llm=MockChatModel(responses=[]),
        session_manager=SessionManager(tmp_path / "sessions", agent_id="wiki"),
        skill_loader=SkillLoader(SkillConfig(
            agent_id="wiki", load_mode=SkillLoadMode.dynamic,
        )),
        agent_id="wiki",
    )


def test_wiki_agent_identity():
    assert WikiAgent.agent_id == "wiki"
    assert "Wiki" in WikiAgent.agent_name
    # 底座能力全继承（不声明专属 max_turns 等即用 CodingBaseAgent 默认）
    assert WikiAgent.mode == "build"
    assert WikiAgent.max_turns == 200


def test_wiki_command_dir_builtin_commands(tmp_path: Path):
    """command_dir 指向 agents/wiki-agent/commands，含 ingest/query/lint。"""
    agent = _make_agent(tmp_path)
    assert agent.command_dir.name == "commands"
    assert agent.command_dir.is_dir()
    commands = discover_commands(Path("/nonexistent"), agent.command_dir)
    assert set(commands) >= {"ingest", "query", "lint"}
    # frontmatter description 提取正常（面板展示用）
    assert commands["ingest"].description


def test_reload_workspace_skills_mounts_nested_project(tmp_path: Path):
    """部署形态：workspace 根 + 一层项目子目录的 .claude/skills。"""
    agent = _make_agent(tmp_path)
    project = tmp_path / "workspaces" / "llm-wiki" / ".claude" / "skills" / "ingest"
    project.mkdir(parents=True)
    (project / "SKILL.md").write_text(
        "---\nname: ingest\ndescription: 摄取技能\n---\n\n摄取正文。",
        encoding="utf-8",
    )
    agent.reload_workspace_skills(tmp_path / "workspaces")

    ids = {s.id for s in agent.skill_loader.list_skills()}
    assert "wiki.ingest" in ids  # ark 约定：agent_id.skill_name
    skill = agent.skill_loader.get_skill("wiki.ingest")
    assert skill is not None and "摄取正文" in skill.content


def test_reload_workspace_skills_same_dirs_skipped(
    tmp_path: Path, monkeypatch,
):
    """目录集合不变时零开销跳过（同会话连续提问常态）。"""
    agent = _make_agent(tmp_path)
    agent.reload_workspace_skills(tmp_path)

    calls = []
    monkeypatch.setattr(
        agent.skill_loader, "load_from_directories",
        lambda *a, **k: calls.append(a) or {},
    )
    agent.reload_workspace_skills(tmp_path)  # 同目录：跳过
    assert calls == []

    other = tmp_path / "other"
    (other / ".claude" / "skills").mkdir(parents=True)
    agent.reload_workspace_skills(other)  # 换目录：重扫
    assert len(calls) == 1


def test_skill_loader_hidden_dirs_skipped(tmp_path: Path):
    """一层扫描跳过隐藏目录（.cache 等）与 node_modules。"""
    agent = _make_agent(tmp_path)
    for noise in (".cache", "node_modules"):
        d = tmp_path / noise / ".claude" / "skills" / "bad"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: bad\ndescription: d\n---\n\nx", encoding="utf-8",
        )
    agent.reload_workspace_skills(tmp_path)
    assert not {s.id for s in agent.skill_loader.list_skills()}
