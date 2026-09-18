"""slash 命令发现与展开测试 — .opencode/commands 优先、$ARGUMENTS、AGENTS.md 注入。"""

from __future__ import annotations

from pathlib import Path

from egis_opencode.workspace import CommandSpec, discover_commands, expand_command


def _make_commands(root: Path) -> None:
    opencode = root / ".opencode/commands"
    opencode.mkdir(parents=True)
    (opencode / "ingest.md").write_text(
        "---\ndescription: 编译 raw 到 wiki\n---\n\n# ingest 命令\n\n"
        "用户参数：$ARGUMENTS\n\n扫描 raw/ 目录。",
        encoding="utf-8",
    )
    (opencode / "query.md").write_text(
        "---\ndescription: 检索回答\n---\n\n回答问题：$ARGUMENTS",
        encoding="utf-8",
    )
    claude = root / ".claude/commands"
    claude.mkdir(parents=True)
    (claude / "query.md").write_text(
        "---\ndescription: claude 版同名命令（应被 opencode 版覆盖优先级挡住）\n---\n\nclaude query",
        encoding="utf-8",
    )
    (claude / "lint.md").write_text("巡检 wiki 健康度。", encoding="utf-8")


def test_discover_opencode_takes_priority(tmp_path: Path):
    _make_commands(tmp_path)
    commands = discover_commands(tmp_path)
    assert set(commands) == {"ingest", "query", "lint"}
    assert commands["query"].description == "检索回答"  # opencode 版优先
    assert commands["lint"].description == ""  # 无 frontmatter


def test_expand_command_with_arguments(tmp_path: Path):
    _make_commands(tmp_path)
    prompt = expand_command("/ingest raw/web/foo.md", tmp_path)
    assert prompt is not None
    assert "用户参数：raw/web/foo.md" in prompt
    assert "$ARGUMENTS" not in prompt


def test_expand_wraps_execution_frame(tmp_path: Path):
    """命中命令时必须包执行指令框架 —— 消除“解读 vs 执行”歧义。

    实测缺陷：命令 md 是说明书文体，无参数触发时模型把展开文
    本误读成“关于命令的资料”，只输出实现方案不执行。
    """
    _make_commands(tmp_path)
    # 无参数（本缺陷的复现形态）
    prompt = expand_command("/ingest", tmp_path)
    assert prompt is not None
    assert prompt.startswith('<slash-command name="ingest">')
    assert "立即按其中流程开始执行" in prompt
    assert "不要只解读" in prompt
    assert prompt.rstrip().endswith("</slash-command>") or "</slash-command>" in prompt
    # 带参数同样带框架，且命令正文在框架内
    prompt = expand_command("/ingest raw/a.md", tmp_path)
    assert prompt is not None
    assert '<slash-command name="ingest">' in prompt
    assert "用户参数：raw/a.md" in prompt


def test_expand_unknown_command_has_no_frame(tmp_path: Path):
    """未命中分支是引导话术（指令已明确），不应套执行框架。"""
    _make_commands(tmp_path)
    prompt = expand_command("/nope", tmp_path)
    assert prompt is not None
    assert "<slash-command" not in prompt
    assert "未知命令 /nope" in prompt


def test_expand_command_without_args_placeholder(tmp_path: Path):
    _make_commands(tmp_path)
    prompt = expand_command("/lint", tmp_path)
    assert prompt is not None
    assert "巡检 wiki 健康度" in prompt


def test_expand_appends_args_when_no_placeholder(tmp_path: Path):
    (tmp_path / ".opencode/commands").mkdir(parents=True)
    (tmp_path / ".opencode/commands/say.md").write_text(
        "复述参数。", encoding="utf-8",
    )
    prompt = expand_command("/say hello world", tmp_path)
    assert "## 用户参数" in prompt
    assert "hello world" in prompt


def test_expand_unknown_command_lists_available(tmp_path: Path):
    _make_commands(tmp_path)
    prompt = expand_command("/nope", tmp_path)
    assert prompt is not None
    assert "未知命令 /nope" in prompt
    assert "/ingest" in prompt and "/query" in prompt and "/lint" in prompt


def test_non_command_message_returns_none(tmp_path: Path):
    _make_commands(tmp_path)
    assert expand_command("读一下 README", tmp_path) is None
    assert expand_command("ingest 不是斜杠开头", tmp_path) is None


def test_expand_injects_agents_md(tmp_path: Path):
    _make_commands(tmp_path)
    (tmp_path / "AGENTS.md").write_text(
        "# 规范\n\n- 始终简体中文\n- raw/ 只读", encoding="utf-8",
    )
    prompt = expand_command("/lint", tmp_path)
    assert prompt is not None
    assert "AGENTS.md" in prompt
    assert "raw/ 只读" in prompt


def test_discover_missing_root_is_empty(tmp_path: Path):
    assert discover_commands(tmp_path / "nope") == {}


# ── agent 内置命令（双层发现）────────────────────


def test_discover_agent_builtin_commands(tmp_path: Path):
    """agent 内置目录独立生效（workspace 无命令也可发现）。"""
    builtin = tmp_path / "agent-commands"
    builtin.mkdir()
    (builtin / "ingest.md").write_text(
        "---\ndescription: 摄取资料\n---\n\n摄取流程正文。", encoding="utf-8",
    )
    commands = discover_commands(tmp_path / "workspace", builtin)
    assert set(commands) == {"ingest"}
    assert commands["ingest"].description == "摄取资料"


def test_agent_builtin_takes_priority_over_workspace(tmp_path: Path):
    """同名时 agent 内置命令优先（平台命令不随工作目录变）。"""
    builtin = tmp_path / "agent-commands"
    builtin.mkdir()
    (builtin / "query.md").write_text("内置版 query", encoding="utf-8")
    ws = tmp_path / "workspace" / ".opencode" / "commands"
    ws.mkdir(parents=True)
    (ws / "query.md").write_text("workspace 版 query", encoding="utf-8")
    (ws / "custom.md").write_text("workspace 自有命令", encoding="utf-8")
    commands = discover_commands(tmp_path / "workspace", builtin)
    assert commands["query"].body == "内置版 query"
    assert "custom" in commands  # workspace 自定义仍可见


def test_expand_command_with_agent_builtin(tmp_path: Path):
    """expand 链路透传 agent 内置目录：/ingest 展开为命令正文。"""
    builtin = tmp_path / "agent-commands"
    builtin.mkdir()
    (builtin / "ingest.md").write_text(
        "摄取 $ARGUMENTS", encoding="utf-8",
    )
    prompt = expand_command("/ingest raw/a.md", tmp_path / "workspace", builtin)
    assert prompt is not None
    assert "摄取 raw/a.md" in prompt
    assert '<slash-command name="ingest">' in prompt


def test_discover_agent_dir_missing_falls_back(tmp_path: Path):
    """内置目录不存在（agent 无 commands）时纯 workspace 发现。"""
    ws = tmp_path / ".claude" / "commands"
    ws.mkdir(parents=True)
    (ws / "hello.md").write_text("你好", encoding="utf-8")
    commands = discover_commands(tmp_path, tmp_path / "nope")
    assert set(commands) == {"hello"}


def test_command_spec_frozen():
    spec = CommandSpec(name="x", description="", body="")
    try:
        spec.name = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("CommandSpec should be frozen")
