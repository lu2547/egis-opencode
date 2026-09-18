"""glob / grep 检索工具测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from ark_agentic.core.types import ToolCall

from egis_opencode.core.tools.search import GlobTool, GrepTool
from egis_opencode.events import TOOL_DIGEST

from tests.helpers import RecordingHandler

CTX = {"user:id": "alice"}


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall.create(name, arguments)


def _ctx(handler: RecordingHandler | None = None) -> dict:
    ctx = dict(CTX)
    if handler is not None:
        ctx["system:event_handler"] = handler
    return ctx


@pytest.fixture
def project(ws_user_root: Path) -> Path:
    proj = ws_user_root / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "main.py").write_text("def main():\n    return 'hello'\n")
    (proj / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (proj / "README.md").write_text("# project\n")
    (proj / ".git").mkdir()
    (proj / ".git" / "config").write_text("ignored")
    return proj


# ── glob ───────────────────────────────────────────────


async def test_glob_recursive_pattern(project):
    result = await GlobTool().execute(
        _call("glob", pattern="**/*.py", path="proj"), _ctx(),
    )
    assert result.is_error is False
    content = str(result.content)
    assert "src/main.py" in content
    assert "src/util.py" in content
    assert ".git" not in content  # 跳过隐藏目录


async def test_glob_no_match(project):
    result = await GlobTool().execute(
        _call("glob", pattern="**/*.rs", path="proj"), _ctx(),
    )
    assert result.is_error is False
    assert "无匹配" in str(result.content)


async def test_glob_missing_pattern(project):
    result = await GlobTool().execute(_call("glob"), _ctx())
    assert result.is_error is True


async def test_glob_escape_rejected(ws_user_root):
    result = await GlobTool().execute(
        _call("glob", pattern="**/*", path="../"), _ctx(),
    )
    assert result.is_error is True


async def test_glob_emits_digest(project, recorder):
    await GlobTool().execute(
        _call("glob", pattern="**/*.py", path="proj"), _ctx(recorder),
    )
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["tool_name"] == "glob"
    assert digest["display_type"] == "search"
    assert digest["result_count"] == 2
    assert len(digest["results_preview"]) == 2


# ── grep ───────────────────────────────────────────────


async def test_grep_matches_with_line_numbers(project):
    result = await GrepTool().execute(
        _call("grep", pattern="def \\w+", path="proj"), _ctx(),
    )
    content = str(result.content)
    assert "src/main.py:1:def main():" in content
    assert "src/util.py:1:def helper():" in content


async def test_grep_include_filter(project):
    result = await GrepTool().execute(
        _call("grep", pattern="project", path="proj", include="*.md"), _ctx(),
    )
    content = str(result.content)
    assert "README.md:1:" in content
    assert "main.py" not in content


async def test_grep_single_file_target(project):
    result = await GrepTool().execute(
        _call("grep", pattern="hello", path="proj/src/main.py"), _ctx(),
    )
    assert result.is_error is False
    assert "main.py:2" in str(result.content)


async def test_grep_invalid_regex(project):
    result = await GrepTool().execute(
        _call("grep", pattern="([unclosed"), _ctx(),
    )
    assert result.is_error is True
    assert "正则不合法" in str(result.content)


async def test_grep_no_hits(project):
    result = await GrepTool().execute(
        _call("grep", pattern="zzz_not_there", path="proj"), _ctx(),
    )
    assert result.is_error is False
    assert "无命中" in str(result.content)


async def test_grep_skips_binary(project):
    (project / "img.png").write_bytes(b"\x89PNG binary")
    result = await GrepTool().execute(
        _call("grep", pattern=".", path="proj"), _ctx(),
    )
    # 二进制被跳过，文本正常命中且不报错
    assert result.is_error is False
    assert "img.png" not in str(result.content)


async def test_grep_emits_digest(project, recorder):
    await GrepTool().execute(
        _call("grep", pattern="def", path="proj"), _ctx(recorder),
    )
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["tool_name"] == "grep"
    assert digest["result_count"] == 2


async def test_glob_digest_covers_more_than_30(project, recorder):
    """preview 全量展示（>30 条尾部标注总数）；llm_digest 带路径概要 —
    compaction 折叠后模型唯一可见的就是 digest，不带内容会导致模型
    "看不到列表"而反复重试。"""
    files_root = project / "many"
    files_root.mkdir(parents=True, exist_ok=True)
    for i in range(35):
        (files_root / f"f{i:02d}.py").write_text("x = 1\n")

    result = await GlobTool().execute(
        _call("glob", pattern="many/*.py", path="proj"), _ctx(recorder),
    )
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["result_count"] == 35
    preview = digest["results_preview"]
    # 前 30 条 + 总数提示行，而非静默截断
    assert len(preview) == 31
    assert preview[-1] == "… 共 35 条"
    # digest 概要带路径（折叠后模型仍可见列表开头）
    assert "many/f00.py" in result.llm_digest
    assert "many/f07.py" in result.llm_digest
