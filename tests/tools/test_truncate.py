"""truncate_output / truncate_line 单元测试。

对齐 opencode tool/truncate.ts 契约：
- 未超限原样返回
- 行数/字节超限：head/tail 方向 preview + 截断量 + 续读指引 + 全文落盘
- 截断文件 7 天保留（懒清理）
- 单行截断（巨行防线）
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time

from egis_opencode.agents.coding.tools.truncate import (
    TRUNCATION_DIRNAME,
    truncate_line,
    truncate_output,
)
from egis_opencode.config import settings


@contextlib.contextmanager
def _override(**overrides):
    """frozen dataclass 的临时字段覆盖（整体替换 __dict__，退出恢复）。"""
    old = dict(settings.__dict__)
    object.__setattr__(
        settings, "__dict__",
        dataclasses.replace(settings, **overrides).__dict__,
    )
    try:
        yield
    finally:
        object.__setattr__(settings, "__dict__", old)

# ── truncate_line ─────────────────────────────


def test_truncate_line_untouched_within_limit():
    assert truncate_line("short line") == "short line"


def test_truncate_line_cuts_huge_line():
    huge = "x" * 5000
    out = truncate_line(huge)
    assert out.startswith("x" * 2000)
    assert out.endswith("... (line truncated to 2000 chars)")
    assert len(out) < 2100


# ── truncate_output：未超限 ─────────────────────────


def test_output_within_limits_unchanged(tmp_path):
    text = "\n".join(f"line{i}" for i in range(100))
    result = truncate_output(text, workspace_root=tmp_path)
    assert result.truncated is False
    assert result.content == text
    assert result.output_path is None
    # 未超限不产生落盘目录
    assert not (tmp_path / TRUNCATION_DIRNAME).exists()


# ── truncate_output：行数超限 ───────────────────────


def test_output_lines_truncated_head(tmp_path):
    text = "\n".join(f"line{i}" for i in range(100))
    with _override(tool_output_max_lines=10):
        result = truncate_output(text, workspace_root=tmp_path, direction="head")

    assert result.truncated is True
    assert "line0" in result.content
    assert "line10" not in result.content
    assert "90 lines truncated" in result.content
    assert "Full output saved to" in result.content
    # 全文落盘且内容完整
    saved = tmp_path / TRUNCATION_DIRNAME
    files = list(saved.glob("tool_*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == text
    assert result.output_path == str(files[0])


def test_output_lines_truncated_tail(tmp_path):
    text = "\n".join(f"line{i}" for i in range(100))
    with _override(tool_output_max_lines=10):
        result = truncate_output(text, workspace_root=tmp_path, direction="tail")

    assert result.truncated is True
    assert "line99" in result.content  # 尾部保留
    assert "line0" not in result.content
    # tail 方向：截断量提示在开头
    assert result.content.startswith("...90 lines truncated...")


# ── truncate_output：字节超限 ───────────────────────


def test_output_bytes_truncated(tmp_path):
    text = "\n".join("y" * 500 for _ in range(20))  # ≈10KB
    with _override(tool_output_max_bytes=1024):
        result = truncate_output(text, workspace_root=tmp_path)

    assert result.truncated is True
    assert "bytes truncated" in result.content
    assert "Use Grep to search the full content" in result.content


def test_output_hint_task_variant(tmp_path):
    """has_task=True：hint 引导 Task 委派（opencode 同款文案分支）。"""
    text = "\n".join(f"l{i}" for i in range(50))
    with _override(tool_output_max_lines=5):
        result = truncate_output(text, workspace_root=tmp_path, has_task=True)

    assert "Use the Task tool" in result.content
    assert "Do NOT read the full file yourself" in result.content


# ── 截断文件保留策略 ─────────────────────────────


def test_stale_truncation_files_cleaned(tmp_path):
    """超过 7 天的截断文件被懒清理，新文件保留。"""
    stale = tmp_path / TRUNCATION_DIRNAME / "tool_stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    old = time.time() - 8 * 24 * 3600
    os.utime(stale, (old, old))

    with _override(tool_output_max_lines=5):
        truncate_output(
            "\n".join(f"l{i}" for i in range(50)), workspace_root=tmp_path,
        )

    assert not stale.exists()
    assert list((tmp_path / TRUNCATION_DIRNAME).glob("tool_*.txt"))
