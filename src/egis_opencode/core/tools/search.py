"""glob / grep 检索工具（标准库实现，无外部依赖）。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ...config import settings
from ...events import diff_preview
from ...workspace import WorkspacePathError
from .base import CodingTool

#: grep 跳过的二进制扩展名（与 files.py 黑名单同源，读取失败亦降级跳过）
#: .docx 亦跳过 —— 提取文本请用 read（grep 不做文档提取）
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".woff", ".woff2",
    ".ttf", ".eot", ".mp3", ".mp4", ".avi", ".mov", ".sqlite", ".db",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}


def _glob_collect(base: Path, pattern: str, cap: int) -> tuple[list[Path], bool]:
    """glob 遍历（阻塞 IO）— 线程池执行体；返回 (matches, truncated)。"""
    matches: list[Path] = []
    for path in sorted(base.glob(pattern)):
        if path.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        matches.append(path)
        if len(matches) >= cap:
            return matches, True
    return matches, False


def _grep_scan(
    target: Path,
    regex: "re.Pattern[str]",
    include: str | None,
    max_results: int,
    workspace: Any,
) -> tuple[list[str], int]:
    """grep 扫描（阻塞 IO/CPU）— 线程池执行体；返回 (hits, scanned)。"""
    if target.is_file():
        files = [target]
    else:
        files = list(
            p for p in target.rglob("*")
            if p.is_file()
            and not any(part in _SKIP_DIRS for part in p.parts)
            and (not include or p.match(include))
        )

    hits: list[str] = []
    scanned = 0
    for file in sorted(files):
        if file.suffix.lower() in _SKIP_SUFFIXES:
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        rel = workspace.relative_display(file)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                if len(hits) >= max_results:
                    break
        if len(hits) >= max_results:
            break
    return hits, scanned


class GlobTool(CodingTool):
    """glob 文件查找 — ``**/*.py`` 风格模式匹配。"""

    name = "glob"
    description = (
        "按 glob 模式查找文件（如 **/*.py、src/**/*.ts）。"
        "返回相对路径列表，按路径排序。"
    )
    parameters = [
        ToolParameter(
            name="pattern", type="string",
            description="glob 模式（支持 ** / * / ?）",
            required=True,
        ),
        ToolParameter(
            name="path", type="string",
            description="起始目录（相对用户 workspace，默认根目录）",
            required=False,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            return self._error(
                tool_call, "pattern 参数缺失", context=context,
            )

        try:
            base = self._resolve(context, str(args.get("path") or "."))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)
        if not base.exists():
            return self._error(
                tool_call, f"目录不存在: {base}", context=context,
            )
        if not base.is_dir():
            return self._error(
                tool_call, f"不是目录: {base}", context=context,
            )

        # 目录遍历是磁盘 IO，大仓库/网络盘会阻塞事件循环 → 线程池
        try:
            matches, truncated = await asyncio.to_thread(
                _glob_collect, base, pattern, settings.search_max_results,
            )
        except (OSError, ValueError) as exc:
            return self._error(
                tool_call, f"glob 失败: {exc}", context=context,
            )

        workspace = self._workspace(context)
        display_paths = [workspace.relative_display(p) for p in matches]
        # preview 全量展示（卡片内部可滚动）；超限尾部标注总数，
        # 避免“12 条结果只列 10 条”式的静默截断
        preview = display_paths[:30]
        if len(display_paths) > 30:
            preview.append(f"… 共 {len(display_paths)} 条")
        self._emit_digest(
            context,
            tool_name="glob", tool_call_id=tool_call.id,
            display_type="search", status="success",
            title=f"glob {pattern}",
            pattern=pattern, result_count=len(matches),
            results_preview=preview,
        )
        if not matches:
            return AgentToolResult.text_result(
                tool_call.id, f"无匹配: {pattern}",
                llm_digest=f"[tool:glob status=ok] {pattern} 无匹配。",
            )
        note = "\n（结果已达上限，可缩小 pattern）" if truncated else ""
        # 不传 llm_digest：ark serialize 时 fallback 到 content 全文 ——
        # 正常轮模型看完整命中列表；compaction 折叠时 ark safe_digest
        # 截断为 500 字符摘要（显式传 digest 会让模型永远看不到列表）
        return AgentToolResult.text_result(
            tool_call.id,
            "\n".join(display_paths) + note,
        )


class GrepTool(CodingTool):
    """grep 内容检索 — 正则逐行匹配，输出 ``path:line:content``。"""

    name = "grep"
    description = (
        "在文件中按正则检索内容，输出 path:行号:命中行。"
        "可选 include 过滤文件名（glob 语法）。"
    )
    parameters = [
        ToolParameter(
            name="pattern", type="string",
            description="正则表达式",
            required=True,
        ),
        ToolParameter(
            name="path", type="string",
            description="检索目标：文件或目录（相对用户 workspace，默认根目录）",
            required=False,
        ),
        ToolParameter(
            name="include", type="string",
            description="文件名过滤 glob（如 *.py），仅目录检索时生效",
            required=False,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            return self._error(
                tool_call, "pattern 参数缺失", context=context,
            )

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return self._error(
                tool_call, f"正则不合法: {exc}", context=context,
            )

        try:
            target = self._resolve(context, str(args.get("path") or "."))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)
        if not target.exists():
            return self._error(
                tool_call, f"目标不存在: {target}", context=context,
            )

        include = str(args.get("include") or "").strip() or None
        max_results = settings.search_max_results
        workspace = self._workspace(context)
        # 列文件 + 逐文件读内容 + 正则扫描全部是磁盘 IO/CPU，
        # 大目录扫描可达秒级 → 整体丢线程池
        hits, scanned = await asyncio.to_thread(
            _grep_scan, target, regex, include, max_results, workspace,
        )

        truncated = len(hits) >= max_results
        self._emit_digest(
            context,
            tool_name="grep", tool_call_id=tool_call.id,
            display_type="search", status="success",
            title=f"grep {pattern}",
            pattern=pattern, result_count=len(hits),
            results_preview=[diff_preview(h) for h in hits[:10]],
        )
        if not hits:
            return AgentToolResult.text_result(
                tool_call.id,
                f"无命中: {pattern}（扫描 {scanned} 个文件）",
                llm_digest=f"[tool:grep status=ok] {pattern} 无命中（{scanned} 文件）。",
            )
        note = "\n（结果已达上限）" if truncated else ""
        # 同 glob：不传 digest，正常轮模型看完整命中行；折叠时 ark 截断
        return AgentToolResult.text_result(
            tool_call.id,
            f"{len(hits)} 处命中（扫描 {scanned} 个文件）:\n" + "\n".join(hits) + note,
        )
