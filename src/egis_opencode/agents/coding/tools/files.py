"""read / write / edit / list 文件操作工具（opencode 语义）。"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ark_agentic.core.tools.base import ToolParameter
from ark_agentic.core.types import AgentToolResult

from ....config import settings
from ....events import diff_preview
from ....workspace import WorkspacePathError
from .base import CodingTool
from .truncate import truncate_line

logger = logging.getLogger(__name__)

# 二进制扩展名黑名单（读文件时拒绝，避免把二进制灌进上下文）
# 注：.docx 不在此列 —— 走 mammoth 文本提取（见 _DOCX_SUFFIXES）
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".woff", ".woff2",
    ".ttf", ".eot", ".mp3", ".mp4", ".avi", ".mov", ".sqlite", ".db",
    ".doc", ".ppt", ".xls", ".pptx", ".xlsx",
}

# 可提取文本的办公文档（mammoth 转 markdown，保留标题层级）
_DOCX_SUFFIXES = {".docx"}

#: 读未知后缀文件时抽检二进制头的大小（字节）
_BINARY_SNIFF_BYTES = 4096

#: mammoth 对带书签标题输出的 <a id="..."></a> 锚点，纯噪音，提取时移除
_DOCX_ANCHOR_RE = re.compile(r'<a id="[^"]*"></a>')

#: docx 内嵌图片的 data URI —— 巨行噪音（实测 1.4MB 文档可产出 150 万
#: 字符 base64，占提取文本 98%），提取时替换为占位符，只保留正文
_DOCX_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")


@dataclass
class _SlicedLines:
    """read 截断结果（对齐 opencode read.ts lines() 的 flags）。"""

    #: 送入 LLM 的行（已做单行截断）
    raw: list[str]
    #: 流过的总行数（行数截断时计数到文件尾；字节截断时停在断点）
    count: int
    #: 是否因累计字节上限提前停
    cut: bool
    #: 是否还有未读内容
    more: bool


class ReadTool(CodingTool):
    """读文件 — 带行号输出（对齐 opencode read.ts 语义）。

    防护链：单行 2000 字符截断 → 累计字节上限截断（带续读提示）→
    行数上限（带续读提示）；docx 提取后同样过链（图片占位符化）。
    """

    name = "read"
    description = (
        "读取工作区内的文本文件，输出带行号的内容。"
        ".docx 会自动提取为 markdown 文本（内嵌图片以占位符替代）。"
        "大文件用 offset/limit 分段读取。"
    )
    parameters = [
        ToolParameter(
            name="path", type="string",
            description="文件路径（相对用户 workspace）",
            required=True,
        ),
        ToolParameter(
            name="offset", type="integer",
            description="起始行号（1-based，默认 1）",
            required=False,
        ),
        ToolParameter(
            name="limit", type="integer",
            description="读取行数（默认 2000）",
            required=False,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        try:
            path = self._resolve(context, str(args.get("path") or ""))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)

        if not path.exists():
            return self._error(
                tool_call, await asyncio.to_thread(_miss_message, path),
                digest=f"[tool:read status=error] 文件不存在: {path.name}",
                context=context,
            )
        if not path.is_file():
            return self._error(
                tool_call, f"不是文件: {path}", context=context,
            )
        if path.suffix.lower() in _BINARY_SUFFIXES:
            return self._error(
                tool_call, f"二进制文件不支持读取: {path.suffix}",
                digest=f"[tool:read status=error] 二进制文件 {path.name} 不可读。",
                context=context,
            )

        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or settings.read_max_lines)

        # 文件 IO / docx 提取一律丢线程池：工具跑在服务进程的事件循环上，
        # 同步读大文件/解析文档会把所有并发会话的流式输出一起卡住
        if path.suffix.lower() in _DOCX_SUFFIXES:
            lines, error = await asyncio.to_thread(_extract_docx, path)
            if error is not None:
                return self._error(
                    tool_call, error,
                    digest=f"[tool:read status=error] {path.name} 提取失败。",
                    context=context,
                )
            if not any(line.strip() for line in lines):
                return self._error(
                    tool_call, f"文档无文本内容: {path.name}",
                    digest=f"[tool:read status=error] {path.name} 无文本。",
                    context=context,
                )
            sliced = await asyncio.to_thread(
                _slice_lines, iter(lines), offset, limit,
            )
        else:
            if await asyncio.to_thread(_looks_binary, path):
                return self._error(
                    tool_call,
                    f"二进制文件不支持读取: {path.name}"
                    f"（未知后缀 {path.suffix or '无后缀'}，含 NUL 字节）",
                    digest=f"[tool:read status=error] 二进制文件 {path.name} 不可读。",
                    context=context,
                )
            # IO 防线：非流式计数需要读完整个文件，超大文件直接拒绝
            if path.stat().st_size > settings.read_max_bytes:
                return self._error(
                    tool_call,
                    f"文件过大（{path.stat().st_size} 字节 > "
                    f"{settings.read_max_bytes}），请用 grep 定位后分段 read",
                    digest=f"[tool:read status=error] 文件过大，建议 grep+分段读取。",
                    context=context,
                )
            sliced = await asyncio.to_thread(
                _read_file_lines, path, offset, limit,
            )

        # offset 越界明确报错（对齐 opencode：空文件 + offset=1 例外）
        if sliced.count < offset and not (sliced.count == 0 and offset == 1):
            return self._error(
                tool_call,
                f"Offset {offset} is out of range for this file "
                f"({sliced.count} lines)",
                digest=(
                    f"[tool:read status=error] offset={offset} 超出范围"
                    f"（共 {sliced.count} 行）。"
                ),
                context=context,
            )

        display = self._display(context, path)
        output = [f"<path>{display}</path>", "<type>file</type>", "<content>"]
        output += [f"{i}: {line}" for i, line in enumerate(sliced.raw, start=offset)]

        last = offset + len(sliced.raw) - 1
        next_offset = last + 1
        if sliced.cut:
            output.append(
                f"\n(Output capped at {settings.tool_output_max_bytes // 1024} KB. "
                f"Showing lines {offset}-{last}. "
                f"Use offset={next_offset} to continue.)"
            )
        elif sliced.more:
            output.append(
                f"\n(Showing lines {offset}-{last} of {sliced.count}. "
                f"Use offset={next_offset} to continue.)"
            )
        else:
            output.append(f"\n(End of file - total {sliced.count} lines)")
        output.append("</content>")
        body = "\n".join(output)

        self._emit_digest(
            context,
            tool_name="read", tool_call_id=tool_call.id,
            display_type="search", status="success",
            title=f"read {display}",
            path=display, result_count=len(sliced.raw),
            results_preview=[
                f"{i}: {line[:200]}"
                for i, line in enumerate(sliced.raw[:8], start=offset)
            ],
            # 对齐 opencode metadata.display：卡片可展示完整读取内容。
            # UI 预览只列 8 行会让用户误以为模型也只读到 8 行 —— 实际
            # 模型收到的是 content 全文，这里把同一份内容给到前端
            file_text="\n".join(sliced.raw),
            line_start=offset if sliced.raw else None,
            line_end=(offset + len(sliced.raw) - 1) if sliced.raw else None,
            total_lines=sliced.count,
            content_truncated=sliced.cut or sliced.more,
        )
        # ark 语义（serialize_messages_for_llm）：tool 消息发给 LLM 的 content
        # 就是 llm_digest —— 显式传 digest 会让模型永远看不到正文（幻觉根源）。
        # 不传 digest：正常轮 fallback 到 content 全文，compaction 折叠时由
        # ark safe_digest 截断为 500 字符摘要 —— 这才是 digest 的正确用法。
        return AgentToolResult.text_result(
            tool_call.id,
            body,
        )


class WriteTool(CodingTool):
    """写文件 — 全量写入（新文件或覆盖）。"""

    name = "write"
    description = (
        "在工作区内创建或完整覆写一个文件。"
        "已有文件建议优先 edit 做局部修改。"
    )
    parameters = [
        ToolParameter(
            name="path", type="string",
            description="目标文件路径（相对用户 workspace）",
            required=True,
        ),
        ToolParameter(
            name="content", type="string",
            description="完整文件内容",
            required=True,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        try:
            path = self._resolve(context, str(args.get("path") or ""))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)

        content = args.get("content")
        if content is None:
            return self._error(
                tool_call, "content 参数缺失", context=context,
            )

        # 旧内容预览 + 写入整体丢线程池（同 read：事件循环不碰文件 IO）
        existed, old_preview, write_error = await asyncio.to_thread(
            _write_file, path, str(content),
        )
        if write_error is not None:
            return self._error(
                tool_call, f"写入失败: {write_error}", context=context,
            )

        display = self._display(context, path)
        self._emit_digest(
            context,
            tool_name="write", tool_call_id=tool_call.id,
            display_type="file_edit", status="success",
            title=f"{'覆写' if existed else '新建'} {display}",
            path=display,
            old_string_preview=old_preview or None,
            new_string_preview=diff_preview(str(content)),
        )
        return AgentToolResult.text_result(
            tool_call.id,
            f"已{'覆写' if existed else '新建'} {display}（{len(str(content))} 字符）",
            llm_digest=f"[tool:write status=ok] {'覆写' if existed else '新建'} {display}。",
        )


class EditTool(CodingTool):
    """编辑文件 — old_string 精确替换（opencode edit 语义）。

    - old_string 必须在文件中唯一（多处命中时报错并提示扩大上下文）
    - replace_all=True 时替换全部命中
    """

    name = "edit"
    description = (
        "对文件做精确局部修改：把唯一的 old_string 替换为 new_string。"
        "old_string 须与文件内容逐字符一致（含缩进），"
        "多处命中时扩大上下文使其唯一，或用 replace_all。"
    )
    parameters = [
        ToolParameter(
            name="path", type="string",
            description="目标文件路径（相对用户 workspace）",
            required=True,
        ),
        ToolParameter(
            name="old_string", type="string",
            description="被替换的原文（须精确匹配，含缩进与空行）",
            required=True,
        ),
        ToolParameter(
            name="new_string", type="string",
            description="替换后的新文本",
            required=True,
        ),
        ToolParameter(
            name="replace_all", type="boolean",
            description="替换全部命中（默认 false，要求唯一命中）",
            required=False,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        try:
            path = self._resolve(context, str(args.get("path") or ""))
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)

        old_string = str(args.get("old_string") or "")
        new_string = str(args.get("new_string") or "")
        replace_all = bool(args.get("replace_all", False))

        if not old_string:
            return self._error(
                tool_call, "old_string 不能为空", context=context,
            )
        if not path.exists():
            return self._error(
                tool_call, f"文件不存在: {path}", context=context,
            )
        if not path.is_file():
            return self._error(
                tool_call, f"不是文件: {path}", context=context,
            )

        try:
            text = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            return self._error(
                tool_call, f"读取失败: {exc}", context=context,
            )

        count = text.count(old_string)
        if count == 0:
            return self._error(
                tool_call,
                "old_string 未在文件中命中。请先 read 目标区域，"
                "确保逐字符一致（含缩进）。",
                digest=f"[tool:edit status=error] 未命中，未修改。",
                context=context,
            )
        if count > 1 and not replace_all:
            return self._error(
                tool_call,
                f"old_string 命中 {count} 处。扩大上下文使其唯一，"
                "或设置 replace_all=true。",
                digest=f"[tool:edit status=error] 多处命中({count})，未修改。",
                context=context,
            )

        if replace_all:
            updated = text.replace(old_string, new_string)
            replaced = count
        else:
            updated = text.replace(old_string, new_string, 1)
            replaced = 1

        try:
            await asyncio.to_thread(path.write_text, updated, encoding="utf-8")
        except OSError as exc:
            return self._error(
                tool_call, f"写入失败: {exc}", context=context,
            )

        diff = _simple_diff(old_string, new_string)
        display = self._display(context, path)
        self._emit_digest(
            context,
            tool_name="edit", tool_call_id=tool_call.id,
            display_type="file_edit", status="success",
            title=f"edit {display}",
            path=display,
            diff=diff,
            old_string_preview=diff_preview(old_string),
            new_string_preview=diff_preview(new_string),
        )
        return AgentToolResult.text_result(
            tool_call.id,
            f"已修改 {display}（替换 {replaced} 处）",
            llm_digest=f"[tool:edit status=ok] 修改 {display}，替换 {replaced} 处。",
        )


class ListTool(CodingTool):
    """列目录 — 一级子项清单（目录/文件/大小）。"""

    name = "list"
    description = "列出目录的一级内容（名称 + 类型 + 大小）。深度遍历用 glob。"
    parameters = [
        ToolParameter(
            name="path", type="string",
            description="目录路径（相对用户 workspace，默认根目录）",
            required=False,
        ),
    ]

    async def execute(self, tool_call, context=None) -> AgentToolResult:
        args = tool_call.arguments or {}
        raw = str(args.get("path") or ".")
        try:
            path = self._resolve(context, raw)
        except WorkspacePathError as exc:
            return self._path_error_result(tool_call, exc)

        if not path.exists():
            return self._error(
                tool_call, f"目录不存在: {path}", context=context,
            )
        if not path.is_dir():
            return self._error(
                tool_call, f"不是目录: {path}", context=context,
            )

        # iterdir + stat 全在磁盘上，目录大时阻塞明显 → 线程池
        entries = await asyncio.to_thread(_list_entries, path)

        display = self._display(context, path)
        # preview 全量展示（卡片内部可滚动）；超限时尾部标注总数，
        # 避免“12 条结果只列 10 条”式的静默截断
        preview = [
            f"{e['name']}/" if e.get("type") == "dir" else str(e["name"])
            for e in entries
        ]
        if len(preview) > 30:
            preview = preview[:30] + [f"… 共 {len(entries)} 项"]
        self._emit_digest(
            context,
            tool_name="list", tool_call_id=tool_call.id,
            display_type="search", status="success",
            title=f"list {display}",
            path=display, result_count=len(entries),
            results_preview=preview,
        )
        # 不传 llm_digest：正常轮模型看完整条目（json fallback）；
        # compaction 折叠时 ark safe_digest 截断为摘要
        return AgentToolResult.json_result(
            tool_call.id,
            {"directory": display, "entries": entries},
        )


def _miss_message(path: Path) -> str:
    """文件不存在时的提示（对齐 opencode miss）：同目录相似名建议。"""
    base = path.name.lower()
    try:
        items = [p.name for p in path.parent.iterdir()]
    except OSError:
        items = []
    similar = [
        name for name in items
        if base in name.lower() or name.lower() in base
    ][:3]
    if similar:
        return (
            f"File not found: {path}\n\nDid you mean one of these?\n"
            + "\n".join(similar)
        )
    return f"File not found: {path}"


def _slice_lines(source, offset: int, limit: int) -> _SlicedLines:
    """流式行切分（对齐 opencode read.ts lines()）。

    三层防护：行数上限（raw 满后继续计数到尾）、单行截断、
    累计字节上限（达到即停）。``source`` 为行迭代器（可含换行符）。
    """
    raw: list[str] = []
    count = 0
    nbytes = 0
    cut = False
    more = False
    for line in source:
        count += 1
        if count < offset:
            continue
        if len(raw) >= limit:
            more = True
            continue
        text = truncate_line(line.rstrip("\n"))
        size = len(text.encode("utf-8", errors="replace")) + (1 if raw else 0)
        if nbytes + size > settings.tool_output_max_bytes:
            cut = True
            more = True
            break
        raw.append(text)
        nbytes += size
    return _SlicedLines(raw=raw, count=count, cut=cut, more=more)


def _read_file_lines(path: Path, offset: int, limit: int) -> _SlicedLines:
    """流式读文件并切分（bytes 达上限即停，不再读文件剩余部分）。"""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return _slice_lines(fh, offset, limit)


def _extract_docx(path: Path) -> tuple[list[str], str | None]:
    """docx → markdown 行列表（mammoth，保留标题层级）。

    内嵌图片的 base64 data URI 替换为占位符 —— 否则一张图就是一行
    几十万字符，直接击穿 LLM 上下文（幻觉源头）。返回 (lines, error)：
    成功时 error=None；损坏/非 docx 返回错误文案。
    """
    try:
        import mammoth

        with path.open("rb") as fh:
            result = mammoth.convert_to_markdown(fh)
        # 真实文档常带 Word 自动书签，mammoth 会转出 <a id="heading_N"></a> 锚点噪音
        text = _DOCX_ANCHOR_RE.sub("", result.value)
        text = _DOCX_IMAGE_RE.sub("[图片内容已省略]", text)
        return text.splitlines(), None
    except ImportError:  # pragma: no cover - 依赖缺失防御
        # 不提示安装：平台依赖由运维统一管理，agent 侧装包会被拒绝
        return [], (
            "服务端未启用 .docx 文本提取能力（mammoth 未安装），"
            "请联系管理员处理；不要尝试自行安装依赖"
        )
    except Exception as exc:  # noqa: BLE001 — 非法 zip/损坏文档
        logger.warning("docx extract failed: %s", path, exc_info=True)
        return [], f"docx 提取失败（文件损坏或非 Word 文档）: {exc}"


def _write_file(path: Path, content: str) -> tuple[bool, str, str | None]:
    """写文件（含旧内容预览提取）；返回 (existed, old_preview, error)。"""
    existed = path.exists()
    old_preview = ""
    if existed and path.is_file():
        try:
            old_preview = diff_preview(
                path.read_text(encoding="utf-8", errors="replace"),
            )
        except OSError:
            old_preview = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return existed, old_preview, str(exc)
    return existed, old_preview, None


def _list_entries(path: Path) -> list[dict[str, Any]]:
    """列一级子项（点文件隐藏，目录优先字母序，超限截断）。"""
    entries: list[dict[str, Any]] = []
    for child in sorted(
        path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()),
    ):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({"name": child.name, "type": "dir"})
        else:
            entries.append(
                {"name": child.name, "type": "file", "bytes": child.stat().st_size},
            )
        if len(entries) >= settings.search_max_results:
            break
    return entries


def _looks_binary(path: Path) -> bool:
    """未知后缀文件的二进制兑底检测（头部含 NUL 字节即判定）。"""
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False


def _simple_diff(old: str, new: str) -> str:
    """unified 风格的最小 diff 预览（删除行 - / 新增行 +）。"""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    parts: list[str] = []
    for line in old_lines:
        parts.append(f"- {line}")
    for line in new_lines:
        parts.append(f"+ {line}")
    return "\n".join(parts)
