"""工具输出统一截断（对齐 opencode tool/truncate.ts 语义）。

所有可能产生大输出的工具（bash 等）在返回前走 ``truncate_output``：
未超限原样返回；超限时保留 head/tail 方向的 preview，全文写入工作区
``.truncation/``（7 天懒清理），并在 preview 后拼接被截断量与续读指引
—— 引导模型用 grep/read 分段处理落盘文件，而不是把全文再灌回上下文。

read 工具不走本服务（它有自己的 offset/limit 续读语义），但复用
``truncate_line`` 做单行截断。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ....config import settings

#: 截断文件保留时长（秒；对齐 opencode RETENTION = 7 天）
_RETENTION_SECONDS = 7 * 24 * 3600

#: 截断文件所在目录名（点开头 —— list 工具隐藏点目录，不会污染工作区视图）
TRUNCATION_DIRNAME = ".truncation"


@dataclass
class TruncatedOutput:
    """截断结果：``content`` 为给模型看的文本，``output_path`` 为全文落盘路径。"""

    content: str
    truncated: bool
    output_path: str | None = None


def truncate_line(line: str) -> str:
    """单行超长截断（防 base64 data URI 等巨行灌爆上下文）。

    对齐 opencode read.ts 的 MAX_LINE_LENGTH：超长行截断到上限并带后缀，
    模型能识别该行不完整（配合 read 的 offset 续读）。
    """
    max_len = settings.tool_output_max_line_length
    if len(line) <= max_len:
        return line
    return f"{line[:max_len]}... (line truncated to {max_len} chars)"


def truncation_dir(workspace_root: Path) -> Path:
    return workspace_root / TRUNCATION_DIRNAME


def _cleanup(dirpath: Path, now: float) -> None:
    """删除超过保留期的截断文件（尽力而为，失败不影响工具结果）。"""
    try:
        for entry in dirpath.iterdir():
            if not entry.name.startswith("tool_"):
                continue
            try:
                if now - entry.stat().st_mtime > _RETENTION_SECONDS:
                    entry.unlink()
            except OSError:
                continue
    except OSError:
        return


def truncate_output(
    text: str, *,
    workspace_root: Path,
    direction: str = "head",
    has_task: bool = False,
) -> TruncatedOutput:
    """对齐 opencode Truncate.output：超限截断 + 全文落盘 + 续读指引。

    ``direction="head"`` 保留开头（默认），``"tail"`` 保留结尾（bash 日志
    的错误信息集中在尾部）。超限判定：行数或总字节任一超出
    ``tool_output_max_lines`` / ``tool_output_max_bytes``。
    """
    max_lines = settings.tool_output_max_lines
    max_bytes = settings.tool_output_max_bytes
    total_bytes = len(text.encode("utf-8", errors="replace"))
    lines = text.split("\n")

    if len(lines) <= max_lines and total_bytes <= max_bytes:
        return TruncatedOutput(content=text, truncated=False)

    out: list[str] = []
    nbytes = 0
    hit_bytes = False

    if direction == "tail":
        for i in range(len(lines) - 1, -1, -1):
            if len(out) >= max_lines:
                break
            size = len(lines[i].encode("utf-8", errors="replace")) + (1 if out else 0)
            if nbytes + size > max_bytes:
                hit_bytes = True
                break
            out.insert(0, lines[i])
            nbytes += size
    else:
        for i in range(min(len(lines), max_lines)):
            size = len(lines[i].encode("utf-8", errors="replace")) + (1 if i else 0)
            if nbytes + size > max_bytes:
                hit_bytes = True
                break
            out.append(lines[i])
            nbytes += size

    removed = total_bytes - nbytes if hit_bytes else len(lines) - len(out)
    unit = "bytes" if hit_bytes else "lines"
    preview = "\n".join(out)

    now = time.time()
    dirpath = truncation_dir(workspace_root)
    filename = f"tool_{int(now * 1000)}_{uuid.uuid4().hex[:8]}.txt"
    filepath = dirpath / filename
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
        filepath.write_text(text, encoding="utf-8", errors="replace")
        _cleanup(dirpath, now)
    except OSError:
        # 落盘失败不拦工具结果：退化为纯 preview + 截断量提示
        filepath = None  # type: ignore[assignment]

    if filepath is not None:
        hint = (
            f"The tool call succeeded but the output was truncated. "
            f"Full output saved to: {filepath}"
        )
        if has_task:
            hint += (
                "\nUse the Task tool to have explore agent process this file "
                "with Grep and Read (with offset/limit). "
                "Do NOT read the full file yourself - delegate to save context."
            )
        else:
            hint += (
                "\nUse Grep to search the full content or Read with "
                "offset/limit to view specific sections."
            )
    else:
        hint = "The tool call succeeded but the output was truncated."

    if direction == "tail":
        content = f"...{removed} {unit} truncated...\n\n{hint}\n\n{preview}"
    else:
        content = f"{preview}\n\n...{removed} {unit} truncated...\n\n{hint}"
    return TruncatedOutput(
        content=content, truncated=True,
        output_path=str(filepath) if filepath is not None else None,
    )
