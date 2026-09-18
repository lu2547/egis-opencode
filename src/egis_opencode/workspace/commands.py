"""Slash 命令发现与展开 — 工作目录下的 ``.opencode/commands`` / ``.claude/commands``。

opencode 生态约定：命令是 Markdown 文件（frontmatter ``description`` +
正文，``$ARGUMENTS`` 占位符）。egis-opencode 在 chat 入口把
``/ingest <args>`` 展开为「命令正文 + AGENTS.md 项目上下文」的完整
提示词再交给 agent —— 命令定义放在哪个工作目录，就由该目录生效
（本地目录会话读该目录的命令，多租户模式读用户 workspace 根）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: 命令目录约定（先到先得：opencode 优先于 claude）
_COMMAND_DIRS = (".opencode/commands", ".claude/commands")

#: AGENTS.md 注入上限（字节）——过大的项目规范截断，避免挤爆上下文
_AGENTS_MAX_BYTES = 32_000

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class CommandSpec:
    """单个 slash 命令（name = 文件名去 .md）。"""

    name: str
    description: str
    body: str


def discover_commands(root: Path) -> dict[str, CommandSpec]:
    """扫描工作目录下的命令定义（同名时 .opencode 优先）。"""
    commands: dict[str, CommandSpec] = {}
    if not root.is_dir():
        return commands
    for rel in _COMMAND_DIRS:
        base = root / rel
        if not base.is_dir():
            continue
        for file in sorted(base.glob("*.md")):
            name = file.stem.strip()
            if not name or name in commands:
                continue
            spec = _parse_command_file(file)
            if spec is not None:
                commands[name] = spec
    return commands


def expand_command(message: str, root: Path) -> str | None:
    """把 ``/name args`` 展开为完整提示词；非命令消息返回 None。

    - 命中：命令正文（``$ARGUMENTS`` → args）+ AGENTS.md 项目上下文
    - 未命中：返回引导话术（附可用命令列表），由 agent 告知用户
    """
    text = message.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(None, 1)
    if not parts or not parts[0]:
        return None
    name = parts[0]
    args = parts[1].strip() if len(parts) > 1 else ""

    commands = discover_commands(root)
    spec = commands.get(name)
    if spec is None:
        available = "、".join(f"/{n}" for n in sorted(commands)) or "（无）"
        return (
            f"用户输入了未知命令 /{name}。请告知该命令不存在，"
            f"并列出当前工作目录可用的命令：{available}。"
        )

    prompt = spec.body
    if "$ARGUMENTS" in prompt:
        prompt = prompt.replace("$ARGUMENTS", args)
    elif args:
        prompt = f"{prompt}\n\n## 用户参数\n\n{args}"

    # 执行指令框架：命令正文常是说明书文体（“# xx 命令”标题 +
    # “用户参数（可选…）:”元信息行），无参数触发时该说明行留空，
    # 模型会把整篇误读成“关于命令的资料”而非执行指令 —— 实测
    # /ingest 无参数触发时 agent 只输出“命令实现方案”不执行。
    # 显式框定为执行指令，消除歧义（命令文件属用户工作区，不改其文体）。
    prompt = (
        f"<slash-command name=\"{name}\">\n"
        f"用户触发了 /{name} 命令。以下全文是该命令的执行指令："
        f"立即按其中流程开始执行并产出结果，不要只解读、总结、复述"
        f"或为它编写实现方案。\n\n"
        f"{prompt}\n"
        f"</slash-command>"
    )

    agents_md = _read_agents_md(root)
    if agents_md:
        prompt = (
            f"{prompt}\n\n---\n\n"
            f"## 项目上下文（AGENTS.md — 当前工作目录的会话规范，必须遵守）\n\n"
            f"{agents_md}"
        )
    return prompt


def _parse_command_file(file: Path) -> CommandSpec | None:
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("command file unreadable: %s", file)
        return None
    description = ""
    match = _FRONTMATTER_RE.match(text)
    if match:
        description = _frontmatter_value(match.group(1), "description")
        text = text[match.end():]
    return CommandSpec(
        name=file.stem, description=description, body=text.strip(),
    )


def _frontmatter_value(block: str, key: str) -> str:
    """极简 frontmatter 解析（单行 ``key: value``；不引 yaml 依赖）。"""
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip() == key:
            return v.strip().strip("\"'")
    return ""


def _read_agents_md(root: Path) -> str:
    """读取工作目录的项目规范（AGENTS.md，回落 CLAUDE.md）。"""
    for candidate in ("AGENTS.md", "CLAUDE.md"):
        file = root / candidate
        try:
            if file.is_file():
                return file.read_text(
                    encoding="utf-8", errors="replace",
                )[:_AGENTS_MAX_BYTES]
        except OSError:
            continue
    return ""


__all__ = ["CommandSpec", "discover_commands", "expand_command"]
