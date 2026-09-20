"""egis-opencode 配置 — 进程环境变量的集中读取点。

所有配置项带默认值，进程启动时读取一次（模块级单例 ``settings``）。
与 ark 的 ``os.getenv`` 直读习惯保持一致，不引入额外配置文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """egis-opencode 运行配置。"""

    # ── workspace ─────────────────────────────────────
    #: 多租户 workspace 根目录；须与 sandbox.json 的 workspace.root 一致
    workspace_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("CODING_WORKSPACE_ROOT", "data/ark_workspaces")
        )
    )
    #: 允许会话绑定本地任意目录（local:<path> 工作目录模式）；
    #: 企业部署可置 false 收紧为仅多租户 sandbox
    local_workspace_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("CODING_ALLOW_LOCAL_WORKSPACE", "true")
            .strip().lower() in ("1", "true", "yes", "on")
        )
    )
    #: 默认工作目录模式 —— 前端未传 workspace_root 且会话无绑定时的
    #: 服务端兜底（对齐 opencode「启动即工作目录」的单机部署形态）：
    #: ``local`` = 锚定 CODING_DEFAULT_WORKSPACE_DIR；``multi``（默认）
    #: = 多租户用户根，行为与未引入该配置时完全一致
    default_workspace_mode: str = field(
        default_factory=lambda: os.getenv(
            "CODING_DEFAULT_WORKSPACE_MODE", "multi",
        ).strip().lower()
    )
    #: 默认工作目录（mode=local 时生效；绝对路径，相对按进程 CWD 解析）
    default_workspace_dir: str = field(
        default_factory=lambda: os.getenv(
            "CODING_DEFAULT_WORKSPACE_DIR", "",
        ).strip()
    )

    # ── 权限 ──────────────────────────────────────────
    #: ask 权限等待应答超时（秒）；超时按拒绝处理
    permission_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CODING_PERMISSION_TIMEOUT", 120)
    )
    #: 静默放行模式：ask 规则视同 allow，不弹审批卡片；
    #: deny 硬禁令（plan 只读、裸 pip 等）不受影响仍拒。
    #: 进程级开关（多租户下全局共享）；企业部署保持 false
    permission_silent_allow: bool = field(
        default_factory=lambda: (
            os.getenv("CODING_PERMISSION_SILENT_ALLOW", "false")
            .strip().lower() in ("1", "true", "yes", "on")
        )
    )

    # ── 文件工具 ──────────────────────────────────────
    #: read 工具单文件读取上限（字节）
    read_max_bytes: int = field(
        default_factory=lambda: _env_int("CODING_READ_MAX_BYTES", 256_000)
    )
    #: read 工具单次最大行数（0 = 不限制）
    read_max_lines: int = field(
        default_factory=lambda: _env_int("CODING_READ_MAX_LINES", 2000)
    )
    #: grep/list 结果条数上限
    search_max_results: int = field(
        default_factory=lambda: _env_int("CODING_SEARCH_MAX_RESULTS", 100)
    )

    # ── LLM 单次输出（对齐 opencode OUTPUT_TOKEN_MAX=32000）──
    #: 单次 LLM 输出 token 上限。ark SamplingConfig 默认 4096 过小：
    #: thinking + 正文 + tool call arguments 共享该预算，写长脚本时
    #: finish_reason="length"，ark 将其按 run 终止处理并丢弃当轮
    #: tool calls（前端表现为“轮次上限”假象 + 脚本写入截断）。
    max_output_tokens: int = field(
        default_factory=lambda: _env_int("CODING_MAX_OUTPUT_TOKENS", 32_000)
    )

    # ── Doom loop 护栏（对齐 opencode DOOM_LOOP_THRESHOLD；防复读退化）──
    #: 连续 N 次完全相同 (tool, args) 调用判定为死循环命中
    doom_loop_threshold: int = field(
        default_factory=lambda: _env_int("CODING_DOOM_LOOP_THRESHOLD", 3)
    )
    #: run 内累计第 N 次命中 → ToolLoopAction.STOP 熔断终止
    doom_loop_max_hits: int = field(
        default_factory=lambda: _env_int("CODING_DOOM_LOOP_MAX_HITS", 5)
    )

    # ── 工具输出统一截断（对齐 opencode tool_output；防上下文爆炸）──
    #: read / bash 等工具输出的行数上限（超出截断 + 续读提示）
    tool_output_max_lines: int = field(
        default_factory=lambda: _env_int("CODING_TOOL_OUTPUT_MAX_LINES", 2000)
    )
    #: 工具输出累计字节上限（50KB；超出截断，全文落盘 .truncation/）
    tool_output_max_bytes: int = field(
        default_factory=lambda: _env_int("CODING_TOOL_OUTPUT_MAX_BYTES", 50 * 1024)
    )
    #: read 单行截断长度（防 base64 等巨行灌爆上下文）
    tool_output_max_line_length: int = field(
        default_factory=lambda: _env_int("CODING_TOOL_OUTPUT_MAX_LINE_LENGTH", 2000)
    )

    # ── bash（本地目录模式直连执行；默认对齐 sandbox.json）──
    #: 本地 bash 单命令超时（秒）
    bash_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CODING_BASH_TIMEOUT", 120)
    )
    #: 本地 bash 原始输出保留上限（字节，stdout/stderr 各自尾部保留）。
    #: 默认 2 倍 tool_output_max_bytes —— 对齐 opencode shell keep 策略：
    #: 字节层先保尾部，行数/字节语义截断再由 truncate_output 收口
    bash_max_output_bytes: int = field(
        default_factory=lambda: _env_int(
            "CODING_BASH_MAX_OUTPUT_BYTES", 100_000
        )
    )

    # ── 标题生成 ──────────────────────────────────────
    #: 会话标题最大长度（字符）
    title_max_chars: int = 50
    #: SSE 关流前等待 title 等尾部事件落帧的上限（秒；超时静默丢帧，
    #: 由 sessions 列表合并 titles 兑底）
    title_sse_grace_seconds: float = 3.0

    @property
    def workspace_root_resolved(self) -> Path:
        """workspace root 的绝对路径（相对路径按进程 CWD 解析）。"""
        root = self.workspace_root
        if not root.is_absolute():
            root = Path.cwd() / root
        return root


settings = Settings()
