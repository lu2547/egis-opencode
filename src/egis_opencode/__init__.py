"""egis-opencode — 基于 ark-agentic 底座的 opencode 式 Coding Agent 平台。

架构分层：
- ``agents/coding``     — CodingAgent（build/plan 双模式）+ coding 工具集
- ``permissions``       — allow/ask/deny 规则引擎 + 审批服务 + before_tool 守卫
- ``workspace``         — 多租户 workspace 路径守卫与项目管理
- ``api``               — CodingPlugin（REST + SSE，扩展 AG-UI 协议）
- ``config``            — 环境变量设置

ark-agentic 为上游框架，只复用不修改。
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

#: agents 目录绝对路径（Bootstrap 自动发现 BaseAgent 子类的根）
AGENTS_ROOT: Path = Path(__file__).resolve().parent / "agents"

__all__ = ["__version__", "AGENTS_ROOT"]
