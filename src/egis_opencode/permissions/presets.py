"""build / plan 双模式的权限预设（对齐 opencode ``agent.ts`` defaults）。

opencode 的默认策略是**默认放行**（``"*": "allow"``）——
bash/edit/write 都不询问，安全性由执行环境兜底（本地是用户自担，
本平台是沙箱容器隔离）；审批只是用户显式收紧规则（配置 ``ask``）
或命中敏感项（读 ``.env``）时才出现的交互。
"""

from __future__ import annotations

from .rules import Rule, Ruleset


def default_ruleset() -> Ruleset:
    """opencode ``agent.ts`` defaults：

    - 全局默认 allow（未知工具 / bash / edit 均不问）
    - 敏感凭证文件读取 ask（``*.env`` / ``*.env.*``；``*.env.example`` 放行）
    """
    return [
        Rule(permission="*", pattern="*", action="allow"),
        # 敏感凭证：ask（opencode read 规则原文如此，非 deny）
        Rule(permission="read", pattern="*.env", action="ask"),
        Rule(permission="read", pattern="*.env.*", action="ask"),
        Rule(permission="read", pattern="*.env.example", action="allow"),
    ]


def build_ruleset() -> Ruleset:
    """build（默认编码模式）= defaults —— 与 opencode build agent 一致。"""
    return default_ruleset()


def plan_ruleset() -> Ruleset:
    """plan（只读规划）= defaults + 写操作 deny。

    opencode plan 仅 deny edit（写计划文件除外）；本平台 plan 是纯只读
    规划模式（写工具根本不注册），bash/task 一并 deny 属平台模式定义。
    """
    return default_ruleset() + [
        Rule(permission="edit", pattern="*", action="deny"),
        Rule(permission="write", pattern="*", action="deny"),
        Rule(permission="bash", pattern="*", action="deny"),
        Rule(permission="task", pattern="*", action="deny"),
    ]


#: 模式名 → 规则集工厂
PRESETS: dict[str, object] = {
    "build": build_ruleset,
    "plan": plan_ruleset,
}


def ruleset_for_mode(mode: str) -> Ruleset:
    """按模式名取规则集；未知模式回落 build。"""
    factory = PRESETS.get(mode, build_ruleset)
    return factory()
