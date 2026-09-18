"""permissions 子包 — allow/ask/deny 权限子系统（对齐 opencode permission）。"""

from .bridge import (
    BRIDGE_KEY,
    PermissionBridge,
    attach_bridge,
    bridge_from_context,
    detach_bridge,
)
from .guard import FRAMEWORK_TOOLS, PermissionGuard
from .presets import (
    build_ruleset,
    default_ruleset,
    plan_ruleset,
    ruleset_for_mode,
)
from .rules import (
    ARITY,
    Action,
    PIP_VIOLATION_GUIDE,
    Rule,
    Ruleset,
    arity_prefix,
    bash_always_patterns,
    bash_patterns,
    bash_tokens,
    evaluate,
    find_pip_violation,
    merge_rulesets,
    pattern_for,
    rules_from_config,
    wildcard_match,
)
from .service import (
    PendingRequest,
    PermissionResolution,
    PermissionService,
    UnknownPermissionRequest,
)

__all__ = [
    "ARITY",
    "BRIDGE_KEY",
    "FRAMEWORK_TOOLS",
    "PermissionBridge",
    "PermissionGuard",
    "PermissionResolution",
    "PermissionService",
    "PendingRequest",
    "Rule",
    "Ruleset",
    "Action",
    "PIP_VIOLATION_GUIDE",
    "arity_prefix",
    "attach_bridge",
    "bash_always_patterns",
    "bash_patterns",
    "bash_tokens",
    "bridge_from_context",
    "build_ruleset",
    "default_ruleset",
    "detach_bridge",
    "evaluate",
    "find_pip_violation",
    "merge_rulesets",
    "pattern_for",
    "plan_ruleset",
    "rules_from_config",
    "ruleset_for_mode",
    "UnknownPermissionRequest",
    "wildcard_match",
]
