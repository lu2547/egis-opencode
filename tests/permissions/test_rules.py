"""权限规则引擎表驱动测试 — findLast 语义 / 通配匹配 / 预设 / bash 解析。

预设断言对齐 opencode ``agent.ts`` defaults：默认 ``*:* allow``，
仅 ``*.env`` 敏感凭证读取 ask。
"""

from __future__ import annotations

import pytest

from egis_opencode.permissions.rules import (
    ARITY,
    Rule,
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
from egis_opencode.permissions.presets import (
    build_ruleset,
    plan_ruleset,
    ruleset_for_mode,
)


# ── evaluate：findLast 语义（后写优先）──────────────────


@pytest.mark.parametrize(
    ("permission", "pattern", "expected_action"),
    [
        # build = opencode defaults：全放行，仅 .env 询问
        ("read", "/ws/alice/main.py", "allow"),
        ("read", "/ws/alice/.env", "ask"),            # *.env ask
        ("read", "/ws/alice/app.env", "ask"),
        ("read", "/ws/alice/app.env.local", "ask"),   # *.env.* ask
        ("read", "/ws/alice/.env.example", "allow"),  # *.env.example 再覆盖回 allow
        ("write", "/ws/alice/new.py", "allow"),       # 默认放行（同 opencode）
        ("edit", "/ws/alice/a.py", "allow"),
        ("bash", "rm -rf /tmp/x", "allow"),           # bash 不问（同 opencode）
        ("bash", "docker ps", "allow"),
        ("glob", "**", "allow"),
        ("grep", "def main", "allow"),
        ("todo", "*", "allow"),
        ("webfetch", "http://x.com", "allow"),        # 未知工具也走全局 allow
    ],
)
def test_build_ruleset_default_allow(permission, pattern, expected_action):
    rule = evaluate(permission, pattern, build_ruleset())
    assert rule.action == expected_action


@pytest.mark.parametrize(
    ("permission", "pattern", "expected_action"),
    [
        ("read", "/ws/alice/main.py", "allow"),       # defaults 保留
        ("read", "/ws/alice/.env", "ask"),            # defaults 保留
        ("glob", "**/*.py", "allow"),
        ("grep", "pattern", "allow"),
        ("list", ".", "allow"),
        ("todo", "*", "allow"),
        ("write", "/ws/alice/x.py", "deny"),          # plan 追加写操作 deny
        ("edit", "/ws/alice/x.py", "deny"),
        ("bash", "ls", "deny"),                       # plan 纯只读（平台定义）
        ("task", "general", "deny"),
    ],
)
def test_plan_ruleset_readonly(permission, pattern, expected_action):
    rule = evaluate(permission, pattern, plan_ruleset())
    assert rule.action == expected_action


def test_unknown_mode_falls_back_to_build():
    assert ruleset_for_mode("no-such-mode") == build_ruleset()


def test_evaluate_no_match_defaults_to_ask():
    rule = evaluate("anything", "value", [])
    assert rule.action == "ask"


def test_later_rule_wins():
    ruleset = [
        Rule(permission="bash", pattern="*", action="allow"),
        Rule(permission="bash", pattern="git push*", action="ask"),
        Rule(permission="bash", pattern="git push*", action="deny"),  # 后写优先
    ]
    assert evaluate("bash", "git push origin", ruleset).action == "deny"
    assert evaluate("bash", "git status", ruleset).action == "allow"


def test_merge_rulesets_order():
    base = [Rule(permission="read", action="ask")]
    extra = [Rule(permission="read", action="allow")]
    merged = merge_rulesets(base, extra)
    assert merged[0].action == "ask"
    assert merged[1].action == "allow"
    assert evaluate("read", "x", merged).action == "allow"


# ── rules_from_config：opencode 风格嵌套配置展开 ────────


def test_rules_from_config_flat_and_nested():
    ruleset = rules_from_config(
        {
            "bash": "ask",
            "read": {"*": "allow", "*.env": "deny"},
        }
    )
    by_pair = {(r.permission, r.pattern): r.action for r in ruleset}
    assert by_pair[("bash", "*")] == "ask"
    assert by_pair[("read", "*")] == "allow"
    assert by_pair[("read", "*.env")] == "deny"


# ── pattern_for：非 bash 工具的参数提取 ────────────────


def test_pattern_for_extracts_path():
    assert pattern_for("read", {"path": "a.py"}) == "a.py"
    assert pattern_for("write", {"file_path": "b.py"}) == "b.py"
    assert pattern_for("edit", {"file": "c.py"}) == "c.py"
    assert pattern_for("glob", {"pattern": "**/*.ts"}) == "**/*.ts"


def test_pattern_for_falls_back_to_star():
    assert pattern_for("todo", {"todos": []}) == "*"
    assert pattern_for("bash", {"command": "ls"}) == "*"


# ── wildcard_match ─────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "pattern", "expected"),
    [
        ("anything", "*", True),
        ("/a/b/c.py", "*.py", True),
        ("/a/b/c.py", "*.ts", False),
        ("/a/b/c.py", "/a/**", True),
        ("git status", "git", False),       # 精确匹配，不是前缀
        ("git", "git", True),
        ("git checkout main", "git checkout *", True),  # arity 前缀 pattern
        ("git status", "git checkout *", False),
        ("/ws/alice/app.env", "*.env", True),
        ("/ws/alice/app.env.local", "*.env.*", True),
        ("/ws/alice/.env.example", "*.env.example", True),
        ("", "*", True),
    ],
)
def test_wildcard_match(value, pattern, expected):
    assert wildcard_match(value, pattern) is expected


def test_rule_describe():
    assert Rule(permission="bash", pattern="git", action="allow").describe() == (
        "bash:git=allow"
    )


# ── bash_patterns：复合命令 → 子命令完整文本 ────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo hi && ls -la", ["echo hi", "ls -la"]),
        ("ls | head -3", ["ls", "head -3"]),
        ("a; b\n c", ["a", "b", "c"]),
        ("git pull || git clone x", ["git pull", "git clone x"]),
        ("pytest 2>&1 | tail -5", ["pytest 2>&1", "tail -5"]),
        # 重定向保留在段文本内（同 opencode redirected_statement）
        ("ls > out.txt", ["ls > out.txt"]),
        ("cat a >> b", ["cat a >> b"]),
        # 命令替换不拆段（留在段文本里，安全性由规则匹配决定）
        ("echo $(rm -rf /)", ["echo $(rm -rf /)"]),
        # cd 类命令跳过（cwd 由工具管理）
        ("cd /tmp && rm -rf x", ["rm -rf x"]),
        ("ls", ["ls"]),
    ],
)
def test_bash_patterns(command, expected):
    assert bash_patterns(command) == expected


def test_bash_patterns_quotes_protected():
    """引号内的操作符不拆（等长掩码保证索引对齐）。"""
    assert bash_patterns('echo "a && b"') == ['echo "a && b"']
    assert bash_patterns("grep 'x|y' file") == ["grep 'x|y' file"]
    assert bash_patterns('echo "a;b" && ls') == ['echo "a;b"', "ls"]
    # 用户截图场景：复合只读命令整体是两条 pattern
    assert bash_patterns('echo "Files in raw/docs:" && ls raw/docs/') == [
        'echo "Files in raw/docs:"', "ls raw/docs/",
    ]


# ── arity_prefix / bash_tokens ─────────────────────────


def test_bash_tokens_strips_env_assignment():
    assert bash_tokens("FOO=bar cmd arg") == ["cmd", "arg"]
    assert bash_tokens("cmd arg") == ["cmd", "arg"]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["git", "checkout", "main"], ["git", "checkout"]),      # git:2
        (["git", "status"], ["git", "status"]),                  # git:2
        (["npm", "run", "dev"], ["npm", "run", "dev"]),          # npm run:3
        (["npm", "install"], ["npm", "install"]),                # npm:2
        (["ls", "-la"], ["ls"]),                                 # 表外回落首词
        (["docker", "compose", "up"], ["docker", "compose", "up"]),  # docker compose:3
        (["redis-cli", "ping"], ["redis-cli", "ping"]),          # redis-cli:2
    ],
)
def test_arity_prefix(tokens, expected):
    assert arity_prefix(tokens) == expected


def test_arity_table_keys():
    assert ARITY["git"] == 2
    assert ARITY["npm run"] == 3
    assert ARITY["redis-cli"] == 2


# ── bash_always_patterns：前缀元数 + " *" ──────────────


def test_bash_always_patterns():
    assert bash_always_patterns("git checkout main && npm run build") == [
        "git checkout *", "npm run build *",
    ]
    assert bash_always_patterns("ls -la") == ["ls *"]
    # cd 段跳过 → 只对 rm 产生记忆 pattern
    assert bash_always_patterns("cd /tmp && rm -rf x") == ["rm *"]


# ── find_pip_violation：裸 pip 硬禁令单一权威检测器 ───


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "pip3 install requests",
        "pip install -r requirements.txt",
        "pip uninstall requests",
        "python -m pip install requests",
        "python3.12 -m pip install requests",
        "sudo pip install requests",
        "env pip install requests",
        "sudo env pip install requests",
        "PIP_INDEX_URL=http://x pip install requests",
        "cd proj && pip install requests",
        "make build || pip install requests",
        "echo hi; pip install requests",
        "cat setup.py | pip install -e .",
    ],
)
def test_find_pip_violation_hits(command):
    assert find_pip_violation(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install requests",          # uv 前缀：head 是 uv
        "uv add requests",
        "uv venv .venv",
        "uv pip install --python .venv requests",
        "uv run pip list",
        "pip list",                          # 只读子命令
        "pip show requests",
        "pip --version",
        "echo install",
        "grep pip install README.md",
        'echo "a && pip install x"',         # 引号内是纯文本，不拆段
        "make install",
    ],
)
def test_find_pip_violation_passes(command):
    assert find_pip_violation(command) is None


def test_find_pip_violation_returns_hit_segment():
    """返回命中的子命令原文（拒绝文案里展示给模型）。"""
    assert find_pip_violation("cd proj && pip install requests") == (
        "pip install requests"
    )
    assert find_pip_violation("sudo pip3 uninstall flask") == (
        "sudo pip3 uninstall flask"
    )
