"""权限规则引擎 — allow/ask/deny 三态评估（对齐 opencode permission 语义）。

规则模型：
- ``permission``：规则维度名。工具名（read/write/edit/bash/glob/grep/…）
  或虚拟维度（``external_directory``）。
- ``pattern``：该维度下的匹配模式，支持 ``*`` / ``**`` 通配。
  - 文件类工具 → 路径（如 ``/workspace/alice/**``）
  - bash → 子命令完整文本（如 ``git checkout main``）
- ``action``：allow / ask / deny

评估顺序：**后写的规则优先**（findLast 语义，同 opencode）——
规则集按 [defaults, user-config, runtime-approved] 依次合并，
末位（用户审批的 "always"）优先级最高。

bash 命令解析对齐 opencode ``tool/shell.ts``：
- ``bash_patterns``：复合命令拆为子命令完整文本（引号内操作符不拆；
  ``2>&1`` 的 ``&`` 是 fd 复制不拆；重定向保留在段文本内）；
  cd 类目录切换命令不产生 pattern（cwd 由工具管理）
- ``bash_always_patterns``："总是允许" 记忆的 pattern =
  命令前缀元数（``ARITY``）+ ``" *"``，如 ``git checkout *``、
  ``npm run *``；表外命令回落首词（``ls *``）
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal

Action = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class Rule:
    """单条权限规则。"""

    permission: str
    pattern: str = "*"
    action: Action = "ask"
    #: 规则说明（deny 时拼进 LLM 可见的错误详情，如 pip 硬禁令指引）
    reason: str = ""

    def describe(self) -> str:
        return f"{self.permission}:{self.pattern}={self.action}"


#: 规则集 = 有序规则列表（评估时从后往前找第一条命中）
Ruleset = list[Rule]


def evaluate(permission: str, pattern: str, ruleset: Ruleset) -> Rule:
    """按 findLast 语义评估规则集；无命中返回默认 ask。"""
    for rule in reversed(ruleset):
        if (
            wildcard_match(permission, rule.permission)
            and wildcard_match(pattern, rule.pattern)
        ):
            return rule
    return Rule(permission=permission, pattern="*", action="ask")


def merge_rulesets(*rulesets: Ruleset) -> Ruleset:
    """顺序拼接规则集（后者优先）。"""
    merged: Ruleset = []
    for ruleset in rulesets:
        merged.extend(ruleset)
    return merged


def rules_from_config(config: dict) -> Ruleset:
    """把嵌套 dict 配置（opencode 风格）展开为规则列表。

    形如::

        {
            "bash": "ask",
            "read": {"*": "allow", "*.env": "deny"},
            "edit": "ask",
        }

    根级 str 值 → pattern="*"；dict 值 → pattern=键。
    """
    rules: Ruleset = []
    for permission, value in config.items():
        if isinstance(value, str):
            rules.append(Rule(permission=permission, pattern="*", action=value))
        elif isinstance(value, dict):
            for pattern, action in value.items():
                rules.append(
                    Rule(permission=permission, pattern=pattern, action=action)
                )
    return rules


def pattern_for(tool_name: str, tool_args: dict) -> str:
    """非 bash 工具的调用参数 → 权限评估 pattern。

    - 文件类工具（read/write/edit/list）→ path/file_path/file 参数
    - 检索类（glob/grep）→ pattern 参数
    - 其他 → "*"
    """
    for key in ("path", "file_path", "file", "pattern"):
        value = tool_args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "*"


# ── bash 命令解析（对齐 opencode tool/shell.ts collect） ──

#: 目录切换命令不产生审批 pattern（cwd 由工具管理，同 opencode CWD）
CWD_COMMANDS = frozenset({"cd", "chdir", "popd", "pushd"})

#: 命令前缀元数表（移植 opencode permission/arity.ts）：
#: 值 = 定义"人类可理解命令"的 token 数，最长前缀优先。
#: 例：git:2 → ``git checkout``；"npm run":3 → ``npm run dev``；
#: 表外命令回落首词（``ls``）。
ARITY: dict[str, int] = {
    "cat": 1,
    "cd": 1,
    "chmod": 1,
    "chown": 1,
    "cp": 1,
    "echo": 1,
    "env": 1,
    "export": 1,
    "grep": 1,
    "kill": 1,
    "killall": 1,
    "ln": 1,
    "ls": 1,
    "mkdir": 1,
    "mv": 1,
    "ps": 1,
    "pwd": 1,
    "rm": 1,
    "rmdir": 1,
    "sleep": 1,
    "source": 1,
    "tail": 1,
    "touch": 1,
    "unset": 1,
    "which": 1,
    "aws": 3,
    "az": 3,
    "bazel": 2,
    "brew": 2,
    "bun": 2,
    "bun run": 3,
    "bun x": 3,
    "cargo": 2,
    "cargo add": 3,
    "cargo run": 3,
    "cdk": 2,
    "cf": 2,
    "cmake": 2,
    "composer": 2,
    "consul": 2,
    "consul kv": 3,
    "crictl": 2,
    "deno": 2,
    "deno task": 3,
    "doctl": 3,
    "docker": 2,
    "docker builder": 3,
    "docker compose": 3,
    "docker container": 3,
    "docker image": 3,
    "docker network": 3,
    "docker volume": 3,
    "eksctl": 2,
    "eksctl create": 3,
    "firebase": 2,
    "flyctl": 2,
    "gcloud": 3,
    "gh": 3,
    "git": 2,
    "git config": 3,
    "git remote": 3,
    "git stash": 3,
    "go": 2,
    "gradle": 2,
    "helm": 2,
    "heroku": 2,
    "hugo": 2,
    "ip": 2,
    "ip addr": 3,
    "ip link": 3,
    "ip netns": 3,
    "ip route": 3,
    "kind": 2,
    "kind create": 3,
    "kubectl": 2,
    "kubectl kustomize": 3,
    "kubectl rollout": 3,
    "kustomize": 2,
    "make": 2,
    "mc": 2,
    "mc admin": 3,
    "minikube": 2,
    "mongosh": 2,
    "mysql": 2,
    "mvn": 2,
    "ng": 2,
    "npm": 2,
    "npm exec": 3,
    "npm init": 3,
    "npm run": 3,
    "npm view": 3,
    "nvm": 2,
    "nx": 2,
    "openssl": 2,
    "openssl req": 3,
    "openssl x509": 3,
    "pip": 2,
    "pipenv": 2,
    "pnpm": 2,
    "pnpm dlx": 3,
    "pnpm exec": 3,
    "pnpm run": 3,
    "poetry": 2,
    "podman": 2,
    "podman container": 3,
    "podman image": 3,
    "psql": 2,
    "pulumi": 2,
    "pulumi stack": 3,
    "pyenv": 2,
    "python": 2,
    "rake": 2,
    "rbenv": 2,
    "redis-cli": 2,
    "rustup": 2,
    "serverless": 2,
    "sfdx": 3,
    "skaffold": 2,
    "sls": 2,
    "sst": 2,
    "swift": 2,
    "systemctl": 2,
    "terraform": 2,
    "terraform workspace": 3,
    "tmux": 2,
    "turbo": 2,
    "ufw": 2,
    "vault": 2,
    "vault auth": 3,
    "vault kv": 3,
    "vercel": 2,
    "volta": 2,
    "wp": 2,
    "yarn": 2,
    "yarn dlx": 3,
    "yarn run": 3,
}

#: 前缀 KEY=VALUE 环境赋值 token（``FOO=bar cmd`` 的赋值段不计入命令）
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def arity_prefix(tokens: list[str]) -> list[str]:
    """按元数表截取"人类可理解命令"token（最长前缀优先，表外回落首词）。"""
    for length in range(len(tokens), 0, -1):
        arity = ARITY.get(" ".join(tokens[:length]))
        if arity is not None:
            return tokens[:arity]
    return tokens[:1] if tokens else []


def bash_tokens(segment: str) -> list[str]:
    """子命令文本 → token 列表（shlex 失败回落空白切分；跳过前缀环境赋值）。"""
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    while tokens and _ENV_ASSIGN_RE.match(tokens[0]):
        tokens = tokens[1:]
    return tokens


def bash_patterns(command: str) -> list[str]:
    """复合命令 → 子命令完整文本列表（opencode ``scan.patterns`` 语义）。

    - 引号内操作符不拆（``echo "a && b"`` 是一条）
    - 引号外 ``&&`` / ``||`` / ``;`` / ``|`` / ``&`` / 换行 切分；
      ``2>&1`` 的 ``&`` 紧跟 ``>`` 是 fd 复制，不切
    - 重定向（``>`` / ``>>`` / ``<``）保留在段文本内（同 opencode
      redirected_statement 取父节点文本）
    - cd 类目录切换命令跳过（不产生审批 pattern）
    """
    # 引号掩码：引号内字符替换为空格（等长对齐），操作符检测只作用于引号外
    masked: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < n:
                masked.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = None
            masked.append(" ")
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            masked.append(" ")
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            masked.append("  ")
            i += 2
            continue
        masked.append(ch)
        i += 1
    masked_text = "".join(masked)

    cuts: list[tuple[int, int]] = []
    i = 0
    while i < n:
        pair = masked_text[i : i + 2]
        if pair in ("&&", "||"):
            cuts.append((i, i + 2))
            i += 2
            continue
        ch = masked_text[i]
        if ch in (";", "|", "&", "\n"):
            # >&N fd 复制中的 & 紧跟 >，不作切分
            if ch == "&" and i > 0 and masked_text[i - 1] == ">":
                i += 1
                continue
            cuts.append((i, i + 1))
            i += 1
            continue
        i += 1

    segments: list[str] = []
    last = 0
    for start, end in cuts:
        seg = command[last:start].strip()
        if seg:
            segments.append(seg)
        last = end
    tail = command[last:].strip()
    if tail:
        segments.append(tail)

    return [
        seg for seg in segments
        if bash_tokens(seg) and bash_tokens(seg)[0] not in CWD_COMMANDS
    ]


def bash_always_patterns(command: str) -> list[str]:
    """每个子命令的 "always" 记忆 pattern：前缀元数 + ``" *"``。

    对齐 opencode ``scan.always``（``BashArity.prefix(tokens).join(" ") + " *"``）：
    审批一次 ``git checkout main``，记住 ``git checkout *``，
    之后 ``git checkout -`` 开头的分支操作免审批。
    """
    return [
        " ".join(arity_prefix(bash_tokens(seg))) + " *"
        for seg in bash_patterns(command)
    ]


# ── 裸 pip 硬禁令（单一权威检测器）─────────────────

#: 裸 pip 拒绝后的替代指引（guard deny 详情与 bash 工具拒绝文案共用）
PIP_VIOLATION_GUIDE = (
    "禁止用裸 pip 安装/卸载依赖（会污染平台自身或全局 Python 环境，"
    "无法被审批绕过）。依赖安装必须先建 venv、在 venv 内做各种包的安装，"
    "一律用 uv：\n"
    "1. 首次初始化（工作区根）：uv venv .venv\n"
    "2. 安装：uv pip install --python .venv "
    "--default-index https://pypi.tuna.tsinghua.edu.cn/simple <package>\n"
    "   （项目用 pyproject 管理时优先 uv add <package>，"
    "镜像网 --default-index 同上）\n"
    "3. 运行：.venv/bin/python …（或先 source .venv/bin/activate）"
)

#: pip 归一化前缀：``python -m pip …`` → ``pip …``
_PYTHON_RE = re.compile(r"^python3?(?:\.\d+)?$")

#: 命令前缀剥离链（sudo / env；KEY=VAL 已由 bash_tokens 剥）
_PREFIX_STRIP = frozenset({"sudo", "env"})

#: pip 只读子命令之外的变更型子命令
_PIP_MUTATING = frozenset({"install", "uninstall"})


def find_pip_violation(command: str) -> str | None:
    """检测裸 ``pip install/uninstall`` 片段（返回命中的子命令；None = 安全）。

    基于 ``bash_patterns``（引号感知切分，引号内的 ``&&`` 不拆）逐子命令
    检测：剥 sudo/env/``KEY=VAL`` 前缀 → ``python -m pip`` 归一化为 ``pip``
    → head 为 pip/pip3 且含 install/uninstall 即命中。放行形态：
    ``uv pip install``（head 是 uv）、``pip list/show`` 等只读子命令、
    引号内的文本。guard 与 bash 工具共用本检测器（单一权威源）。
    """
    for segment in bash_patterns(command):
        tokens = bash_tokens(segment)
        while tokens and tokens[0] in _PREFIX_STRIP:
            tokens = tokens[1:]
        if not tokens:
            continue
        head = tokens[0]
        if (
            _PYTHON_RE.fullmatch(head)
            and len(tokens) >= 3
            and tokens[1] == "-m"
            and tokens[2] in {"pip", "pip3"}
        ):
            tokens = [tokens[2], *tokens[3:]]
            head = tokens[0]
        if head not in {"pip", "pip3"}:
            continue  # uv pip install / 其他命令天然不命中
        if any(t in _PIP_MUTATING for t in tokens[1:]):
            return segment.strip()
    return None


# ── 通配符匹配 ──────────────────────────────────────────


def wildcard_match(value: str, pattern: str) -> bool:
    """shell 风格通配匹配：``*`` 任意字符段、``**`` 含路径分隔符、大小写敏感。

    实现上 ``*`` 与 ``**`` 同义（正则 ``.*``），区别仅在语义文档；
    opencode 的 Wildcard 同样如此（fnmatch 路径语义）。
    """
    if pattern == "*":
        return True
    regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    return re.match(regex, value) is not None
