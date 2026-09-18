"""多租户 workspace 路径守卫。

所有文件工具的路径参数必须先经 ``resolve_under_workspace`` 解析：
- 相对路径按用户 workspace 根解析
- 拒绝 ``..`` 穿越、绝对路径越界、symlink 逃逸
- 解析结果始终落在 ``<root>/<user_id>`` 内

设计约定：与 ark SandboxPlugin 的 workspace 布局（``<root>/<user_id>``）
保持一致，两套体系操作同一目录。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class WorkspacePathError(ValueError):
    """路径越界或不合法。"""


def safe_segment(value: str) -> str:
    """把 user_id 等外部输入清洗为安全的单段目录名（对齐 ark sandbox）。"""
    text = _SAFE_SEGMENT_RE.sub("_", str(value or "default")).strip("._")
    return text or "default"


@dataclass(frozen=True)
class WorkspacePaths:
    """单个用户的 workspace 路径解析器。

    两种形态：
    - 多租户（默认）：``user_root = <root>/<user_id>``
    - 锚定（``anchored``）：``user_root = root`` 本身 —— 会话绑定了
      本地目录 / workspace 内项目目录时使用，不再拼 user_id 段。
    """

    root: Path
    user_id: str

    @classmethod
    def anchored(cls, root: Path) -> "WorkspacePaths":
        """以 root 自身为工作区根（本地/项目目录模式）。"""
        return cls(root=root, user_id="")

    @property
    def user_root(self) -> Path:
        """该用户的 workspace 根目录 ``<root>/<user_id>``。"""
        if not self.user_id:
            return self.root
        return self.root / safe_segment(self.user_id)

    def ensure_user_root(self) -> Path:
        """确保用户 workspace 目录存在并返回。"""
        self.user_root.mkdir(parents=True, exist_ok=True)
        return self.user_root

    def resolve(self, raw: str) -> Path:
        """解析工具传入的路径参数为 workspace 内绝对路径。

        Raises:
            WorkspacePathError: 路径为空、穿越越界或指向 symlink 逃逸。
        """
        return resolve_under_workspace(self.user_root, raw)

    def relative_display(self, path: Path) -> str:
        """相对用户 workspace 根的展示路径。"""
        try:
            return str(path.relative_to(self.user_root))
        except ValueError:
            return str(path)


def resolve_under_workspace(user_root: Path, raw: str) -> Path:
    """把 ``raw`` 解析为 ``user_root`` 内的绝对路径（守卫核心）。

    规则：
    1. 空路径报错
    2. 绝对路径：必须在 user_root 内（展示为直接命中）
    3. 相对路径：按 user_root 拼接后 resolve
    4. resolve 后必须仍落在 user_root 内（挡 ``..`` 与 symlink 逃逸）
    """
    if raw is None or not str(raw).strip():
        raise WorkspacePathError("path is empty")

    text = str(raw).strip()
    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = user_root / candidate

    # strict=False：允许指向尚不存在的目标（write/edit 前置检查）
    resolved = resolved.resolve(strict=False)

    root = user_root.resolve(strict=False)
    if not _is_relative_to(resolved, root):
        raise WorkspacePathError(
            f"path escapes workspace: {raw!r} (resolved: {resolved})"
        )
    return resolved


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
