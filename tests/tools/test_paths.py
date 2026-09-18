"""WorkspacePaths 路径守卫测试 — 穿越 / 绝对路径越界 / symlink 逃逸 / 清洗。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from egis_opencode.workspace.paths import (
    WorkspacePathError,
    WorkspacePaths,
    resolve_under_workspace,
    safe_segment,
)


@pytest.fixture
def user_root(tmp_path: Path) -> Path:
    root = tmp_path / "ws" / "alice"
    root.mkdir(parents=True)
    return root


# ── resolve_under_workspace ────────────────────────────


def test_relative_path_resolves_inside(user_root):
    resolved = resolve_under_workspace(user_root, "proj/a.py")
    assert resolved == (user_root / "proj" / "a.py").resolve()
    assert user_root in resolved.parents


def test_absolute_path_inside_root_allowed(user_root):
    resolved = resolve_under_workspace(user_root, str(user_root / "a.py"))
    assert resolved == (user_root / "a.py").resolve()


def test_dotdot_traversal_rejected(user_root):
    with pytest.raises(WorkspacePathError, match="escapes"):
        resolve_under_workspace(user_root, "../bob/secret.txt")


def test_deep_dotdot_traversal_rejected(user_root):
    with pytest.raises(WorkspacePathError, match="escapes"):
        resolve_under_workspace(user_root, "proj/../../bob/secret.txt")


def test_absolute_path_outside_root_rejected(user_root, tmp_path):
    outside = tmp_path / "etc" / "passwd"
    with pytest.raises(WorkspacePathError, match="escapes"):
        resolve_under_workspace(user_root, str(outside))


def test_empty_path_rejected(user_root):
    with pytest.raises(WorkspacePathError, match="empty"):
        resolve_under_workspace(user_root, "")
    with pytest.raises(WorkspacePathError, match="empty"):
        resolve_under_workspace(user_root, "   ")


def test_nonexistent_target_allowed(user_root):
    """strict=False：write/edit 前置解析允许指向尚不存在的目标。"""
    resolved = resolve_under_workspace(user_root, "new_dir/new.py")
    assert not resolved.exists()
    assert resolved.parent == (user_root / "new_dir").resolve()


def test_symlink_escape_rejected(user_root, tmp_path):
    """workspace 内的 symlink 指向外部 → resolve 后逃逸被拒。"""
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    link = user_root / "leak"
    os.symlink(secret, link)
    with pytest.raises(WorkspacePathError, match="escapes"):
        resolve_under_workspace(user_root, "leak")


def test_symlink_inside_root_allowed(user_root):
    target = user_root / "real"
    target.mkdir()
    link = user_root / "alias"
    os.symlink(target, link)
    resolved = resolve_under_workspace(user_root, "alias/x.py")
    assert resolved.parent == target.resolve()


def test_root_itself_resolves(user_root):
    assert resolve_under_workspace(user_root, ".") == user_root.resolve()


# ── WorkspacePaths ─────────────────────────────────────


def test_user_root_segment_sanitized(tmp_path):
    """user_id 经 safe_segment 清洗（挡路径注入）。"""
    paths = WorkspacePaths(root=tmp_path, user_id="../etc/passwd")
    # 斜杠被替换、首尾 ./._ 被剥离 → 单段安全名
    assert paths.user_root == tmp_path / "etc_passwd"
    assert ".." not in paths.user_root.parts
    assert "/" not in paths.user_root.name


def test_ensure_user_root_creates(tmp_path):
    paths = WorkspacePaths(root=tmp_path, user_id="bob")
    root = paths.ensure_user_root()
    assert root.is_dir()
    assert root == tmp_path / "bob"


def test_relative_display(tmp_path):
    paths = WorkspacePaths(root=tmp_path, user_id="alice")
    display = paths.relative_display(tmp_path / "alice" / "proj" / "a.py")
    assert display == str(Path("proj") / "a.py")
    # 外部路径原样返回
    assert paths.relative_display(Path("/etc/passwd")) == "/etc/passwd"


# ── safe_segment ───────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("alice", "alice"),
        ("user-001", "user-001"),
        ("u_1.2", "u_1.2"),
        ("a/b", "a_b"),
        ("..", "default"),       # 清洗后空 → default
        ("", "default"),
        (None, "default"),
        ("  spaced  ", "spaced"),
        ("中文用户", "default"),    # 非 ASCII 全替换后剥离为空 → default
    ],
)
def test_safe_segment(raw, expected):
    assert safe_segment(raw) == expected
