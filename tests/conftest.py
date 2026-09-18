"""egis-opencode 测试公共基建（fixtures）。

- ``ws_root``：把全局 ``settings.workspace_root`` 指向 tmp 目录
  （frozen dataclass 实例经 ``__dict__`` 整体替换，monkeypatch 自动恢复）。
- ``recorder``：SSE handler 录制器（见 ``tests/helpers.py``）。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from egis_opencode.config import settings

from .helpers import RecordingHandler


@pytest.fixture
def ws_root(tmp_path: Path) -> Path:
    """把 settings.workspace_root 重定向到 tmp（含恢复）。

    frozen dataclass 拦截普通 setattr（含 ``__dict__``），
    故用 ``object.__setattr__`` 整体替换实例 ``__dict__``。
    """
    root = tmp_path / "workspaces"
    old_dict = dict(settings.__dict__)
    object.__setattr__(
        settings, "__dict__",
        dataclasses.replace(settings, workspace_root=root).__dict__,
    )
    yield root
    object.__setattr__(settings, "__dict__", old_dict)


@pytest.fixture
def ws_user_root(ws_root: Path) -> Path:
    """预建 ``<root>/alice`` 用户 workspace 并返回。"""
    user_root = ws_root / "alice"
    user_root.mkdir(parents=True, exist_ok=True)
    return user_root


@pytest.fixture
def recorder() -> RecordingHandler:
    return RecordingHandler()


__all__ = ["RecordingHandler", "recorder", "ws_root", "ws_user_root"]
