"""工作目录绑定测试 — resolve_binding 校验 / anchored 路径形态 / 工具锚定行为。"""

from __future__ import annotations

from pathlib import Path

import pytest
from ark_agentic.core.types import ToolCall

from egis_opencode.config import settings
from egis_opencode.workspace import WorkspacePathError, WorkspacePaths
from egis_opencode.workspace.service import WorkspaceService


@pytest.fixture
def local_dir(tmp_path: Path) -> Path:
    d = tmp_path / "llm-wiki"
    d.mkdir()
    (d / "wiki").mkdir()
    return d


@pytest.fixture
def service() -> WorkspaceService:
    return WorkspaceService()


# ── resolve_binding 校验 ───────────────────────────────


def test_local_binding_resolves(service: WorkspaceService, local_dir: Path):
    assert service.resolve_binding(f"local:{local_dir}") == local_dir.resolve()


def test_local_binding_rejects_relative(service: WorkspaceService, local_dir: Path):
    with pytest.raises(WorkspacePathError, match="绝对路径"):
        service.resolve_binding("local:relative/dir")


def test_local_binding_rejects_missing_dir(service: WorkspaceService, tmp_path: Path):
    with pytest.raises(WorkspacePathError, match="目录不存在"):
        service.resolve_binding(f"local:{tmp_path / 'nope'}")


def test_local_binding_rejects_root(service: WorkspaceService):
    with pytest.raises(WorkspacePathError, match="整个文件系统"):
        service.resolve_binding("local:/")


def test_local_binding_rejects_file(service: WorkspaceService, tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(WorkspacePathError):
        service.resolve_binding(f"local:{f}")


def test_unknown_mode_rejected(service: WorkspaceService, local_dir: Path):
    with pytest.raises(WorkspacePathError, match="不支持"):
        service.resolve_binding(f"remote:{local_dir}")
    with pytest.raises(WorkspacePathError):
        service.resolve_binding("garbage-no-colon")


def test_local_binding_disabled(service: WorkspaceService, local_dir: Path):
    old = settings.local_workspace_enabled
    object.__setattr__(settings, "local_workspace_enabled", False)
    try:
        with pytest.raises(WorkspacePathError, match="已禁用"):
            service.resolve_binding(f"local:{local_dir}")
    finally:
        object.__setattr__(settings, "local_workspace_enabled", old)


# ── anchored 路径形态 ──────────────────────────────────


def test_anchored_paths_root_is_user_root(local_dir: Path):
    paths = WorkspacePaths.anchored(local_dir)
    assert paths.user_root == local_dir
    assert paths.resolve("wiki/index.md") == local_dir / "wiki" / "index.md"


def test_anchored_paths_still_guards_escape(local_dir: Path):
    paths = WorkspacePaths.anchored(local_dir)
    with pytest.raises(WorkspacePathError):
        paths.resolve("../outside.txt")


def test_multi_tenant_paths_unchanged(ws_root: Path):
    paths = WorkspacePaths(root=ws_root, user_id="alice")
    assert paths.user_root == ws_root / "alice"


# ── 工具锚定（read 在 anchored root 下解析）────────────


async def test_read_tool_anchored_to_local_dir(local_dir: Path):
    from egis_opencode.agents.coding.tools.files import ReadTool

    (local_dir / "wiki" / "index.md").write_text("# 首页\n", encoding="utf-8")
    tool = ReadTool()
    ctx = {"user:id": "alice", "workspace:root": str(local_dir)}
    result = await tool.execute(
        ToolCall.create("read", {"path": "wiki/index.md"}), ctx,
    )
    assert result.is_error is False
    assert "1: # 首页" in str(result.content)


async def test_read_tool_rejects_escape_from_anchored(local_dir: Path):
    from egis_opencode.agents.coding.tools.files import ReadTool

    tool = ReadTool()
    ctx = {"user:id": "alice", "workspace:root": str(local_dir)}
    result = await tool.execute(
        ToolCall.create("read", {"path": "../../etc/passwd"}), ctx,
    )
    assert result.is_error is True


async def test_write_tool_anchored_creates_under_local_dir(local_dir: Path):
    from egis_opencode.agents.coding.tools.files import WriteTool

    tool = WriteTool()
    ctx = {"user:id": "alice", "workspace:root": str(local_dir)}
    result = await tool.execute(
        ToolCall.create(
            "write", {"path": "wiki/new.md", "content": "新页面"},
        ),
        ctx,
    )
    assert result.is_error is False
    assert (local_dir / "wiki" / "new.md").read_text(encoding="utf-8") == "新页面"


async def test_anchored_root_missing_dir_falls_back_multitenant(
    ws_root: Path, tmp_path: Path,
):
    """workspace:root 指向不存在目录 → 回落多租户（is_dir 守卫）。"""
    from egis_opencode.agents.coding.tools.files import ReadTool

    (ws_root / "alice").mkdir(parents=True, exist_ok=True)
    (ws_root / "alice" / "note.md").write_text("多租户内容", encoding="utf-8")
    tool = ReadTool()
    ctx = {
        "user:id": "alice",
        "workspace:root": str(tmp_path / "deleted-dir"),
    }
    result = await tool.execute(
        ToolCall.create("read", {"path": "note.md"}), ctx,
    )
    assert result.is_error is False
    assert "多租户内容" in str(result.content)
