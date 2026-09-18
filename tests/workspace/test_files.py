"""文件浏览服务测试 — file_tree（结构/过滤/懒加载/限额/守卫）+ read_file（预览）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from egis_opencode.workspace import WorkspacePathError
from egis_opencode.workspace.service import WorkspaceService


@pytest.fixture
def local_wiki(tmp_path: Path) -> Path:
    """模拟 llm-wiki 目录：wiki 子树 + 命令 + 噪音目录。"""
    d = tmp_path / "llm-wiki"
    (d / "wiki" / "concepts").mkdir(parents=True)
    (d / "wiki" / "index.md").write_text("# 首页\n", encoding="utf-8")
    (d / "wiki" / "concepts" / "rag.md").write_text("# RAG\n", encoding="utf-8")
    (d / "raw").mkdir()
    (d / "raw" / "article.md").write_text("素材", encoding="utf-8")
    (d / "AGENTS.md").write_text("# 规范\n", encoding="utf-8")
    (d / ".opencode" / "commands").mkdir(parents=True)
    (d / ".opencode" / "commands" / "ingest.md").write_text("cmd", encoding="utf-8")
    (d / ".git" / "objects").mkdir(parents=True)  # 噪音：应被过滤
    (d / "node_modules" / "pkg").mkdir(parents=True)  # 噪音：应被过滤
    return d


@pytest.fixture
def service() -> WorkspaceService:
    return WorkspaceService()


def _find(nodes: list, path: str):
    for node in nodes:
        if node.path == path:
            return node
        if node.children:
            hit = _find(node.children, path)
            if hit:
                return hit
    return None


# ── file_tree ──────────────────────────────────────────


async def test_tree_local_binding_structure(service, local_wiki):
    nodes, truncated = await service.file_tree(
        "alice", f"local:{local_wiki}", depth=2,
    )
    assert truncated is False
    # 顶层：目录在前、文件在后；dot 目录排普通目录之后（不抢占顶层视线）
    names = [n.name for n in nodes]
    assert names == ["raw", "wiki", ".opencode", "AGENTS.md"]
    assert ".git" not in names and "node_modules" not in names

    # depth=2：wiki 的子层已加载（concepts 目录 + index.md）
    wiki = _find(nodes, "wiki")
    assert wiki is not None and wiki.type == "dir"
    assert {c.name for c in wiki.children} == {"concepts", "index.md"}
    # 深度耗尽：concepts 的 children=None（懒加载标记）
    concepts = _find(nodes, "wiki/concepts")
    assert concepts.children is None


async def test_tree_depth_one_marks_all_dirs_lazy(service, local_wiki):
    nodes, _ = await service.file_tree("alice", f"local:{local_wiki}", depth=1)
    wiki = _find(nodes, "wiki")
    assert wiki is not None
    assert wiki.children is None  # 单层：所有目录都懒加载


async def test_tree_subpath_listing(service, local_wiki):
    """懒加载：仅列 wiki 子树，path 相对根。"""
    nodes, _ = await service.file_tree(
        "alice", f"local:{local_wiki}", path="wiki/concepts", depth=1,
    )
    assert [n.name for n in nodes] == ["rag.md"]
    assert nodes[0].type == "file"
    assert nodes[0].size > 0


async def test_tree_multitenant_user_root(ws_user_root, service):
    """空绑定串 = 多租户用户根（ws_user_root 先解析：service 构造时固化 root）。"""
    (ws_user_root / "proj").mkdir()
    (ws_user_root / "proj" / "README.md").write_text("x", encoding="utf-8")
    nodes, _ = await service.file_tree("alice", "", depth=2)
    assert [n.name for n in nodes] == ["proj"]
    assert [c.name for c in nodes[0].children] == ["README.md"]


async def test_tree_rejects_escape(service, local_wiki):
    with pytest.raises(WorkspacePathError):
        await service.file_tree("alice", f"local:{local_wiki}", path="../other")


async def test_tree_rejects_missing_dir(service, local_wiki):
    with pytest.raises(WorkspacePathError, match="目录不存在"):
        await service.file_tree("alice", f"local:{local_wiki}", path="nope")


async def test_tree_node_limit_truncates(service, local_wiki, monkeypatch):
    from egis_opencode.workspace import service as svc_mod
    monkeypatch.setattr(svc_mod, "_TREE_NODE_LIMIT", 3)
    nodes, truncated = await service.file_tree(
        "alice", f"local:{local_wiki}", depth=1,
    )
    assert truncated is True
    assert len(nodes) <= 3


# ── read_file ──────────────────────────────────────────


def test_read_file_utf8(service, local_wiki):
    content = service.read_file("alice", f"local:{local_wiki}", "wiki/index.md")
    assert content.content == "# 首页\n"
    assert content.truncated is False
    assert content.size == len("# 首页\n".encode("utf-8"))


def test_read_file_binary_returns_placeholder(service, local_wiki):
    """二进制（docx/图片）不再报错：返回占位标记，前端展示“不支持预览”。"""
    binary = local_wiki / "logo.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    content = service.read_file("alice", f"local:{local_wiki}", "logo.png")
    assert content.binary is True
    assert content.content == ""
    assert content.truncated is False
    assert content.size > 0


def test_read_file_truncates_oversize(service, local_wiki, monkeypatch):
    from egis_opencode.workspace import service as svc_mod
    monkeypatch.setattr(svc_mod, "_PREVIEW_MAX_BYTES", 10)
    big = local_wiki / "big.md"
    big.write_text("abcdefghij" * 3, encoding="utf-8")  # 30 字节
    content = service.read_file("alice", f"local:{local_wiki}", "big.md")
    assert content.truncated is True
    assert len(content.content) == 10
    assert content.size == 30


def test_read_file_rejects_missing_and_escape(service, local_wiki):
    with pytest.raises(WorkspacePathError, match="文件不存在"):
        service.read_file("alice", f"local:{local_wiki}", "nope.md")
    with pytest.raises(WorkspacePathError):
        service.read_file("alice", f"local:{local_wiki}", "../etc/passwd")
