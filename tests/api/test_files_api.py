"""文件浏览 REST 集成测试 — /workspaces/tree 与 /workspaces/file。"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.fixture
def local_wiki(tmp_path: Path) -> Path:
    d = tmp_path / "llm-wiki"
    (d / "wiki" / "concepts").mkdir(parents=True)
    (d / "wiki" / "index.md").write_text("# 首页\n\n内容", encoding="utf-8")
    (d / "raw").mkdir()
    (d / "raw" / "article.md").write_text("素材", encoding="utf-8")
    return d


async def test_tree_endpoint_local(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/workspaces/tree",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "depth": 2,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated"] is False
    names = [n["name"] for n in body["nodes"]]
    assert "wiki" in names and "raw" in names
    wiki = next(n for n in body["nodes"] if n["name"] == "wiki")
    assert {c["name"] for c in wiki["children"]} == {"concepts", "index.md"}
    concepts = next(c for c in wiki["children"] if c["name"] == "concepts")
    assert concepts["children"] is None  # 深度耗尽 → 懒加载标记


async def test_tree_endpoint_lazy_depth1(make_app, local_wiki):
    """懒加载：depth=1 只列一层，目录 children=None。"""
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/workspaces/tree",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "wiki",
                "depth": 1,
            },
        )
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    assert {n["name"] for n in nodes} == {"concepts", "index.md"}
    concepts = next(n for n in nodes if n["name"] == "concepts")
    assert concepts["children"] is None


async def test_tree_endpoint_rejects_escape(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/workspaces/tree",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "../outside",
            },
        )
    assert resp.status_code == 400


async def test_file_endpoint_preview(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/coding/workspaces/file",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "wiki/index.md",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "# 首页\n\n内容"
    assert body["truncated"] is False
    assert body["path"] == "wiki/index.md"


async def test_file_endpoint_errors(make_app, local_wiki):
    app, _ = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        missing = await client.get(
            "/api/coding/workspaces/file",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "nope.md",
            },
        )
        escape = await client.get(
            "/api/coding/workspaces/file",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "../etc/passwd",
            },
        )
        empty = await client.get(
            "/api/coding/workspaces/file",
            params={
                "user_id": "alice",
                "workspace_root": f"local:{local_wiki}",
                "path": "  ",
            },
        )
    # 语义区分：资源消失是 404（文件不存在）；参数非法/越界是 400
    assert missing.status_code == 404
    assert escape.status_code == 400
    assert empty.status_code == 400
