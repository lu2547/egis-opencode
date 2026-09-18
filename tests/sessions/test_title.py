"""TitleStore / TitleGenerator 测试 — 原子写 / 幂等 / LLM 成功与回落。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from egis_opencode.events import TITLE_GENERATED
from egis_opencode.permissions import BRIDGE_KEY, PermissionBridge
from egis_opencode.sessions.title import (
    TitleGenerator,
    TitleStore,
    _normalize_title,
)

from tests.helpers import MockChatModel, RecordingHandler


# ── TitleStore ─────────────────────────────────────────


def test_store_set_get_roundtrip(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    assert store.get("s1") is None

    store.set("s1", "修复登录 bug")
    assert store.get("s1") == "修复登录 bug"
    assert store.titles() == {"s1": "修复登录 bug"}

    # 落盘内容正确
    raw = json.loads((tmp_path / "titles.json").read_text(encoding="utf-8"))
    assert raw == {"s1": "修复登录 bug"}


def test_store_idempotent_when_same_title(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    store.set("s1", "title")
    mtime_first = (tmp_path / "titles.json").stat().st_mtime_ns
    store.set("s1", "title")  # 相同标题不重写
    assert (tmp_path / "titles.json").stat().st_mtime_ns == mtime_first


def test_store_overwrite_and_delete(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    store.set("s1", "old")
    store.set("s1", "new")
    assert store.get("s1") == "new"
    store.delete("s1")
    assert store.get("s1") is None
    # 删除不存在的 id 无副作用
    store.delete("nope")


def test_store_loads_existing_file(tmp_path):
    path = tmp_path / "titles.json"
    path.write_text(json.dumps({"s1": "已有"}), encoding="utf-8")
    store = TitleStore(path)
    assert store.get("s1") == "已有"


def test_store_unreadable_file_falls_back_empty(tmp_path):
    path = tmp_path / "titles.json"
    path.write_text("not json {{{", encoding="utf-8")
    store = TitleStore(path)
    assert store.titles() == {}


# ── _normalize_title ───────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("简单标题", "简单标题"),
        ('"带引号标题"', "带引号标题"),
        ("  带空白  ", "带空白"),
        ("「括号标题」", "括号标题"),
    ],
)
def test_normalize_title(raw, expected):
    assert _normalize_title(raw) == expected


# ── TitleGenerator（after_agent 回调形态）───────────────


class _Ctx:
    """duck-typed CallbackContext（title 路径触碰 session/user_input/llm/桥）。"""

    def __init__(
        self, user_input: str, llm,
        input_context: dict | None = None,
    ) -> None:
        self.user_input = user_input
        self.session = type("S", (), {"session_id": "sess-1"})()
        self._llm = llm
        self.input_context = input_context

    def llm(self, role: str = "callback"):
        return self._llm


async def test_generator_llm_success(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    llm = MockChatModel(responses=[AIMessage(content="生成会话标题")])
    generator = TitleGenerator(store)

    await generator(_Ctx("帮我给 README 加安装说明", llm))
    # 后台 task：等它完成
    await _drain_tasks(generator)

    assert store.get("sess-1") == "生成会话标题"


async def test_generator_emits_title_event(tmp_path, recorder):
    store = TitleStore(tmp_path / "titles.json")
    llm = MockChatModel(responses=[AIMessage(content="关于部署的问答")])
    generator = TitleGenerator(store)

    # 事件通道：after_agent hook 拿不到 handler，经 input_context 注入 bridge
    ctx = _Ctx(
        "部署问题", llm,
        input_context={BRIDGE_KEY: PermissionBridge(recorder)},
    )
    await generator(ctx)
    await _drain_tasks(generator)

    events = recorder.of_type(TITLE_GENERATED)
    assert len(events) == 1
    assert events[0]["session_id"] == "sess-1"
    assert events[0]["title"] == "关于部署的问答"


async def test_generator_llm_failure_falls_back_to_input(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    llm = MockChatModel()  # 无 responses → ainvoke 抛错
    generator = TitleGenerator(store)

    await generator(_Ctx("一段非常长的用户输入" * 10, llm))
    await _drain_tasks(generator)

    title = store.get("sess-1")
    assert title is not None
    assert len(title) <= 50  # title_max_chars
    assert title.startswith("一段非常长的用户输入")


async def test_generator_skips_when_title_exists(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    store.set("sess-1", "已有标题")
    llm = MockChatModel(responses=[AIMessage(content="不该被调用")])
    generator = TitleGenerator(store)

    await generator(_Ctx("新输入", llm))

    await _drain_tasks(generator)
    assert store.get("sess-1") == "已有标题"
    assert llm.call_count == 0


async def test_generator_truncates_long_llm_title(tmp_path):
    store = TitleStore(tmp_path / "titles.json")
    long_title = "标" * 120
    llm = MockChatModel(responses=[AIMessage(content=long_title)])
    generator = TitleGenerator(store)

    await generator(_Ctx("输入", llm))
    await _drain_tasks(generator)

    assert len(store.get("sess-1")) == 50


async def _drain_tasks(generator: TitleGenerator) -> None:
    """等待后台 title task 全部结束。"""
    for _ in range(200):
        if not generator._tasks:
            return
        import asyncio

        await asyncio.sleep(0.01)
    raise AssertionError("title generation tasks did not finish")
