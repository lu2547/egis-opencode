"""会话标题子系统 — after_agent 回调 + LLM 异步生成 + JSON 存储。

- ``TitleStore``：``CONFIG_DIR/coding/titles.json``（session_id → title），
  原子写入，进程内单例。
- ``TitleGenerator``：挂 ``RunnerCallbacks.after_agent``；仅对尚无标题的
  会话生成一次，后台 task 调用 ``ctx.llm(CALLBACK)``（未配置时回落
  MAIN），完成后写存储并经 ``input_context["temp:permission_bridge"]``
  通道（同权限守卫，run_hooks 不传 handler 给 hook）发 ``title_generated``
  事件（chat 端点的 SSE 收尾窗口会等它落帧）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..config import settings
from ..events import TITLE_GENERATED, TitleGeneratedPayload
from ..permissions.bridge import PermissionBridge, bridge_from_context

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "你是会话标题生成器。根据用户的第一条请求，生成一个不超过"
    "{max_chars} 个字符的简短标题（中文，概括用户意图）。"
    "直接输出标题本身：不要引号、不要句号、不要任何解释。\n\n"
    "用户请求：{user_input}"
)


class TitleStore:
    """session_id → title 的 JSON 文件存储（原子写）。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] | None = None

    def get(self, session_id: str) -> str | None:
        return self._load().get(session_id)

    def set(self, session_id: str, title: str) -> None:
        data = self._load()
        if data.get(session_id) == title:
            return
        data[session_id] = title
        self._save(data)

    def delete(self, session_id: str) -> None:
        data = self._load()
        if session_id in data:
            del data[session_id]
            self._save(data)

    def titles(self) -> dict[str, str]:
        return dict(self._load())

    # ── 内部 ───────────────────────────────────────────

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._cache = {
                str(k): str(v) for k, v in raw.items()
            } if isinstance(raw, dict) else {}
        except FileNotFoundError:
            self._cache = {}
        except (OSError, ValueError):
            logger.warning("titles file unreadable: %s", self._path, exc_info=True)
            self._cache = {}
        return self._cache

    def _save(self, data: dict[str, str]) -> None:
        self._cache = data
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self._path.parent, suffix=".tmp",
                delete=False, encoding="utf-8",
            ) as tmp:
                json.dump(data, tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, self._path)
        except OSError:
            logger.warning("titles save failed: %s", self._path, exc_info=True)


def default_title_store() -> TitleStore:
    """``CONFIG_DIR/coding/titles.json``（与 sandbox.json 同级布局）。"""
    from ark_agentic.core.paths import get_agent_config_dir

    return TitleStore(get_agent_config_dir("coding") / "titles.json")


class TitleGenerator:
    """after_agent 回调：首条 run 结束后异步生成会话标题。

    非阻塞：LLM 调用跑在后台 task，不拖延 run_finished 事件；
    生成完成经 bridge 发 ``title_generated``（流已断开时静默）。
    """

    def __init__(self, store: TitleStore) -> None:
        self._store = store
        self._tasks: set[asyncio.Task[None]] = set()

    async def __call__(
        self, ctx: Any, *, response: Any = None, result: Any = None,
        **kwargs: Any,
    ) -> None:
        session_id = getattr(ctx.session, "session_id", "")
        if not session_id or self._store.get(session_id):
            return None
        bridge = bridge_from_context(getattr(ctx, "input_context", None))
        if bridge is not None:
            bridge.defer()  # chat 端点关流前等本 task 落帧（drained 信号）
        task = asyncio.create_task(
            self._generate(ctx, session_id, bridge),
            name=f"title-{session_id[:12]}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if bridge is not None:
            task.add_done_callback(lambda _t: bridge.settle())
        return None

    async def _generate(
        self, ctx: Any, session_id: str, bridge: PermissionBridge | None,
    ) -> None:
        user_input = str(getattr(ctx, "user_input", "") or "").strip()
        title = ""
        try:
            llm = ctx.llm()  # CALLBACK role，未配置时由 registry 回落 MAIN
            response = await llm.ainvoke(
                _TITLE_PROMPT.format(
                    max_chars=settings.title_max_chars,
                    user_input=user_input[:500],
                )
            )
            title = _normalize_title(
                str(getattr(response, "content", "") or ""),
            )
        except Exception:  # noqa: BLE001 — 标题生成失败回落截断
            logger.info("title LLM generation failed; fallback", exc_info=True)
        if not title:
            title = user_input[: settings.title_max_chars] or "新会话"
        try:
            self._store.set(session_id, title)
        except Exception:  # noqa: BLE001
            return
        if bridge is not None and bridge.available:
            try:
                bridge.emit(
                    TITLE_GENERATED,
                    TitleGeneratedPayload(
                        session_id=session_id, title=title,
                    ).model_dump(),
                )
            except Exception:  # noqa: BLE001 — 流已断开
                pass


def _normalize_title(raw: str) -> str:
    text = raw.strip().strip("\"'“”‘’「」『』").strip()
    return text[: settings.title_max_chars]


#: 进程级单例 — TitleGenerator（写入）与 sessions REST（读取）共享。
title_store = default_title_store()


__all__ = [
    "TitleGenerator", "TitleStore", "default_title_store", "title_store",
]
