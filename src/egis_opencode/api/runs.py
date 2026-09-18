"""RunRegistry — 进行中 run 的进程内注册表（abort / 并发防抖）。

session_id → asyncio.Task。同一 session 同时只允许一个 run；
``POST /chat/abort`` 经 registry cancel 对应 task。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class RunRegistry:
    """session 维度的 run 注册表（进程内单例语义）。"""

    def __init__(self) -> None:
        self._runs: dict[str, asyncio.Task] = {}

    def register(self, session_id: str, task: asyncio.Task) -> None:
        self._runs[session_id] = task

    def unregister(self, session_id: str, task: asyncio.Task) -> None:
        """仅当注册者仍是该 task 时注销（防竞态误删后继 run）。"""
        if self._runs.get(session_id) is task:
            del self._runs[session_id]

    def is_running(self, session_id: str) -> bool:
        task = self._runs.get(session_id)
        return task is not None and not task.done()

    def cancel(self, session_id: str) -> bool:
        task = self._runs.get(session_id)
        if task is None or task.done():
            return False
        task.cancel()
        logger.info("Run aborted: session=%s", session_id)
        return True


#: 进程级单例
run_registry = RunRegistry()


__all__ = ["RunRegistry", "run_registry"]
