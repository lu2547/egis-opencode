"""SessionRunLock — 跨进程会话互斥（PG advisory lock；ark 乐观锁的上层前置）。

背景：ark 用 session version 乐观锁（``ConcurrentSessionUpdate``）兜底
并发写冲突，但报错时 run 已消耗 token。进程内已有 ``RunRegistry``
（``api/runs.py``）防抖，多副本部署时进程间仍会撞 —— PG 在场时用
``pg_try_advisory_lock`` 在 run 开始前抢占，拿不到立即 409，把冲突
挡在模型调用之前（同 opencode ``assertNotBusy`` 的前置语义）。

实现要点：
- **独立 asyncpg 短连接**而非 ark engine 池：advisory lock 绑定连接
  backend，池化连接归还未断开时锁会残留（同 session 下一个 run 误判
  busy）；一次性连接 close 即释放，且不占 ark 会话池
- **降级路径**（fail-open）：非 PG 后端（SQLite）、PG 不可达 →
  noop handle，退化为仅 ``RunRegistry`` 进程内防抖 —— 可用性优先，
  极端漏网仍由 ark 乐观锁兜底（ConcurrentSessionUpdate 透传 409）
- 锁 key：``crc32(session_id)``（uint32 → PG bigint；同 session 稳定
  同 key，不同 session 撞 key 概率 2^-32 可忽略，且撞上只是误 409
  不损数据）
"""

from __future__ import annotations

import asyncio
import logging
import os
import zlib
from typing import Any

logger = logging.getLogger(__name__)


class RunLockHandle:
    """已持有的运行锁（release 幂等）。

    ``noop=True`` 表示降级形态（非 PG / PG 不可达）—— 不产生任何
    跨进程互斥，调用方继续依赖 RunRegistry + ark 乐观锁。
    """

    __slots__ = ("_conn", "noop")

    def __init__(self, conn: Any | None = None, *, noop: bool = False) -> None:
        self._conn = conn
        self.noop = noop

    async def release(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            await conn.close()  # 断开 backend → advisory lock 必然释放
        except Exception:  # noqa: BLE001 — 释放失败只记日志（连接终会回收）
            logger.warning("run lock release failed", exc_info=True)


def _pg_dsn() -> str | None:
    """``DB_CONNECTION_STR`` → asyncpg DSN；非 postgresql 前缀返回 None。"""
    conn = os.environ.get("DB_CONNECTION_STR", "").strip()
    if not conn.startswith("postgresql"):
        return None
    # sqlalchemy 形态（postgresql+asyncpg://…）→ asyncpg 原生 DSN
    if "+asyncpg" in conn:
        conn = conn.replace("postgresql+asyncpg", "postgresql", 1)
    return conn


class SessionRunLock:
    """PG advisory lock 封装（进程级单例语义）。"""

    def __init__(self, *, connect_timeout: float = 2.0) -> None:
        self._connect_timeout = connect_timeout

    async def acquire(self, session_id: str) -> RunLockHandle | None:
        """抢占会话锁；返回 handle（含 noop 降级），None = 会话正忙。

        宁可降级不可阻塞请求：连接失败不影响可用性（RunRegistry
        与 ark 乐观锁兜底），只在真正抢到/抢不到之间二值返回。
        """
        dsn = _pg_dsn()
        if dsn is None:
            return RunLockHandle(noop=True)
        try:
            import asyncpg

            conn = await asyncio.wait_for(
                asyncpg.connect(dsn), timeout=self._connect_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — 降级路径
            logger.warning(
                "session run lock degraded to in-process registry "
                "(pg unreachable): %s", exc,
            )
            return RunLockHandle(noop=True)
        try:
            key = zlib.crc32(session_id.encode("utf-8"))
            got = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", key,
            )
        except Exception as exc:  # noqa: BLE001 — 降级路径
            logger.warning("pg_try_advisory_lock failed: %s", exc)
            await conn.close()
            return RunLockHandle(noop=True)
        if not got:
            await conn.close()
            return None
        return RunLockHandle(conn)


#: 进程级单例
session_run_lock = SessionRunLock()


__all__ = ["RunLockHandle", "SessionRunLock", "session_run_lock"]
