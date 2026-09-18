"""SessionRunLock 测试 — PG advisory lock 互斥 + 降级形态（3.4）。

- PG 可达（weknora）：同 session 二次 acquire 得 None、release 后可重
  新 acquire、不同 session 互不干扰、跨"进程"（两个独立连接）互斥
- 非 PG / 无 env：noop handle（退化为 RunRegistry + ark 乐观锁）
- PG 不可达：fail-open noop（本机 docker 未起时跳过 PG 组）
"""

from __future__ import annotations

import asyncio

import pytest

from egis_opencode.api.run_lock import SessionRunLock, _pg_dsn

#: 现成 weknora 库（本机 docker；连不上即 skip PG 组）
_TEST_PG_DSN = "postgresql+asyncpg://weknora:weknora123@localhost:5432/weknora"


async def _pg_available() -> bool:
    try:
        import asyncpg

        conn = await asyncio.wait_for(
            asyncpg.connect(_TEST_PG_DSN.replace(
                "postgresql+asyncpg", "postgresql",
            )),
            timeout=2.0,
        )
    except Exception:
        return False
    await conn.close()
    return True


@pytest.fixture
async def pg_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """DB_CONNECTION_STR 指向 weknora PG；不可达则整组 skip。"""
    if not await _pg_available():
        pytest.skip("weknora PG unreachable (docker not running)")
    monkeypatch.setenv("DB_CONNECTION_STR", _TEST_PG_DSN)


# ── DSN 归一化 ─────────────────────────────────────


def test_pg_dsn_none_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DB_CONNECTION_STR", raising=False)
    assert _pg_dsn() is None


def test_pg_dsn_sqlite_degrades(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_CONNECTION_STR", "sqlite:///data/ark.db")
    assert _pg_dsn() is None


def test_pg_dsn_normalizes_sqlalchemy_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_CONNECTION_STR", _TEST_PG_DSN)
    assert _pg_dsn() == "postgresql://weknora:weknora123@localhost:5432/weknora"


# ── 降级形态（非 PG）─────────────────────────────


async def test_sqlite_degrades_to_noop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_CONNECTION_STR", "sqlite:///data/ark.db")
    lock = SessionRunLock()

    handle = await lock.acquire("sess-1")

    assert handle is not None
    assert handle.noop is True
    await handle.release()  # noop release 幂等无害
    await handle.release()


async def test_pg_unreachable_degrades_to_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    """fail-open：PG 不可达返回 noop（请求不因锁服务故障被拒）。"""
    monkeypatch.setenv(
        "DB_CONNECTION_STR",
        "postgresql+asyncpg://nobody:nopass@127.0.0.1:59999/nope",
    )
    lock = SessionRunLock(connect_timeout=0.5)

    handle = await lock.acquire("sess-1")

    assert handle is not None
    assert handle.noop is True


# ── PG 真锁互斥 ──────────────────────────────────


async def test_same_session_mutually_exclusive(pg_live):
    lock = SessionRunLock()

    handle = await lock.acquire("sess-lock-1")
    assert handle is not None and handle.noop is False

    try:
        second = await lock.acquire("sess-lock-1")
        assert second is None  # 同 session 抢不到
    finally:
        await handle.release()

    # 释放后可重新抢到
    again = await lock.acquire("sess-lock-1")
    assert again is not None
    await again.release()


async def test_different_sessions_do_not_interfere(pg_live):
    lock = SessionRunLock()

    h1 = await lock.acquire("sess-a")
    h2 = await lock.acquire("sess-b")
    assert h1 is not None
    assert h2 is not None  # 不同 session 各持各锁

    await h1.release()
    await h2.release()


async def test_cross_instance_mutual_exclusion(pg_live):
    """两个独立 SessionRunLock（模拟多副本进程）同 session 互斥。"""
    lock_a = SessionRunLock()
    lock_b = SessionRunLock()

    handle = await lock_a.acquire("sess-cross")
    assert handle is not None

    try:
        assert await lock_b.acquire("sess-cross") is None
    finally:
        await handle.release()

    assert await lock_b.acquire("sess-cross") is not None
    # 收尾：再释放一次由 lock_b 持有的
    handle_b = await lock_b.acquire("sess-cross")
    if handle_b is not None:
        await handle_b.release()


async def test_release_is_idempotent(pg_live):
    lock = SessionRunLock()
    handle = await lock.acquire("sess-rel")
    assert handle is not None

    await handle.release()
    await handle.release()  # 二次 release 不抛

    # 锁已释放，立即可再取
    handle2 = await lock.acquire("sess-rel")
    assert handle2 is not None
    await handle2.release()
