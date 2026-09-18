"""api 集成测试装配 — 轻量 FastAPI + 真 CodingAgent + MockChatModel。

不经 Bootstrap lifespan：手动 ``app.state.ctx = SimpleNamespace(agent_registry=...)``
（与 deps.py 的请求期解析对齐）；callbacks 复刻 CodingAgent.build_callbacks，
但 TitleStore 落 tmp（不触碰 CONFIG_DIR 全局单例）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from ark_agentic.core.runtime.callbacks import RunnerCallbacks
from ark_agentic.core.runtime.registry import AgentRegistry
from ark_agentic.core.session import SessionManager

from egis_opencode.agents.coding.agent import CodingAgent, CodingPlanAgent
from egis_opencode.core.tools import create_coding_tools
from egis_opencode.api.plugin import CodingPlugin
from egis_opencode.permissions.guard import PermissionGuard
from egis_opencode.permissions.presets import build_ruleset, plan_ruleset
from egis_opencode.permissions.rules import Rule
from egis_opencode.permissions.service import permission_service
from egis_opencode.sessions.title import TitleGenerator, TitleStore
from egis_opencode.sessions.workspace_binding import workspace_binding_store

from tests.helpers import MockChatModel

DEFAULT_USER = "alice"


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def titles_path(tmp_path: Path) -> Path:
    return tmp_path / "titles.json"


@pytest.fixture
def binding_path(tmp_path: Path) -> Path:
    """全局 binding 单例重定向到 tmp（chat/routes 共享同一实例）。"""
    path = tmp_path / "session_workspaces.json"
    old_path = workspace_binding_store._path
    old_cache = workspace_binding_store._cache
    workspace_binding_store._path = path
    workspace_binding_store._cache = None
    yield path
    workspace_binding_store._path = old_path
    workspace_binding_store._cache = old_cache


@pytest.fixture
def make_app(
    tmp_path: Path,
    sessions_dir: Path,
    titles_path: Path,
    binding_path: Path,
    ws_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[FastAPI, MockChatModel]]:
    """测试 app 工厂：返回 (app, llm)；llm.responses 可后续追加脚本。

    依赖 ``ws_root``：保证 settings.workspace_root 已重定向到 tmp
    后再构造文件工具（工具构造时读 settings，顺序不能颠倒）。
    """

    def _make(
        responses: list[AIMessage] | None = None,
        *,
        register_plan: bool = False,
        extra_rules: list[Rule] | None = None,
    ) -> tuple[FastAPI, MockChatModel]:
        """extra_rules：build 模式预设后追加的收紧规则
        （默认预设全放行，ask 流程测试注入 ``write=ask`` 等）。"""
        llm = MockChatModel(responses=responses)
        # 缩短审批超时：防测试意外挂起 120s（应答都由轮询 responder 驱动）
        permission_service._timeout = 15

        registry = AgentRegistry()
        agents: dict[str, CodingAgent] = {}
        for cls, ruleset in (
            (CodingAgent, build_ruleset() + (extra_rules or [])),
            (CodingPlanAgent, plan_ruleset()),
        ):
            if cls is CodingPlanAgent and not register_plan:
                continue
            agent = cls._construct(
                llm=llm,
                session_manager=SessionManager(
                    sessions_dir, agent_id=cls.agent_id,
                ),
                agent_id=cls.agent_id,
            )
            for tool in create_coding_tools(agent, mode=cls.mode):
                agent.tool_registry.register(tool)
            guard = PermissionGuard(
                agent=agent,
                service=permission_service,
                base_ruleset=ruleset,
            )

            async def _cleanup_run(ctx: Any, **kwargs: Any) -> None:
                guard.discard_run(ctx.run_id)

            agent._callbacks = RunnerCallbacks(
                before_tool=[guard],
                after_agent=[_cleanup_run, TitleGenerator(TitleStore(titles_path))],
            )
            registry.register(cls.agent_id, agent)
            agents[cls.agent_id] = agent

        app = FastAPI()
        CodingPlugin().install_routes(app)
        app.state.ctx = SimpleNamespace(agent_registry=registry)
        # sessions 路由读全局 title_store → 指到 tmp
        monkeypatch.setattr(
            "egis_opencode.api.routes.sessions.title_store",
            TitleStore(titles_path),
        )
        return app, llm

    return _make


def chat_payload(
    message: str = "hi",
    *,
    user_id: str = DEFAULT_USER,
    session_id: str | None = None,
    stream: bool = True,
    agent_id: str = "coding",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "message": message,
        "stream": stream,
        "agent_id": agent_id,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


__all__ = ["chat_payload", "make_app", "titles_path"]
