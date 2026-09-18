"""POST /api/coding/chat — 自建 AG-UI SSE 端点。

与 ark ``plugins/api/chat.py`` 的差异（为什么不能复用）：
- run 纳入 ``RunRegistry``（session 级并发防抖 + abort 令牌）
- run_agent 协程结束即注销，abort 后 emit_failed("aborted")
- 其余结构（resolve_session → bus → run_agent → SSE 循环）与 ark 对齐
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ark_agentic.core.runtime import ConcurrentSessionUpdate, RunResult
from ark_agentic.core.stream.event_bus import StreamEventBus
from ark_agentic.core.stream.events import AgentStreamEvent
from ark_agentic.core.stream.output_formatter import create_formatter
from ark_agentic.core.types import RunOptions, ToolResultType

from ..config import settings
from ..permissions import PermissionBridge, attach_bridge, detach_bridge
from ..questions import question_service
from ..sessions.workspace_binding import workspace_binding_store
from ..workspace import WorkspacePathError, expand_command
from ..workspace.commands import _read_agents_md
from ..workspace.service import WorkspaceService
from .deps import get_agent
from .run_lock import session_run_lock
from .runs import run_registry
from .schemas import AbortRequest, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=None)
async def chat(request: ChatRequest, http_request: Request):
    """对话端点：流式（AG-UI SSE）与非流式。"""
    agent = get_agent(http_request, request.agent_id)

    user_id = request.user_id
    if run_registry.is_running(request.session_id or ""):
        raise HTTPException(
            status_code=409,
            detail=f"session {request.session_id!r} already has a running request",
        )

    input_context: dict[str, Any] = {}
    if request.context:
        for k, v in request.context.items():
            input_context[f"user:{k}" if ":" not in k else k] = v
    input_context["user:id"] = user_id

    session_id = await agent.resolve_session(request.session_id, user_id)

    # 跨进程会话互斥（PG advisory lock；非 PG/不可达时降级 noop，
    # 仅剩 RunRegistry 进程内防抖 + ark 乐观锁兑底）。前置到模型调用前，
    # 拿不到锁直接 409 —— 与上方 is_running 防抖同语义。release 在
    # run 真正结束（run_agent finally / 非流式 finally）才执行
    run_lock_handle = await session_run_lock.acquire(session_id)
    if run_lock_handle is None:
        raise HTTPException(
            status_code=409,
            detail=f"session {session_id!r} already has a running request",
        )

    # 工作目录绑定：workspace_root 三态（None=沿用 / ""=解绑 / local:=锚定）。
    # 解析出目录后注入 input_context["workspace:root"]，文件/bash 工具
    # 钩在该目录（见 core/tools/base.py）；slash 命令与 workspace skills
    # 也读该目录。
    anchored_root = _resolve_session_workspace(
        session_id, request.workspace_root,
    )
    if anchored_root is not None:
        input_context["workspace:root"] = str(anchored_root)

    # 工作目录解析后：重挂 workspace skills（.opencode/skills 等，
    # 目录集合不变时零开销；ark read_skill 工具即刻可见）
    command_root = anchored_root or WorkspaceService().paths_for(
        user_id,
    ).ensure_user_root()
    if hasattr(agent, "reload_workspace_skills"):
        agent.reload_workspace_skills(command_root)

    # slash 命令展开：agent 内置 commands 优先，workspace 自定义在后
    message = expand_command(
        request.message, command_root,
        agent_commands=getattr(agent, "command_dir", None),
    ) or request.message

    # AGENTS.md 项目规范：走 ark 内建 ``user:`` state → 系统提示 context 通道
    # （``merge_input_context`` 合入 session.state，每次 run
    # ``_build_system_prompt`` 渲染 —— 等价 opencode instruction 机制，
    # 不污染落盘消息）。关键：每次 run 显式带键 —— merge 只 overwrite
    # 不清除，缺键会让旧目录的规范残留到解绑后的会话；
    # 无绑定/无文件时注入空串覆盖旧值。多会话隔离由 state 按会话存天然成立
    input_context["user:agents_md"] = _read_agents_md(command_root)

    run_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    # question 工具的交互通道标识（temp: 前缀 → ark 落盘时自动剥离）：
    # 工具经 executor 注入的 system:event_handler 发事件，
    # REST 应答按 session_id 索引 pending、abort 时按 run_id 清理
    input_context["temp:session_id"] = session_id
    input_context["temp:run_id"] = run_id

    if not request.stream:
        try:
            result = await agent.run(
                session_id=session_id,
                user_input=message,
                user_id=user_id,
                input_context=input_context,
                stream=False,
            )
        except ConcurrentSessionUpdate as exc:
            raise HTTPException(
                status_code=409,
                detail=f"session {exc.session_id!r} is busy, please retry",
            )
        finally:
            await run_lock_handle.release()
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            response=result.response.content or "",
            tool_calls=_tool_calls_payload(result),
            turns=result.turns,
        )

    queue: asyncio.Queue[AgentStreamEvent] = asyncio.Queue()
    done_event = asyncio.Event()
    bus = StreamEventBus(run_id=run_id, session_id=session_id, queue=queue)
    formatter = create_formatter(request.protocol, agent_id=agent.agent_id)
    # 权限审批通道：guard（before_tool hook 拿不到 handler）经 input_context
    # 取回 bridge → 向本 SSE 流发 permission_request 等事件。token 为纯
    # str（session_id）：ark 把 input_context 写进 user 消息 metadata 落盘，
    # 不可序列化对象会炸 transcript；bridge 对象存进程级注册表。
    bridge = PermissionBridge(bus)
    input_context["temp:permission_bridge"] = attach_bridge(session_id, bridge)

    async def run_agent() -> None:
        task = asyncio.current_task()
        run_registry.register(session_id, task)  # type: ignore[arg-type]
        bus.emit_created("收到您的消息，正在处理中…")
        try:
            result = await agent.run(
                session_id=session_id,
                user_input=message,
                user_id=user_id,
                input_context=input_context,
                stream=True,
                run_options=RunOptions(sampling_override=None),
                handler=bus,
            )
            bus.emit_completed(
                message=result.response.content or "",
                tool_calls=_tool_calls_payload(result) or None,
                turns=result.turns,
                card_description=_collect_card_description(result),
                context=request.context or None,
                outcome=result.outcome,
                suggestions=result.suggestions,
            )
        except asyncio.CancelledError:
            bus.emit_failed("run aborted")
            raise
        except Exception as exc:
            logger.exception("Agent run error: %s", exc)
            bus.emit_failed(str(exc))
        finally:
            if task is not None:
                run_registry.unregister(session_id, task)
            detach_bridge(session_id)
            await run_lock_handle.release()
            # question 工具的挂起提问兑底清理（abort 时 wait_answer 被取消，
            # 残留 pending 在此落定 —— 正常路径已由 timeout/respond 处理）
            question_service.discard_run(run_id)
            # 收尾窗口：等 title 等尾部事件落帧（drained 信号，带超时上限
            # 防慢 LLM 拖住断流）。abort（CancelledError）时也必须 set
            # done_event —— 否则 event_stream 循环永远等不到退出条件。
            try:
                await asyncio.wait_for(
                    bridge.drained.wait(),
                    timeout=settings.title_sse_grace_seconds,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "deferred events drain timeout: %s", session_id,
                )
            except asyncio.CancelledError:
                done_event.set()
                raise
            done_event.set()

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(run_agent())
        try:
            while True:
                if done_event.is_set() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    sse_line = formatter.format(event)
                    if sse_line is not None:
                        yield sse_line
                except asyncio.TimeoutError:
                    continue
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/abort")
async def abort(body: AbortRequest) -> dict[str, Any]:
    """取消进行中的 run（前端 fetch abort 的服务端对位）。"""
    cancelled = run_registry.cancel(body.session_id)
    return {"session_id": body.session_id, "aborted": cancelled}


def _tool_calls_payload(result: RunResult) -> list[dict[str, Any]]:
    return [
        {"name": tc.name, "arguments": tc.arguments}
        for tc in (result.tool_calls or [])
    ]


def _resolve_session_workspace(
    session_id: str, workspace_root: str | None,
) -> Path | None:
    """解析会话的有效工作目录（绑定串 → 目录），并同步持久化存储。

    解析优先级：显式传参 > 会话已存绑定 > .env 默认（
    CODING_DEFAULT_WORKSPACE_MODE/DIR，前端不传时的服务端兑底）。
    返回 None 表示多租户模式（不注入 workspace:root）。无效绑定
    （目录被删/配置关闭）不中断对话：清除存储并回落多租户。
    """
    service = WorkspaceService()
    if workspace_root is not None:
        if workspace_root:
            try:
                anchored = service.resolve_binding(workspace_root)
            except WorkspacePathError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            workspace_binding_store.set(session_id, workspace_root)
            return anchored
        workspace_binding_store.delete(session_id)
        return _default_workspace_or_none(service)

    stored = workspace_binding_store.get(session_id)
    if not stored:
        # 前端未传且会话无显式绑定：回落 .env 默认工作目录。
        # 不持久化 —— 配置变更重启后未显式绑定的会话自动跟随
        return _default_workspace_or_none(service)
    try:
        return service.resolve_binding(stored)
    except WorkspacePathError:
        logger.warning(
            "stored workspace binding invalid, fallback: %s", stored,
        )
        workspace_binding_store.delete(session_id)
        return _default_workspace_or_none(service)


def _default_workspace_or_none(service: WorkspaceService) -> Path | None:
    """.env 默认工作目录（有效返回目录；未配置/无效返回 None=多租户）。"""
    default_raw = service.default_binding()
    if not default_raw:
        return None
    try:
        return service.resolve_binding(default_raw)
    except WorkspacePathError:  # pragma: no cover — default_binding 已校验
        logger.warning("default workspace invalid: %r", default_raw)
        return None


def _collect_card_description(result: RunResult) -> list[dict[str, Any]]:
    """messages_snapshot 的 module 列表（A2UI 卡片 digest + 最终文本）。"""
    modules = [
        {"module_type": "text", "module_desc": r.llm_digest}
        for r in result.tool_results
        if r.result_type == ToolResultType.A2UI
    ]
    final_text = result.response.content
    if final_text:
        modules.append({"module_type": "text", "module_desc": final_text})
    return modules


__all__ = ["abort", "chat", "router"]
