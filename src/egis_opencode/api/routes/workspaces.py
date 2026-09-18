"""GET/POST /api/coding/workspaces — 项目 / 工作目录绑定 / 文件浏览。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...sessions.workspace_binding import workspace_binding_store
from ...workspace import WorkspacePathError
from ...workspace.service import FileNode
from ..deps import get_workspace_service
from ..schemas import (
    CloneRequest,
    FileContentResponse,
    FileNodeResponse,
    FileTreeResponse,
    ProjectStatusResponse,
    WorkspaceBindRequest,
    WorkspaceBindingResponse,
)

router = APIRouter()


def _node_response(node: FileNode) -> FileNodeResponse:
    """FileNode dataclass → 响应模型（递归 children）。"""
    return FileNodeResponse(
        name=node.name,
        path=node.path,
        type=node.type,
        size=node.size,
        children=(
            [_node_response(child) for child in node.children]
            if node.children is not None else None
        ),
    )


@router.get("/workspaces")
async def list_workspaces(user_id: str) -> list[ProjectStatusResponse]:
    """用户 workspace 下的项目列表（含 git 状态）。"""
    service = get_workspace_service()
    projects = await service.list_projects(user_id)
    return [ProjectStatusResponse(**p.__dict__) for p in projects]


@router.post("/workspaces/clone")
async def clone_workspace(body: CloneRequest) -> ProjectStatusResponse:
    """克隆远程仓库到用户 workspace（同名目录已存在时报错）。"""
    service = get_workspace_service()
    try:
        status = await service.clone_project(
            body.user_id,
            body.repo_url,
            project_name=body.project_name,
            branch=body.branch,
        )
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ProjectStatusResponse(**status.__dict__)


@router.post("/workspaces/bind")
async def bind_workspace(body: WorkspaceBindRequest) -> WorkspaceBindingResponse:
    """校验并绑定/解绑会话工作目录（local:<绝对路径>）。"""
    service = get_workspace_service()
    if not body.workspace_root:
        if body.session_id:
            workspace_binding_store.delete(body.session_id)
        # 解绑 = 清除显式绑定，回落 .env 默认工作目录（而非直接
        # 多租户）—— 与后续 getWorkspaceBinding / chat 的解析结果
        # 保持一致，前端 UI 不会解绑后仍显示旧目录
        fallback = service.default_binding()
        if fallback:
            try:
                status = await service.local_status(fallback)
            except WorkspacePathError:  # pragma: no cover — 已校验
                status = None
            if status is not None:
                return WorkspaceBindingResponse(
                    workspace_root=fallback,
                    status=ProjectStatusResponse(**status.__dict__),
                    is_default=True,
                )
        return WorkspaceBindingResponse(workspace_root="")
    try:
        status = await service.local_status(body.workspace_root)
    except WorkspacePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if body.session_id:
        workspace_binding_store.set(body.session_id, body.workspace_root)
    return WorkspaceBindingResponse(
        workspace_root=body.workspace_root,
        status=ProjectStatusResponse(**status.__dict__),
    )


@router.get("/workspaces/default")
async def get_workspace_default() -> WorkspaceBindingResponse:
    """服务端默认工作目录（.env CODING_DEFAULT_WORKSPACE_MODE/DIR）。

    前端启动时优先取此端点：有默认则直接展示（is_default=True，
    chat 不携带 workspace_root，后端每轮解析，.env 变更立即跟随）；
    空串 = 未配置/无效，前端回落本地 localStorage 自动恢复。
    """
    service = get_workspace_service()
    raw = service.default_binding()
    if not raw:
        return WorkspaceBindingResponse(workspace_root="")
    try:
        status = await service.local_status(raw)
    except WorkspacePathError:  # pragma: no cover — default_binding 已校验
        return WorkspaceBindingResponse(workspace_root="")
    return WorkspaceBindingResponse(
        workspace_root=raw,
        status=ProjectStatusResponse(**status.__dict__),
        is_default=True,
    )


@router.get("/workspaces/binding")
async def get_workspace_binding(session_id: str) -> WorkspaceBindingResponse:
    """会话当前的工作目录绑定（显式绑定 > .env 默认 > 空串多租户）。"""
    service = get_workspace_service()
    stored = workspace_binding_store.get(session_id) or ""
    if stored:
        try:
            status = await service.local_status(stored)
        except WorkspacePathError:
            # 显式绑定的目录已失效：清掉脏绑定，继续走默认回落
            workspace_binding_store.delete(session_id)
        else:
            return WorkspaceBindingResponse(
                workspace_root=stored,
                status=ProjectStatusResponse(**status.__dict__),
            )
    # 无显式绑定（或已失效）：回落 .env 默认 —— UI 文件树/命令面板
    # 与 chat 实际锚定目录同源，避免“卡片显示多租户、工具落默认目录”
    default_raw = service.default_binding()
    if not default_raw:
        return WorkspaceBindingResponse(workspace_root="")
    try:
        status = await service.local_status(default_raw)
    except WorkspacePathError:  # pragma: no cover — default_binding 已校验
        return WorkspaceBindingResponse(workspace_root="")
    return WorkspaceBindingResponse(
        workspace_root=default_raw,
        status=ProjectStatusResponse(**status.__dict__),
        is_default=True,
    )


@router.get("/workspaces/tree")
async def workspace_tree(
    user_id: str,
    workspace_root: str = "",
    path: str = "",
    depth: int = Query(default=2, ge=1, le=4),
) -> FileTreeResponse:
    """工作目录文件树（限深；目录在前；truncated = 触达节点上限）。

    ``workspace_root`` 空串 = 多租户用户根；``local:<绝对路径>`` = 锁定目录。
    """
    service = get_workspace_service()
    try:
        nodes, truncated = await service.file_tree(
            user_id, workspace_root, path=path, depth=depth,
        )
    except WorkspacePathError as exc:
        # “目录不存在”是 404（资源消失）；其余（绑定串非法/越界）是 400
        status = 404 if "不存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))
    return FileTreeResponse(
        nodes=[_node_response(node) for node in nodes],
        truncated=truncated,
    )


@router.get("/workspaces/file")
async def workspace_file(
    user_id: str,
    workspace_root: str = "",
    path: str = "",
) -> FileContentResponse:
    """读工作目录内文件做预览（utf-8；超限截断；二进制占位）。"""
    if not path.strip():
        raise HTTPException(status_code=400, detail="path is required")
    service = get_workspace_service()
    try:
        content = service.read_file(user_id, workspace_root, path)
    except WorkspacePathError as exc:
        # “文件不存在”是 404（资源消失）；其余（绑定串非法/越界）是 400
        status = 404 if "不存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))
    return FileContentResponse(
        path=content.path,
        content=content.content,
        size=content.size,
        truncated=content.truncated,
        binary=content.binary,
    )


__all__ = ["router"]
