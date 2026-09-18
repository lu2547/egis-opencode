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


@router.get("/workspaces/binding")
async def get_workspace_binding(session_id: str) -> WorkspaceBindingResponse:
    """会话当前的工作目录绑定（无绑定返回空串 = 多租户模式）。"""
    stored = workspace_binding_store.get(session_id) or ""
    if not stored:
        return WorkspaceBindingResponse(workspace_root="")
    service = get_workspace_service()
    try:
        status = await service.local_status(stored)
    except WorkspacePathError:
        # 目录已失效：清掉脏绑定，按多租户返回
        workspace_binding_store.delete(session_id)
        return WorkspaceBindingResponse(workspace_root="")
    return WorkspaceBindingResponse(
        workspace_root=stored,
        status=ProjectStatusResponse(**status.__dict__),
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
