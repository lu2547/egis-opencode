"""WorkspaceService — 用户 workspace 内的项目管理（clone / 列表 / git 状态 / 文件浏览）。

项目目录约定：``<workspace_root>/<user_id>/<project_name>``。
git 操作通过 ``git`` CLI 在用户 workspace 内执行（子进程，带超时）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from .paths import WorkspacePathError, WorkspacePaths, safe_segment

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECONDS = 300.0

#: 文件树扫描时忽略的目录/文件名（噪音目录，无浏览价值）
_IGNORED_NAMES = {
    ".git", ".DS_Store", "node_modules", "__pycache__",
    ".venv", ".pytest_cache", ".ruff_cache",
}

#: 单次树扫描的节点总数上限（防大目录把响应撑爆）
_TREE_NODE_LIMIT = 2000

#: 文件预览的大小上限（字节）；超出截断返回
_PREVIEW_MAX_BYTES = 200_000


@dataclass(frozen=True)
class ProjectStatus:
    """单个项目目录的状态摘要。"""

    name: str
    path: str
    is_git_repo: bool
    branch: str = ""
    head_short: str = ""
    dirty_files: int = 0
    untracked_files: int = 0


@dataclass
class FileNode:
    """文件树节点（path 为相对工作目录根的 posix 路径）。

    目录节点 ``children``：``None`` = 未加载（前端懒加载标记），
    ``[]`` = 已加载但为空。最深层目录始终返回 None。
    """

    name: str
    path: str
    type: str  # "dir" | "file"
    size: int = 0
    children: list[FileNode] | None = None


@dataclass(frozen=True)
class FileContent:
    """文件预览内容（truncated = 超过预览上限被截断；binary = 二进制不可预览）。"""

    path: str
    content: str
    size: int
    truncated: bool
    binary: bool = False


class WorkspaceService:
    """用户 workspace 内的项目目录管理。"""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or settings.workspace_root_resolved

    @property
    def root(self) -> Path:
        return self._root

    def paths_for(self, user_id: str) -> WorkspacePaths:
        """某用户的路径解析器。"""
        return WorkspacePaths(root=self._root, user_id=user_id)

    # ── 工作目录绑定 ──────────────────────────────

    def resolve_binding(self, raw: str) -> Path:
        """把 workspace_root 绑定串解析为目录绝对路径（校验后返回）。

        当前仅支持 ``local:<绝对路径>``：本地目录直连（须存在且为目录，
        禁止 "/"；需 ``CODING_ALLOW_LOCAL_WORKSPACE``）。其余形态报
        ``WorkspacePathError``。空串合法（表示解绑，返回 None 语义由
        调用方处理——本方法不接空串）。
        """
        if not raw or ":" not in raw:
            raise WorkspacePathError(
                f"workspace_root 不合法: {raw!r}（应为 local:<绝对路径>）"
            )
        mode, _, value = raw.partition(":")
        if mode != "local":
            raise WorkspacePathError(
                f"不支持的 workspace_root 模式: {mode!r}（当前仅 local）"
            )
        if not settings.local_workspace_enabled:
            raise WorkspacePathError(
                "本地目录模式已禁用（CODING_ALLOW_LOCAL_WORKSPACE=false）"
            )
        path = Path(value.strip())
        if not path.is_absolute():
            raise WorkspacePathError(
                f"local 工作目录必须是绝对路径: {value!r}"
            )
        resolved = path.resolve(strict=False)
        if resolved == Path(resolved.root):
            raise WorkspacePathError("不允许把整个文件系统绑定为工作目录")
        if not resolved.is_dir():
            raise WorkspacePathError(f"目录不存在或不是目录: {resolved}")
        return resolved

    async def dir_status(self, directory: Path) -> ProjectStatus:
        """任意目录的状态摘要（git 信息；名称取目录名）。"""
        return await self._dir_status(directory, directory.name)

    async def local_status(self, raw_binding: str) -> ProjectStatus:
        """校验 local 绑定串并返回目录状态（不做会话持久化）。"""
        return await self.dir_status(self.resolve_binding(raw_binding))

    def default_binding(self) -> str:
        """服务端默认工作目录的绑定串（.env 驱动；未配置/无效返回空串）。

        前端未显式传 workspace_root 且会话无绑定时的兑底 —— 对齐
        opencode「启动即工作目录」的单机部署形态：本地/单人部署在
        .env 配 CODING_DEFAULT_WORKSPACE_MODE=local + DIR，前端无需
        绑定操作开箱即用。不持久化到会话：.env 修改重启后，未显式
        绑定的会话立即跟随新值。
        """
        if settings.default_workspace_mode != "local":
            return ""
        raw_dir = settings.default_workspace_dir
        if not raw_dir:
            logger.warning(
                "CODING_DEFAULT_WORKSPACE_MODE=local 但未配置 "
                "CODING_DEFAULT_WORKSPACE_DIR，回落多租户",
            )
            return ""
        # 相对路径按进程 CWD 解析（与 workspace_root_resolved 同约定）
        path = Path(raw_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        raw = f"local:{path}"
        try:
            self.resolve_binding(raw)
        except WorkspacePathError as exc:
            logger.warning("默认工作目录无效，回落多租户: %s", exc)
            return ""
        return raw

    # ── 文件浏览（目录树 + 预览） ────────────────

    def workspace_paths(self, user_id: str, workspace_root: str) -> WorkspacePaths:
        """按绑定串取路径解析器："" = 多租户用户根；``local:*`` = 锁定目录。"""
        if not workspace_root:
            return self.paths_for(user_id)
        return WorkspacePaths.anchored(self.resolve_binding(workspace_root))

    async def file_tree(
        self,
        user_id: str,
        workspace_root: str,
        path: str = "",
        depth: int = 2,
    ) -> tuple[list[FileNode], bool]:
        """列出 ``path`` 目录下的文件树（限深 + 节点上限）。

        返回 ``(nodes, truncated)``：目录在前、文件在后，各自字母序；
        最深层目录 ``children=None``（前端展开时再请求该层）。
        节点 path 一律相对 workspace 根（子目录请求也带父前缀，
        前端直接拿去 read_file/toggleDir，否则深层文件 404）。
        """
        paths = self.workspace_paths(user_id, workspace_root)
        target = paths.user_root if not path.strip() else paths.resolve(path)
        if not target.is_dir():
            raise WorkspacePathError(f"目录不存在或不是目录: {path or '.'}")
        depth = max(1, min(int(depth), 4))
        budget = [_TREE_NODE_LIMIT]
        prefix = path.strip().strip("/")
        nodes = self._scan_dir(target, prefix, depth, budget)
        return nodes, budget[0] <= 0

    def read_file(
        self, user_id: str, workspace_root: str, path: str,
    ) -> FileContent:
        """读工作目录内文件做预览（utf-8；超限截断；二进制返回占位标记）。"""
        paths = self.workspace_paths(user_id, workspace_root)
        target = paths.resolve(path)
        if not target.is_file():
            raise WorkspacePathError(f"文件不存在: {path}")
        size = target.stat().st_size
        limit = min(size, _PREVIEW_MAX_BYTES)
        with target.open("rb") as fh:
            head = fh.read(limit)
        if b"\x00" in head:
            # 二进制（docx/图片等）不是错误：返回占位标记，前端展示
            # “不支持预览”卡片而非报错（HTTP 400 只留给路径不合法）
            return FileContent(
                path=paths.relative_display(target),
                content="",
                size=size,
                truncated=False,
                binary=True,
            )
        return FileContent(
            path=paths.relative_display(target),
            content=head.decode("utf-8", "replace"),
            size=size,
            truncated=size > _PREVIEW_MAX_BYTES,
        )

    def _scan_dir(
        self, directory: Path, prefix: str, depth: int, budget: list[int],
    ) -> list[FileNode]:
        """扫描单层目录；depth>1 时目录递归，耗尽层返回 children=None。"""
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            logger.warning("workspace scan failed: %s", directory, exc_info=True)
            return []

        dirs, files = [], []
        for entry in entries:
            if entry.name in _IGNORED_NAMES:
                continue
            (dirs if entry.is_dir() else files).append(entry)

        # 排序：目录在前、文件在后，各自组内普通条目在前、dot 条目在后
        # （.claude/.opencode 等配置目录不抢占顶层视线，同 IDE 文件树惯例）
        def sort_key(item: Path) -> tuple[bool, str]:
            return (item.name.startswith("."), item.name.lower())

        nodes: list[FileNode] = []
        for entry in sorted(dirs, key=sort_key) + sorted(files, key=sort_key):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                children = (
                    self._scan_dir(entry, rel, depth - 1, budget)
                    if depth > 1 else None
                )
                nodes.append(FileNode(name=entry.name, path=rel, type="dir", children=children))
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                nodes.append(FileNode(name=entry.name, path=rel, type="file", size=size))
        return nodes

    # ── 项目管理 ────────────────────────────────────────

    async def clone_project(
        self,
        user_id: str,
        repo_url: str,
        *,
        project_name: str | None = None,
        branch: str | None = None,
    ) -> ProjectStatus:
        """把远程仓库 clone 到用户 workspace（已存在同名目录则报错）。"""
        if not repo_url or not repo_url.strip():
            raise WorkspacePathError("repo_url is required")
        name = safe_segment(project_name or _infer_project_name(repo_url))
        user_root = self.paths_for(user_id).ensure_user_root()
        target = user_root / name
        if target.exists():
            raise WorkspacePathError(
                f"project directory already exists: {name}"
            )

        cmd = ["git", "clone"]
        if branch:
            cmd += ["--branch", branch]
        cmd += ["--", repo_url, str(target)]

        await self._run_git(cmd, cwd=user_root)
        return await self.project_status(user_id, name)

    async def import_local_project(
        self, user_id: str, project_name: str,
    ) -> ProjectStatus:
        """把 workspace 内已存在的目录登记为项目（本地导入）。"""
        name = safe_segment(project_name)
        paths = self.paths_for(user_id)
        target = paths.resolve(name)
        if not target.is_dir():
            raise WorkspacePathError(f"project directory not found: {name}")
        return await self.project_status(user_id, name)

    async def list_projects(self, user_id: str) -> list[ProjectStatus]:
        """列出用户 workspace 下的一级子目录及其 git 状态。"""
        paths = self.paths_for(user_id)
        user_root = paths.user_root
        if not user_root.is_dir():
            return []
        projects = []
        for child in sorted(user_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                status = await self.project_status(user_id, child.name)
            except Exception:  # noqa: BLE001 — 单项目失败不影响列表
                logger.warning("project status failed: %s", child, exc_info=True)
                continue
            projects.append(status)
        return projects

    async def project_status(self, user_id: str, project_name: str) -> ProjectStatus:
        """读取单个项目目录的 git 状态（非 git 目录返回基础信息）。"""
        name = safe_segment(project_name)
        paths = self.paths_for(user_id)
        target = paths.resolve(name)
        if not target.is_dir():
            raise WorkspacePathError(f"project directory not found: {name}")
        return await self._dir_status(target, name)

    # ── 内部 ────────────────────────────────

    async def _dir_status(self, directory: Path, name: str) -> ProjectStatus:
        """目录状态内核：.git 存在时读分支/dirty 计数。"""
        if not (directory / ".git").exists():
            return ProjectStatus(
                name=name,
                path=str(directory),
                is_git_repo=False,
            )

        branch, head = await self._rev_parse(directory)
        dirty, untracked = await self._dirty_counts(directory)
        return ProjectStatus(
            name=name,
            path=str(directory),
            is_git_repo=True,
            branch=branch,
            head_short=head[:8],
            dirty_files=dirty,
            untracked_files=untracked,
        )

    # ── git 命令 ──────────────────────────────────────────

    async def _rev_parse(self, repo: Path) -> tuple[str, str]:
        out = await self._git_output(
            ["rev-parse", "--abbrev-ref", "HEAD"], repo,
        )
        branch = out.strip()
        head = await self._git_output(["rev-parse", "--short", "HEAD"], repo)
        return branch, head.strip()

    async def _dirty_counts(self, repo: Path) -> tuple[int, int]:
        out = await self._git_output(["status", "--porcelain"], repo)
        dirty = 0
        untracked = 0
        for line in out.splitlines():
            if not line.strip():
                continue
            if line.startswith("??"):
                untracked += 1
            else:
                dirty += 1
        return dirty, untracked

    async def _run_git(self, cmd: list[str], cwd: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_GIT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise WorkspacePathError(f"git command timed out: {cmd[1]}")
        if proc.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            raise WorkspacePathError(
                f"git command failed: {detail or proc.returncode}"
            )

    async def _git_output(self, args: list[str], repo: Path) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise WorkspacePathError(f"git command timed out: {args[0]}")
        if proc.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            raise WorkspacePathError(
                f"git command failed: {detail or proc.returncode}"
            )
        return (stdout or b"").decode("utf-8", "replace")


def _infer_project_name(repo_url: str) -> str:
    """从 git URL 推断项目名（去 .git 后缀）。"""
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    return safe_segment(tail or "project")
