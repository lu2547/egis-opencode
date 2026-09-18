"""会话工作目录绑定存储 — session_id → workspace_root 串。

- 绑定串形态：``"local:<绝对路径>"``（本地目录直连）；
  空串/删除 = 回落多租户模式（``<workspace_root>/<user_id>``）。
- 持久化：``CONFIG_DIR/coding/session_workspaces.json``（原子写，进程单例）。
- chat 端点每次请求据此把 ``workspace:root`` 注入 input_context，
  文件/bash 工具即锚定到该目录（见 agents/coding/tools/base.py）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceBindingStore:
    """session_id → workspace_root 绑定串的 JSON 文件存储（原子写）。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] | None = None

    def get(self, session_id: str) -> str | None:
        return self._load().get(session_id)

    def set(self, session_id: str, workspace_root: str) -> None:
        if not workspace_root:
            self.delete(session_id)
            return
        data = self._load()
        if data.get(session_id) == workspace_root:
            return
        data[session_id] = workspace_root
        self._save(data)

    def delete(self, session_id: str) -> None:
        data = self._load()
        if session_id in data:
            del data[session_id]
            self._save(data)

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
            logger.warning(
                "workspace bindings unreadable: %s", self._path, exc_info=True,
            )
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
            logger.warning(
                "workspace bindings save failed: %s", self._path, exc_info=True,
            )


def default_workspace_binding_store() -> WorkspaceBindingStore:
    """``CONFIG_DIR/coding/session_workspaces.json``（与 titles.json 同级）。"""
    from ark_agentic.core.paths import get_agent_config_dir

    return WorkspaceBindingStore(
        get_agent_config_dir("coding") / "session_workspaces.json",
    )


#: 进程级单例 — chat 端点（写入）与 REST binding 查询（读取）共享。
workspace_binding_store = default_workspace_binding_store()


__all__ = [
    "WorkspaceBindingStore",
    "default_workspace_binding_store",
    "workspace_binding_store",
]
