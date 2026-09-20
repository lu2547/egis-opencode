"""CodingBaseAgent — coding 底座基类（能力下沉到 core，agents/ 只放业务壳）。

底座沉淀通用编码能力（工具集 / 权限 / run 钮子 / 标题生成），业务
agent 只声明身份（agent_id / prompt / 内置 commands / 私有 skills）：

- ``agents/coding``   薄壳：CodingAgent / CodingPlanAgent（build/plan）
- ``agents/wiki-agent`` 业务 agent：WikiAgent（LLM Wiki 维护）

不声明 ``agent_id`` —— ark discovery 跳过中间抽象类（仅为代码共享
存在），注册的永远是子类。

能力分工（与 workspace/commands.py 的边界）：
- **commands**（用户显式 ``/xxx`` 触发）：``agents/<agent>/commands/*.md``
  内置 + workspace ``.opencode/commands``（expand_command 扔 user prompt）
- **skills**（模型自主触发）：ark SkillLoader 挂 agent 私有 skills +
  workspace ``.opencode/skills`` / ``.claude/skills``；dynamic 模式下
  ark 自动注册 ``read_skill`` 工具，正文经 ``<active_skill>`` 注入
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from ark_agentic import BaseAgent
from ark_agentic.core.llm.factory import create_chat_model_from_env
from ark_agentic.core.llm.sampling import SamplingConfig
from ark_agentic.core.runtime.callbacks import RunnerCallbacks
from ark_agentic.core.runtime._runner_types import RunnerConfig
from ark_agentic.core.skills.base import SkillConfig
from ark_agentic.core.skills.loader import SkillLoader
from ark_agentic.core.skills.matcher import SkillMatcher
from ark_agentic.core.types import SkillLoadMode

from ..config import settings
from ..permissions import PermissionGuard, ruleset_for_mode
from ..permissions.service import permission_service
from ..sessions.title import TitleGenerator, title_store
from .doom_loop import DoomLoopGuard
from .tools import CodingMode, create_coding_tools

logger = logging.getLogger(__name__)

_BASE_PROMPTS_DIR = Path(__file__).parent / "prompts"

#: workspace 下可挂 skills 的相对目录（agent 私有 skills 之后，优先级递减）
_WORKSPACE_SKILL_DIRS = (".opencode/skills", ".claude/skills")

#: 一层子目录扫描时跳过的噪音目录
_SKILL_SKIP_SUBDIRS = {"node_modules", ".truncation"}


def base_prompt(name: str) -> str:
    """读底座 prompts（core/prompts/）—— 子类声明的入口。"""
    return (_BASE_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_mode_from_env() -> SkillLoadMode:
    raw = os.getenv("SKILL_LOAD_MODE", "dynamic").strip().lower()
    return SkillLoadMode.dynamic if raw == "dynamic" else SkillLoadMode.full


class CodingBaseAgent(BaseAgent):
    """编码底座：工具集 / 权限 / run 钮子 / 标题生成 / skills 挂载。

    子类最小声明（参照 agents/wiki-agent）::

        class WikiAgent(CodingBaseAgent):
            agent_id = "wiki"
            agent_name = "…"
            agent_description = "…"
            system_protocol = base_prompt("system.md")  # 或自带 prompts/
    """

    #: 权限模式（build=全量工具 / plan=只读）→ 工具集与权限预设
    mode: ClassVar[CodingMode] = "build"
    #: ReAct 循环上限。opencode 主循环无轮数限制（模型自然结束/用户 abort），
    #: 此处仅作失控安全阀 —— 取值需覆盖长任务的探索+执行全程
    #: （实测中型任务纯探索期即可耗 60+ 轮），不应成为正常任务的边界。
    max_turns: ClassVar[int] = 200
    enable_subtasks: ClassVar[bool] = False

    # ── 内置命令（slash 面板）──────────────────────────

    @property
    def command_dir(self) -> Path:
        """agent 内置命令目录：本模块所在 agent 目录下的 ``commands/``。

        沿用 ark ``skills_dir`` 约定推导（skills_dir = agent 目录/skills），
        目录不存在时自然无效（discover_commands 只扫存在的目录）。
        """
        return self.skills_dir.parent / "commands"

    # ── Skills（复用 ark：私有目录 + workspace 动态挂载）──────────

    def _init_skill_subsystem(self) -> None:
        """ark 默认只扫 agent 私有 skills/；底座额外支持 workspace 挂载。

        workspace 目录在 chat 入口按会话工作目录 ``reload_workspace_skills``
        动态合入（多租户/本地目录会话各自生效）；read_skill 工具持有
        同一 loader 引用，reload 后立即可见。
        """
        self._skill_config = SkillConfig(
            skill_directories=self._skill_directories(),
            agent_id=self.agent_id,
            enable_eligibility_check=True,
            load_mode=_load_mode_from_env(),
            description_max_chars=self.skill_description_max_chars,
        )
        self.skill_loader: SkillLoader | None = SkillLoader(self._skill_config)
        try:
            self.skill_loader.load_from_directories()
            all_skills = self.skill_loader.list_skills(include_disabled=True)
            enabled_count = sum(1 for s in all_skills if s.enabled)
            logger.info(
                "Loaded %d skills (%d enabled) for agent '%s'",
                len(all_skills),
                enabled_count,
                self.agent_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load skills for agent '%s': %s", self.agent_id, exc,
            )
        self.skill_matcher: SkillMatcher | None = SkillMatcher(self.skill_loader)
        self._skill_dirs_loaded: tuple[str, ...] = ()

    def _skill_directories(self, workspace_root: Path | None = None) -> list[str]:
        """agent 私有 skills 在前（优先级最高），workspace 目录在后。

        部署形态是 workspace 根 + 一层项目子目录（如
        ``<workspace>/llm-wiki/.claude/skills``），故除根外还扫一层
        非隐藏子目录 —— 同名 skill 靠前的目录优先（ark loader 约定）。
        """
        dirs: list[str] = []
        if self.skills_dir.is_dir():
            dirs.append(str(self.skills_dir))
        if workspace_root is not None and workspace_root.is_dir():
            bases = [workspace_root] + sorted(
                d for d in workspace_root.iterdir()
                if d.is_dir()
                and not d.name.startswith(".")
                and d.name not in _SKILL_SKIP_SUBDIRS
            )
            for base in bases:
                for rel in _WORKSPACE_SKILL_DIRS:
                    candidate = base / rel
                    if candidate.is_dir():
                        dirs.append(str(candidate))
        return dirs

    def reload_workspace_skills(self, workspace_root: Path | None) -> None:
        """按会话工作目录重挂 skills（目录集合不变时零开销跳过）。

        chat 入口解析出 ``workspace:root`` 后调用 —— 本地目录会话读
        该目录的 ``.opencode/skills`` / ``.claude/skills``，多租户读
        用户 workspace 根；目录集合相同（常态：同一会话连续提问）
        直接返回，不重扫。

        兼容 ``_construct`` 显式注入路径（skills 子系统未初始化时
        ``_skill_dirs_loaded`` / ``_skill_config`` 可能缺席，getattr 兑底）。
        """
        loader = self.skill_loader
        if loader is None:
            return
        dirs = self._skill_directories(workspace_root)
        key = tuple(dirs)
        if key == getattr(self, "_skill_dirs_loaded", ()):
            return
        self._skill_dirs_loaded = key
        config = getattr(self, "_skill_config", None)
        if config is not None:
            config.skill_directories = list(dirs)
        try:
            loader.load_from_directories(dirs)
            logger.info(
                "Reloaded %d skills for agent '%s' (dirs=%s)",
                len(loader.list_skills()),
                self.agent_id,
                dirs,
            )
        except Exception as exc:
            logger.warning(
                "Failed to reload workspace skills for agent '%s': %s",
                self.agent_id, exc,
            )

    # ── 工具 / 执行 / 回调（原 CodingAgent 全量平移）────────────

    def build_llm(self):
        """MAIN LLM 输出预算与采样参数对齐 opencode / Qwen 官方推荐。

        ark ``SamplingConfig`` 默认 ``max_tokens=4096`` + 低温 0.1：
        - 预算：thinking + 正文 + tool call arguments 共享，写长脚本极易
          ``finish_reason="length"``，ark 将其按 run 终止处理并丢弃当轮
          tool calls（前端“轮次上限”假象 + 写入半截脚本）；
        - 采样：qwen3.5 thinking 官方推荐 temperature=0.6 / top_p=0.95，
          0.1 属复读退化高危区（官方明确 “DO NOT use greedy”）；
          presence_penalty 官方不建议设，0.6 会抗乱长输出。
        """
        return create_chat_model_from_env(
            sampling=SamplingConfig.for_chat(
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                max_tokens=settings.max_output_tokens,
            ),
        )

    def build_tools(self):
        return create_coding_tools(self, mode=self.mode)

    def build_runner_config(self) -> RunnerConfig:
        """执行钮子定制：工具层超时对齐长耗时工具。

        框架默认 ``tool_timeout=30`` 会把 task（并行读多个 docx）整体
        cancel 掉 —— 且取消发生在事件发完后，前端子任务卡片永久转圈。
        600s 覆盖并行子任务全程；bash 自身另有 timeout 参数控制长命令，
        此处仅作外层安全阀。
        """
        config = super().build_runner_config()
        return RunnerConfig(
            **{
                **config.__dict__,
                "tool_timeout": 600.0,
            }
        )

    def build_callbacks(self) -> RunnerCallbacks | None:
        guard = PermissionGuard(
            agent=self,
            service=permission_service,
            base_ruleset=ruleset_for_mode(self.mode),
        )
        doom = DoomLoopGuard(agent=self)

        async def _cleanup_run(ctx: Any, **kwargs: Any) -> None:
            """after_agent：run 结束清理授权缓存与 doom 观察态。"""
            guard.discard_run(ctx.run_id)
            doom.discard_run(ctx.run_id)

        return RunnerCallbacks(
            before_tool=[guard, doom],
            after_agent=[_cleanup_run, TitleGenerator(title_store)],
        )


__all__ = ["CodingBaseAgent", "base_prompt"]
