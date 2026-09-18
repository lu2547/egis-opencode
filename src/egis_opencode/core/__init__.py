"""core — coding 底座（工具集 / 权限 / prompts / skills 挂载）。

agents/ 下只放业务壳（身份声明 + 内置 commands / 私有 skills）；
不声明 agent_id 的底座基类不会被 ark discovery 注册。
"""

from .agent import CodingBaseAgent, base_prompt

__all__ = ["CodingBaseAgent", "base_prompt"]
