"""egis-opencode agents 根包。

ark discovery（``pkgutil.walk_packages``）从这里扫描所有声明了
``agent_id`` 的 ``BaseAgent`` 子类并自动注册；本包不显式导入
agent 模块，避免 discovery 前的副作用。
"""
