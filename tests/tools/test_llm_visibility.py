"""LLM 视角的工具输出回归测试（幻觉根因防护）。

此前所有测试只断言 ``AgentToolResult.content``，从未断言过模型真正
看到什么。ark 的 ``serialize_messages_for_llm`` 取 ``tr.llm_digest``
作为 tool 消息 content —— 显式传 digest 时 content 全文被旁路。

历史 bug：egis coding 工具给 read/grep/glob/list 传了"一行摘要"式
digest，导致 21K 字符的文档全文从未进入模型上下文，模型只能靠训练
记忆编造文档内容（知识污染型幻觉）。本组测试锁死"模型可见性"契约：

1. read/grep/glob/list：不传 digest → LLM 看到 content 全文
2. 折叠态：compaction safe_digest → 截断为 500 字符摘要（原设计意图）
3. bash：显式 digest 携带完整 stdout/stderr（JSON content 转义不可读）
4. write/edit：短摘要（模型只需确认，opencode 语义）
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ark_agentic.core.runtime._runner_helpers import serialize_messages_for_llm
from ark_agentic.core.session.compaction import safe_digest
from ark_agentic.core.types import AgentMessage, MessageRole, SessionEntry

from egis_opencode.agents.coding.tools.bash import BashTool
from egis_opencode.agents.coding.tools.files import ListTool, ReadTool, WriteTool
from egis_opencode.agents.coding.tools.search import GlobTool, GrepTool
from egis_opencode.events import TOOL_DIGEST

from ..helpers import RecordingHandler

# 测试样例复用 test_files 的假件风格


@pytest.fixture
def tool() -> BashTool:
    return BashTool(agent_id="coding")


def _call(name: str, **arguments):
    from unittest.mock import MagicMock

    call = MagicMock()
    call.id = f"tc_{name}"
    call.name = name
    call.arguments = arguments
    return call


def _ctx(recorder: RecordingHandler | None = None) -> dict:
    ctx: dict = {"user:id": "alice"}
    if recorder is not None:
        ctx["system:event_handler"] = recorder
    return ctx


def _llm_messages(*tool_results) -> list[dict]:
    """走 ark 真实序列化路径：SessionEntry.messages → LLM 消息列表。"""
    session = SessionEntry(
        session_id="s1",
        messages=[
            AgentMessage(role=MessageRole.USER, content="task"),
            AgentMessage(role=MessageRole.TOOL, tool_results=list(tool_results)),
        ],
    )
    return serialize_messages_for_llm(session, system_prompt="sys")


def _llm_tool_content(messages: list[dict], tool_call_id: str) -> str:
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
            return str(msg["content"])
    raise AssertionError(f"tool message not found: {tool_call_id}")


# ── read：文档全文必须进 LLM 视角 ─────────────────────────


async def test_read_digest_event_carries_full_content(ws_user_root, recorder):
    """tool_digest 事件携带完整内容（对齐 opencode metadata.display）。

    UI 卡片与模型看到同一份全文，消除“只读了 8 行”的误导；
    流式事件与历史回放同构。
    """
    doc = "\n".join(f"line-{i}" for i in range(1, 31))
    (ws_user_root / "full.md").write_text(doc, encoding="utf-8")

    await ReadTool().execute(
        _call("read", path="full.md"), _ctx(recorder),
    )
    digests = recorder.of_type(TOOL_DIGEST)
    assert digests, "成功路径必须发射 tool_digest 事件"
    payload = digests[-1]

    assert payload["file_text"].count("\n") == 29  # 30 行全文
    assert payload["line_start"] == 1
    assert payload["line_end"] == 30
    assert payload["total_lines"] == 30
    assert payload["content_truncated"] is False
    # 预览仍保留（卡片默认折叠态展示）
    assert len(payload["results_preview"]) == 8


async def test_read_full_content_visible_to_llm(ws_user_root):
    """read 不传 digest → serialize 后模型看到带行号的全文（幻觉根因回归）。"""
    doc = "\n".join(f"line-{i}: AgentScope Wiki Agent 架构内容" for i in range(1, 21))
    (ws_user_root / "arch.md").write_text(doc, encoding="utf-8")

    result = await ReadTool().execute(_call("read", path="arch.md"), _ctx())
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)

    assert "line-20: AgentScope Wiki Agent 架构内容" in content
    assert "<content>" in content
    assert "(End of file - total 20 lines)" in content


async def test_read_compaction_folds_to_digest(ws_user_root):
    """折叠态：read 无显式 digest → safe_digest 截断 content 为 500 字符。

    这正是原设计意图：正常轮全文，长会话 compaction 后摘要。
    """
    doc = "\n".join(f"line-{i}_" + "x" * 60 for i in range(1, 40))
    (ws_user_root / "long.md").write_text(doc, encoding="utf-8")

    result = await ReadTool().execute(_call("read", path="long.md"), _ctx())
    assert result._llm_digest is None  # 未显式传 digest
    folded = safe_digest(result)
    assert len(folded) <= 500 + len("...")  # ark 截断上限


async def test_read_docx_text_visible_to_llm(monkeypatch, ws_user_root):
    """docx 提取文本（图片占位符化后）必须进 LLM 视角。"""
    from types import SimpleNamespace

    fake_mammoth = SimpleNamespace(
        convert_to_markdown=lambda fh: SimpleNamespace(
            value=(
                "# Wiki Agent 架构\n\nTask Orchestrator 编排细节。\n\n"
                "![](data:image/png;base64," + "A" * 200_000 + ")\n\n"
                "Agent Runtime 分层说明。\n"
            ),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "mammoth", fake_mammoth)

    (ws_user_root / "wiki.docx").write_bytes(b"placeholder")
    result = await ReadTool().execute(_call("read", path="wiki.docx"), _ctx())
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)

    assert "Task Orchestrator 编排细节" in content
    assert "Agent Runtime 分层说明" in content
    assert "base64" not in content
    assert "[图片内容已省略]" in content


# ── grep / glob：命中内容必须进 LLM 视角 ─────────────────


async def test_grep_hits_visible_to_llm(ws_user_root):
    (ws_user_root / "a.py").write_text(
        "alpha = 1\nbeta = 2\ngamma_hit = 'target'\n", encoding="utf-8",
    )
    result = await GrepTool().execute(
        _call("grep", pattern="target"), _ctx(),
    )
    assert result._llm_digest is None
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "gamma_hit" in content
    assert "target" in content


async def test_glob_matches_visible_to_llm(ws_user_root):
    (ws_user_root / "src").mkdir()
    (ws_user_root / "src" / "a.py").write_text("", encoding="utf-8")
    (ws_user_root / "src" / "b.py").write_text("", encoding="utf-8")
    result = await GlobTool().execute(_call("glob", pattern="src/*.py"), _ctx())
    assert result._llm_digest is None
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "src/a.py" in content
    assert "src/b.py" in content


async def test_list_entries_visible_to_llm(ws_user_root):
    (ws_user_root / "docs").mkdir()
    (ws_user_root / "docs" / "readme.md").write_text("x", encoding="utf-8")
    result = await ListTool().execute(_call("list", path="docs"), _ctx())
    assert result._llm_digest is None
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "readme.md" in content


# ── bash：digest 携带完整输出 ────────────────────────────


async def test_bash_stdout_fully_visible_to_llm(tool, tmp_path):
    """bash 输出（截断收口后）完整进 LLM 视角，不再只有尾部 300 字。"""
    result = await tool.execute(
        _call("bash", command="printf 'out-line-%s\\n' $(seq 1 500)"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "exit_code=0" in content
    assert "out-line-1" in content    # 头部可见
    assert "out-line-500" in content  # 尾部可见（500 行 × ~13 字 < 50KB 不截断）


async def test_bash_stderr_visible_to_llm(tool, tmp_path):
    result = await tool.execute(
        _call("bash", command="echo boom >&2; false"),
        {**_ctx(), "workspace:root": str(tmp_path)},
    )
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "exit_code=1" in content
    assert "boom" in content
    assert "stderr:" in content


# ── write：短摘要语义保持 ────────────────────────────────


async def test_write_digest_stays_summary(ws_user_root):
    """write 保持短 digest：模型只需确认成功，无需回显全文（opencode 语义）。"""
    result = await WriteTool().execute(
        _call("write", path="out.py", content="print('hello')"), _ctx(),
    )
    assert result._llm_digest is not None
    content = _llm_tool_content(_llm_messages(result), result.tool_call_id)
    assert "hello" not in content  # 正文不回显
    assert "out.py" in content     # 确认信息可见
