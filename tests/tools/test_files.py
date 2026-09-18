"""read / write / edit / list 文件工具测试（真实文件操作 + 事件契约 + 越界拒绝）。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from ark_agentic.core.types import ToolCall

from egis_opencode.core.tools.files import (
    EditTool,
    ListTool,
    ReadTool,
    WriteTool,
)
from egis_opencode.events import TOOL_DIGEST
from egis_opencode.workspace import WorkspacePathError

from tests.helpers import RecordingHandler

CTX = {"user:id": "alice"}


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall.create(name, arguments)


def _ctx(handler: RecordingHandler | None = None) -> dict:
    ctx = dict(CTX)
    if handler is not None:
        ctx["system:event_handler"] = handler
    return ctx


def _make_docx(path: Path, title: str, body: str, *, bookmarked: bool = False) -> None:
    """构造最小合法 docx（Heading1 标题段 + 正文段，可选书签）。

    bookmarked=True 时标题段带 Word 自动书签（模拟真实文档目录锚点，
    mammoth 会转出 <a id="..."></a> 噪音）。
    """
    bookmark_start = (
        '<w:bookmarkStart w:id="0" w:name="_Toc12345"/>' if bookmarked else ""
    )
    bookmark_end = '<w:bookmarkEnd w:id="0"/>' if bookmarked else ""
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
        f"{bookmark_start}"
        f"<w:r><w:t>{title}</w:t></w:r>"
        f"{bookmark_end}</w:p>"
        f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.'
        'wordprocessingml.document.main+xml"/></Types>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("word/document.xml", doc_xml)


# ── read ───────────────────────────────────────────────


async def test_read_returns_numbered_lines(ws_user_root):
    (ws_user_root / "a.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    result = await ReadTool().execute(_call("read", path="a.py"), _ctx())

    assert result.is_error is False
    content = str(result.content)
    # opencode read.ts 输出契约：<path>/<type>/<content> 标签 + "N: line" 行号
    assert "<path>a.py</path>" in content
    assert "<type>file</type>" in content
    assert "<content>" in content
    assert "1: line1" in content
    assert "2: line2" in content
    assert "(End of file - total 3 lines)" in content


async def test_read_offset_limit(ws_user_root):
    (ws_user_root / "a.txt").write_text("\n".join(f"l{i}" for i in range(1, 51)))
    result = await ReadTool().execute(
        _call("read", path="a.txt", offset=10, limit=5), _ctx(),
    )
    content = str(result.content)
    assert "10: l10" in content
    assert "14: l14" in content
    assert "15: l15" not in content
    assert "(Showing lines 10-14 of 50. Use offset=15 to continue.)" in content


async def test_read_missing_file_error(ws_user_root):
    result = await ReadTool().execute(_call("read", path="nope.py"), _ctx())
    assert result.is_error is True
    assert "File not found" in str(result.content)


async def test_read_binary_rejected(ws_user_root):
    (ws_user_root / "logo.png").write_bytes(b"\x89PNG...")
    result = await ReadTool().execute(_call("read", path="logo.png"), _ctx())
    assert result.is_error is True
    assert "二进制" in str(result.content)


async def test_read_path_escape_rejected(ws_user_root):
    result = await ReadTool().execute(
        _call("read", path="../bob/secret.txt"), _ctx(),
    )
    assert result.is_error is True
    assert "越出 workspace" in str(result.content)


async def test_read_emits_digest_event(ws_user_root, recorder):
    (ws_user_root / "a.py").write_text("x = 1\ny = 2\n")
    await ReadTool().execute(
        _call("read", path="a.py"), _ctx(recorder),
    )
    digests = recorder.of_type(TOOL_DIGEST)
    assert len(digests) == 1
    assert digests[0]["tool_name"] == "read"
    assert digests[0]["display_type"] == "search"
    assert digests[0]["status"] == "success"
    assert digests[0]["path"] == "a.py"
    # 内容预览（前端卡片直接展示，不用展开猜测）
    assert digests[0]["results_preview"] == ["1: x = 1", "2: y = 2"]


# ── read .docx 提取 ─────────────────────────────


async def test_read_docx_extracts_markdown(ws_user_root):
    """docx 经 mammoth 提取：标题层级保留 + 行号输出。"""
    _make_docx(ws_user_root / "素材.docx", "平安养老险", "公司实力介绍素材。")
    result = await ReadTool().execute(
        _call("read", path="素材.docx"), _ctx(),
    )
    assert result.is_error is False
    content = str(result.content)
    assert "# 平安养老险" in content
    assert "公司实力介绍素材" in content
    assert "1: " in content  # 行号输出形态


async def test_read_docx_anchored_local_dir(ws_user_root, tmp_path):
    """本地目录会话下读 docx（llm-wiki raw 场景）。"""
    local = tmp_path / "llm-wiki" / "raw" / "docs"
    local.mkdir(parents=True)
    _make_docx(local / "介绍.docx", "受托优势", "受托投资介绍内容。")
    ctx = {"user:id": "alice", "workspace:root": str(tmp_path / "llm-wiki")}
    result = await ReadTool().execute(
        _call("read", path="raw/docs/介绍.docx"), ctx,
    )
    assert result.is_error is False
    assert "受托投资介绍内容" in str(result.content)


async def test_read_docx_bookmark_anchors_stripped(ws_user_root):
    """带书签 docx（真实文档常见）：锚点噪音被清理，标题层级保留。"""
    _make_docx(
        ws_user_root / "带目录.docx", "标题一", "正文。", bookmarked=True,
    )
    result = await ReadTool().execute(
        _call("read", path="带目录.docx"), _ctx(),
    )
    assert result.is_error is False
    content = str(result.content)
    assert "<a id=" not in content
    assert "# 标题一" in content
    assert "正文。" in content


async def test_read_docx_broken_file_error(ws_user_root):
    """损坏 docx（非法 zip）：明确报错而非乱码。"""
    (ws_user_root / "broken.docx").write_bytes(b"not a zip file")
    result = await ReadTool().execute(
        _call("read", path="broken.docx"), _ctx(),
    )
    assert result.is_error is True
    assert "docx 提取失败" in str(result.content)


async def test_read_docx_empty_document_error(ws_user_root):
    """合法 zip 但无正文段：报无文本。"""
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body/></w:document>"
    )
    with zipfile.ZipFile(ws_user_root / "empty.docx", "w") as z:
        z.writestr("word/document.xml", doc_xml)
    result = await ReadTool().execute(
        _call("read", path="empty.docx"), _ctx(),
    )
    assert result.is_error is True
    assert "无文本" in str(result.content)


async def test_read_unknown_binary_sniffed(ws_user_root):
    """未知后缀二进制（含 NUL）：兑底检测拒绝，不灌乱码。"""
    (ws_user_root / "data.dat").write_bytes(b"\x00\x01\x02binary")
    result = await ReadTool().execute(
        _call("read", path="data.dat"), _ctx(),
    )
    assert result.is_error is True
    assert "二进制" in str(result.content)


async def test_read_long_line_truncated(ws_user_root):
    """巨行（base64 data URI 形态）：单行截断到 2000 字符 + 后缀提示。"""
    huge = "x" * 60_000  # 模拟 base64 巨行
    (ws_user_root / "big.md").write_text(f"head\n{huge}\ntail\n", encoding="utf-8")
    result = await ReadTool().execute(_call("read", path="big.md"), _ctx())

    assert result.is_error is False
    content = str(result.content)
    assert len(content) < 10_000  # 巨行不再整体进上下文
    assert "... (line truncated to 2000 chars)" in content
    assert "head" in content


async def test_read_byte_cap_truncates_with_hint(ws_user_root):
    """累计字节达 50KB 上限：截断 + "capped ... Use offset=N" 续读提示。"""
    # 300 行 x 300B ≈ 90KB，超过 tool_output_max_bytes(50KB)
    body = "\n".join(f"line-{i:04d}-" + "y" * 280 for i in range(300))
    (ws_user_root / "fat.txt").write_text(body, encoding="utf-8")
    result = await ReadTool().execute(
        _call("read", path="fat.txt", limit=2000), _ctx(),
    )

    assert result.is_error is False
    content = str(result.content)
    assert "(Output capped at 50 KB." in content
    assert "Use offset=" in content
    assert len(content) < 60_000


async def test_read_line_limit_shows_remaining(ws_user_root):
    """行数达 limit：提示剩余行数与续读 offset（对齐 opencode）。"""
    body = "\n".join(f"l{i}" for i in range(1, 51))
    (ws_user_root / "many.txt").write_text(body, encoding="utf-8")
    result = await ReadTool().execute(
        _call("read", path="many.txt", limit=10), _ctx(),
    )

    assert result.is_error is False
    content = str(result.content)
    assert "(Showing lines 1-10 of 50. Use offset=11 to continue.)" in content


async def test_read_offset_out_of_range_error(ws_user_root):
    """offset 超出文件行数：明确报错（对齐 opencode 而非空区间文案）。"""
    (ws_user_root / "small.txt").write_text("a\nb\n", encoding="utf-8")
    result = await ReadTool().execute(
        _call("read", path="small.txt", offset=99), _ctx(),
    )
    assert result.is_error is True
    assert "out of range" in str(result.content)
    assert "2 lines" in str(result.content)


async def test_read_missing_suggests_similar(ws_user_root):
    """文件不存在 + 同目录相似名：Did-you-mean 建议（对齐 opencode miss）。"""
    (ws_user_root / "config.json.bak").write_text("{}", encoding="utf-8")
    result = await ReadTool().execute(
        _call("read", path="config.json"), _ctx(),
    )
    assert result.is_error is True
    content = str(result.content)
    assert "Did you mean one of these?" in content
    assert "config.json.bak" in content


async def test_read_docx_image_placeholder(monkeypatch, ws_user_root):
    """docx 内嵌图片（base64 data URI）：占位符化，巨行噪音不进上下文。

    幻觉根因回归：mammoth 默认把内嵌图片转成 ![](data:image/png;base64,…)
    内联巨行（实测 1.4MB 文档可产出 150 万字符 base64，占提取文本 98%），
    直接击穿 LLM 上下文后模型只能靠训练记忆编造文档内容。
    """
    from types import SimpleNamespace

    fake_mammoth = SimpleNamespace(
        convert_to_markdown=lambda fh: SimpleNamespace(
            value=(
                "# 标题\n\n正文段落。\n\n"
                "![](data:image/png;base64,"
                + "A" * 300_000
                + ")\n\n结尾。\n"
            ),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "mammoth", fake_mammoth)

    (ws_user_root / "img.docx").write_bytes(b"placeholder")
    result = await ReadTool().execute(
        _call("read", path="img.docx"), _ctx(),
    )

    assert result.is_error is False
    content = str(result.content)
    assert "data:image" not in content
    assert "base64" not in content
    assert "[图片内容已省略]" in content
    assert "# 标题" in content
    assert "结尾。" in content


async def test_read_legacy_office_rejected(ws_user_root):
    """.doc/.pptx/.xlsx 老格式：明确二进制拒绝提示。"""
    (ws_user_root / "老文档.doc").write_bytes(b"\xd0\xcf\x11\xe0")
    result = await ReadTool().execute(
        _call("read", path="老文档.doc"), _ctx(),
    )
    assert result.is_error is True
    assert "二进制" in str(result.content)


# ── write ──────────────────────────────────────────────


async def test_write_creates_new_file(ws_root, ws_user_root, recorder):
    result = await WriteTool().execute(
        _call("write", path="proj/new.py", content="print('hi')\n"),
        _ctx(recorder),
    )

    assert result.is_error is False
    assert (ws_user_root / "proj" / "new.py").read_text() == "print('hi')\n"
    digests = recorder.of_type(TOOL_DIGEST)
    assert digests[0]["display_type"] == "file_edit"
    assert digests[0]["status"] == "success"
    assert digests[0]["title"].startswith("新建")


async def test_write_overwrites_existing(ws_user_root, recorder):
    target = ws_user_root / "a.txt"
    target.write_text("old")
    await WriteTool().execute(
        _call("write", path="a.txt", content="new"), _ctx(recorder),
    )
    assert target.read_text() == "new"
    assert recorder.of_type(TOOL_DIGEST)[0]["title"].startswith("覆写")
    # 覆写摘要带旧内容预览
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["old_string_preview"] == "old"
    assert digest["new_string_preview"] == "new"


async def test_write_escape_rejected(ws_user_root):
    result = await WriteTool().execute(
        _call("write", path="../../etc/cron", content="x"), _ctx(),
    )
    assert result.is_error is True
    assert "越出 workspace" in str(result.content)


async def test_write_missing_content_error(ws_user_root):
    result = await WriteTool().execute(_call("write", path="a.txt"), _ctx())
    assert result.is_error is True


# ── edit ───────────────────────────────────────────────


async def test_edit_unique_replacement(ws_user_root, recorder):
    target = ws_user_root / "conf.py"
    target.write_text("DEBUG = False\nVERBOSE = False\n")

    result = await EditTool().execute(
        _call("edit", path="conf.py", old_string="DEBUG = False",
              new_string="DEBUG = True"),
        _ctx(recorder),
    )

    assert result.is_error is False
    assert target.read_text() == "DEBUG = True\nVERBOSE = False\n"
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["display_type"] == "file_edit"
    assert "- DEBUG = False" in digest["diff"]
    assert "+ DEBUG = True" in digest["diff"]


async def test_edit_ambiguous_match_rejected(ws_user_root):
    target = ws_user_root / "dup.txt"
    target.write_text("same\nsame\n")
    result = await EditTool().execute(
        _call("edit", path="dup.txt", old_string="same", new_string="x"), _ctx(),
    )
    assert result.is_error is True
    assert "2 处" in str(result.content)
    assert target.read_text() == "same\nsame\n"  # 未被修改


async def test_edit_replace_all(ws_user_root):
    target = ws_user_root / "dup.txt"
    target.write_text("same\nsame\n")
    result = await EditTool().execute(
        _call("edit", path="dup.txt", old_string="same", new_string="x",
              replace_all=True),
        _ctx(),
    )
    assert result.is_error is False
    assert target.read_text() == "x\nx\n"


async def test_edit_no_match_rejected(ws_user_root):
    (ws_user_root / "a.txt").write_text("content")
    result = await EditTool().execute(
        _call("edit", path="a.txt", old_string="absent", new_string="x"), _ctx(),
    )
    assert result.is_error is True
    assert "未在文件中命中" in str(result.content)


async def test_edit_missing_file_rejected(ws_user_root):
    result = await EditTool().execute(
        _call("edit", path="nope.txt", old_string="a", new_string="b"), _ctx(),
    )
    assert result.is_error is True


# ── list ───────────────────────────────────────────────


async def test_list_directory_entries(ws_user_root, recorder):
    (ws_user_root / "sub").mkdir()
    (ws_user_root / "a.py").write_text("x" * 10)
    (ws_user_root / "b.md").write_text("y" * 5)
    (ws_user_root / ".hidden").write_text("z")  # 隐藏项不列

    result = await ListTool().execute(_call("list", path="."), _ctx(recorder))

    assert result.is_error is False
    entries = result.content["entries"]
    names = [e["name"] for e in entries]
    assert names == ["sub", "a.py", "b.md"]  # 目录在前
    by_name = {e["name"]: e for e in entries}
    assert by_name["sub"]["type"] == "dir"
    assert by_name["a.py"]["bytes"] == 10

    # digest 预览：目录带 / 后缀，一眼能看清结构
    digest = recorder.of_type(TOOL_DIGEST)[0]
    assert digest["results_preview"] == ["sub/", "a.py", "b.md"]


async def test_list_default_root(ws_user_root):
    (ws_user_root / "only.txt").write_text("x")
    result = await ListTool().execute(_call("list"), _ctx())
    assert result.is_error is False


async def test_list_missing_directory_error(ws_user_root):
    result = await ListTool().execute(_call("list", path="no-dir"), _ctx())
    assert result.is_error is True


# ── 多用户隔离 ─────────────────────────────────────────


async def test_user_isolation(ws_root, ws_user_root):
    """alice 与 bob 的 workspace 完全隔离（同名文件互不可见）。"""
    bob_root = ws_root / "bob"
    bob_root.mkdir(parents=True)
    (bob_root / "shared.txt").write_text("bob's secret")
    (ws_user_root / "shared.txt").write_text("alice's file")

    # alice 经绝对路径读 bob 文件 → 越界拒绝
    result = await ReadTool().execute(
        _call("read", path=str(bob_root / "shared.txt")), _ctx(),
    )
    assert result.is_error is True
    assert "越出 workspace" in str(result.content)


async def test_default_user_when_context_empty(ws_root, ws_user_root):
    """无用户标识回落 default 用户目录（不误伤他人 workspace）。"""
    (ws_root / "default").mkdir(parents=True, exist_ok=True)
    (ws_root / "default" / "mine.txt").write_text("default user file")
    result = await ReadTool().execute(
        _call("read", path="mine.txt"), {},
    )
    assert result.is_error is False
    assert "default user file" in str(result.content)


async def test_read_error_emits_error_digest(ws_user_root, recorder):
    """错误路径必须发 error digest 事件（tool_call_result 帧只带裸 content，
    前端无法区分成败 —— 错误不发 digest 会显示成"成功（无详情）"）。"""
    result = await ReadTool().execute(
        _call("read", path="nope.md"), _ctx(recorder),
    )
    assert result.is_error is True

    digests = recorder.of_type(TOOL_DIGEST)
    assert digests, "错误路径必须发射 tool_digest 事件"
    error_digest = digests[-1]
    assert error_digest["status"] == "error"
    assert "File not found" in error_digest["note"]
