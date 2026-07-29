"""M1b Runtime Document Adapter 测试：四格式 fixture + 字符偏移回读 + 无文本层 PDF 报错。

关键不变量（命门）：``metadata["runtime_adapter"]["blocks"]`` 的每个字符区间都必须能用
``NormalizedDocument.slice`` 从 ``text`` 精确回读；且 ``normalized_content_hash`` 与
``text`` 严格一致。适配器是既有 MD/TXT 入口的**等价扩展**：下游一行不改。
"""

from __future__ import annotations

import hashlib
import io

import pytest

from evidence.documents import NormalizedDocument, normalize_text
from evidence.runtime_adapter import (
    ADAPTER_VERSION,
    NoTextLayerError,
    UnsupportedInputError,
    adapt_file,
    adapt_path,
    detect_source_type,
    provenance_for_range,
)


# ── 最小合法 PDF 写入器（TEST-ONLY，含文本层） ──────────────────────────────
def _make_pdf(pages: list[str]) -> bytes:
    """构造每页一个文本对象的最小 PDF；pages 为各页可见文本。"""
    page_ids, content_ids, nid = [], [], 4
    for _ in pages:
        page_ids.append(nid)
        content_ids.append(nid + 1)
        nid += 2
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    parts: dict[int, str] = {
        1: "<</Type/Catalog/Pages 2 0 R>>",
        2: f"<</Type/Pages/Kids[{kids}]/Count {len(pages)}>>",
        3: "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    }
    for pid, cid, txt in zip(page_ids, content_ids, pages):
        esc = txt.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({esc}) Tj ET"
        parts[pid] = (
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]"
            f"/Contents {cid} 0 R/Resources<</Font<</F1 3 0 R>>>>>>"
        )
        parts[cid] = f"<</Length {len(stream)}>>\nstream\n{stream}\nendstream"
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(parts):
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n{parts[oid]}\nendobj\n".encode("latin-1"))
    xref_pos = out.tell()
    count = len(parts) + 1
    out.write(f"xref\n0 {count}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for oid in range(1, count):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<</Size {count}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    return out.getvalue()


def _make_pdf_no_text() -> bytes:
    """无内容流的单页 PDF（扫描件语义：无文本层）。"""
    parts = {
        1: "<</Type/Catalog/Pages 2 0 R>>",
        2: "<</Type/Pages/Kids[3 0 R]/Count 1>>",
        3: "<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>",
    }
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(parts):
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n{parts[oid]}\nendobj\n".encode("latin-1"))
    xref_pos = out.tell()
    count = len(parts) + 1
    out.write(f"xref\n0 {count}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for oid in range(1, count):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<</Size {count}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    return out.getvalue()


def _make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("报告标题", level=1)
    doc.add_paragraph("第一段 retry 内容。")
    doc.add_heading("子节标题", level=2)
    doc.add_paragraph("第二段 backoff 说明。")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "列A"
    table.rows[0].cells[1].text = "列B"
    table.rows[1].cells[0].text = "v1"
    table.rows[1].cells[1].text = "v2"
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ── fixtures：四格式写入 tmp 目录 ─────────────────────────────────────────
@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "note.md").write_text(
        "# 建设目标\n\n平台面向工程建设。\n\n## 子节\n\nretry 内容说明。\n",
        encoding="utf-8",
    )
    (root / "plain.txt").write_text(
        "第一段现状问题。\n\n第二段需求说明。\n", encoding="utf-8"
    )
    (root / "scan.pdf").write_bytes(
        _make_pdf(["First page retry strategy.", "Second page backoff requeue."])
    )
    (root / "report.docx").write_bytes(_make_docx())
    return root


# ── 通用不变量：偏移回读 + hash 一致 ──────────────────────────────────────
def _assert_offsets_and_hash(nd: NormalizedDocument) -> None:
    # normalized_content_hash 与 text 严格一致。
    assert nd.normalized_content_hash == hashlib.sha256(nd.text.encode("utf-8")).hexdigest()
    # text 无 CR、无 BOM（下游 exact_locator 物化前提）。
    assert "\r" not in nd.text
    assert not nd.text.startswith("﻿")
    blocks = nd.metadata["runtime_adapter"]["blocks"]
    assert blocks, "adapter 必须产出至少一个 block"
    for b in blocks:
        start, end = b["start_char"], b["end_char"]
        assert 0 <= start < end <= len(nd.text)
        # 命门：block 区间必须能从 text 精确回读，且回读内容等于其自身规范化。
        segment = nd.slice(start, end)
        assert segment == normalize_text(segment)
        assert segment.strip(), "block 区间不应是纯空白"


@pytest.mark.parametrize(
    ("name", "expected_type"),
    (("note.md", "md"), ("plain.txt", "txt"), ("scan.pdf", "pdf"), ("report.docx", "docx")),
)
def test_four_formats_adapt_with_readable_offsets(corpus, name, expected_type):
    nd = adapt_file(corpus / name)
    assert nd.source_type == expected_type
    assert nd.metadata["runtime_adapter"]["version"] == ADAPTER_VERSION
    assert nd.metadata["runtime_adapter"]["original_filename"] == name
    _assert_offsets_and_hash(nd)


def test_md_equivalent_to_frozen_from_path_entry(corpus):
    # 适配器对 MD/TXT 必须与冻结的 NormalizedDocument.from_path 逐字节等价（等价替换前提）。
    nd = adapt_file(corpus / "note.md")
    frozen = NormalizedDocument.from_path(corpus / "note.md")
    assert nd.text == frozen.text
    assert nd.normalized_content_hash == frozen.normalized_content_hash


def test_txt_equivalent_to_frozen_from_path_entry(corpus):
    nd = adapt_file(corpus / "plain.txt")
    frozen = NormalizedDocument.from_path(corpus / "plain.txt")
    assert nd.text == frozen.text
    assert nd.normalized_content_hash == frozen.normalized_content_hash


def test_md_headings_captured_in_blocks(corpus):
    nd = adapt_file(corpus / "note.md")
    headings = [
        b for b in nd.metadata["runtime_adapter"]["blocks"] if b["kind"] == "heading"
    ]
    assert [nd.slice(b["start_char"], b["end_char"]).lstrip("# ").strip() for b in headings] == [
        "建设目标",
        "子节",
    ]
    # heading_path 反映层级栈：## 子节 位于 # 建设目标 之下。
    assert headings[1]["heading_path"] == ["建设目标", "子节"]


def test_docx_heading_and_table_blocks(corpus):
    nd = adapt_file(corpus / "report.docx")
    kinds = [b["kind"] for b in nd.metadata["runtime_adapter"]["blocks"]]
    assert "heading" in kinds and "table" in kinds
    # 表格单元以 \t/\n 结构进入 text，可回读。
    table_block = next(
        b for b in nd.metadata["runtime_adapter"]["blocks"] if b["kind"] == "table"
    )
    seg = nd.slice(table_block["start_char"], table_block["end_char"])
    assert "列A" in seg and "v2" in seg


def test_pdf_two_pages_page_provenance(corpus):
    nd = adapt_file(corpus / "scan.pdf")
    ra = nd.metadata["runtime_adapter"]
    assert ra["page_count"] == 2
    page_blocks = [b for b in ra["blocks"] if b["kind"] == "page"]
    assert [b["page"] for b in page_blocks] == [1, 2]
    # provenance_for_range 能据字符区间回答页码。
    prov = provenance_for_range(nd, page_blocks[1]["start_char"], page_blocks[1]["end_char"])
    assert prov["pages"] == [2]


def test_no_text_layer_pdf_raises(tmp_path):
    scanned = tmp_path / "scanned.pdf"
    scanned.write_bytes(_make_pdf_no_text())
    with pytest.raises(NoTextLayerError):
        adapt_file(scanned)


def test_unsupported_extension_rejected(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        detect_source_type(bad)


def test_magic_byte_mismatch_rejected(tmp_path):
    # 扩展名 .pdf 但内容非 %PDF → 报错（不误判）。
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf at all")
    with pytest.raises(UnsupportedInputError):
        detect_source_type(fake)


def test_adapt_path_directory_is_sorted_and_recursive(corpus):
    docs = adapt_path(corpus)
    assert len(docs) == 4
    # 按路径排序稳定。
    assert [d.source_path for d in docs] == sorted(d.source_path for d in docs)


def test_auto_document_id_is_content_addressed_and_stable(corpus):
    # 无显式 id 时，document_id 由文件名 + 规范化正文哈希派生 → 同内容稳定复现、异内容相异。
    first = adapt_file(corpus / "plain.txt")
    second = adapt_file(corpus / "plain.txt")
    other = adapt_file(corpus / "note.md")
    assert first.document_id == second.document_id
    assert first.document_id.startswith("doc-")
    assert first.document_id != other.document_id
