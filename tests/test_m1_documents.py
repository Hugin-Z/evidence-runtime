"""M1 Foundation Slice：规范化文档、无损 Chunk 与 EvidenceSpan。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

from evidence import EvidenceSpan, NormalizedDocument, chunk_document


def _write_document(
    tmp_path: Path,
    data: bytes,
    *,
    name: str = "material.txt",
    **kwargs,
) -> NormalizedDocument:
    path = tmp_path / name
    path.write_bytes(data)
    return NormalizedDocument.from_path(path, **kwargs)


def test_txt_reads_utf8_without_changing_content(tmp_path):
    document = _write_document(tmp_path, "项目材料\n第二行\n".encode("utf-8"))
    assert document.text == "项目材料\n第二行\n"
    assert document.source_type == "txt"


@pytest.mark.parametrize("name", ["material.md", "material.markdown"])
def test_markdown_suffixes_are_supported(tmp_path, name):
    document = _write_document(tmp_path, b"# Heading\nbody", name=name)
    assert document.text == "# Heading\nbody"
    assert document.source_type == "md"


def test_utf8_bom_is_consumed_but_preserved_in_source_hash(tmp_path):
    raw = b"\xef\xbb\xbfalpha\r\nbeta"
    document = _write_document(tmp_path, raw)
    assert document.text == "alpha\nbeta"
    assert document.source_content_hash == hashlib.sha256(raw).hexdigest()


def test_only_line_endings_are_normalized(tmp_path):
    raw = "  第一行  \r\n\r第二行\t\r\n".encode("utf-8")
    document = _write_document(tmp_path, raw)
    assert document.text == "  第一行  \n\n第二行\t\n"
    assert document.normalized_content_hash == hashlib.sha256(
        document.text.encode("utf-8")
    ).hexdigest()


def test_document_id_is_deterministic_without_absolute_path(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = _write_document(left, b"same", name="same.txt")
    second = _write_document(right, b"same", name="same.txt")
    changed = _write_document(right, b"changed", name="changed.txt", source_path="same.txt")
    assert first.document_id == second.document_id
    assert str(left) not in first.document_id
    assert changed.document_id != first.document_id


def test_caller_document_id_is_preserved(tmp_path):
    document = _write_document(tmp_path, b"content", document_id="project-doc-01")
    assert document.document_id == "project-doc-01"


def test_normalized_markdown_can_represent_a_manually_converted_pdf(tmp_path):
    document = _write_document(
        tmp_path,
        b"normalized source",
        name="normalized.md",
        source_path="original/project.pdf",
        source_type="pdf",
    )
    assert document.source_type == "pdf"
    assert document.source_path == "original/project.pdf"


@pytest.mark.parametrize("name", ["material.pdf", "material.docx", "material.json"])
def test_unsupported_normalized_file_type_fails(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"not read")
    with pytest.raises(ValueError, match="normalized document path must end"):
        NormalizedDocument.from_path(path)


def test_non_utf8_input_fails_clearly(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        NormalizedDocument.from_path(path)


def test_chinese_offsets_use_python_characters(tmp_path):
    document = _write_document(tmp_path, "甲乙\n丙丁".encode("utf-8"))
    assert document.slice(1, 4) == "乙\n丙"
    assert document.line_range(1, 4) == (1, 2)


def test_line_index_handles_first_middle_last_and_empty_line(tmp_path):
    document = _write_document(tmp_path, "第一\n\n第三".encode("utf-8"))
    assert document.line_starts == (0, 3, 4)
    assert document.line_number_at(0) == 1
    assert document.line_number_at(2) == 1  # newline belongs to the preceding line
    assert document.line_number_at(3) == 2
    assert document.line_number_at(4) == 3
    assert document.line_number_at(len(document.text)) == 3


def test_trailing_newline_document_end_is_empty_last_line(tmp_path):
    document = _write_document(tmp_path, b"a\n")
    assert document.line_starts == (0, 2)
    assert document.line_number_at(len(document.text)) == 2
    assert document.line_range(len(document.text), len(document.text)) == (2, 2)


def test_empty_document_has_one_logical_position(tmp_path):
    document = _write_document(tmp_path, b"")
    assert document.line_starts == (0,)
    assert document.line_number_at(0) == 1
    assert document.line_range(0, 0) == (1, 1)
    assert document.slice(0, 0) == ""
    assert chunk_document(document, max_chars=10) == ()


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("line_number_at", (-1,)),
        ("line_number_at", (4,)),
        ("slice", (-1, 1)),
        ("slice", (2, 1)),
        ("slice", (0, 4)),
        ("line_range", (0, 4)),
    ],
)
def test_line_index_rejects_out_of_bounds(tmp_path, method, args):
    document = _write_document(tmp_path, b"abc")
    with pytest.raises(ValueError, match="char"):
        getattr(document, method)(*args)


@pytest.mark.parametrize(
    "text",
    [
        "普通第一段\n普通第二段\n\n第三段",
        "x" * 57,
        "中" * 57,
        "第一行\n\n\n第四行\n",
        "1234567890",
        "12345678901",
    ],
)
def test_chunks_are_a_lossless_exact_partition(tmp_path, text):
    document = _write_document(tmp_path, text.encode("utf-8"))
    chunks = chunk_document(document, max_chars=10)
    assert "".join(chunk.text for chunk in chunks) == document.text
    assert all(0 < len(chunk.text) <= 10 for chunk in chunks)
    assert all(
        document.slice(chunk.start_char, chunk.end_char) == chunk.text
        for chunk in chunks
    )
    assert [chunk.start_char for chunk in chunks] == [
        0,
        *[chunk.end_char for chunk in chunks[:-1]],
    ]


def test_long_single_line_is_hard_split(tmp_path):
    document = _write_document(tmp_path, b"a" * 31)
    chunks = chunk_document(document, max_chars=10)
    assert [len(chunk.text) for chunk in chunks] == [10, 10, 10, 1]


def test_paragraph_boundary_is_preferred(tmp_path):
    document = _write_document(tmp_path, b"abc\n\ndefgh")
    chunks = chunk_document(document, max_chars=6)
    assert chunks[0].text == "abc\n\n"


def test_chunk_identity_and_hashes_are_stable(tmp_path):
    document = _write_document(tmp_path, "中文\n材料".encode("utf-8"))
    first = chunk_document(document, max_chars=3)
    second = chunk_document(document, max_chars=3)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    for chunk in first:
        assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5])
def test_chunker_rejects_invalid_max_chars(tmp_path, max_chars):
    document = _write_document(tmp_path, b"abc")
    with pytest.raises(ValueError, match="max_chars.*integer >= 1"):
        chunk_document(document, max_chars=max_chars)


def test_evidence_span_exactly_reads_document_slice(tmp_path):
    document = _write_document(tmp_path, "第一行\n第二行\n第三行".encode("utf-8"))
    start = document.text.index("一行")
    end = document.text.index("第三")
    span = EvidenceSpan.from_document(
        document,
        start,
        end,
        discovery_sources=("vector", "ripgrep"),
    )
    assert span.text == document.text[start:end]
    assert span.document_id == document.document_id
    assert (span.start_line, span.end_line) == (1, 2)
    assert span.page is None
    assert span.discovery_sources == ("vector", "ripgrep")
    assert span.metadata["normalized_content_hash"] == document.normalized_content_hash


def test_evidence_id_is_stable_for_same_document_range(tmp_path):
    document = _write_document(tmp_path, "甲乙丙丁".encode("utf-8"))
    first = EvidenceSpan.from_document(document, 1, 3)
    second = EvidenceSpan.from_document(
        document,
        1,
        3,
        discovery_sources=("ripgrep",),
    )
    assert first.evidence_id == second.evidence_id
    assert first.text == "乙丙"


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 0), (1, 1), (2, 1), (-1, 1), (0, 4)],
)
def test_evidence_span_rejects_empty_or_invalid_ranges(tmp_path, start, end):
    document = _write_document(tmp_path, b"abc")
    with pytest.raises(ValueError, match="EvidenceSpan|character range"):
        EvidenceSpan.from_document(document, start, end)


def test_evidence_span_cannot_accept_caller_supplied_text_or_document_id(tmp_path):
    document = _write_document(tmp_path, b"abc")
    with pytest.raises(TypeError):
        EvidenceSpan(  # type: ignore[call-arg]
            evidence_id="fake",
            document_id="wrong",
            text="wrong",
            start_char=0,
            end_char=1,
            start_line=1,
            end_line=1,
            page=None,
            discovery_sources=(),
            metadata={},
        )
    with pytest.raises(TypeError, match="from_document"):
        EvidenceSpan()
    span = EvidenceSpan.from_document(document, 0, 1)
    with pytest.raises(FrozenInstanceError):
        span.text = "wrong"  # type: ignore[misc]
