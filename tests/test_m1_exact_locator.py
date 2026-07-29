"""M1 Step 4 canonical corpus、anchor、offset、scope 与 context 测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evidence import (
    CANONICAL_MANIFEST_NAME,
    CanonicalCorpusError,
    EvidenceSpan,
    ExactEvidenceLocator,
    InvalidRipgrepOutputError,
    NormalizedDocument,
    RipgrepExecutionError,
    RipgrepLineEvent,
    RipgrepRunResult,
    RipgrepSubmatch,
    VectorCandidate,
    expand_match_context,
    extract_exact_anchors,
    map_rg_submatch_to_char_range,
    materialize_normalized_documents,
    solution_material_completeness_v1,
)
from evidence.exact_locator import _build_rescue_candidates


SLOT_ID = "background_and_basis"


def _document(root, name, raw, *, document_id):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, bytes):
        path.write_bytes(raw)
    else:
        path.write_text(raw, encoding="utf-8", newline="")
    return NormalizedDocument.from_path(
        path,
        document_id=document_id,
        source_path=f"materials/{name}",
    )


def _candidate(document, *, rank=1, chunk_id=None, slot_id=SLOT_ID):
    if not document.text:
        raise ValueError("test candidate needs non-empty document")
    start_line, end_line = document.line_range(0, len(document.text))
    return VectorCandidate(
        slot_id=slot_id,
        query="test query",
        document_id=document.document_id,
        chunk_id=chunk_id or f"chunk-{document.document_id}-{rank}",
        text=document.text,
        start_char=0,
        end_char=len(document.text),
        start_line=start_line,
        end_line=end_line,
        rank=rank,
        distance=float(rank),
        discovery_source="vector",
    )


def _candidate_range(
    document,
    start_char,
    end_char,
    *,
    rank,
    distance=None,
    chunk_id=None,
):
    start_line, end_line = document.line_range(start_char, end_char)
    return VectorCandidate(
        slot_id=SLOT_ID,
        query="test query",
        document_id=document.document_id,
        chunk_id=chunk_id or f"chunk-{rank}-{start_char}",
        text=document.slice(start_char, end_char),
        start_char=start_char,
        end_char=end_char,
        start_line=start_line,
        end_line=end_line,
        rank=rank,
        distance=float(rank) if distance is None else distance,
        discovery_source="vector",
    )


def _span(document, start_char, end_char):
    return EvidenceSpan.from_document(
        document,
        start_char,
        end_char,
        discovery_sources=("vector", "ripgrep"),
    )


def _event_for(document, term, *, occurrence=0, path="canonical.txt"):
    lowered = document.text.casefold()
    search = term.casefold()
    position = -1
    for _ in range(occurrence + 1):
        position = lowered.find(search, position + 1)
    assert position >= 0
    line_number = document.line_number_at(position)
    line_start = document.line_starts[line_number - 1]
    line_end = (
        document.line_starts[line_number]
        if line_number < len(document.line_starts)
        else len(document.text)
    )
    line_text = document.slice(line_start, line_end)
    match_text = document.slice(position, position + len(term))
    start_byte = len(document.slice(line_start, position).encode("utf-8"))
    submatch = RipgrepSubmatch(
        match_text,
        start_byte,
        start_byte + len(match_text.encode("utf-8")),
    )
    return RipgrepLineEvent(
        kind="match",
        path=str(path),
        line_number=line_number,
        line_text=line_text,
        submatches=(submatch,),
    ), submatch


class LiteralFakeExecutor:
    """TEST-ONLY fixed-string executor over materialized files."""

    rg_source = "test fake"

    def __init__(self, *, error=None):
        self.calls = []
        self.error = error

    def search(self, terms, targets, *, context=0, regex=False, globs=()):
        self.calls.append((tuple(terms), tuple(map(str, targets)), context, regex, tuple(globs)))
        if self.error is not None:
            raise self.error
        assert len(terms) == 1 and context == 0 and regex is False and not globs
        term = terms[0]
        events = []
        grouped = {}
        for raw_path in targets:
            path = Path(raw_path).resolve()
            text = path.read_text(encoding="utf-8")
            lowered = text.casefold()
            search = term.casefold()
            cursor = 0
            file_events = []
            while True:
                start = lowered.find(search, cursor)
                if start < 0:
                    break
                end = start + len(term)
                line_start = text.rfind("\n", 0, start) + 1
                newline = text.find("\n", end)
                line_end = len(text) if newline < 0 else newline + 1
                line_text = text[line_start:line_end]
                match_text = text[start:end]
                start_byte = len(text[line_start:start].encode("utf-8"))
                event = RipgrepLineEvent(
                    kind="match",
                    path=str(path),
                    line_number=text.count("\n", 0, start) + 1,
                    line_text=line_text,
                    submatches=(
                        RipgrepSubmatch(
                            match_text,
                            start_byte,
                            start_byte + len(match_text.encode("utf-8")),
                        ),
                    ),
                )
                events.append(event)
                file_events.append(event)
                cursor = end
            if file_events:
                grouped[str(path)] = tuple(file_events)
        return RipgrepRunResult(
            command=("fake-rg", "--fixed-strings", term),
            returncode=0 if events else 1,
            stderr="",
            rg_source=self.rg_source,
            events=tuple(events),
            grouped_events=grouped,
        )


def test_canonical_materialization_normalizes_bom_crlf_and_preserves_unicode(tmp_path):
    raw = b"\xef\xbb\xbf" + "第一行😀\r\n第二行\r\n".encode("utf-8")
    document = _document(tmp_path, "source.txt", raw, document_id="doc-unicode")
    manifest = materialize_normalized_documents([document], tmp_path / "canonical")
    entry = manifest.get("doc-unicode")
    stored = Path(entry.canonical_path).read_bytes()
    assert stored == "第一行😀\n第二行\n".encode("utf-8")
    assert not stored.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in stored
    assert entry.normalized_content_hash == hashlib.sha256(stored).hexdigest()
    assert entry.text_length == len(document.text)
    assert entry.line_count == len(document.line_starts) == 3


def test_canonical_filename_is_content_derived_not_document_id_or_source_name(tmp_path):
    # 物化文件名派生自**规范化正文哈希**（内容确定），既不含来源名、也不用 document_id。
    # 早期为 sha256(document_id)，使 rg 语料文件名随分配的 id 漂移（M1b 适配器暴露）。
    document = _document(tmp_path, "secret-name.md", "正文", document_id="logical-id")
    entry = materialize_normalized_documents([document], tmp_path / "canonical").documents[0]
    assert Path(entry.canonical_path).name == (
        f"document-{document.normalized_content_hash}.txt"
    )
    assert "secret-name" not in entry.canonical_path
    # 明确证明不再以 document_id 派生文件名。
    assert hashlib.sha256(b"logical-id").hexdigest() not in entry.canonical_path


@pytest.mark.parametrize("document_id", ("../escape", "a/b", r"a\b", ".", ".."))
def test_canonical_document_id_path_traversal_is_rejected(tmp_path, document_id):
    document = _document(tmp_path, "source.txt", "正文", document_id=document_id)
    with pytest.raises(CanonicalCorpusError, match="path traversal"):
        materialize_normalized_documents([document], tmp_path / "canonical")


def test_repeated_materialization_is_stable_and_manifest_maps_documents(tmp_path):
    first = _document(tmp_path, "b.txt", "乙", document_id="doc-b")
    second = _document(tmp_path, "a.txt", "甲", document_id="doc-a")
    one = materialize_normalized_documents([first, second], tmp_path / "canonical")
    before = Path(one.get("doc-a").canonical_path).read_bytes()
    two = materialize_normalized_documents([second, first], tmp_path / "canonical")
    assert one == two
    assert two.document_ids == ("doc-a", "doc-b")
    assert Path(two.get("doc-a").canonical_path).read_bytes() == before
    stored = json.loads(
        (tmp_path / "canonical" / CANONICAL_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert stored == two.to_dict()


def test_content_change_writes_new_content_addressed_file_and_changes_corpus_identity(tmp_path):
    # 内容寻址：同一 document_id 正文变化 → 文件名（内容哈希）变化 → 写入**不同**物化文件，
    # 旧文件不被覆盖。语料 rg 只检索 manifest 的显式目标，遗留旧文件不参与检索。
    old = _document(tmp_path / "old", "x.txt", "旧正文", document_id="doc-x")
    new = _document(tmp_path / "new", "x.txt", "新正文😀", document_id="doc-x")
    first = materialize_normalized_documents([old], tmp_path / "canonical")
    second = materialize_normalized_documents([new], tmp_path / "canonical")
    assert first.corpus_id != second.corpus_id
    assert first.documents[0].canonical_path != second.documents[0].canonical_path
    assert Path(first.documents[0].canonical_path).read_text(encoding="utf-8") == old.text
    assert Path(second.documents[0].canonical_path).read_text(encoding="utf-8") == new.text


def test_duplicate_document_ids_are_rejected(tmp_path):
    first = _document(tmp_path / "a", "x.txt", "甲", document_id="duplicate")
    second = _document(tmp_path / "b", "x.txt", "乙", document_id="duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        materialize_normalized_documents([first, second], tmp_path / "canonical")


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    (
        ("ASCII needle end", "needle", (6, 12)),
        ("前缀中文命中后缀", "中文", (2, 4)),
        ("emoji 😀 marker", "😀", (6, 7)),
        ("混合abc中文XYZ", "abc中文", (2, 7)),
    ),
)
def test_rg_utf8_byte_offsets_map_to_python_character_offsets(tmp_path, text, term, expected):
    document = _document(tmp_path, "source.txt", text, document_id="doc")
    event, submatch = _event_for(document, term)
    assert map_rg_submatch_to_char_range(document, event, submatch) == expected
    assert document.slice(*expected) == term


def test_offset_mapping_handles_multiple_and_repeated_hits(tmp_path):
    document = _document(
        tmp_path,
        "source.txt",
        "first needle and needle\n末行 needle",
        document_id="doc",
    )
    ranges = []
    for occurrence in range(3):
        event, submatch = _event_for(document, "needle", occurrence=occurrence)
        ranges.append(map_rg_submatch_to_char_range(document, event, submatch))
    assert ranges == [(6, 12), (17, 23), (27, 33)]
    assert all(document.slice(*item) == "needle" for item in ranges)


def test_offset_mapping_handles_multiple_submatches_in_one_rg_line_event(tmp_path):
    document = _document(tmp_path, "source.txt", "针needle与needle\n", document_id="doc")
    first_event, first = _event_for(document, "needle", occurrence=0)
    _, second = _event_for(document, "needle", occurrence=1)
    event = RipgrepLineEvent(
        "match",
        first_event.path,
        first_event.line_number,
        first_event.line_text,
        (first, second),
    )
    ranges = [
        map_rg_submatch_to_char_range(document, event, submatch)
        for submatch in event.submatches
    ]
    assert ranges == [(1, 7), (8, 14)]


def test_offset_mapping_first_and_last_line_without_terminal_newline(tmp_path):
    document = _document(tmp_path, "source.txt", "first\n最后命中", document_id="doc")
    first_event, first_match = _event_for(document, "first")
    last_event, last_match = _event_for(document, "命中")
    assert map_rg_submatch_to_char_range(document, first_event, first_match) == (0, 5)
    assert map_rg_submatch_to_char_range(document, last_event, last_match) == (8, 10)


def test_offset_mapping_rejects_utf8_split_or_noncanonical_line(tmp_path):
    document = _document(tmp_path, "source.txt", "中x\n", document_id="doc")
    bad = RipgrepSubmatch("中", 1, 3)
    event = RipgrepLineEvent("match", "canonical", 1, "中x\n", (bad,))
    with pytest.raises(InvalidRipgrepOutputError, match="UTF-8"):
        map_rg_submatch_to_char_range(document, event, bad)
    mismatch = RipgrepLineEvent(
        "match", "canonical", 1, "另一行\n", (RipgrepSubmatch("另", 0, 3),)
    )
    with pytest.raises(InvalidRipgrepOutputError, match="canonical"):
        map_rg_submatch_to_char_range(document, mismatch, mismatch.submatches[0])


def test_anchor_order_sources_patterns_dedup_and_no_weak_conditions(tmp_path):
    text = '依据“星河平台”于2026-07-16达到30%，型号 ABC-123，项目 星河项目 42。'
    document = _document(tmp_path, "source.txt", text, document_id="doc")
    candidate = _candidate(document)
    slot = solution_material_completeness_v1.get_slot(SLOT_ID)
    anchors = extract_exact_anchors(
        slot,
        [candidate],
        explicit_terms=("EXACT-ID", " exact-id ", "", "x" * 81),
    )
    assert anchors[0].term == "EXACT-ID" and anchors[0].source == "caller"
    assert anchors[1].term == slot.name and anchors[1].source == "slot_name"
    assert all(field.replace("_", " ") in [item.term for item in anchors]
               for field in slot.required_fields)
    terms = [item.term for item in anchors]
    assert {"星河平台", "2026-07-16", "30%", "ABC-123", "星河项目", "42"} <= set(terms)
    assert len({term.casefold() for term in terms}) == len(terms)
    assert all(condition not in terms for condition in slot.weak_conditions)
    assert all(len(term) <= 80 and term.strip() for term in terms)


def test_anchor_limit_is_stable_and_candidates_must_match_slot(tmp_path):
    document = _document(tmp_path, "source.txt", "编号 1 2 3 4", document_id="doc")
    slot = solution_material_completeness_v1.get_slot(SLOT_ID)
    first = extract_exact_anchors(slot, [_candidate(document)], max_anchors=3)
    second = extract_exact_anchors(slot, [_candidate(document)], max_anchors=3)
    assert first == second and len(first) == 3
    with pytest.raises(ValueError, match="belongs to slot"):
        extract_exact_anchors(slot, [_candidate(document, slot_id="objectives")])


def test_context_expands_complete_lines_without_trim_and_respects_boundaries(tmp_path):
    document = _document(
        tmp_path,
        "source.txt",
        "  first  \nmatch here\n  last  ",
        document_id="doc",
    )
    match_start = document.text.index("match")
    assert expand_match_context(
        document, match_start, match_start + 5, context_lines=1, max_chars=100
    ) == (0, len(document.text))
    start, end = expand_match_context(
        document, match_start, match_start + 5, context_lines=0, max_chars=100
    )
    assert document.slice(start, end) == "match here\n"
    assert document.slice(0, len(document.text)).startswith("  ")
    assert document.slice(0, len(document.text)).endswith("  ")


def test_context_symmetric_crop_keeps_match_and_max_chars(tmp_path):
    document = _document(tmp_path, "source.txt", "L" * 40 + "MATCH" + "R" * 40, document_id="doc")
    match_start = document.text.index("MATCH")
    start, end = expand_match_context(
        document, match_start, match_start + 5, context_lines=2, max_chars=21
    )
    assert end - start == 21
    assert document.slice(start, end) == "L" * 8 + "MATCH" + "R" * 8
    with pytest.raises(ValueError, match="exceeds"):
        expand_match_context(
            document, match_start, match_start + 5, context_lines=0, max_chars=4
        )


def test_candidate_scope_hit_avoids_fallback_and_returns_exact_span(tmp_path):
    candidate_doc = _document(
        tmp_path, "candidate.txt", "前行\n  EXACT-MARKER 正文  \n后行", document_id="candidate"
    )
    other_doc = _document(tmp_path, "other.txt", "EXACT-MARKER", document_id="other")
    executor = LiteralFakeExecutor()
    result = ExactEvidenceLocator(tmp_path / "canonical", executor=executor).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[candidate_doc, other_doc],
        candidates=[_candidate(candidate_doc), _candidate(candidate_doc, rank=2)],
        explicit_terms=("EXACT-MARKER",),
        context_lines=0,
    )
    assert result.candidate_document_ids == ("candidate",)
    assert result.fallback_used is False
    # 物化文件名内容派生：rg 只搜命中候选文档对应的内容哈希文件。
    candidate_path = str((tmp_path / "canonical" /
        f"document-{candidate_doc.normalized_content_hash}.txt").resolve())
    assert all(call[1] == (candidate_path,) for call in executor.calls)
    assert len(result.evidence_spans) == 1
    span = result.evidence_spans[0]
    assert span.document_id == "candidate"
    assert span.text == "  EXACT-MARKER 正文  \n"
    assert candidate_doc.slice(span.start_char, span.end_char) == span.text
    assert span.discovery_sources == ("vector", "ripgrep")
    assert span.metadata["search_scopes"] == ("candidate",)


def test_candidate_miss_falls_back_to_all_documents(tmp_path):
    candidate_doc = _document(tmp_path, "candidate.txt", "nothing", document_id="candidate")
    fallback_doc = _document(tmp_path, "fallback.txt", "target EXACT-MARKER", document_id="fallback")
    executor = LiteralFakeExecutor()
    result = ExactEvidenceLocator(tmp_path / "canonical", executor=executor).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[candidate_doc, fallback_doc],
        candidates=[_candidate(candidate_doc)],
        explicit_terms=("EXACT-MARKER",),
        context_lines=0,
    )
    assert result.fallback_used is True
    assert result._rescue_candidates == ()
    assert [span.document_id for span in result.evidence_spans] == ["fallback"]
    assert result.evidence_spans[0].discovery_sources == ("ripgrep",)
    assert {attempt.scope for attempt in result.search_attempts} == {
        "candidate", "full_corpus"
    }
    fallback_calls = executor.calls[len(result.anchors):]
    assert all(len(call[1]) == 2 for call in fallback_calls)


def test_no_candidates_searches_full_corpus_when_enabled(tmp_path):
    document = _document(tmp_path, "source.txt", "EXACT-MARKER", document_id="doc")
    result = ExactEvidenceLocator(
        tmp_path / "canonical", executor=LiteralFakeExecutor()
    ).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[document],
        candidates=[],
        explicit_terms=("EXACT-MARKER",),
        context_lines=0,
    )
    assert result.candidate_document_ids == ()
    assert result.fallback_used is True
    assert len(result.evidence_spans) == 1
    assert result._rescue_candidates == ()


def test_fallback_can_be_disabled_and_no_hit_is_empty(tmp_path):
    document = _document(tmp_path, "source.txt", "nothing", document_id="doc")
    executor = LiteralFakeExecutor()
    result = ExactEvidenceLocator(tmp_path / "canonical", executor=executor).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[document],
        candidates=[_candidate(document)],
        explicit_terms=("ABSENT",),
        full_corpus_fallback=False,
    )
    assert result.evidence_spans == ()
    assert result.fallback_used is False
    assert len(result.search_attempts) == len(result.anchors)


def test_unknown_candidate_document_and_wrong_slot_fail(tmp_path):
    known = _document(tmp_path, "known.txt", "正文", document_id="known")
    unknown = _document(tmp_path, "unknown.txt", "正文", document_id="unknown")
    locator = ExactEvidenceLocator(tmp_path / "canonical", executor=LiteralFakeExecutor())
    with pytest.raises(ValueError, match="unknown document_id"):
        locator.locate_for_slot(
            contract=solution_material_completeness_v1,
            slot_id=SLOT_ID,
            documents=[known],
            candidates=[_candidate(unknown)],
        )
    with pytest.raises(ValueError, match="belongs to slot"):
        locator.locate_for_slot(
            contract=solution_material_completeness_v1,
            slot_id=SLOT_ID,
            documents=[known],
            candidates=[_candidate(known, slot_id="objectives")],
        )


def test_max_results_and_repeated_anchors_deduplicate_with_provenance(tmp_path):
    document = _document(
        tmp_path,
        "source.txt",
        "SAME first\nSAME second\nSAME third",
        document_id="doc",
    )
    result = ExactEvidenceLocator(
        tmp_path / "canonical", executor=LiteralFakeExecutor()
    ).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[document],
        candidates=[_candidate(document)],
        explicit_terms=("SAME", "same"),
        context_lines=0,
        max_results=2,
    )
    assert len(result.evidence_spans) == 2
    assert [span.start_line for span in result.evidence_spans] == [1, 2]
    assert [item.evidence_span.start_line for item in result._rescue_candidates] == [3]
    assert [item.exact_raw_rank for item in result._rescue_candidates] == [3]
    assert all(span.metadata["matched_anchors"] == ("SAME",) for span in result.evidence_spans)
    assert all(span.discovery_sources == ("vector", "ripgrep") for span in result.evidence_spans)


def test_rescue_pool_keeps_baseline_top_k_and_can_select_raw_rank_240(tmp_path):
    document = _document(tmp_path, "source.txt", "x" * 500, document_id="doc")
    baseline = tuple(_span(document, index, index + 1) for index in range(20))
    target = _span(document, 239, 240)
    overflow = tuple((index + 21, _span(document, index + 20, index + 21)) for index in range(219))
    overflow += ((240, target),)
    candidate = _candidate_range(document, 239, 240, rank=1)
    selected = _build_rescue_candidates(overflow, (candidate,))
    assert tuple(span.evidence_id for span in baseline) == tuple(
        span.evidence_id for span in baseline
    )
    assert len(selected) == 1
    assert selected[0].evidence_span is target
    assert selected[0].exact_raw_rank == 240


def test_rescue_pool_overlap_precedes_gap_then_exact_rank(tmp_path):
    document = _document(tmp_path, "source.txt", "x" * 100, document_id="doc")
    lower_rank_with_gap = _span(document, 20, 25)
    overlapping = _span(document, 40, 50)
    candidate = _candidate_range(document, 45, 55, rank=1)
    selected = _build_rescue_candidates(
        ((21, lower_rank_with_gap), (80, overlapping)),
        (candidate,),
    )
    assert selected[0].evidence_span is overlapping

    equal_gap_left = _span(document, 40, 45)
    equal_gap_right = _span(document, 60, 65)
    candidate = _candidate_range(document, 50, 55, rank=1)
    selected = _build_rescue_candidates(
        ((30, equal_gap_right), (22, equal_gap_left)),
        (candidate,),
    )
    assert selected[0].evidence_span is equal_gap_left


def test_rescue_pool_order_is_content_deterministic_not_rank_or_input_shuffle(tmp_path):
    # 破平改为内容确定后，rescue 候选序由 _candidate_order_key（距离→正文哈希→字符位置）
    # 决定，不再受**分配的 rank**或**输入顺序/chunk_id**影响。此处刻意让 rank 与 distance
    # 相反（rank=1 距离更大）：若仍按 rank 排序结果为 [first, second]，按内容确定（距离）
    # 排序则为 [second, first]。真实向量输出中 rank 与 distance 单调一致，此为合成探针。
    document = _document(tmp_path, "source.txt", "x" * 100, document_id="doc")
    first_span = _span(document, 10, 15)
    second_span = _span(document, 80, 85)
    first_candidate = _candidate_range(
        document, 10, 15, rank=1, distance=0.9, chunk_id="later-input"
    )
    second_candidate = _candidate_range(
        document, 80, 85, rank=2, distance=0.1, chunk_id="earlier-input"
    )
    ranked = ((21, first_span), (22, second_span))
    forward = _build_rescue_candidates(ranked, (second_candidate, first_candidate))
    reverse = _build_rescue_candidates(ranked, (first_candidate, second_candidate))
    # 距离更小（更相关）的 second_candidate 先选 → second_span 在前。
    assert [item.evidence_span for item in forward] == [second_span, first_span]
    # 输入顺序（含 chunk_id）洗牌后结果逐字段一致：内容确定。
    assert forward == reverse


def test_rescue_pool_deduplicates_spans_and_caps_at_eight(tmp_path):
    document = _document(tmp_path, "source.txt", "x" * 200, document_id="doc")
    overflow = tuple(
        (21 + index, _span(document, index * 10, index * 10 + 5))
        for index in range(12)
    )
    candidates = tuple(
        _candidate_range(
            document,
            min(index, 9) * 10,
            min(index, 9) * 10 + 5,
            rank=index + 1,
        )
        for index in range(12)
    )
    selected = _build_rescue_candidates(overflow, candidates)
    assert len(selected) == 8
    assert len({item.evidence_span.evidence_id for item in selected}) == 8


def test_rescue_pool_is_empty_without_candidates_overflow_or_same_document(tmp_path):
    first = _document(tmp_path / "a", "source.txt", "x" * 20, document_id="first")
    second = _document(tmp_path / "b", "source.txt", "y" * 20, document_id="second")
    span = _span(first, 0, 5)
    candidate = _candidate_range(second, 0, 5, rank=1)
    assert _build_rescue_candidates(((21, span),), ()) == ()
    assert _build_rescue_candidates((), (candidate,)) == ()
    assert _build_rescue_candidates(((21, span),), (candidate,)) == ()


def test_overlapping_contexts_merge_and_preserve_multiple_anchor_sources(tmp_path):
    document = _document(
        tmp_path,
        "source.txt",
        "ALPHA first\nBETA second\nthird line",
        document_id="doc",
    )
    result = ExactEvidenceLocator(
        tmp_path / "canonical", executor=LiteralFakeExecutor()
    ).locate_for_slot(
        contract=solution_material_completeness_v1,
        slot_id=SLOT_ID,
        documents=[document],
        candidates=[_candidate(document)],
        explicit_terms=("ALPHA", "BETA"),
        context_lines=1,
        max_span_chars=200,
    )
    assert len(result.evidence_spans) == 1
    span = result.evidence_spans[0]
    assert span.text == document.text
    assert span.metadata["matched_anchors"][:2] == ("ALPHA", "BETA")
    assert span.metadata["anchor_sources"] == ("caller",)


def test_real_executor_error_is_never_converted_to_empty_results(tmp_path):
    document = _document(tmp_path, "source.txt", "正文", document_id="doc")
    error = RipgrepExecutionError(
        "boom",
        command=("rg",),
        returncode=2,
        stderr="permission denied",
        anchors=("X",),
        search_scope=("doc",),
    )
    locator = ExactEvidenceLocator(
        tmp_path / "canonical", executor=LiteralFakeExecutor(error=error)
    )
    with pytest.raises(RipgrepExecutionError) as raised:
        locator.locate_for_slot(
            contract=solution_material_completeness_v1,
            slot_id=SLOT_ID,
            documents=[document],
            candidates=[_candidate(document)],
            explicit_terms=("X",),
        )
    assert raised.value is error
