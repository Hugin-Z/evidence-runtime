from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence.contract import solution_material_completeness_v1
from evidence.documents import NormalizedDocument
from evidence.exact_locator import ExactLocationResult
from evidence.gap_checker import (
    DEFAULT_CONTRACT_PATH,
    EvidenceGapChecker,
    _attempt_span_rescue,
    _post_swap_guard_failures,
    _verification_safety_signature,
    build_argument_parser,
    load_normalized_document_manifest,
    main,
)
from evidence.gap_report import GapReport, GapReportBuilder
from evidence.types import EvidenceSpan, SlotStatus, VectorCandidate
from evidence.vector_discovery import VectorIndexManifest
from evidence.verifier import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    DeterministicSlotVerifier,
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
)


def _write_manifest(tmp_path, documents):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _entry(path="materials/project.md", **changes):
    value = {
        "document_id": "doc-001",
        "title": "项目材料",
        "source_path": path,
        "source_type": "md",
        "normalized_path": path,
        "metadata": {"domain": "demo"},
    }
    value.update(changes)
    return value


@pytest.mark.parametrize(
    ("filename", "source_type"),
    [("project.md", "md"), ("project.markdown", "md"), ("project.txt", "txt")],
)
def test_manifest_loads_supported_normalized_files(tmp_path, filename, source_type):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / filename).write_text("第一行\n第二行", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [_entry(f"materials/{filename}", source_type=source_type)],
    )
    (document,) = load_normalized_document_manifest(manifest)
    assert document.document_id == "doc-001"
    assert document.normalized_path == f"materials/{filename}"
    assert not Path(document.normalized_path).is_absolute()
    assert document.source_type == source_type


def test_manifest_normalizes_bom_and_crlf_without_losing_source_hash(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    path = materials / "project.md"
    path.write_bytes(b"\xef\xbb\xbfline1\r\nline2\r\n")
    manifest = _write_manifest(tmp_path, [_entry()])
    (document,) = load_normalized_document_manifest(manifest)
    assert document.text == "line1\nline2\n"
    assert document.source_content_hash != document.normalized_content_hash


def test_manifest_document_id_and_hash_are_stable(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "project.md").write_text("稳定正文", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [_entry()])
    first = load_normalized_document_manifest(manifest)[0]
    second = load_normalized_document_manifest(manifest)[0]
    assert first.document_id == second.document_id == "doc-001"
    assert first.normalized_content_hash == second.normalized_content_hash


def test_manifest_rejects_duplicate_document_ids(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "project.md").write_text("x", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [_entry(), _entry(title="duplicate")])
    with pytest.raises(ValueError, match="duplicate document_id"):
        load_normalized_document_manifest(manifest)


def test_manifest_rejects_missing_file(tmp_path):
    manifest = _write_manifest(tmp_path, [_entry()])
    with pytest.raises(ValueError, match="does not exist"):
        load_normalized_document_manifest(manifest)


def test_manifest_rejects_unsupported_extension(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "project.pdf").write_bytes(b"not-pdf")
    manifest = _write_manifest(
        tmp_path,
        [_entry("materials/project.pdf", source_type="pdf")],
    )
    with pytest.raises(ValueError, match="unsupported extension"):
        load_normalized_document_manifest(manifest)


def test_manifest_rejects_source_type_extension_conflict(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "project.txt").write_text("x", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [_entry("materials/project.txt")])
    with pytest.raises(ValueError, match="conflicts with extension"):
        load_normalized_document_manifest(manifest)


def test_manifest_rejects_absolute_and_parent_escape_paths(tmp_path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("x", encoding="utf-8")
    absolute = _write_manifest(
        tmp_path,
        [_entry(str(outside), normalized_path=str(outside))],
    )
    with pytest.raises(ValueError, match="must be relative"):
        load_normalized_document_manifest(absolute)
    escaped = _write_manifest(
        tmp_path,
        [_entry("../outside.md", normalized_path="../outside.md")],
    )
    with pytest.raises(ValueError, match="stay within"):
        load_normalized_document_manifest(escaped)


def test_manifest_rejects_unknown_or_missing_schema_fields(tmp_path):
    with pytest.raises(ValueError, match="fields must be exactly"):
        load_normalized_document_manifest(
            _write_manifest(tmp_path, [_entry(extra="x")])
        )
    manifest = tmp_path / "bad.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fields must be exactly"):
        load_normalized_document_manifest(manifest)


def _document(tmp_path):
    path = tmp_path / "input.md"
    path.write_text("项目材料正文。", encoding="utf-8")
    return NormalizedDocument.from_path(
        path,
        document_id="doc-input",
        source_path="materials/input.md",
        normalized_path="materials/input.md",
    )


RESCUE_SLOT_ID = "solution_content_and_capabilities"


def _rescue_fixture(tmp_path, *, count=22):
    lines = tuple(f"evidence segment {index:02d}" for index in range(count))
    path = tmp_path / "rescue.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    document = NormalizedDocument.from_path(
        path,
        document_id="doc-rescue",
        source_path="materials/rescue.md",
        normalized_path="materials/rescue.md",
    )
    spans = []
    cursor = 0
    for line in lines:
        spans.append(
            EvidenceSpan.from_document(
                document,
                cursor,
                cursor + len(line),
                discovery_sources=("vector", "ripgrep"),
            )
        )
        cursor += len(line) + 1
    return document, tuple(spans)


def _fact(span, *, value="positive"):
    return {
        "subject": "subject",
        "predicate": "support",
        "value": value,
        "normalized_value": value,
        "evidence_id": span.evidence_id,
    }


def _slot_result(
    spans,
    *,
    status,
    covered=(),
    reason_codes=(),
    facts=(),
    conflicts=(),
    quantitative=(),
    semantic_gate=None,
    missing_source_context=(),
):
    spans = tuple(spans)
    slot = solution_material_completeness_v1.get_slot(RESCUE_SLOT_ID)
    covered = frozenset(covered)
    coverage = tuple(
        FieldCoverage(
            field_id,
            field_id in covered,
            (spans[0].evidence_id,) if field_id in covered else (),
            None if field_id in covered else "缺少字段",
        )
        for field_id in slot.required_fields
    )
    gate = semantic_gate or {
        "applicable": False,
        "evaluated": False,
        "passed": None,
        "failed_aspects": (),
        "reason": "not applicable",
        "preliminary_status": status.value,
    }
    return SlotVerificationResult(
        contract_id=solution_material_completeness_v1.contract_id,
        slot_id=RESCUE_SLOT_ID,
        status=status,
        evidence_ids=tuple(span.evidence_id for span in spans),
        field_coverage=coverage,
        reason_codes=tuple(reason_codes),
        reason="test result",
        conflicts=tuple(conflicts),
        trace={
            "covered_fields": tuple(item.field_id for item in coverage if item.covered),
            "missing_fields": tuple(item.field_id for item in coverage if not item.covered),
            "detected_facts": tuple(facts),
            "quantitative_claims": tuple(quantitative),
            "missing_source_context_evidence_ids": tuple(missing_source_context),
            "semantic_gate": gate,
        },
    )


def _filled_result(spans, **changes):
    fields = solution_material_completeness_v1.get_slot(
        RESCUE_SLOT_ID
    ).required_fields
    return _slot_result(
        spans,
        status=SlotStatus.FILLED,
        covered=fields,
        **changes,
    )


def _weak_result(spans, **changes):
    first_field = solution_material_completeness_v1.get_slot(
        RESCUE_SLOT_ID
    ).required_fields[0]
    return _slot_result(
        spans,
        status=SlotStatus.WEAK,
        covered=(first_field,),
        reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
        **changes,
    )


def _conflict_for(spans):
    positive = DetectedFact(
        subject="subject",
        predicate="support",
        value="支持",
        normalized_value="positive",
        evidence_id=spans[0].evidence_id,
    )
    negative = DetectedFact(
        subject="subject",
        predicate="support",
        value="不支持",
        normalized_value="negative",
        evidence_id=spans[1].evidence_id,
    )
    return DetectedConflict(
        conflict_type="opposite_assertion",
        subject="subject",
        predicate="support",
        facts=(positive, negative),
        evidence_ids=(positive.evidence_id, negative.evidence_id),
        reason="test conflict",
    )


def _rescue_candidate(span, *, rank=21):
    return SimpleNamespace(evidence_span=span, exact_raw_rank=rank)


class _ScriptedSlotVerifier:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def verify_slot(self, *, contract, slot_id, evidence_spans):
        assert contract is solution_material_completeness_v1
        assert slot_id == RESCUE_SLOT_ID
        spans = tuple(evidence_spans)
        self.calls.append(tuple(span.evidence_id for span in spans))
        return self.callback(spans)


@pytest.mark.parametrize(
    ("initial_kind", "with_rescue", "expected_attempt"),
    (
        ("filled", True, False),
        ("conflicting", True, False),
        ("weak", True, True),
        ("missing", True, True),
        ("weak-no-missing", True, False),
        ("weak", False, False),
    ),
)
def test_span_rescue_trigger_conditions(
    tmp_path,
    initial_kind,
    with_rescue,
    expected_attempt,
):
    _, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    if initial_kind == "filled":
        initial = _filled_result(baseline)
    elif initial_kind == "conflicting":
        conflict = _conflict_for(baseline)
        initial = _slot_result(
            baseline,
            status=SlotStatus.CONFLICTING,
            covered=(),
            reason_codes=(VerificationReasonCode.CONFLICTING_FACTS,),
            facts=tuple(
                {
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "value": item.value,
                    "normalized_value": item.normalized_value,
                    "evidence_id": item.evidence_id,
                }
                for item in conflict.facts
            ),
            conflicts=(conflict,),
        )
    elif initial_kind == "missing":
        initial = _slot_result(
            baseline,
            status=SlotStatus.MISSING,
            covered=(),
            reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
        )
    elif initial_kind == "weak-no-missing":
        fields = solution_material_completeness_v1.get_slot(
            RESCUE_SLOT_ID
        ).required_fields
        initial = _slot_result(
            baseline,
            status=SlotStatus.WEAK,
            covered=fields,
            reason_codes=(VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM,),
        )
    else:
        initial = _weak_result(baseline)

    def callback(current):
        if rescue.evidence_id in {item.evidence_id for item in current}:
            return _filled_result(current)
        if initial.status is SlotStatus.MISSING:
            return _slot_result(
                current,
                status=SlotStatus.MISSING,
                covered=(),
                reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
            )
        return _weak_result(current)

    verifier = _ScriptedSlotVerifier(callback)
    _, _, trace = _attempt_span_rescue(
        verifier,
        contract=solution_material_completeness_v1,
        slot_id=RESCUE_SLOT_ID,
        baseline_spans=baseline,
        rescue_candidates=(_rescue_candidate(rescue),) if with_rescue else (),
        initial=initial,
    )
    assert trace.rescue_attempted is expected_attempt
    assert bool(verifier.calls) is expected_attempt


def test_safety_signature_is_order_stable_and_compares_all_guarded_outputs(tmp_path):
    _, spans = _rescue_fixture(tmp_path)
    baseline = spans[:20]
    facts = (_fact(spans[0], value="a"), _fact(spans[1], value="b"))
    first = _weak_result(baseline, facts=facts)
    reordered = _weak_result(baseline, facts=tuple(reversed(facts)))
    assert _verification_safety_signature(first) == _verification_safety_signature(
        reordered
    )

    variants = (
        _slot_result(
            baseline,
            status=SlotStatus.MISSING,
            covered=(
                solution_material_completeness_v1.get_slot(
                    RESCUE_SLOT_ID
                ).required_fields[0],
            ),
            reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
            facts=facts,
        ),
        _slot_result(
            baseline,
            status=SlotStatus.WEAK,
            covered=(),
            reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
            facts=facts,
        ),
        _slot_result(
            baseline,
            status=SlotStatus.WEAK,
            covered=(
                solution_material_completeness_v1.get_slot(
                    RESCUE_SLOT_ID
                ).required_fields[0],
            ),
            reason_codes=(VerificationReasonCode.PROMOTIONAL_ONLY,),
            facts=facts,
        ),
        _weak_result(baseline, facts=(_fact(spans[0], value="changed"),)),
        _weak_result(
            baseline,
            facts=facts,
            quantitative=({"role": "configuration", "verified": True},),
        ),
        _weak_result(
            baseline,
            facts=facts,
            semantic_gate={
                "applicable": True,
                "evaluated": False,
                "passed": None,
                "failed_aspects": (),
                "reason": "changed",
                "preliminary_status": "weak",
            },
        ),
        _weak_result(
            baseline,
            facts=facts,
            missing_source_context=(spans[0].evidence_id,),
        ),
    )
    signature = _verification_safety_signature(first)
    assert all(_verification_safety_signature(item) != signature for item in variants)


def test_safety_signature_detects_conflict_detail_changes(tmp_path):
    _, spans = _rescue_fixture(tmp_path)
    baseline = spans[:20]
    first_conflict = _conflict_for((spans[0], spans[1]))
    second_conflict = _conflict_for((spans[2], spans[3]))
    first = _slot_result(
        baseline,
        status=SlotStatus.CONFLICTING,
        covered=(),
        reason_codes=(VerificationReasonCode.CONFLICTING_FACTS,),
        conflicts=(first_conflict,),
    )
    second = _slot_result(
        baseline,
        status=SlotStatus.CONFLICTING,
        covered=(),
        reason_codes=(VerificationReasonCode.CONFLICTING_FACTS,),
        conflicts=(second_conflict,),
    )
    assert _verification_safety_signature(first) != _verification_safety_signature(
        second
    )


@pytest.mark.parametrize("initial_status", (SlotStatus.WEAK, SlotStatus.MISSING))
def test_post_swap_guard_accepts_only_complete_field_safe_improvement(
    tmp_path,
    initial_status,
):
    _, spans = _rescue_fixture(tmp_path)
    baseline = spans[:20]
    if initial_status is SlotStatus.WEAK:
        initial = _weak_result(baseline, facts=(_fact(spans[0]),))
    else:
        initial = _slot_result(
            baseline,
            status=SlotStatus.MISSING,
            covered=(),
            reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
        )
    final_facts = tuple(initial.trace["detected_facts"]) + (_fact(spans[20]),)
    final = _filled_result(baseline[:-1] + (spans[20],), facts=final_facts)
    assert _post_swap_guard_failures(initial, final) == ()


@pytest.mark.parametrize(
    "failure_kind",
    (
        "status",
        "field",
        "fact",
        "conflict",
        "reason",
        "quantitative",
        "semantic",
        "source-context",
    ),
)
def test_post_swap_guard_rejects_each_regression(tmp_path, failure_kind):
    _, spans = _rescue_fixture(tmp_path)
    baseline = spans[:20]
    initial = _weak_result(baseline, facts=(_fact(spans[0]),))
    final_spans = baseline[:-1] + (spans[20],)
    final_kwargs = {"facts": tuple(initial.trace["detected_facts"])}
    if failure_kind == "status":
        final = _weak_result(final_spans, facts=final_kwargs["facts"])
    elif failure_kind == "field":
        final = _slot_result(
            final_spans,
            status=SlotStatus.WEAK,
            covered=(),
            reason_codes=(VerificationReasonCode.MISSING_REQUIRED_FIELD,),
            facts=final_kwargs["facts"],
        )
    elif failure_kind == "fact":
        final = _filled_result(final_spans, facts=())
    elif failure_kind == "conflict":
        conflict = _conflict_for(final_spans)
        final = _slot_result(
            final_spans,
            status=SlotStatus.CONFLICTING,
            covered=(),
            reason_codes=(VerificationReasonCode.CONFLICTING_FACTS,),
            facts=tuple(
                {
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "value": item.value,
                    "normalized_value": item.normalized_value,
                    "evidence_id": item.evidence_id,
                }
                for item in conflict.facts
            ),
            conflicts=(conflict,),
        )
    elif failure_kind == "reason":
        final = _filled_result(
            final_spans,
            facts=final_kwargs["facts"],
            reason_codes=(VerificationReasonCode.PROMOTIONAL_ONLY,),
        )
    elif failure_kind == "quantitative":
        final = _filled_result(
            final_spans,
            facts=final_kwargs["facts"],
            quantitative=(
                {
                    "evidence_id": spans[20].evidence_id,
                    "role": "outcome_claim",
                    "requires_outcome_support": True,
                    "verified": False,
                },
            ),
        )
    elif failure_kind == "semantic":
        final = _filled_result(
            final_spans,
            facts=final_kwargs["facts"],
            semantic_gate={
                "applicable": True,
                "evaluated": True,
                "passed": False,
                "failed_aspects": ("project_specific_target_content",),
                "reason": "failed",
                "preliminary_status": "filled",
            },
        )
    else:
        final = _filled_result(
            final_spans,
            facts=final_kwargs["facts"],
            missing_source_context=(spans[20].evidence_id,),
        )
    assert _post_swap_guard_failures(initial, final)


def test_single_swap_preserves_order_count_identity_and_stops_after_adoption(tmp_path):
    document, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    initial = _weak_result(baseline, facts=(_fact(spans[0]),))

    def callback(current):
        if rescue.evidence_id in {item.evidence_id for item in current}:
            return _filled_result(current, facts=tuple(initial.trace["detected_facts"]))
        return _weak_result(current, facts=tuple(initial.trace["detected_facts"]))

    verifier = _ScriptedSlotVerifier(callback)
    final_spans, final, trace = _attempt_span_rescue(
        verifier,
        contract=solution_material_completeness_v1,
        slot_id=RESCUE_SLOT_ID,
        baseline_spans=baseline,
        rescue_candidates=(_rescue_candidate(rescue), _rescue_candidate(spans[21], rank=22)),
        initial=initial,
    )
    assert final.status is SlotStatus.FILLED
    assert trace.rescue_applied is True and trace.swap_attempt_count == 1
    assert trace.evicted_evidence_id == baseline[-1].evidence_id
    assert trace.inserted_evidence_id == rescue.evidence_id
    assert len(final_spans) == len(baseline) == 20
    assert final_spans[:-1] == baseline[:-1]
    assert final_spans[-1] is rescue
    assert document.slice(rescue.start_char, rescue.end_char) == rescue.text


def test_non_neutral_evictions_stop_at_three_and_restore_exact_baseline(tmp_path):
    _, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    initial = _weak_result(baseline)

    def callback(current):
        return _weak_result(current, facts=(_fact(current[0]),))

    verifier = _ScriptedSlotVerifier(callback)
    final_spans, final, trace = _attempt_span_rescue(
        verifier,
        contract=solution_material_completeness_v1,
        slot_id=RESCUE_SLOT_ID,
        baseline_spans=baseline,
        rescue_candidates=(_rescue_candidate(rescue),),
        initial=initial,
    )
    assert trace.eviction_candidates_checked == 3
    assert trace.neutral_eviction_count == 0
    assert trace.swap_attempt_count == 0
    assert trace.guard_failure_reasons == ("EVICTION_NOT_NEUTRAL",)
    assert final_spans is baseline and final is initial


@pytest.mark.parametrize("regression", ("quantitative-insert", "semantic-delete"))
def test_b1_regressions_are_rejected_and_baseline_is_restored(tmp_path, regression):
    _, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    initial = _weak_result(baseline)

    def callback(current):
        ids = {item.evidence_id for item in current}
        if rescue.evidence_id in ids:
            return _filled_result(
                current,
                quantitative=(
                    {
                        "evidence_id": rescue.evidence_id,
                        "role": "outcome_claim",
                        "requires_outcome_support": True,
                        "verified": False,
                    },
                ),
                reason_codes=(VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM,),
            )
        if regression == "semantic-delete" and len(current) == 19:
            return _weak_result(
                current,
                semantic_gate={
                    "applicable": True,
                    "evaluated": False,
                    "passed": None,
                    "failed_aspects": (),
                    "reason": "changed by deletion",
                    "preliminary_status": "weak",
                },
            )
        return _weak_result(current)

    verifier = _ScriptedSlotVerifier(callback)
    final_spans, final, trace = _attempt_span_rescue(
        verifier,
        contract=solution_material_completeness_v1,
        slot_id=RESCUE_SLOT_ID,
        baseline_spans=baseline,
        rescue_candidates=(_rescue_candidate(rescue),),
        initial=initial,
    )
    assert trace.rescue_applied is False
    assert final_spans is baseline and final is initial
    if regression == "quantitative-insert":
        assert "NEW_REASON_CODE" in trace.guard_failure_reasons
        assert "QUANTITATIVE_RISK" in trace.guard_failure_reasons
    else:
        assert trace.swap_attempt_count == 0


def test_span_rescue_trace_is_json_serializable_and_repeatable(tmp_path):
    _, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    initial = _weak_result(baseline)

    def run_once():
        verifier = _ScriptedSlotVerifier(
            lambda current: (
                _filled_result(current)
                if rescue.evidence_id in {item.evidence_id for item in current}
                else _weak_result(current)
            )
        )
        return _attempt_span_rescue(
            verifier,
            contract=solution_material_completeness_v1,
            slot_id=RESCUE_SLOT_ID,
            baseline_spans=baseline,
            rescue_candidates=(_rescue_candidate(rescue),),
            initial=initial,
        )

    first = run_once()
    second = run_once()
    assert first[0] == second[0] and first[1] == second[1] and first[2] == second[2]
    json.dumps(first[2].to_dict(), ensure_ascii=False)


def test_span_rescue_verifier_exception_propagates(tmp_path):
    _, spans = _rescue_fixture(tmp_path)
    baseline, rescue = spans[:20], spans[20]
    initial = _weak_result(baseline)

    def fail(current):
        raise RuntimeError("verifier backend failed")

    with pytest.raises(RuntimeError, match="verifier backend failed"):
        _attempt_span_rescue(
            _ScriptedSlotVerifier(fail),
            contract=solution_material_completeness_v1,
            slot_id=RESCUE_SLOT_ID,
            baseline_spans=baseline,
            rescue_candidates=(_rescue_candidate(rescue),),
            initial=initial,
        )


class _FakeVector:
    def __init__(self, events, *, include_first_candidate=False, fail=False):
        self.events = events
        self.include_first_candidate = include_first_candidate
        self.fail = fail

    def index_documents(self, documents, *, max_chars):
        self.events.append(("index", tuple(item.document_id for item in documents)))
        if self.fail:
            raise RuntimeError("embedding backend failed")
        ordered = tuple(
            sorted(
                (item.document_id, item.normalized_content_hash)
                for item in documents
            )
        )
        return VectorIndexManifest(
            fingerprint="a" * 64,
            collection_name="m1-test",
            index_schema_version="test-v1",
            ordered_documents=ordered,
            chunker_version="test-v1",
            max_chars=max_chars,
            embedding_backend="injected:test",
            embedding_model="fake-model",
            embedding_dimension=2,
            chunk_count=len(documents),
        )

    def discover_contract(
        self, contract, *, top_k_per_slot, manifest, allowed_document_ids=None
    ):
        self.events.append(
            ("discover", manifest.fingerprint, top_k_per_slot, allowed_document_ids)
        )
        result = {slot.slot_id: () for slot in contract.slots}
        if self.include_first_candidate:
            slot = contract.slots[0]
            result[slot.slot_id] = (
                VectorCandidate(
                    slot_id=slot.slot_id,
                    query="测试查询",
                    document_id="doc-input",
                    chunk_id="chunk-1",
                    text="项目",
                    start_char=0,
                    end_char=2,
                    start_line=1,
                    end_line=1,
                    rank=1,
                    distance=0.2,
                    discovery_source="vector",
                ),
            )
        return result


class _FakeLocator:
    def __init__(self, events):
        self.events = events

    def locate_for_slot(self, **kwargs):
        candidates = tuple(kwargs["candidates"])
        slot_id = kwargs["slot_id"]
        self.events.append(("locate", slot_id, len(candidates)))
        return ExactLocationResult(
            slot_id=slot_id,
            anchors=(),
            evidence_spans=(),
            candidate_document_ids=tuple(
                dict.fromkeys(item.document_id for item in candidates)
            ),
            fallback_used=not candidates,
            search_attempts=(),
            rg_source="fake",
        )


class _InvalidSpanLocator(_FakeLocator):
    def locate_for_slot(self, **kwargs):
        result = super().locate_for_slot(**kwargs)
        candidate = VectorCandidate(
            slot_id=kwargs["slot_id"],
            query="测试",
            document_id="doc-input",
            chunk_id=f"invalid-{kwargs['slot_id']}",
            text="项目",
            start_char=0,
            end_char=2,
            start_line=1,
            end_line=1,
            rank=1,
            distance=0.1,
            discovery_source="vector",
        )
        return ExactLocationResult(
            slot_id=result.slot_id,
            anchors=result.anchors,
            evidence_spans=(candidate,),
            candidate_document_ids=result.candidate_document_ids,
            fallback_used=result.fallback_used,
            search_attempts=result.search_attempts,
            rg_source=result.rg_source,
        )


class _InspectingVerifier:
    def __init__(self, events, *, weak_first=False):
        self.events = events
        self.weak_first = weak_first

    def verify_contract(self, *, contract, evidence_by_slot):
        self.events.append(("verify", tuple(evidence_by_slot)))
        assert all(
            all(not isinstance(value, VectorCandidate) for value in spans)
            for spans in evidence_by_slot.values()
        )
        results = []
        for index, slot in enumerate(contract.slots):
            status = SlotStatus.WEAK if self.weak_first and index == 0 else SlotStatus.MISSING
            results.append(
                SlotVerificationResult(
                    contract_id=contract.contract_id,
                    slot_id=slot.slot_id,
                    status=status,
                    evidence_ids=(),
                    field_coverage=tuple(
                        FieldCoverage(field_id, False, (), "缺少字段")
                        for field_id in slot.required_fields
                    ),
                    reason_codes=(
                        VerificationReasonCode.MISSING_REQUIRED_FIELD
                        if status is SlotStatus.WEAK
                        else VerificationReasonCode.NO_EVIDENCE,
                    ),
                    reason="测试判定",
                    conflicts=(),
                    trace={},
                )
            )
        return ContractVerificationResult(
            contract_id=contract.contract_id,
            status="incomplete",
            slots=tuple(results),
        )


class _RecordingBuilder:
    def __init__(self, events):
        self.events = events
        self.real = GapReportBuilder()

    def build(self, **kwargs):
        self.events.append(("build", kwargs["verification"].status))
        return self.real.build(**kwargs)


def _checker(events, **vector_options):
    return EvidenceGapChecker(
        vector_discovery=_FakeVector(events, **vector_options),
        exact_locator=_FakeLocator(events),
        verifier=_InspectingVerifier(events, weak_first=True),
        report_builder=_RecordingBuilder(events),
    )


def test_pipeline_uses_post_swap_verification_and_final_evidence_without_reordering(
    tmp_path,
):
    lines = (
        "建设内容包括数据平台模块。",
        "建设内容包括审批功能模块。",
        "系统提供数据汇聚能力。",
        "平台包含统一管理模块。",
        "该模块用于满足审批业务需求。",
    )
    path = tmp_path / "integration.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    document = NormalizedDocument.from_path(
        path,
        document_id="doc-input",
        source_path="materials/integration.md",
        normalized_path="materials/integration.md",
    )
    spans = []
    cursor = 0
    for line in lines:
        spans.append(EvidenceSpan.from_document(document, cursor, cursor + len(line)))
        cursor += len(line) + 1
    baseline = tuple(spans[:4])
    rescue = spans[4]

    class Locator:
        def locate_for_slot(self, **kwargs):
            slot_id = kwargs["slot_id"]
            is_target = slot_id == RESCUE_SLOT_ID
            return ExactLocationResult(
                slot_id=slot_id,
                anchors=(),
                evidence_spans=baseline if is_target else (),
                candidate_document_ids=(),
                fallback_used=True,
                search_attempts=(),
                rg_source="fake",
                _rescue_candidates=(
                    _rescue_candidate(rescue, rank=94),
                )
                if is_target
                else (),
            )

    checker = EvidenceGapChecker(
        vector_discovery=_FakeVector([]),
        exact_locator=Locator(),
        verifier=DeterministicSlotVerifier(),
    )
    run = checker.run(
        contract=solution_material_completeness_v1,
        documents=(document,),
        max_chars=800,
        top_k_per_slot=8,
        context_lines=2,
        max_span_chars=1200,
        max_results_per_slot=20,
    )
    result = next(
        item for item in run.verification.slots if item.slot_id == RESCUE_SLOT_ID
    )
    slot_trace = run.report.trace["per_slot"][RESCUE_SLOT_ID]
    assert result.status is SlotStatus.FILLED
    assert run.exact_results_by_slot[RESCUE_SLOT_ID].evidence_spans == baseline
    assert slot_trace["initial_verification_status"] == "weak"
    assert slot_trace["final_verification_status"] == "filled"
    assert slot_trace["rescue_applied"] is True
    assert slot_trace["inserted_evidence_id"] == rescue.evidence_id
    assert tuple(item.slot_id for item in run.report.slots) == (
        solution_material_completeness_v1.slot_ids
    )


def test_pipeline_runs_all_slots_in_strict_stage_order(tmp_path):
    events = []
    checker = _checker(events, include_first_candidate=True)
    document = _document(tmp_path)
    run = checker.run(
        contract=solution_material_completeness_v1,
        documents=(document,),
        max_chars=800,
        top_k_per_slot=8,
        context_lines=2,
        max_span_chars=1200,
        max_results_per_slot=20,
    )
    assert [item[0] for item in events] == [
        "index",
        "discover",
        *("locate" for _ in solution_material_completeness_v1.slots),
        "verify",
        "build",
    ]
    assert tuple(run.candidates_by_slot) == solution_material_completeness_v1.slot_ids
    assert tuple(run.exact_results_by_slot) == solution_material_completeness_v1.slot_ids


def test_pipeline_empty_vector_results_still_use_exact_fallback(tmp_path):
    events = []
    checker = _checker(events)
    run = checker.run(
        contract=solution_material_completeness_v1,
        documents=(_document(tmp_path),),
        max_chars=500,
        top_k_per_slot=3,
        context_lines=0,
        max_span_chars=300,
        max_results_per_slot=4,
    )
    locate_events = [item for item in events if item[0] == "locate"]
    assert len(locate_events) == 8
    assert all(item[2] == 0 for item in locate_events)
    assert all(item.fallback_used for item in run.exact_results_by_slot.values())
    assert all(item.status is SlotStatus.MISSING for item in run.verification.slots[1:])


def test_pipeline_trace_has_only_auditable_counts_and_configuration(tmp_path):
    events = []
    document = _document(tmp_path)
    run = _checker(events).run(
        contract=solution_material_completeness_v1,
        documents=(document,),
        max_chars=321,
        top_k_per_slot=4,
        context_lines=1,
        max_span_chars=654,
        max_results_per_slot=7,
    )
    trace = run.report.trace
    assert trace["document_ids"] == [document.document_id]
    assert trace["document_count"] == 1
    assert trace["vector_index_fingerprint"] == "a" * 64
    assert trace["max_chars"] == 321
    assert trace["top_k_per_slot"] == 4
    assert trace["context_lines"] == 1
    assert trace["max_span_chars"] == 654
    assert trace["max_results_per_slot"] == 7
    assert len(trace["per_slot"]) == 8
    assert "embedding" not in json.dumps(trace, ensure_ascii=False).lower()


def test_pipeline_backend_errors_propagate_without_material_downgrade(tmp_path):
    checker = _checker([], fail=True)
    with pytest.raises(RuntimeError, match="backend failed"):
        checker.run(
            contract=solution_material_completeness_v1,
            documents=(_document(tmp_path),),
            max_chars=800,
            top_k_per_slot=8,
            context_lines=2,
            max_span_chars=1200,
            max_results_per_slot=20,
        )


def test_pipeline_blocks_vector_candidate_at_exact_to_verifier_boundary(tmp_path):
    events = []
    checker = EvidenceGapChecker(
        vector_discovery=_FakeVector(events),
        exact_locator=_InvalidSpanLocator(events),
        verifier=_InspectingVerifier(events),
    )
    with pytest.raises(RuntimeError, match="non-EvidenceSpan"):
        checker.run(
            contract=solution_material_completeness_v1,
            documents=(_document(tmp_path),),
            max_chars=800,
            top_k_per_slot=8,
            context_lines=2,
            max_span_chars=1200,
            max_results_per_slot=20,
        )
    assert not any(item[0] == "verify" for item in events)


def test_pipeline_does_not_mutate_inputs_and_contract_controls_slot_order(tmp_path):
    document = _document(tmp_path)
    documents = [document]
    before = list(documents)
    run = _checker([]).run(
        contract=solution_material_completeness_v1,
        documents=documents,
        max_chars=800,
        top_k_per_slot=8,
        context_lines=2,
        max_span_chars=1200,
        max_results_per_slot=20,
    )
    assert documents == before
    assert tuple(item.slot_id for item in run.report.slots) == (
        solution_material_completeness_v1.slot_ids
    )
    assert run.report.status == "incomplete"


def test_pipeline_document_input_order_does_not_change_contract_or_trace_order(tmp_path):
    first = _document(tmp_path)
    second_path = tmp_path / "second.md"
    second_path.write_text("另一份材料。", encoding="utf-8")
    second = NormalizedDocument.from_path(
        second_path,
        document_id="doc-a",
        source_path="materials/second.md",
        normalized_path="materials/second.md",
    )
    runs = []
    for documents in ((first, second), (second, first)):
        runs.append(
            _checker([]).run(
                contract=solution_material_completeness_v1,
                documents=documents,
                max_chars=800,
                top_k_per_slot=8,
                context_lines=2,
                max_span_chars=1200,
                max_results_per_slot=20,
            )
        )
    assert [item.slot_id for item in runs[0].report.slots] == [
        item.slot_id for item in runs[1].report.slots
    ]
    assert runs[0].report.trace["document_ids"] == runs[1].report.trace["document_ids"]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_chars", 0),
        ("top_k_per_slot", 0),
        ("context_lines", -1),
        ("max_span_chars", 0),
        ("max_results_per_slot", 0),
    ],
)
def test_pipeline_rejects_invalid_numeric_configuration(tmp_path, keyword, value):
    arguments = {
        "contract": solution_material_completeness_v1,
        "documents": (_document(tmp_path),),
        "max_chars": 800,
        "top_k_per_slot": 8,
        "context_lines": 2,
        "max_span_chars": 1200,
        "max_results_per_slot": 20,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        _checker([]).run(**arguments)


def _minimal_report(status):
    return GapReport(
        contract_id=solution_material_completeness_v1.contract_id,
        contract_version=solution_material_completeness_v1.version,
        status=status,
        ready=status == "complete",
        slots=(),
        evidence=(),
        pending=(),
        trace={},
    )


class _CliChecker:
    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error

    def run(self, **kwargs):
        if self.error:
            raise self.error
        return SimpleNamespace(report=self.report)


def _cli_inputs(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "project.md").write_text("材料", encoding="utf-8")
    return _write_manifest(tmp_path, [_entry()]), tmp_path / "output"


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("complete", 0), ("incomplete", 1), ("conflicted", 1)],
)
def test_cli_exit_codes_and_writes_both_reports(tmp_path, capsys, status, exit_code):
    manifest, output = _cli_inputs(tmp_path)
    code = main(
        ["--manifest", str(manifest), "--output-dir", str(output)],
        checker_factory=lambda settings: _CliChecker(_minimal_report(status)),
    )
    assert code == exit_code
    assert (output / "gap-report.json").is_file()
    assert (output / "gap-report.md").is_file()
    stdout = capsys.readouterr().out
    assert f"status: {status}" in stdout
    assert "filled=" in stdout and "pending:" in stdout
    assert "gap-report.json" in stdout and "gap-report.md" in stdout


def test_cli_runtime_error_is_exit_two_and_redacts_embedding_secret(
    tmp_path,
    capsys,
    monkeypatch,
):
    manifest, output = _cli_inputs(tmp_path)
    secret = "super-secret-token"
    monkeypatch.setattr("evidence.gap_checker.DEFAULT_SETTINGS.embedding_api_key", secret)
    code = main(
        ["--manifest", str(manifest), "--output-dir", str(output)],
        checker_factory=lambda settings: _CliChecker(
            error=RuntimeError(f"backend rejected {secret}")
        ),
    )
    stderr = capsys.readouterr().err
    assert code == 2
    assert secret not in stderr
    assert "***" in stderr


def test_cli_input_error_is_exit_two(tmp_path, capsys):
    code = main(
        [
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_default_contract_is_stable_and_exists():
    parser = build_argument_parser()
    contract_action = next(item for item in parser._actions if item.dest == "contract")
    assert Path(contract_action.default) == DEFAULT_CONTRACT_PATH
    assert DEFAULT_CONTRACT_PATH.is_absolute()
    assert DEFAULT_CONTRACT_PATH.is_file()
