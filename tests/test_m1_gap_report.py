from __future__ import annotations

import json

import pytest

from evidence.contract import solution_material_completeness_v1
from evidence.documents import NormalizedDocument
from evidence.gap_report import (
    EvidenceReference,
    GapReportBuilder,
    PendingActionType,
    gap_report_to_dict,
    gap_report_to_json,
    render_gap_report_markdown,
    write_gap_report_json,
    write_gap_report_markdown,
)
from evidence.types import EvidenceSpan, SlotStatus
from evidence.verifier import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
)


def _span(tmp_path, name: str, text: str) -> EvidenceSpan:
    path = tmp_path / f"{name}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    document = NormalizedDocument.from_path(
        path,
        document_id=f"doc-{name}",
        source_path=f"materials/{name}.md",
        normalized_path=f"materials/{name}.md",
    )
    return EvidenceSpan.from_document(
        document,
        0,
        len(document.text),
        discovery_sources=("exact",),
    )


def _verification_bundle(tmp_path, overrides=None):
    contract = solution_material_completeness_v1
    overrides = overrides or {}
    results = []
    evidence_by_slot = {}
    for index, slot in enumerate(contract.slots):
        status = overrides.get(slot.slot_id, SlotStatus.FILLED)
        first = _span(tmp_path, f"{index}-a", f"{slot.name}的中文证据。")
        spans = (first,)
        conflicts = ()
        if status is SlotStatus.MISSING:
            spans = ()
        elif status is SlotStatus.CONFLICTING:
            second = _span(tmp_path, f"{index}-b", f"{slot.name}的另一份证据。")
            spans = (first, second)
            facts = (
                DetectedFact("系统", "状态", "已上线", "true", first.evidence_id),
                DetectedFact("系统", "状态", "未上线", "false", second.evidence_id),
            )
            conflicts = (
                DetectedConflict(
                    conflict_type="opposite_assertion",
                    subject="系统",
                    predicate="状态",
                    facts=facts,
                    evidence_ids=tuple(item.evidence_id for item in facts),
                    reason="系统状态存在冲突",
                ),
            )
        evidence_ids = tuple(item.evidence_id for item in spans)
        coverage = []
        for field_index, field_id in enumerate(slot.required_fields):
            covered = bool(spans) and (
                status is not SlotStatus.WEAK or field_index == 0
            )
            coverage.append(
                FieldCoverage(
                    field_id=field_id,
                    covered=covered,
                    evidence_ids=(first.evidence_id,) if covered else (),
                    reason=None if covered else "缺少字段",
                )
            )
        if status is SlotStatus.FILLED:
            reason_codes = ()
        elif status is SlotStatus.MISSING:
            reason_codes = (VerificationReasonCode.NO_EVIDENCE,)
        elif status is SlotStatus.CONFLICTING:
            reason_codes = (VerificationReasonCode.CONFLICTING_FACTS,)
        else:
            reason_codes = (VerificationReasonCode.MISSING_REQUIRED_FIELD,)
        results.append(
            SlotVerificationResult(
                contract_id=contract.contract_id,
                slot_id=slot.slot_id,
                status=status,
                evidence_ids=evidence_ids,
                field_coverage=tuple(coverage),
                reason_codes=reason_codes,
                reason=f"{slot.name}判定为{status.value}",
                conflicts=conflicts,
                trace={"fixture": True},
            )
        )
        evidence_by_slot[slot.slot_id] = spans
    if any(item.status is SlotStatus.CONFLICTING for item in results):
        overall = "conflicted"
    elif all(item.status is SlotStatus.FILLED for item in results):
        overall = "complete"
    else:
        overall = "incomplete"
    verification = ContractVerificationResult(
        contract_id=contract.contract_id,
        status=overall,
        slots=tuple(results),
    )
    report = GapReportBuilder().build(
        contract=contract,
        verification=verification,
        evidence_by_slot=evidence_by_slot,
        trace={"document_count": 8, "label": "中文"},
    )
    return contract, verification, evidence_by_slot, report


@pytest.mark.parametrize(
    ("status", "overall", "ready", "action"),
    [
        (SlotStatus.FILLED, "complete", True, None),
        (SlotStatus.WEAK, "incomplete", False, PendingActionType.REQUEST_MATERIAL),
        (SlotStatus.MISSING, "incomplete", False, PendingActionType.REQUEST_MATERIAL),
        (SlotStatus.CONFLICTING, "conflicted", False, PendingActionType.RESOLVE_CONFLICT),
    ],
)
def test_report_projects_four_statuses(tmp_path, status, overall, ready, action):
    slot_id = solution_material_completeness_v1.slots[0].slot_id
    _, _, _, report = _verification_bundle(tmp_path, {slot_id: status})
    assert report.status == overall
    assert report.ready is ready
    assert tuple(item.slot_id for item in report.slots) == (
        solution_material_completeness_v1.slot_ids
    )
    matching = tuple(item for item in report.pending if item.slot_id == slot_id)
    assert (matching[0].action if matching else None) is action


def test_pending_priority_is_conflict_then_missing_then_weak(tmp_path):
    slots = solution_material_completeness_v1.slots
    overrides = {
        slots[0].slot_id: SlotStatus.WEAK,
        slots[1].slot_id: SlotStatus.MISSING,
        slots[2].slot_id: SlotStatus.CONFLICTING,
    }
    _, _, _, report = _verification_bundle(tmp_path, overrides)
    assert [(item.slot_id, item.priority) for item in report.pending] == [
        (slots[2].slot_id, 1),
        (slots[1].slot_id, 2),
        (slots[0].slot_id, 3),
    ]


def test_missing_fields_are_direct_field_coverage_projection(tmp_path):
    slot = solution_material_completeness_v1.slots[0]
    _, _, _, report = _verification_bundle(tmp_path, {slot.slot_id: SlotStatus.WEAK})
    projected = report.slots[0]
    assert projected.covered_fields == slot.required_fields[:1]
    assert projected.missing_fields == slot.required_fields[1:]


def test_evidence_references_are_deduplicated_by_first_reference(tmp_path):
    contract, verification, evidence_by_slot, _ = _verification_bundle(tmp_path)
    shared = evidence_by_slot[contract.slots[0].slot_id][0]
    shared_path = tmp_path / "0-a.md"
    same_document = NormalizedDocument.from_path(
        shared_path,
        document_id=shared.document_id,
    )
    alternate_discovery = EvidenceSpan.from_document(
        same_document,
        shared.start_char,
        shared.end_char,
        discovery_sources=("different-slot-route",),
        metadata={"slot_specific": True},
    )
    assert alternate_discovery.evidence_id == shared.evidence_id
    assert alternate_discovery != shared
    evidence_by_slot[contract.slots[1].slot_id] = (alternate_discovery,)
    second = verification.slots[1]
    coverage = tuple(
        FieldCoverage(item.field_id, True, (shared.evidence_id,))
        for item in second.field_coverage
    )
    replacement = SlotVerificationResult(
        contract_id=second.contract_id,
        slot_id=second.slot_id,
        status=second.status,
        evidence_ids=(shared.evidence_id,),
        field_coverage=coverage,
        reason_codes=second.reason_codes,
        reason=second.reason,
        conflicts=(),
        trace=second.trace,
    )
    verification = ContractVerificationResult(
        contract_id=verification.contract_id,
        status=verification.status,
        slots=(verification.slots[0], replacement, *verification.slots[2:]),
    )
    report = GapReportBuilder().build(
        contract=contract,
        verification=verification,
        evidence_by_slot=evidence_by_slot,
        trace={},
    )
    assert [item.evidence_id for item in report.evidence].count(shared.evidence_id) == 1
    assert report.evidence[0].evidence_id == shared.evidence_id
    assert report.evidence[0].discovery_sources == shared.discovery_sources


def test_evidence_reference_cannot_be_created_from_free_text():
    with pytest.raises(TypeError, match="EvidenceSpan"):
        EvidenceReference()


def test_unknown_evidence_id_fails_instead_of_becoming_missing(tmp_path):
    contract, verification, evidence_by_slot, _ = _verification_bundle(tmp_path)
    evidence_by_slot[contract.slots[0].slot_id] = ()
    with pytest.raises(ValueError, match="unresolved evidence"):
        GapReportBuilder().build(
            contract=contract,
            verification=verification,
            evidence_by_slot=evidence_by_slot,
            trace={},
        )


def test_unknown_slot_and_wrong_verification_order_fail(tmp_path):
    contract, verification, evidence_by_slot, _ = _verification_bundle(tmp_path)
    with pytest.raises(ValueError, match="unknown slot"):
        GapReportBuilder().build(
            contract=contract,
            verification=verification,
            evidence_by_slot={**evidence_by_slot, "unknown": ()},
            trace={},
        )
    reordered = ContractVerificationResult(
        contract_id=verification.contract_id,
        status=verification.status,
        slots=tuple(reversed(verification.slots)),
    )
    with pytest.raises(ValueError, match="contract order"):
        GapReportBuilder().build(
            contract=contract,
            verification=reordered,
            evidence_by_slot=evidence_by_slot,
            trace={},
        )


@pytest.mark.parametrize("bad_trace", [{"value": object()}, {"value": float("nan")}, {1: "x"}])
def test_non_json_trace_is_rejected(tmp_path, bad_trace):
    contract, verification, evidence_by_slot, _ = _verification_bundle(tmp_path)
    with pytest.raises(ValueError, match="JSON"):
        GapReportBuilder().build(
            contract=contract,
            verification=verification,
            evidence_by_slot=evidence_by_slot,
            trace=bad_trace,
        )


def test_json_is_stable_utf8_and_has_machine_values(tmp_path):
    slot_id = solution_material_completeness_v1.slots[0].slot_id
    _, _, _, report = _verification_bundle(tmp_path, {slot_id: SlotStatus.WEAK})
    first = gap_report_to_json(report)
    assert first == gap_report_to_json(report)
    assert "中文" in first and "\\u4e2d" not in first
    assert first.endswith("\n")
    parsed = json.loads(first)
    assert parsed["status"] == "incomplete"
    assert parsed["ready"] is False
    assert parsed["slots"][0]["status"] == "weak"
    assert parsed["pending"][0]["action"] == "request_material"
    assert list(parsed) == [
        "contract_id",
        "contract_version",
        "status",
        "ready",
        "slots",
        "evidence",
        "pending",
        "trace",
    ]
    assert gap_report_to_dict(report)["status"] == parsed["status"]


def test_json_and_markdown_writers_use_lf_and_terminal_newline(tmp_path):
    _, _, _, report = _verification_bundle(tmp_path)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_gap_report_json(report, json_path)
    write_gap_report_markdown(report, md_path)
    assert json_path.read_bytes().endswith(b"\n")
    assert md_path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in md_path.read_bytes()


def test_markdown_pending_precedes_all_slot_details_and_matches_report(tmp_path):
    slot = solution_material_completeness_v1.slots[0]
    _, _, _, report = _verification_bundle(tmp_path, {slot.slot_id: SlotStatus.WEAK})
    markdown = render_gap_report_markdown(report)
    assert markdown.index("## 优先补料事项") < markdown.index("## Slot 明细")
    assert "总体状态：`incomplete`" in markdown
    assert "状态：`weak`" in markdown
    assert slot.required_fields[1] in markdown
    assert report.evidence[0].evidence_id in markdown
    assert f"行 {report.evidence[0].start_line}-{report.evidence[0].end_line}" in markdown
    assert "中文证据" in markdown
    assert markdown.endswith("\n")


def test_markdown_conflict_shows_both_raw_values_and_evidence_ids(tmp_path):
    slot = solution_material_completeness_v1.slots[0]
    _, _, _, report = _verification_bundle(
        tmp_path,
        {slot.slot_id: SlotStatus.CONFLICTING},
    )
    markdown = render_gap_report_markdown(report)
    conflict = report.slots[0].conflicts[0]
    for fact in conflict.facts:
        assert fact.value in markdown
        assert fact.evidence_id in markdown
    assert "可以认为" not in markdown
    assert "建议编写" not in markdown


def test_markdown_covers_all_slot_statuses_and_matches_json_pending_count(tmp_path):
    slots = solution_material_completeness_v1.slots
    _, _, _, report = _verification_bundle(
        tmp_path,
        {
            slots[0].slot_id: SlotStatus.WEAK,
            slots[1].slot_id: SlotStatus.MISSING,
            slots[2].slot_id: SlotStatus.CONFLICTING,
        },
    )
    markdown = render_gap_report_markdown(report)
    parsed = json.loads(gap_report_to_json(report))
    for status in SlotStatus:
        assert f"状态：`{status.value}`" in markdown
    assert f"待办数量：{len(parsed['pending'])}" in markdown
