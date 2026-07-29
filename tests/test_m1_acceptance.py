from __future__ import annotations

import json

import pytest

from evidence.core.config import Settings
from evidence.contract import solution_material_completeness_v1
from evidence.documents import NormalizedDocument
from evidence.gap_report import EvidenceReference, GapReport, SlotGapResult
from evidence.types import EvidenceSpan, SlotStatus
from evidence.verifier import (
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
)
from eval.task_acceptance.run_acceptance import (
    AcceptanceInputError,
    ExpectedDataset,
    ExpectedSlotLabel,
    _write_text,
    build_summary,
    check_evidence_readback,
    compare_report,
    load_expected_labels,
    redact_secrets,
    render_comparison_markdown,
    render_summary_markdown,
    run_acceptance,
)


CONTRACT = solution_material_completeness_v1


def _labels(status="missing", overall="incomplete"):
    return {
        "dataset_id": "dataset-a",
        "description": "人工标注的数据集",
        "contract_id": CONTRACT.contract_id,
        "slots": {
            slot.slot_id: {
                "expected_status": status,
                "reason": f"人工判断 {slot.name}",
            }
            for slot in CONTRACT.slots
        },
        "overall_expected_status": overall,
        "notes": "",
    }


def _write_labels(tmp_path, value):
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _report(tmp_path, statuses, *, overall="incomplete", broken_text=False):
    path = tmp_path / "material.md"
    path.write_text("人工材料证据。", encoding="utf-8")
    document = NormalizedDocument.from_path(
        path,
        document_id="doc-real",
        source_path="materials/material.md",
        normalized_path="materials/material.md",
    )
    span = EvidenceSpan.from_document(
        document,
        0,
        len(document.text),
        discovery_sources=("exact",),
    )
    reference = EvidenceReference.from_span(span)
    if broken_text:
        object.__setattr__(reference, "text", "被篡改")
    projected = []
    for slot in CONTRACT.slots:
        status = statuses.get(slot.slot_id, SlotStatus.MISSING)
        filled = status is SlotStatus.FILLED
        evidence_ids = (span.evidence_id,) if filled else ()
        coverage = tuple(
            FieldCoverage(
                field_id=field_id,
                covered=filled,
                evidence_ids=evidence_ids if filled else (),
                reason=None if filled else "缺少字段",
            )
            for field_id in slot.required_fields
        )
        result = SlotVerificationResult(
            contract_id=CONTRACT.contract_id,
            slot_id=slot.slot_id,
            status=status,
            evidence_ids=evidence_ids,
            field_coverage=coverage,
            reason_codes=(
                ()
                if filled
                else (VerificationReasonCode.NO_EVIDENCE,)
            ),
            reason=f"系统判定 {status.value}",
            conflicts=(),
            trace={},
        )
        projected.append(SlotGapResult.from_verification(CONTRACT, result))
    report = GapReport(
        contract_id=CONTRACT.contract_id,
        contract_version=CONTRACT.version,
        status=overall,
        ready=overall == "complete",
        slots=tuple(projected),
        evidence=(reference,) if any(status is SlotStatus.FILLED for status in statuses.values()) else (),
        pending=(),
        trace={
            "per_slot": {
                slot.slot_id: {
                    "vector_candidate_count": 1,
                    "anchor_count": 1,
                    "exact_span_count": 1,
                    "fallback_used": False,
                    "verification_status": statuses.get(slot.slot_id, SlotStatus.MISSING).value,
                }
                for slot in CONTRACT.slots
            }
        },
    )
    return report, (document,)


def _expected(statuses, overall="incomplete"):
    return ExpectedDataset(
        dataset_id="dataset-a",
        description="人工标注的数据集",
        contract_id=CONTRACT.contract_id,
        slots={
            slot.slot_id: ExpectedSlotLabel(
                expected_status=statuses.get(slot.slot_id, "missing"),
                reason=f"人工判断 {slot.name}",
                references=(),
            )
            for slot in CONTRACT.slots
        },
        overall_expected_status=overall,
        notes="",
    )


def test_expected_schema_loads_all_eight_human_labels(tmp_path):
    loaded = load_expected_labels(_write_labels(tmp_path, _labels()), CONTRACT)
    assert tuple(loaded.slots) == CONTRACT.slot_ids
    assert all(item.reason.startswith("人工判断") for item in loaded.slots.values())


def test_repository_label_example_matches_current_contract():
    loaded = load_expected_labels(
        "eval/task_acceptance/label_schema.example.json",
        CONTRACT,
    )
    assert tuple(loaded.slots) == CONTRACT.slot_ids


def test_expected_schema_rejects_any_missing_slot(tmp_path):
    value = _labels()
    value["slots"].pop(CONTRACT.slot_ids[-1])
    with pytest.raises(AcceptanceInputError, match="label every contract slot"):
        load_expected_labels(_write_labels(tmp_path, value), CONTRACT)


@pytest.mark.parametrize(
    ("field", "value"),
    [("slot", "unknown"), ("overall", "partial")],
)
def test_expected_schema_rejects_illegal_status(tmp_path, field, value):
    labels = _labels()
    if field == "slot":
        labels["slots"][CONTRACT.slot_ids[0]]["expected_status"] = value
    else:
        labels["overall_expected_status"] = value
    with pytest.raises(AcceptanceInputError, match="must be"):
        load_expected_labels(_write_labels(tmp_path, labels), CONTRACT)


def test_comparison_records_exact_status_fields_and_evidence(tmp_path):
    slot_id = CONTRACT.slot_ids[0]
    report, documents = _report(tmp_path, {slot_id: SlotStatus.FILLED})
    expected = _expected({slot_id: "filled"})
    comparison = compare_report(
        expected,
        report,
        evidence_readback_errors=check_evidence_readback(report, documents),
    )
    first = comparison["slots"][0]
    assert first["match"] is True
    assert first["actual_status"] == "filled"
    assert first["actual_reason_codes"] == []
    assert first["actual_missing_fields"] == []
    assert len(first["actual_evidence_ids"]) == 1
    assert comparison["metrics"]["exact_status_matches"] == 8


def test_filled_false_positive_is_counted_separately(tmp_path):
    slot_id = CONTRACT.slot_ids[0]
    report, _ = _report(tmp_path, {slot_id: SlotStatus.FILLED})
    comparison = compare_report(_expected({slot_id: "weak"}), report)
    assert comparison["metrics"]["filled_false_positive_count"] == 1


def test_conflict_missed_is_counted_and_attributed(tmp_path):
    slot_id = CONTRACT.slot_ids[0]
    report, _ = _report(tmp_path, {})
    comparison = compare_report(_expected({slot_id: "conflicting"}), report)
    assert comparison["metrics"]["conflict_missed_count"] == 1
    assert comparison["slots"][0]["error_attribution"] == "CONFLICT_RULE_LIMIT"


def test_evidence_readback_reports_exact_text_mismatch(tmp_path):
    slot_id = CONTRACT.slot_ids[0]
    report, documents = _report(
        tmp_path,
        {slot_id: SlotStatus.FILLED},
        broken_text=True,
    )
    assert check_evidence_readback(report, documents) == (
        f"{report.evidence[0].evidence_id}: text does not match canonical slice",
    )


def test_summary_is_stable_and_hard_gate_uses_false_positives(tmp_path):
    slot_id = CONTRACT.slot_ids[0]
    clean_report, _ = _report(tmp_path, {})
    false_report, _ = _report(tmp_path, {slot_id: SlotStatus.FILLED})
    clean = compare_report(_expected({}), clean_report)
    clean["dataset_id"] = "b"
    false_positive = compare_report(_expected({slot_id: "weak"}), false_report)
    false_positive["dataset_id"] = "a"
    summary = build_summary([clean, false_positive])
    assert [item["dataset_id"] for item in summary["datasets"]] == ["a", "b"]
    assert summary["filled_false_positive_count"] == 1
    assert summary["hard_gate_passed"] is False


def test_secret_redaction_masks_credentials_and_url_query():
    text = (
        "api_key=abc123 Authorization: Bearer xyz987 token=qwerty "
        "password=hunter2 https://example.test/path?sig=secret&x=1 explicit-value"
    )
    redacted = redact_secrets(text, explicit_secrets=("explicit-value",))
    for secret in ("abc123", "xyz987", "qwerty", "hunter2", "sig=secret", "explicit-value"):
        assert secret not in redacted
    assert redacted.count("***") >= 6


def test_markdown_and_json_writes_have_terminal_lf(tmp_path):
    report, _ = _report(tmp_path, {})
    comparison = compare_report(_expected({}), report)
    summary = build_summary([comparison])
    paths = {
        tmp_path / "comparison.md": render_comparison_markdown(comparison),
        tmp_path / "summary.md": render_summary_markdown(summary),
        tmp_path / "summary.json": json.dumps(summary, ensure_ascii=False, indent=2),
    }
    for path, content in paths.items():
        _write_text(path, content)
        assert path.read_bytes().endswith(b"\n")
        assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("existing_empty", [False, True])
def test_missing_or_empty_dataset_root_fails_before_backend_access(tmp_path, existing_empty):
    root = tmp_path / "datasets"
    if existing_empty:
        root.mkdir()
    expected_message = "does not exist" if not existing_empty else "no dataset directories"
    with pytest.raises(AcceptanceInputError, match=expected_message):
        run_acceptance(
            dataset_root=root,
            output_dir=tmp_path / "output",
            contract=CONTRACT,
            settings=Settings(),
            max_chars=800,
            top_k_per_slot=8,
            context_lines=2,
            max_span_chars=1200,
            max_results_per_slot=20,
        )
