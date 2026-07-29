"""Blind AI-label validation, comparison, adjudication, and expected freezing.

This module never imports the production verifier or gap report.  Expected
labels are validated only against the fixed Contract shape and canonical
normalized material lines, and must be frozen before any actual run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidence.contract import solution_material_completeness_v1


SLOT_IDS = solution_material_completeness_v1.slot_ids
CONTRACT_ID = solution_material_completeness_v1.contract_id
SLOT_STATUSES = frozenset({"filled", "weak", "missing", "conflicting"})
OVERALL_STATUSES = frozenset({"complete", "incomplete", "conflicted"})
FORBIDDEN_BLIND_KEYS = (
    "actual",
    "verifier",
    "gap_report",
    "comparison",
    "acceptance_result",
)


class AILabelError(ValueError):
    """AI Judge, adjudication, or expected-freeze input is invalid."""


@dataclass(frozen=True)
class ConsensusSlot:
    slot_id: str
    status: str | None
    reason: str
    evidence: tuple[Mapping[str, object], ...]
    confidence: float
    source: str
    unresolved: bool
    human_review_question: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_blind_input(value: object, location: str = "judge_input") -> None:
    """Reject any actual-system field before a Judge sees a packet."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(forbidden in normalized for forbidden in FORBIDDEN_BLIND_KEYS):
                raise AILabelError(f"{location} contains forbidden actual-system field")
            validate_blind_input(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_blind_input(item, f"{location}[{index}]")


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AILabelError(f"{location} must be an object")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AILabelError(f"{location} must be a non-empty string")
    return value


def _load_material_lines(dataset_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    lines_by_filename: dict[str, list[str]] = {}
    document_id_by_filename: dict[str, str] = {}
    for item in manifest["documents"]:
        path = dataset_dir / item["normalized_path"]
        filename = path.name
        lines_by_filename[filename] = path.read_text(encoding="utf-8-sig").splitlines()
        document_id_by_filename[filename] = item["document_id"]
    return lines_by_filename, document_id_by_filename


def _validate_evidence(
    value: object,
    *,
    location: str,
    lines_by_filename: Mapping[str, list[str]],
    document_id_by_filename: Mapping[str, str],
) -> dict[str, object]:
    evidence = _object(value, location)
    expected = {"document_id", "filename", "start_line", "end_line", "excerpt"}
    if set(evidence) != expected:
        raise AILabelError(f"{location} fields must be exactly {sorted(expected)!r}")
    filename = _string(evidence["filename"], f"{location}.filename")
    document_id = _string(evidence["document_id"], f"{location}.document_id")
    if filename not in lines_by_filename:
        raise AILabelError(f"{location}.filename is not in dataset manifest")
    if document_id_by_filename[filename] != document_id:
        raise AILabelError(f"{location}.document_id does not match manifest")
    start = evidence["start_line"]
    end = evidence["end_line"]
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise AILabelError(f"{location} line range is invalid")
    lines = lines_by_filename[filename]
    if end > len(lines):
        raise AILabelError(f"{location} line range exceeds material")
    excerpt = _string(evidence["excerpt"], f"{location}.excerpt")
    exact_range = "\n".join(lines[start - 1:end])
    if excerpt not in exact_range:
        raise AILabelError(f"{location}.excerpt cannot be read back from line range")
    return dict(evidence)


def validate_judge_output(
    value: object,
    *,
    dataset_dir: Path,
    expected_judge_id: str | None = None,
) -> dict[str, object]:
    data = _object(value, "judge")
    expected_keys = {
        "dataset_id",
        "judge_id",
        "judge_model",
        "judge_independence",
        "contract_id",
        "slots",
        "overall_status",
        "notes",
    }
    if set(data) != expected_keys:
        raise AILabelError("judge fields are incomplete or unknown")
    if _string(data["dataset_id"], "judge.dataset_id") != dataset_dir.name:
        raise AILabelError("judge.dataset_id must match dataset directory")
    judge_id = _string(data["judge_id"], "judge.judge_id")
    if expected_judge_id is not None and judge_id != expected_judge_id:
        raise AILabelError("judge.judge_id does not match isolated session")
    _string(data["judge_model"], "judge.judge_model")
    _string(data["judge_independence"], "judge.judge_independence")
    if _string(data["contract_id"], "judge.contract_id") != CONTRACT_ID:
        raise AILabelError("judge.contract_id does not match fixed Contract")
    lines_by_filename, document_ids = _load_material_lines(dataset_dir)
    slots = _object(data["slots"], "judge.slots")
    if set(slots) != set(SLOT_IDS):
        raise AILabelError("judge.slots must contain all eight fixed Slot IDs exactly")
    validated_slots: dict[str, object] = {}
    for slot_id in SLOT_IDS:
        location = f"judge.slots.{slot_id}"
        slot = _object(slots[slot_id], location)
        if set(slot) != {"status", "reason", "evidence", "confidence"}:
            raise AILabelError(f"{location} has invalid fields")
        status = _string(slot["status"], f"{location}.status")
        if status not in SLOT_STATUSES:
            raise AILabelError(f"{location}.status is illegal")
        reason = _string(slot["reason"], f"{location}.reason")
        raw_evidence = slot["evidence"]
        if not isinstance(raw_evidence, list):
            raise AILabelError(f"{location}.evidence must be an array")
        evidence = [
            _validate_evidence(
                item,
                location=f"{location}.evidence[{index}]",
                lines_by_filename=lines_by_filename,
                document_id_by_filename=document_ids,
            )
            for index, item in enumerate(raw_evidence)
        ]
        if status == "filled" and not evidence:
            raise AILabelError(f"{location}: filled requires source evidence")
        if status == "conflicting" and len(evidence) < 2:
            raise AILabelError(f"{location}: conflicting requires both sides")
        confidence = slot["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise AILabelError(f"{location}.confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise AILabelError(f"{location}.confidence must be between 0 and 1")
        validated_slots[slot_id] = {
            "status": status,
            "reason": reason,
            "evidence": evidence,
            "confidence": confidence,
        }
    overall = _string(data["overall_status"], "judge.overall_status")
    if overall not in OVERALL_STATUSES:
        raise AILabelError("judge.overall_status is illegal")
    if not isinstance(data["notes"], list) or any(
        not isinstance(item, str) for item in data["notes"]
    ):
        raise AILabelError("judge.notes must be an array of strings")
    return {**data, "slots": validated_slots}


def aggregate_overall(statuses: Sequence[str]) -> str:
    if len(statuses) != len(SLOT_IDS) or any(item not in SLOT_STATUSES for item in statuses):
        raise AILabelError("overall aggregation requires eight legal Slot statuses")
    if "conflicting" in statuses:
        return "conflicted"
    if all(item == "filled" for item in statuses):
        return "complete"
    return "incomplete"


def _evidence_keys(slot: Mapping[str, object]) -> set[tuple[str, int, int]]:
    evidence = slot.get("evidence", [])
    if not isinstance(evidence, list):
        return set()
    return {
        (str(item["filename"]), int(item["start_line"]), int(item["end_line"]))
        for item in evidence
        if isinstance(item, Mapping)
    }


def compare_judges(
    judge_a: Mapping[str, object],
    judge_b: Mapping[str, object],
    *,
    audit_status_by_filename: Mapping[str, str] | None = None,
    low_confidence_threshold: float = 0.65,
) -> dict[str, object]:
    audit_status_by_filename = audit_status_by_filename or {}
    if judge_a["dataset_id"] != judge_b["dataset_id"]:
        raise AILabelError("Judge dataset IDs differ")
    slots_a = judge_a["slots"]
    slots_b = judge_b["slots"]
    assert isinstance(slots_a, Mapping) and isinstance(slots_b, Mapping)
    rows: list[dict[str, object]] = []
    for slot_id in SLOT_IDS:
        a = slots_a[slot_id]
        b = slots_b[slot_id]
        assert isinstance(a, Mapping) and isinstance(b, Mapping)
        status_match = a["status"] == b["status"]
        keys_a = _evidence_keys(a)
        keys_b = _evidence_keys(b)
        union = keys_a | keys_b
        overlap = len(keys_a & keys_b) / len(union) if union else 1.0
        confidence_a = float(a["confidence"])
        confidence_b = float(b["confidence"])
        evidence_files = {item[0] for item in union}
        audit_risk = any(
            audit_status_by_filename.get(filename) == "blocked"
            or (
                audit_status_by_filename.get(filename) == "warning"
                and min(confidence_a, confidence_b) < 0.75
            )
            for filename in evidence_files
        )
        if not status_match:
            classification = "STATUS_DISAGREEMENT"
        elif min(confidence_a, confidence_b) < low_confidence_threshold:
            classification = "LOW_CONFIDENCE"
        elif audit_risk:
            classification = "CONVERSION_QUALITY_RISK"
        elif overlap < 1.0:
            classification = "MINOR_EVIDENCE_DIFFERENCE"
        else:
            classification = "AGREED"
        rows.append(
            {
                "slot_id": slot_id,
                "status_a": a["status"],
                "status_b": b["status"],
                "status_match": status_match,
                "evidence_overlap": overlap,
                "reason_summary": f"A: {a['reason']} | B: {b['reason']}",
                "confidence_a": confidence_a,
                "confidence_b": confidence_b,
                "classification": classification,
                "needs_adjudication": classification
                in {
                    "STATUS_DISAGREEMENT",
                    "LOW_CONFIDENCE",
                    "MATERIAL_AMBIGUITY",
                    "CONVERSION_QUALITY_RISK",
                },
            }
        )
    return {
        "dataset_id": judge_a["dataset_id"],
        "judge_a_id": judge_a["judge_id"],
        "judge_b_id": judge_b["judge_id"],
        "slots": rows,
    }


def validate_adjudication(
    value: object,
    *,
    dataset_dir: Path,
    disputed_slot_ids: Sequence[str],
) -> dict[str, object]:
    data = _object(value, "adjudication")
    expected = {
        "dataset_id",
        "adjudicator_model",
        "judge_independence",
        "slots",
        "notes",
    }
    if set(data) != expected:
        raise AILabelError("adjudication fields are invalid")
    if data["dataset_id"] != dataset_dir.name:
        raise AILabelError("adjudication dataset ID differs")
    _string(data["adjudicator_model"], "adjudication.adjudicator_model")
    _string(data["judge_independence"], "adjudication.judge_independence")
    slots = _object(data["slots"], "adjudication.slots")
    if set(slots) != set(disputed_slot_ids):
        raise AILabelError("adjudication must resolve exactly the disputed Slots")
    lines, document_ids = _load_material_lines(dataset_dir)
    validated: dict[str, object] = {}
    for slot_id in disputed_slot_ids:
        location = f"adjudication.slots.{slot_id}"
        item = _object(slots[slot_id], location)
        required = {
            "final_status",
            "reason",
            "supporting_evidence",
            "confidence",
            "requires_human_review",
            "human_review_question",
        }
        if set(item) != required:
            raise AILabelError(f"{location} fields are invalid")
        final_status = item["final_status"]
        requires_human = item["requires_human_review"]
        if type(requires_human) is not bool:
            raise AILabelError(f"{location}.requires_human_review must be bool")
        if requires_human:
            if final_status is not None:
                raise AILabelError(f"{location}.final_status must be null when unresolved")
            _string(item["human_review_question"], f"{location}.human_review_question")
        elif final_status not in SLOT_STATUSES:
            raise AILabelError(f"{location}.final_status is illegal")
        reason = _string(item["reason"], f"{location}.reason")
        raw_evidence = item["supporting_evidence"]
        if not isinstance(raw_evidence, list):
            raise AILabelError(f"{location}.supporting_evidence must be an array")
        evidence = [
            _validate_evidence(
                raw,
                location=f"{location}.supporting_evidence[{index}]",
                lines_by_filename=lines,
                document_id_by_filename=document_ids,
            )
            for index, raw in enumerate(raw_evidence)
        ]
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise AILabelError(f"{location}.confidence must be numeric")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise AILabelError(f"{location}.confidence is outside 0..1")
        validated[slot_id] = {
            **item,
            "reason": reason,
            "supporting_evidence": evidence,
            "confidence": confidence,
        }
    return {**data, "slots": validated}


def resolve_consensus(
    judge_a: Mapping[str, object],
    judge_b: Mapping[str, object],
    comparison: Mapping[str, object],
    adjudication: Mapping[str, object] | None = None,
) -> tuple[ConsensusSlot, ...]:
    slots_a = judge_a["slots"]
    slots_b = judge_b["slots"]
    rows = comparison["slots"]
    assert isinstance(slots_a, Mapping) and isinstance(slots_b, Mapping)
    assert isinstance(rows, list)
    row_by_id = {str(item["slot_id"]): item for item in rows}
    adjudicated = adjudication.get("slots", {}) if adjudication else {}
    assert isinstance(adjudicated, Mapping)
    result: list[ConsensusSlot] = []
    for slot_id in SLOT_IDS:
        row = row_by_id[slot_id]
        a = slots_a[slot_id]
        b = slots_b[slot_id]
        assert isinstance(a, Mapping) and isinstance(b, Mapping)
        if not bool(row["needs_adjudication"]):
            result.append(
                ConsensusSlot(
                    slot_id=slot_id,
                    status=str(a["status"]),
                    reason=str(a["reason"]),
                    evidence=tuple(a["evidence"]),
                    confidence=min(float(a["confidence"]), float(b["confidence"])),
                    source="judge_consensus",
                    unresolved=False,
                    human_review_question="",
                )
            )
            continue
        item = adjudicated.get(slot_id)
        if not isinstance(item, Mapping) or bool(item["requires_human_review"]):
            question = (
                str(item.get("human_review_question", ""))
                if isinstance(item, Mapping)
                else "AI Judges 未能完成独立裁决，请确认 Slot 状态。"
            )
            result.append(
                ConsensusSlot(
                    slot_id=slot_id,
                    status=None,
                    reason=str(item.get("reason", row["reason_summary"]))
                    if isinstance(item, Mapping)
                    else str(row["reason_summary"]),
                    evidence=tuple(item.get("supporting_evidence", ()))
                    if isinstance(item, Mapping)
                    else (),
                    confidence=float(item.get("confidence", 0.0))
                    if isinstance(item, Mapping)
                    else 0.0,
                    source="unresolved",
                    unresolved=True,
                    human_review_question=question,
                )
            )
            continue
        result.append(
            ConsensusSlot(
                slot_id=slot_id,
                status=str(item["final_status"]),
                reason=str(item["reason"]),
                evidence=tuple(item["supporting_evidence"]),
                confidence=float(item["confidence"]),
                source="adjudicator",
                unresolved=False,
                human_review_question="",
            )
        )
    return tuple(result)


def _material_manifest_sha256(dataset_dir: Path) -> str:
    return _sha256_bytes((dataset_dir / "manifest.json").read_bytes())


def _references(evidence: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        f"materials/{item['filename']}:{item['start_line']}-{item['end_line']}"
        for item in evidence
    ]


def freeze_expected(
    *,
    dataset_dir: Path,
    consensus: Sequence[ConsensusSlot],
    judge_a: Mapping[str, object],
    judge_b: Mapping[str, object],
    adjudication: Mapping[str, object] | None,
    generated_timestamp: str,
) -> dict[str, object]:
    if {item.slot_id for item in consensus} != set(SLOT_IDS):
        raise AILabelError("consensus does not cover all eight Slots")
    unresolved = [item for item in consensus if item.unresolved or item.status is None]
    material_hash = _material_manifest_sha256(dataset_dir)
    decision_payload = {
        "dataset_id": dataset_dir.name,
        "slots": {
            item.slot_id: {
                "status": item.status,
                "reason": item.reason,
                "references": _references(item.evidence),
            }
            for item in consensus
        },
    }
    decision_hash = _sha256_bytes(_canonical_json(decision_payload).encode("utf-8"))
    metadata = {
        "label_source": "ai_consensus",
        "judge_a_model": judge_a["judge_model"],
        "judge_b_model": judge_b["judge_model"],
        "adjudicator_model": adjudication["adjudicator_model"] if adjudication else None,
        "judge_independence": judge_a["judge_independence"],
        "generated_before_actual_run": True,
        "generated_timestamp": generated_timestamp,
        "expected_sha256": decision_hash,
        "material_manifest_sha256": material_hash,
    }
    if unresolved:
        draft = {
            "dataset_id": dataset_dir.name,
            "contract_id": CONTRACT_ID,
            "slots": decision_payload["slots"],
            "overall_expected_status": None,
            "unresolved_slots": [item.slot_id for item in unresolved],
            "metadata": metadata,
        }
        (dataset_dir / "expected.ai-draft.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        expected_path = dataset_dir / "expected.json"
        if expected_path.exists():
            expected_path.unlink()
        return {"resolved": False, "unresolved_slots": draft["unresolved_slots"], "metadata": metadata}
    statuses = [str(item.status) for item in consensus]
    overall = aggregate_overall(statuses)
    expected = {
        "dataset_id": dataset_dir.name,
        "description": "由两个盲评 AI Judge 与必要的独立裁决基于真实规范化材料生成。",
        "contract_id": CONTRACT_ID,
        "slots": {
            item.slot_id: {
                "expected_status": item.status,
                "reason": item.reason,
                "references": _references(item.evidence),
            }
            for item in consensus
        },
        "overall_expected_status": overall,
        "notes": _canonical_json({"ai_label_provenance": metadata}),
    }
    rendered = json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    expected_path = dataset_dir / "expected.json"
    expected_path.write_text(rendered, encoding="utf-8")
    draft_path = dataset_dir / "expected.ai-draft.json"
    if draft_path.exists():
        draft_path.unlink()
    provenance = {
        **metadata,
        "expected_file_sha256": _sha256_bytes(expected_path.read_bytes()),
        "expected_frozen_at_ns": expected_path.stat().st_mtime_ns,
    }
    (dataset_dir / "expected.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "resolved": True,
        "unresolved_slots": [],
        "overall_expected_status": overall,
        "expected_file_sha256": provenance["expected_file_sha256"],
        "metadata": metadata,
    }


def assert_results_postdate_expected(
    provenance_path: Path,
    result_paths: Sequence[Path],
) -> None:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    frozen_at = provenance.get("expected_frozen_at_ns")
    if type(frozen_at) is not int:
        raise AILabelError("expected provenance has no freeze timestamp")
    for path in result_paths:
        if path.stat().st_mtime_ns < frozen_at:
            raise AILabelError("actual result predates frozen expected labels")


def load_audit_statuses(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        item["normalized_filename"]: item["automated_review_status"]
        for item in value["files"]
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and freeze blind AI labels")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--comparison-out", required=True)
    parser.add_argument("--adjudication")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--generated-timestamp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve(strict=True)
    judge_a = validate_judge_output(
        json.loads(Path(args.judge_a).read_text(encoding="utf-8-sig")),
        dataset_dir=dataset_dir,
        expected_judge_id="judge-a",
    )
    judge_b = validate_judge_output(
        json.loads(Path(args.judge_b).read_text(encoding="utf-8-sig")),
        dataset_dir=dataset_dir,
        expected_judge_id="judge-b",
    )
    comparison = compare_judges(
        judge_a,
        judge_b,
        audit_status_by_filename=load_audit_statuses(Path(args.audit)),
    )
    _write_json(Path(args.comparison_out), comparison)
    if not args.finalize:
        print(json.dumps({"needs_adjudication": [item["slot_id"] for item in comparison["slots"] if item["needs_adjudication"]]}, ensure_ascii=False))
        return 0
    disputed = [item["slot_id"] for item in comparison["slots"] if item["needs_adjudication"]]
    adjudication = None
    if args.adjudication:
        adjudication = validate_adjudication(
            json.loads(Path(args.adjudication).read_text(encoding="utf-8-sig")),
            dataset_dir=dataset_dir,
            disputed_slot_ids=disputed,
        )
    consensus = resolve_consensus(judge_a, judge_b, comparison, adjudication)
    timestamp = args.generated_timestamp or datetime.now(timezone.utc).isoformat()
    result = freeze_expected(
        dataset_dir=dataset_dir,
        consensus=consensus,
        judge_a=judge_a,
        judge_b=judge_b,
        adjudication=adjudication,
        generated_timestamp=timestamp,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
