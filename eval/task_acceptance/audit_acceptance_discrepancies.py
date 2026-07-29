"""Freeze acceptance artifacts and build a readback-verified Slot matrix.

This is an offline audit utility.  It only reads the already frozen Contract,
manifests, AI labels, actual reports, and comparisons.  It never imports or
runs the verifier, never regenerates expected labels, and writes exclusively
to the caller-provided audit directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.task_acceptance.ai_label_acceptance import validate_judge_output
from evidence.gap_checker import load_normalized_document_manifest


DATASET_IDS = ("complete", "missing-case-or-outcome", "promotional-heavy")


class DiscrepancyAuditError(ValueError):
    """An immutable acceptance artifact is missing or internally inconsistent."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscrepancyAuditError(f"unable to read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise DiscrepancyAuditError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def freeze_inputs(
    *,
    acceptance_root: Path,
    results_root: Path,
    contract_path: Path,
    output_dir: Path,
    generated_timestamp: str,
) -> dict[str, object]:
    """Record byte hashes without changing any acceptance input or result."""

    frozen_path = output_dir / "frozen-inputs.json"
    if frozen_path.exists():
        raise DiscrepancyAuditError("frozen-inputs.json already exists; refusing to overwrite")
    contract_hash = _sha256(contract_path)
    datasets: list[dict[str, object]] = []
    for dataset_id in DATASET_IDS:
        dataset_dir = acceptance_root / dataset_id
        result_dir = results_root / dataset_id
        manifest_path = dataset_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        documents = manifest.get("documents")
        if not isinstance(documents, list):
            raise DiscrepancyAuditError(f"{dataset_id}: manifest documents must be an array")
        materials: list[dict[str, object]] = []
        for item in documents:
            if not isinstance(item, dict) or not isinstance(item.get("normalized_path"), str):
                raise DiscrepancyAuditError(f"{dataset_id}: invalid manifest document")
            relative = Path(item["normalized_path"])
            material_path = (dataset_dir / relative).resolve(strict=True)
            try:
                material_path.relative_to(dataset_dir.resolve(strict=True))
            except ValueError as exc:
                raise DiscrepancyAuditError(f"{dataset_id}: material path escapes dataset") from exc
            materials.append(
                {
                    "document_id": item.get("document_id"),
                    "normalized_path": relative.as_posix(),
                    "sha256": _sha256(material_path),
                }
            )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "material_manifest_sha256": _sha256(manifest_path),
                "expected_sha256": _sha256(dataset_dir / "expected.json"),
                "actual_report_sha256": _sha256(result_dir / "actual-gap-report.json"),
                "comparison_sha256": _sha256(result_dir / "comparison.json"),
                "contract_sha256": contract_hash,
                "materials": materials,
            }
        )
    frozen = {
        "generated_timestamp": generated_timestamp,
        "contract_path": contract_path.as_posix(),
        "contract_sha256": contract_hash,
        "datasets": datasets,
    }
    _write_json(frozen_path, frozen)
    lines = [
        "# Acceptance Frozen Inputs",
        "",
        f"Generated: `{generated_timestamp}`",
        "",
        "| Dataset | Manifest SHA-256 | Expected SHA-256 | Actual SHA-256 | Comparison SHA-256 | Materials |",
        "|---|---|---|---|---|---:|",
    ]
    for item in datasets:
        lines.append(
            f"| `{item['dataset_id']}` | `{item['material_manifest_sha256']}` | "
            f"`{item['expected_sha256']}` | `{item['actual_report_sha256']}` | "
            f"`{item['comparison_sha256']}` | {len(item['materials'])} |"
        )
    lines.extend(["", f"Contract SHA-256: `{contract_hash}`", ""])
    (output_dir / "frozen-inputs.md").write_text("\n".join(lines), encoding="utf-8")
    return frozen


_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_AMOUNT = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|元|RMB|CNY|￥|¥)", re.I)


def redact_sensitive(value: str) -> str:
    """Mask contact, network, and precise currency values in Markdown output."""

    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _PHONE.sub("[REDACTED_PHONE]", value)
    value = _IPV4.sub("[REDACTED_IP]", value)
    return _AMOUNT.sub("[REDACTED_AMOUNT]", value)


def _judge_evidence_readback(evidence: object, dataset_dir: Path) -> bool:
    if not isinstance(evidence, list):
        return False
    manifest = _read_json(dataset_dir / "manifest.json")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        return False
    by_filename: dict[str, tuple[str, list[str]]] = {}
    for item in documents:
        if not isinstance(item, dict):
            return False
        filename = Path(str(item.get("normalized_path", ""))).name
        path = dataset_dir / str(item.get("normalized_path", ""))
        by_filename[filename] = (
            str(item.get("document_id", "")),
            path.read_text(encoding="utf-8-sig").splitlines(),
        )
    for raw in evidence:
        if not isinstance(raw, dict):
            return False
        filename = raw.get("filename")
        if not isinstance(filename, str) or filename not in by_filename:
            return False
        document_id, lines = by_filename[filename]
        start, end, excerpt = raw.get("start_line"), raw.get("end_line"), raw.get("excerpt")
        if raw.get("document_id") != document_id:
            return False
        if type(start) is not int or type(end) is not int or start < 1 or end < start or end > len(lines):
            return False
        if not isinstance(excerpt, str) or excerpt not in "\n".join(lines[start - 1 : end]):
            return False
    return True


def _actual_evidence_readback(
    evidence: Sequence[Mapping[str, object]], dataset_dir: Path
) -> tuple[bool, dict[str, Mapping[str, object]]]:
    documents = load_normalized_document_manifest(dataset_dir / "manifest.json")
    by_document = {item.document_id: item for item in documents}
    by_id: dict[str, Mapping[str, object]] = {}
    valid = True
    for item in evidence:
        evidence_id = item.get("evidence_id")
        document_id = item.get("document_id")
        start = item.get("start_char")
        end = item.get("end_char")
        text = item.get("text")
        if not isinstance(evidence_id, str) or evidence_id in by_id:
            valid = False
            continue
        by_id[evidence_id] = item
        document = by_document.get(str(document_id))
        if (
            document is None
            or type(start) is not int
            or type(end) is not int
            or not isinstance(text, str)
            or document.slice(start, end) != text
        ):
            valid = False
    return valid, by_id


def build_mismatch_matrix(
    *,
    acceptance_root: Path,
    results_root: Path,
    ai_review_root: Path,
    contract_path: Path,
) -> dict[str, object]:
    contract = _read_json(contract_path)
    raw_slots = contract.get("slots")
    if not isinstance(raw_slots, list):
        raise DiscrepancyAuditError("Contract slots must be an array")
    contract_slots = {
        str(item["slot_id"]): item for item in raw_slots if isinstance(item, dict) and "slot_id" in item
    }
    rows: list[dict[str, object]] = []
    for dataset_id in DATASET_IDS:
        dataset_dir = acceptance_root / dataset_id
        result_dir = results_root / dataset_id
        review_dir = ai_review_root / dataset_id
        expected = _read_json(dataset_dir / "expected.json")
        actual = _read_json(result_dir / "actual-gap-report.json")
        comparison = _read_json(result_dir / "comparison.json")
        judge_a = validate_judge_output(
            _read_json(review_dir / "judge-a.json"),
            dataset_dir=dataset_dir,
            expected_judge_id="judge-a",
        )
        judge_b = validate_judge_output(
            _read_json(review_dir / "judge-b.json"),
            dataset_dir=dataset_dir,
            expected_judge_id="judge-b",
        )
        adjudication_path = review_dir / "adjudication.json"
        adjudication = _read_json(adjudication_path) if adjudication_path.exists() else None
        expected_slots = expected.get("slots")
        actual_slots_raw = actual.get("slots")
        comparison_slots_raw = comparison.get("slots")
        trace = actual.get("trace")
        actual_evidence_raw = actual.get("evidence")
        if not isinstance(expected_slots, dict) or not isinstance(actual_slots_raw, list):
            raise DiscrepancyAuditError(f"{dataset_id}: invalid expected/actual slots")
        if not isinstance(comparison_slots_raw, list) or not isinstance(trace, dict):
            raise DiscrepancyAuditError(f"{dataset_id}: invalid comparison/trace")
        if not isinstance(actual_evidence_raw, list) or not all(isinstance(x, dict) for x in actual_evidence_raw):
            raise DiscrepancyAuditError(f"{dataset_id}: invalid actual evidence")
        all_actual_readback, actual_evidence_by_id = _actual_evidence_readback(actual_evidence_raw, dataset_dir)
        actual_slots = {str(item["slot_id"]): item for item in actual_slots_raw if isinstance(item, dict)}
        comparison_slots = {
            str(item["slot_id"]): item for item in comparison_slots_raw if isinstance(item, dict)
        }
        per_slot_trace = trace.get("per_slot")
        if not isinstance(per_slot_trace, dict):
            raise DiscrepancyAuditError(f"{dataset_id}: missing per-Slot trace")
        for slot_id, contract_slot in contract_slots.items():
            expected_slot = expected_slots.get(slot_id)
            actual_slot = actual_slots.get(slot_id)
            comparison_slot = comparison_slots.get(slot_id)
            trace_slot = per_slot_trace.get(slot_id)
            if not all(isinstance(item, dict) for item in (expected_slot, actual_slot, comparison_slot, trace_slot)):
                raise DiscrepancyAuditError(f"{dataset_id}/{slot_id}: incomplete Slot records")
            a = judge_a["slots"][slot_id]
            b = judge_b["slots"][slot_id]
            assert isinstance(a, dict) and isinstance(b, dict)
            adjudicated = None
            adjudication_evidence: list[object] = []
            adjudicator_confidence = None
            if adjudication is not None and isinstance(adjudication.get("slots"), dict):
                decision = adjudication["slots"].get(slot_id)
                if isinstance(decision, dict):
                    adjudicated = decision.get("final_status")
                    raw_support = decision.get("supporting_evidence", [])
                    adjudication_evidence = raw_support if isinstance(raw_support, list) else []
                    adjudicator_confidence = decision.get("confidence")
            actual_ids = actual_slot.get("evidence_ids", [])
            if not isinstance(actual_ids, list):
                raise DiscrepancyAuditError(f"{dataset_id}/{slot_id}: evidence_ids must be an array")
            actual_excerpt_records = [actual_evidence_by_id[item] for item in actual_ids if item in actual_evidence_by_id]
            actual_slot_readback = all_actual_readback and len(actual_excerpt_records) == len(actual_ids)
            judge_evidence = {
                "judge_a": a.get("evidence", []),
                "judge_b": b.get("evidence", []),
                "adjudicator": adjudication_evidence,
            }
            judge_readback = all(
                _judge_evidence_readback(value, dataset_dir) for value in judge_evidence.values()
            )
            row = {
                "dataset_id": dataset_id,
                "slot_id": slot_id,
                "slot_name": contract_slot.get("name"),
                "expected_status": expected_slot.get("expected_status"),
                "actual_status": actual_slot.get("status"),
                "status_match": comparison_slot.get("match"),
                "judge_a_status": a.get("status"),
                "judge_b_status": b.get("status"),
                "adjudicated_status": adjudicated,
                "judge_confidence": {
                    "judge_a": a.get("confidence"),
                    "judge_b": b.get("confidence"),
                    "adjudicator": adjudicator_confidence,
                },
                "judge_independence": judge_a.get("judge_independence"),
                "contract_required_fields": contract_slot.get("required_fields", []),
                "judge_reason": expected_slot.get("reason"),
                "judge_evidence": judge_evidence,
                "judge_evidence_readback_valid": judge_readback,
                "actual_reason_codes": actual_slot.get("reason_codes", []),
                "actual_covered_fields": actual_slot.get("covered_fields", []),
                "actual_missing_fields": actual_slot.get("missing_fields", []),
                "actual_evidence_ids": actual_ids,
                "actual_evidence_excerpts": actual_excerpt_records,
                "actual_evidence_readback_valid": actual_slot_readback,
                "vector_candidate_count": trace_slot.get("vector_candidate_count"),
                "anchor_count": trace_slot.get("anchor_count"),
                "exact_span_count": trace_slot.get("exact_span_count"),
                "fallback_used": trace_slot.get("fallback_used"),
                "initial_error_attribution": comparison_slot.get("error_attribution"),
            }
            rows.append(row)
    if len(rows) != len(DATASET_IDS) * len(contract_slots):
        raise DiscrepancyAuditError("matrix does not cover all dataset/Slot combinations")
    return {
        "dataset_count": len(DATASET_IDS),
        "slot_count": len(rows),
        "mismatch_count": sum(not bool(item["status_match"]) for item in rows),
        "rows": rows,
    }


def write_matrix(output_dir: Path, matrix: Mapping[str, object]) -> None:
    _write_json(output_dir / "mismatch-matrix.json", matrix)
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise DiscrepancyAuditError("matrix rows must be an array")
    lines = [
        "# Acceptance Mismatch Matrix",
        "",
        f"Slots: {matrix.get('slot_count')}; mismatches: {matrix.get('mismatch_count')}",
        "",
        "| Dataset | Slot | Expected | Actual | Match | Judge A/B/Adjudicated | Readback | Trace V/A/E/F | Attribution |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        confidence = raw.get("judge_confidence", {})
        trace = f"{raw.get('vector_candidate_count')}/{raw.get('anchor_count')}/{raw.get('exact_span_count')}/{raw.get('fallback_used')}"
        labels = f"{raw.get('judge_a_status')}/{raw.get('judge_b_status')}/{raw.get('adjudicated_status') or '-'}"
        readback = f"J:{raw.get('judge_evidence_readback_valid')} A:{raw.get('actual_evidence_readback_valid')}"
        lines.append(
            f"| `{raw.get('dataset_id')}` | `{raw.get('slot_id')}` | `{raw.get('expected_status')}` | "
            f"`{raw.get('actual_status')}` | {raw.get('status_match')} | `{labels}` | `{readback}` | "
            f"`{trace}` | `{raw.get('initial_error_attribution') or '-'}` |"
        )
        lines.extend(
            [
                "",
                f"### {raw.get('dataset_id')} / {raw.get('slot_id')}",
                "",
                f"- Slot name: {redact_sensitive(str(raw.get('slot_name') or ''))}",
                f"- Required fields: `{json.dumps(raw.get('contract_required_fields'), ensure_ascii=False)}`",
                f"- Judge confidence: `{json.dumps(confidence, ensure_ascii=False)}`",
                f"- Judge reason: {redact_sensitive(str(raw.get('judge_reason') or ''))}",
                f"- Actual covered fields: `{json.dumps(raw.get('actual_covered_fields'), ensure_ascii=False)}`",
                f"- Actual missing fields: `{json.dumps(raw.get('actual_missing_fields'), ensure_ascii=False)}`",
                f"- Actual reason codes: `{json.dumps(raw.get('actual_reason_codes'), ensure_ascii=False)}`",
                f"- Actual evidence IDs: `{json.dumps(raw.get('actual_evidence_ids'), ensure_ascii=False)}`",
                "- Actual evidence excerpts:",
            ]
        )
        excerpts = raw.get("actual_evidence_excerpts", [])
        if isinstance(excerpts, list) and excerpts:
            for item in excerpts:
                if isinstance(item, dict):
                    lines.append(
                        f"  - `{item.get('evidence_id')}` / `{item.get('document_id')}` / "
                        f"lines `{item.get('start_line')}-{item.get('end_line')}`: "
                        "[PRIVATE_EVIDENCE_REDACTED; see local JSON]"
                    )
        else:
            lines.append("  - none")
        lines.append("")
    (output_dir / "mismatch-matrix.md").write_text("\n".join(lines), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and audit acceptance discrepancies")
    parser.add_argument("--acceptance-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--ai-review-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generated-timestamp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    acceptance_root = Path(args.acceptance_root).resolve(strict=True)
    results_root = Path(args.results_root).resolve(strict=True)
    ai_review_root = Path(args.ai_review_root).resolve(strict=True)
    contract_path = Path(args.contract).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = args.generated_timestamp or datetime.now(timezone.utc).isoformat()
    freeze_inputs(
        acceptance_root=acceptance_root,
        results_root=results_root,
        contract_path=contract_path,
        output_dir=output_dir,
        generated_timestamp=timestamp,
    )
    matrix = build_mismatch_matrix(
        acceptance_root=acceptance_root,
        results_root=results_root,
        ai_review_root=ai_review_root,
        contract_path=contract_path,
    )
    write_matrix(output_dir, matrix)
    print(json.dumps({"slot_count": matrix["slot_count"], "mismatch_count": matrix["mismatch_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
