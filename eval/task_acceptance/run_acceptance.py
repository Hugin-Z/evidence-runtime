"""Run task acceptance against private, human-labelled normalized materials.

This is an acceptance harness, not a second pipeline.  Runtime execution uses
the real VectorDiscovery, ExactEvidenceLocator, DeterministicSlotVerifier and
GapReportBuilder.  The pure label/comparison helpers are separately testable
without accessing private materials or model services.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidence.core.config import DEFAULT_SETTINGS, Settings
from evidence.contract import load_evidence_contract
from evidence.exact_locator import ExactEvidenceLocator
from evidence.gap_checker import load_normalized_document_manifest
from evidence.gap_report import (
    GapReport,
    GapReportBuilder,
    gap_report_to_json,
    render_gap_report_markdown,
)
from evidence.types import EvidenceContract, SlotStatus
from evidence.vector_discovery import VectorDiscovery
from evidence.verifier import DeterministicSlotVerifier
from evidence.gap_checker import EvidenceGapChecker, build_default_gap_checker
from evidence.library import FilesystemLibrary, SourceType


# 使用包资源解析器定位契约，不硬编码源码树路径。
from evidence.resources import contract_file

DEFAULT_CONTRACT_PATH = contract_file("solution_material_completeness_v1.json")
EXPECTED_STATUSES = frozenset(item.value for item in SlotStatus)
EXPECTED_OVERALL_STATUSES = frozenset({"complete", "incomplete", "conflicted"})
_EXPECTED_KEYS = frozenset(
    {
        "dataset_id",
        "description",
        "contract_id",
        "slots",
        "overall_expected_status",
        "notes",
    }
)
_EXPECTED_SLOT_REQUIRED_KEYS = frozenset({"expected_status", "reason"})
_EXPECTED_SLOT_OPTIONAL_KEYS = frozenset({"references"})


class AcceptanceInputError(ValueError):
    """Private dataset or human-label input is invalid or incomplete."""


@dataclass(frozen=True)
class ExpectedSlotLabel:
    expected_status: str
    reason: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class ExpectedDataset:
    dataset_id: str
    description: str
    contract_id: str
    slots: Mapping[str, ExpectedSlotLabel]
    overall_expected_status: str
    notes: str


def _non_empty(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceInputError(f"{location} must be a non-empty string")
    return value


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AcceptanceInputError(f"{location} must be a JSON object")
    return value


def load_expected_labels(
    path: str | Path,
    contract: EvidenceContract,
) -> ExpectedDataset:
    """Strictly load all eight human labels; no missing value is defaulted."""

    label_path = Path(path)
    try:
        raw = json.loads(label_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceInputError(f"unable to read expected labels {str(label_path)!r}") from exc
    data = _object(raw, "expected")
    if set(data) != _EXPECTED_KEYS:
        raise AcceptanceInputError(
            f"expected fields must be exactly {sorted(_EXPECTED_KEYS)!r}; "
            f"got {sorted(data)!r}"
        )
    dataset_id = _non_empty(data["dataset_id"], "expected.dataset_id")
    description = _non_empty(data["description"], "expected.description")
    contract_id = _non_empty(data["contract_id"], "expected.contract_id")
    if contract_id != contract.contract_id:
        raise AcceptanceInputError(
            f"expected.contract_id {contract_id!r} does not match {contract.contract_id!r}"
        )
    raw_slots = _object(data["slots"], "expected.slots")
    if set(raw_slots) != set(contract.slot_ids):
        missing = tuple(item for item in contract.slot_ids if item not in raw_slots)
        unknown = tuple(item for item in raw_slots if item not in contract.slot_ids)
        raise AcceptanceInputError(
            "expected.slots must label every contract slot exactly once; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    labels: dict[str, ExpectedSlotLabel] = {}
    for slot in contract.slots:
        location = f"expected.slots.{slot.slot_id}"
        item = _object(raw_slots[slot.slot_id], location)
        keys = set(item)
        if not _EXPECTED_SLOT_REQUIRED_KEYS <= keys or not keys <= (
            _EXPECTED_SLOT_REQUIRED_KEYS | _EXPECTED_SLOT_OPTIONAL_KEYS
        ):
            raise AcceptanceInputError(
                f"{location} requires expected_status/reason and optional references"
            )
        status = _non_empty(item["expected_status"], f"{location}.expected_status")
        if status not in EXPECTED_STATUSES:
            raise AcceptanceInputError(
                f"{location}.expected_status must be one of {sorted(EXPECTED_STATUSES)!r}"
            )
        reason = _non_empty(item["reason"], f"{location}.reason")
        raw_references = item.get("references", [])
        if not isinstance(raw_references, list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_references
        ):
            raise AcceptanceInputError(f"{location}.references must be an array of strings")
        labels[slot.slot_id] = ExpectedSlotLabel(
            expected_status=status,
            reason=reason,
            references=tuple(raw_references),
        )
    overall = _non_empty(
        data["overall_expected_status"],
        "expected.overall_expected_status",
    )
    if overall not in EXPECTED_OVERALL_STATUSES:
        raise AcceptanceInputError(
            "expected.overall_expected_status must be complete, incomplete or conflicted"
        )
    notes = data["notes"]
    if not isinstance(notes, str):
        raise AcceptanceInputError("expected.notes must be a string")
    return ExpectedDataset(
        dataset_id=dataset_id,
        description=description,
        contract_id=contract_id,
        slots=labels,
        overall_expected_status=overall,
        notes=notes,
    )


def check_evidence_readback(report: GapReport, documents: Sequence[object]) -> tuple[str, ...]:
    """Return stable errors for every report span that cannot be exactly read back."""

    document_by_id = {getattr(item, "document_id", None): item for item in documents}
    errors: list[str] = []
    for evidence in report.evidence:
        document = document_by_id.get(evidence.document_id)
        if document is None:
            errors.append(f"{evidence.evidence_id}: unknown document {evidence.document_id}")
            continue
        try:
            actual = document.slice(evidence.start_char, evidence.end_char)
        except (AttributeError, ValueError) as exc:
            errors.append(f"{evidence.evidence_id}: invalid range ({type(exc).__name__})")
            continue
        if actual != evidence.text:
            errors.append(f"{evidence.evidence_id}: text does not match canonical slice")
    return tuple(errors)


def _attribution(
    *,
    expected_status: str,
    actual_status: str,
    reason_codes: Sequence[str],
    missing_fields: Sequence[str],
    trace: Mapping[str, object],
) -> str | None:
    if expected_status == actual_status:
        return None
    if expected_status == "conflicting" and actual_status != "conflicting":
        return "CONFLICT_RULE_LIMIT"
    if trace.get("vector_candidate_count") == 0:
        return "RETRIEVAL_MISS"
    if trace.get("anchor_count") == 0:
        return "ANCHOR_MISS"
    if trace.get("exact_span_count") == 0:
        return "EXACT_LOCALIZATION_MISS"
    if "PROMOTIONAL_ONLY" in reason_codes:
        return "PROMOTIONAL_RULE_ERROR"
    if "UNVERIFIED_QUANTITATIVE_CLAIM" in reason_codes:
        return "QUANTITATIVE_RULE_ERROR"
    if missing_fields:
        return "FIELD_COVERAGE_MISS"
    return "MATERIAL_QUALITY"


def compare_report(
    expected: ExpectedDataset,
    report: GapReport,
    *,
    evidence_readback_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Compare status only; it never changes report or verifier decisions."""

    if expected.contract_id != report.contract_id:
        raise AcceptanceInputError("expected and actual contract IDs differ")
    actual_by_slot = {item.slot_id: item for item in report.slots}
    missing_output = tuple(
        slot_id for slot_id in expected.slots if slot_id not in actual_by_slot
    )
    unknown_output = tuple(
        slot_id for slot_id in actual_by_slot if slot_id not in expected.slots
    )
    per_slot_trace = report.trace.get("per_slot", {})
    if not isinstance(per_slot_trace, Mapping):
        per_slot_trace = {}
    rows: list[dict[str, object]] = []
    exact_matches = 0
    false_positives = 0
    conflict_missed = 0
    for slot_id, label in expected.slots.items():
        actual = actual_by_slot.get(slot_id)
        if actual is None:
            rows.append(
                {
                    "slot_id": slot_id,
                    "expected_status": label.expected_status,
                    "expected_reason": label.reason,
                    "expected_references": list(label.references),
                    "actual_status": None,
                    "match": False,
                    "actual_reason_codes": [],
                    "actual_missing_fields": [],
                    "actual_evidence_ids": [],
                    "error_attribution": "MATERIAL_QUALITY",
                }
            )
            continue
        matched = label.expected_status == actual.status.value
        exact_matches += int(matched)
        is_false_positive = (
            label.expected_status in {"weak", "missing", "conflicting"}
            and actual.status is SlotStatus.FILLED
        )
        is_conflict_missed = (
            label.expected_status == "conflicting"
            and actual.status is not SlotStatus.CONFLICTING
        )
        false_positives += int(is_false_positive)
        conflict_missed += int(is_conflict_missed)
        trace = per_slot_trace.get(slot_id, {})
        if not isinstance(trace, Mapping):
            trace = {}
        rows.append(
            {
                "slot_id": slot_id,
                "expected_status": label.expected_status,
                "expected_reason": label.reason,
                "expected_references": list(label.references),
                "actual_status": actual.status.value,
                "match": matched,
                "actual_reason_codes": list(actual.reason_codes),
                "actual_missing_fields": list(actual.missing_fields),
                "actual_evidence_ids": list(actual.evidence_ids),
                "error_attribution": _attribution(
                    expected_status=label.expected_status,
                    actual_status=actual.status.value,
                    reason_codes=actual.reason_codes,
                    missing_fields=actual.missing_fields,
                    trace=trace,
                ),
            }
        )
    total_slots = len(expected.slots)
    return {
        "dataset_id": expected.dataset_id,
        "description": expected.description,
        "contract_id": expected.contract_id,
        "overall_expected_status": expected.overall_expected_status,
        "overall_actual_status": report.status,
        "overall_status_match": expected.overall_expected_status == report.status,
        "slots": rows,
        "metrics": {
            "total_slots": total_slots,
            "exact_status_matches": exact_matches,
            "status_match_rate": exact_matches / total_slots if total_slots else 0.0,
            "filled_false_positive_count": false_positives,
            "conflict_missed_count": conflict_missed,
            "evidence_readback_error_count": len(evidence_readback_errors),
            "missing_slot_output_count": len(missing_output),
            "unknown_slot_output_count": len(unknown_output),
        },
        "evidence_readback_errors": list(evidence_readback_errors),
    }


def build_summary(comparisons: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ordered = sorted(comparisons, key=lambda item: str(item["dataset_id"]))
    totals = {
        "total_slots": 0,
        "exact_status_matches": 0,
        "filled_false_positive_count": 0,
        "conflict_missed_count": 0,
        "evidence_readback_error_count": 0,
        "missing_slot_output_count": 0,
        "pipeline_error_count": 0,
        "overall_status_matches": 0,
    }
    datasets: list[dict[str, object]] = []
    for comparison in ordered:
        metrics = comparison.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("comparison metrics must be a mapping")
        for key in (
            "total_slots",
            "exact_status_matches",
            "filled_false_positive_count",
            "conflict_missed_count",
            "evidence_readback_error_count",
            "missing_slot_output_count",
        ):
            totals[key] += int(metrics[key])
        totals["overall_status_matches"] += int(bool(comparison["overall_status_match"]))
        datasets.append(
            {
                "dataset_id": comparison["dataset_id"],
                "overall_status_match": comparison["overall_status_match"],
                "status_match_rate": metrics["status_match_rate"],
                "filled_false_positive_count": metrics["filled_false_positive_count"],
                "conflict_missed_count": metrics["conflict_missed_count"],
            }
        )
    total_slots = totals["total_slots"]
    hard_gate_passed = bool(ordered) and (
        totals["evidence_readback_error_count"] == 0
        and totals["pipeline_error_count"] == 0
        and totals["missing_slot_output_count"] == 0
        and totals["filled_false_positive_count"] == 0
    )
    return {
        "dataset_count": len(ordered),
        **totals,
        "status_match_rate": (
            totals["exact_status_matches"] / total_slots if total_slots else 0.0
        ),
        "hard_gate_passed": hard_gate_passed,
        "datasets": datasets,
    }


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(content.rstrip("\n") + "\n")


def render_comparison_markdown(comparison: Mapping[str, object]) -> str:
    metrics = comparison["metrics"]
    rows = comparison["slots"]
    assert isinstance(metrics, Mapping) and isinstance(rows, list)
    lines = [
        f"# Evidence Runtime 验收比较：{comparison['dataset_id']}",
        "",
        f"- 人工总体状态：`{comparison['overall_expected_status']}`",
        f"- 系统总体状态：`{comparison['overall_actual_status']}`",
        f"- 总体匹配：`{str(comparison['overall_status_match']).lower()}`",
        f"- Slot 匹配率：{float(metrics['status_match_rate']):.4f}",
        f"- Filled false positive：{metrics['filled_false_positive_count']}",
        f"- Conflict missed：{metrics['conflict_missed_count']}",
        f"- Evidence 回读错误：{metrics['evidence_readback_error_count']}",
        "",
        "## 分槽比较",
        "",
        "| Slot | Expected | Actual | Match | Attribution |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['slot_id']}` | `{row['expected_status']}` | "
            f"`{row['actual_status']}` | {str(row['match']).lower()} | "
            f"{row['error_attribution'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def render_summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Evidence Runtime 私有材料验收汇总",
        "",
        f"- 数据集：{summary['dataset_count']}",
        f"- Slot 总数：{summary['total_slots']}",
        f"- 精确状态匹配：{summary['exact_status_matches']}",
        f"- 状态匹配率：{float(summary['status_match_rate']):.4f}",
        f"- Filled false positive：{summary['filled_false_positive_count']}",
        f"- Conflict missed：{summary['conflict_missed_count']}",
        f"- Evidence 回读错误：{summary['evidence_readback_error_count']}",
        f"- Pipeline 错误：{summary['pipeline_error_count']}",
        f"- 缺失 Slot 输出：{summary['missing_slot_output_count']}",
        f"- 硬门槛通过：`{str(summary['hard_gate_passed']).lower()}`",
        "",
    ]
    return "\n".join(lines)


_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/-]{6,})"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|密码)\s*[:=]\s*)([^\s,;]+)"),
)
_URL_QUERY_PATTERN = re.compile(r"(?i)(https?://[^\s?#]+)\?[^\s]+")


def redact_secrets(text: str, *, explicit_secrets: Sequence[str] = ()) -> str:
    result = text
    for secret in explicit_secrets:
        if secret:
            result = result.replace(secret, "***")
    result = _URL_QUERY_PATTERN.sub(lambda match: match.group(1) + "?***", result)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda match: match.group(1) + "***", result)
    return result


def _require_credential_free(text: str, settings: Settings) -> None:
    if redact_secrets(
        text,
        explicit_secrets=(settings.embedding_api_key,),
    ) != text:
        raise RuntimeError(
            "acceptance output contains credential-shaped content; redact source materials first"
        )


def _real_checker(settings: Settings) -> EvidenceGapChecker:
    """验收必须跑**出厂默认配置**，不得自组一套。

    此处曾手工构造 ``EvidenceGapChecker`` 并漏传 ``single_doc_cap_ratio``，
    导致验收与公开 CLI 使用不同配置。现在直接委托出厂装配函数，保持配置同源。
    """

    return build_default_gap_checker(settings)


def _library_documents_and_checker(
    *,
    dataset_dir: Path,
    library_root: Path,
    settings: Settings,
) -> tuple[tuple[object, ...], EvidenceGapChecker]:
    """把数据集材料装进 L1 项目，返回 (documents, 该项目的 checker)。"""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    library = FilesystemLibrary(library_root, dataset_dir.name)
    library.init_project()
    for entry in manifest["documents"]:
        relative = entry.get("normalized_path") or entry["source_path"]
        library.add_material(dataset_dir / relative, source_type=SourceType.OTHER)
    documents = library.documents()
    if len(documents) != len(manifest["documents"]):
        raise AcceptanceInputError(
            f"dataset {dataset_dir.name!r}: library holds {len(documents)} current materials "
            f"but manifest declares {len(manifest['documents'])}"
        )
    project_settings = replace(settings, vector_db_path=str(library.index_root()))
    library.mark_indexed()
    return documents, _real_checker(project_settings)


def run_acceptance(
    *,
    dataset_root: Path,
    output_dir: Path,
    contract: EvidenceContract,
    settings: Settings,
    max_chars: int,
    top_k_per_slot: int,
    context_lines: int,
    max_span_chars: int,
    max_results_per_slot: int,
    library_root: Path | None = None,
) -> dict[str, object]:
    """``library_root`` 非空时使用 L1 项目模式。

    每个数据集建一个同名 L1 项目，把 ``manifest.json`` 指向的规范化材料入库，改由
    ``library.documents()`` 供料，向量索引落在该项目自己的 ``index/`` 下。material_id 是
    **内容派生哈希**、与 manifest 里的人写 slug id 完全不同——因此逐槽结果一致即证明
    L1 是地基替换而非判断变更（A==B 确定性经由 L1 再验一次）。
    """

    if not dataset_root.is_dir():
        raise AcceptanceInputError(f"dataset root does not exist: {str(dataset_root)!r}")
    dataset_dirs = tuple(
        path
        for path in sorted(dataset_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    )
    if not dataset_dirs:
        raise AcceptanceInputError("dataset root contains no dataset directories")
    for dataset_dir in dataset_dirs:
        missing = tuple(
            name
            for name in ("manifest.json", "expected.json")
            if not (dataset_dir / name).is_file()
        )
        if missing:
            raise AcceptanceInputError(
                f"dataset {dataset_dir.name!r} is missing required files: {missing!r}"
            )
    checker = None if library_root is not None else _real_checker(settings)
    comparisons: list[dict[str, object]] = []
    for dataset_dir in dataset_dirs:
        expected = load_expected_labels(dataset_dir / "expected.json", contract)
        if expected.dataset_id != dataset_dir.name:
            raise AcceptanceInputError(
                f"dataset_id {expected.dataset_id!r} must match directory {dataset_dir.name!r}"
            )
        if library_root is None:
            documents = load_normalized_document_manifest(dataset_dir / "manifest.json")
        else:
            documents, checker = _library_documents_and_checker(
                dataset_dir=dataset_dir,
                library_root=library_root,
                settings=settings,
            )
        run = checker.run(
            contract=contract,
            documents=documents,
            max_chars=max_chars,
            top_k_per_slot=top_k_per_slot,
            context_lines=context_lines,
            max_span_chars=max_span_chars,
            max_results_per_slot=max_results_per_slot,
        )
        readback_errors = check_evidence_readback(run.report, documents)
        comparison = compare_report(
            expected,
            run.report,
            evidence_readback_errors=readback_errors,
        )
        dataset_output = output_dir / expected.dataset_id
        dataset_output.mkdir(parents=True, exist_ok=True)
        report_json = gap_report_to_json(run.report)
        report_markdown = render_gap_report_markdown(run.report)
        comparison_json = _json_text(comparison)
        comparison_markdown = render_comparison_markdown(comparison)
        for content in (
            report_json,
            report_markdown,
            comparison_json,
            comparison_markdown,
        ):
            _require_credential_free(content, settings)
        _write_text(dataset_output / "actual-gap-report.json", report_json)
        _write_text(dataset_output / "actual-gap-report.md", report_markdown)
        _write_text(dataset_output / "comparison.json", comparison_json)
        _write_text(
            dataset_output / "comparison.md",
            comparison_markdown,
        )
        comparisons.append(comparison)
    summary = build_summary(comparisons)
    summary_json = _json_text(summary)
    summary_markdown = render_summary_markdown(summary)
    _require_credential_free(summary_json, settings)
    _require_credential_free(summary_markdown, settings)
    _write_text(output_dir / "acceptance-summary.json", summary_json)
    _write_text(output_dir / "acceptance-summary.md", summary_markdown)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private real-material acceptance")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--top-k-per-slot", type=int, default=8)
    parser.add_argument("--context-lines", type=int, default=2)
    parser.add_argument("--max-span-chars", type=int, default=1200)
    parser.add_argument("--max-results-per-slot", type=int, default=20)
    parser.add_argument(
        "--via-library",
        metavar="LIBRARY_ROOT",
        help="改由 L1 材料库供料（每数据集一个同名项目，索引隔离在项目 index/ 下）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_argument_parser().parse_args(argv)
        contract = load_evidence_contract(args.contract)
        summary = run_acceptance(
            dataset_root=Path(args.dataset_root),
            output_dir=Path(args.output_dir),
            contract=contract,
            settings=DEFAULT_SETTINGS,
            max_chars=args.max_chars,
            top_k_per_slot=args.top_k_per_slot,
            context_lines=args.context_lines,
            max_span_chars=args.max_span_chars,
            max_results_per_slot=args.max_results_per_slot,
            library_root=Path(args.via_library) if args.via_library else None,
        )
    except Exception as exc:
        safe = redact_secrets(
            f"{type(exc).__name__}: {exc}",
            explicit_secrets=(DEFAULT_SETTINGS.embedding_api_key,),
        )
        print(f"error: {safe}", file=sys.stderr)
        return 2
    print(f"datasets: {summary['dataset_count']}")
    print(f"status_match_rate: {float(summary['status_match_rate']):.4f}")
    print(f"filled_false_positive: {summary['filled_false_positive_count']}")
    print(f"conflict_missed: {summary['conflict_missed_count']}")
    print(f"hard_gate_passed: {str(summary['hard_gate_passed']).lower()}")
    return 0 if summary["hard_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
