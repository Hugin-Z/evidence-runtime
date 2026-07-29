"""Mechanical conversion audit for private acceptance materials.

The audit is deliberately separate from production ingestion and retrieval.  It
compares already-normalized Markdown/TXT with the immutable source inventory,
emits no source excerpts, and never changes source files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Mapping, Sequence


class AutomatedReviewStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TextAudit:
    status: AutomatedReviewStatus
    human_review_required: bool
    text_length: int
    line_count: int
    replacement_character_count: int
    control_character_count: int
    abnormal_character_ratio: float
    duplicate_line_ratio: float
    longest_line_length: int
    heading_count: int
    markdown_table_row_count: int
    numeric_token_count: int
    amount_or_unit_token_count: int
    detected_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = self.status.value
        value["detected_issues"] = list(self.detected_issues)
        return value


_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")
_AMOUNT_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|元|万元|亿元|台|个|人|户|月|年|天|GB|TB|MB|Mbps)",
    re.IGNORECASE,
)


def _abnormal_character_count(text: str) -> int:
    return sum(
        1
        for char in text
        if char == "\ufffd"
        or unicodedata.category(char) in {"Co", "Cs"}
        or (unicodedata.category(char) == "Cc" and char not in "\n\r\t")
    )


def _table_widths(text: str) -> tuple[int, ...]:
    return tuple(
        max(0, line.count("|") - 1)
        for line in text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    )


def audit_normalized_text(text: str) -> TextAudit:
    """Classify normalized text without retaining any source excerpt."""

    if not isinstance(text, str):
        raise ValueError("text must be str")
    lines = text.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    counts = Counter(nonempty)
    duplicate_instances = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_ratio = duplicate_instances / len(nonempty) if nonempty else 0.0
    replacement_count = text.count("\ufffd")
    control_count = sum(
        1
        for char in text
        if unicodedata.category(char) == "Cc" and char not in "\n\r\t"
    )
    abnormal_count = _abnormal_character_count(text)
    abnormal_ratio = abnormal_count / len(text) if text else 0.0
    table_widths = _table_widths(text)
    issues: list[str] = []
    blockers: list[str] = []
    if not text.strip():
        blockers.append("empty_normalized_text")
    elif len(text.strip()) < 100:
        blockers.append("abnormally_short_document")
    elif len(text.strip()) < 400:
        issues.append("short_document_requires_context_caution")
    if replacement_count >= 10 or abnormal_ratio > 0.02:
        blockers.append("excessive_garbled_or_abnormal_characters")
    elif replacement_count or abnormal_count:
        issues.append("isolated_abnormal_characters")
    if duplicate_ratio > 0.25:
        issues.append("repeated_headers_or_lines")
    longest_line = max((len(line) for line in lines), default=0)
    if longest_line > 5000:
        issues.append("continuous_very_long_line")
    if table_widths and len(set(table_widths)) > 1:
        issues.append("markdown_table_column_count_inconsistent")
    if blockers:
        status = AutomatedReviewStatus.BLOCKED
    elif issues:
        status = AutomatedReviewStatus.WARNING
    else:
        status = AutomatedReviewStatus.PASSED
    return TextAudit(
        status=status,
        human_review_required=status is AutomatedReviewStatus.BLOCKED,
        text_length=len(text),
        line_count=len(lines),
        replacement_character_count=replacement_count,
        control_character_count=control_count,
        abnormal_character_ratio=abnormal_ratio,
        duplicate_line_ratio=duplicate_ratio,
        longest_line_length=longest_line,
        heading_count=len(_HEADING_RE.findall(text)),
        markdown_table_row_count=len(_TABLE_RE.findall(text)),
        numeric_token_count=len(_NUMBER_RE.findall(text)),
        amount_or_unit_token_count=len(_AMOUNT_UNIT_RE.findall(text)),
        detected_issues=tuple(dict.fromkeys(blockers + issues)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_status(
    text_audit: TextAudit,
    source_issues: Sequence[str],
    *,
    source_blocked: bool = False,
) -> tuple[AutomatedReviewStatus, bool]:
    if text_audit.status is AutomatedReviewStatus.BLOCKED or source_blocked:
        return AutomatedReviewStatus.BLOCKED, True
    if text_audit.status is AutomatedReviewStatus.WARNING or source_issues:
        return AutomatedReviewStatus.WARNING, False
    return AutomatedReviewStatus.PASSED, False


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _sample_overlap(sample: str, normalized: str, width: int = 6) -> float:
    source = _compact(sample)
    target = _compact(normalized)
    if len(source) < width:
        return 1.0 if source and source in target else 0.0
    shingles = {source[index:index + width] for index in range(len(source) - width + 1)}
    return sum(item in target for item in shingles) / len(shingles) if shingles else 0.0


def _comparable_pdf_sample_overlap(sample: str, normalized: str) -> float | None:
    """Ignore cover/closing samples too short to represent document body quality."""

    return _sample_overlap(sample, normalized) if len(sample.strip()) >= 20 else None


def _evaluate_pdf_overlaps(overlaps: Sequence[float]) -> tuple[list[str], bool]:
    if overlaps and max(overlaps) < 0.15:
        return ["pdf_sample_text_not_recoverable_from_markdown"], True
    if overlaps and min(overlaps) < 0.45:
        return ["pdf_sample_text_partially_recoverable"], False
    return [], False


def _audit_pdf(source: Path, text: str) -> tuple[dict[str, object], list[str], bool]:
    try:
        import pdfplumber
    except ImportError:
        return {"source_probe": "unavailable"}, ["pdf_source_probe_unavailable"], False
    page_lengths: list[int] = []
    overlaps: list[float] = []
    sampled_page_count = 0
    low_text_sample_count = 0
    with pdfplumber.open(source) as pdf:
        page_count = len(pdf.pages)
        sample_indices = sorted({0, page_count // 2, page_count - 1}) if page_count else []
        for index, page in enumerate(pdf.pages):
            extracted = page.extract_text() or ""
            page_lengths.append(len(extracted.strip()))
            if index in sample_indices:
                sampled_page_count += 1
                overlap = _comparable_pdf_sample_overlap(extracted, text)
                if overlap is None:
                    low_text_sample_count += 1
                else:
                    overlaps.append(overlap)
    text_pages = sum(value >= 20 for value in page_lengths)
    ratio = text_pages / len(page_lengths) if page_lengths else 0.0
    min_overlap = min(overlaps, default=0.0)
    issues: list[str] = []
    blocked = False
    if not page_lengths or ratio < 0.30:
        blocked = True
        issues.append("pdf_effective_text_layer_insufficient")
    elif ratio < 0.70:
        issues.append("pdf_contains_many_image_or_low_text_pages")
    overlap_issues, overlap_blocked = _evaluate_pdf_overlaps(overlaps)
    issues.extend(overlap_issues)
    blocked = blocked or overlap_blocked
    return {
        "page_count": len(page_lengths),
        "text_page_count": text_pages,
        "text_page_ratio": ratio,
        "sample_page_count": sampled_page_count,
        "comparable_sample_page_count": len(overlaps),
        "low_text_sample_page_count": low_text_sample_count,
        "minimum_sample_text_overlap": min_overlap,
        "maximum_sample_text_overlap": max(overlaps, default=0.0),
    }, issues, blocked


def _docx_counts(path: Path) -> dict[str, int]:
    from docx import Document

    document = Document(path)
    media_count = 0
    try:
        import zipfile

        with zipfile.ZipFile(path) as archive:
            media_count = sum(name.startswith("word/media/") for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        pass
    return {
        "paragraph_count": len(document.paragraphs),
        "table_count": len(document.tables),
        "embedded_media_count": media_count,
    }


def _audit_word(
    source: Path,
    text: str,
    conversion_record: Mapping[str, object],
    preparation_root: Path,
) -> tuple[dict[str, object], list[str], bool]:
    probe = source
    chain = "DOCX"
    if source.suffix.lower() == ".doc":
        probe = preparation_root / "intermediate" / f"{source.stem}.docx"
        chain = "DOC→Word DOCX→Markdown"
    issues: list[str] = []
    blocked = False
    metrics: dict[str, object] = {"conversion_chain": chain}
    if probe.is_file():
        metrics.update(_docx_counts(probe))
    else:
        issues.append("word_structure_probe_unavailable")
    baseline = conversion_record.get("structural_checks", {})
    if isinstance(baseline, Mapping):
        baseline_length = baseline.get("baseline_text_length")
        if isinstance(baseline_length, int) and baseline_length > 0:
            metrics["normalized_to_source_text_ratio"] = len(text) / baseline_length
            if len(text) / baseline_length < 0.35:
                blocked = True
                issues.append("word_body_text_severely_truncated")
            elif len(text) / baseline_length < 0.70:
                issues.append("word_body_text_partially_truncated")
    table_count = metrics.get("table_count", 0)
    if isinstance(table_count, int) and table_count > 0 and not _TABLE_RE.search(text):
        issues.append("word_tables_not_preserved_as_markdown_tables")
    if metrics.get("embedded_media_count", 0):
        issues.append("embedded_image_text_not_semantically_audited")
    return metrics, issues, blocked


def _audit_xlsx(source: Path, text: str) -> tuple[dict[str, object], list[str], bool]:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return {"source_probe": "unavailable"}, ["xlsx_source_probe_unavailable"], False
    values = load_workbook(source, data_only=True, read_only=False)
    formulas = load_workbook(source, data_only=False, read_only=False)
    issues: list[str] = []
    sheet_names = values.sheetnames
    missing_sheet_names = [name for name in sheet_names if f"工作表：{name}" not in text]
    merged_count = sum(len(sheet.merged_cells.ranges) for sheet in values.worksheets)
    formula_count = 0
    missing_formula_display = 0
    samples = 0
    recovered = 0
    order_offsets: list[int] = []
    for value_sheet, formula_sheet in zip(values.worksheets, formulas.worksheets):
        for row in formula_sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formula_count += 1
                    if value_sheet[cell.coordinate].value is None:
                        missing_formula_display += 1
        for row in value_sheet.iter_rows():
            for cell in row:
                if cell.value in (None, "") or samples >= 80:
                    continue
                token = str(cell.value).strip()
                if len(token) < 2:
                    continue
                samples += 1
                offset = text.find(token)
                if offset >= 0:
                    recovered += 1
                    order_offsets.append(offset)
    values.close()
    formulas.close()
    recovery_ratio = recovered / samples if samples else 0.0
    blocked = bool(missing_sheet_names) or recovery_ratio < 0.35 or missing_formula_display > 0
    if missing_sheet_names:
        issues.append("xlsx_sheet_names_missing")
    if recovery_ratio < 0.35:
        issues.append("xlsx_display_values_severely_missing")
    elif recovery_ratio < 0.75:
        issues.append("xlsx_display_values_partially_missing_or_reformatted")
    if missing_formula_display:
        issues.append("xlsx_formula_display_values_missing")
    if merged_count:
        issues.append("xlsx_merged_cell_layout_flattened")
    if any(right < left for left, right in zip(order_offsets, order_offsets[1:])):
        issues.append("xlsx_sample_value_order_requires_caution")
    return {
        "sheet_count": len(sheet_names),
        "sheet_names": sheet_names,
        "merged_range_count": merged_count,
        "formula_count": formula_count,
        "formula_display_missing_count": missing_formula_display,
        "sample_value_count": samples,
        "sample_value_recovery_ratio": recovery_ratio,
    }, issues, blocked


def audit_material(
    *,
    source: Path,
    normalized: Path,
    conversion_record: Mapping[str, object],
    preparation_root: Path,
) -> dict[str, object]:
    raw = normalized.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = ""
        text_audit = audit_normalized_text(text)
        return {
            "source_filename": source.name,
            "normalized_filename": normalized.name,
            **text_audit.to_dict(),
            "detected_issues": ["normalized_text_not_utf8"],
            "source_checks": {},
        }
    text_audit = audit_normalized_text(text)
    source_checks: dict[str, object]
    source_issues: list[str]
    source_blocked: bool
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        source_checks, source_issues, source_blocked = _audit_pdf(source, text)
    elif suffix in {".doc", ".docx"}:
        source_checks, source_issues, source_blocked = _audit_word(
            source, text, conversion_record, preparation_root
        )
    elif suffix == ".xlsx":
        source_checks, source_issues, source_blocked = _audit_xlsx(source, text)
    else:
        source_checks, source_issues, source_blocked = {}, [], False
    status, human_required = _merge_status(
        text_audit, source_issues, source_blocked=source_blocked
    )
    combined_issues = list(dict.fromkeys([*text_audit.detected_issues, *source_issues]))
    return {
        "source_filename": source.name,
        "source_sha256": _sha256(source),
        "normalized_filename": normalized.name,
        "normalized_sha256": _sha256(normalized),
        "automated_review_status": status.value,
        "human_review_required": human_required,
        "text_checks": text_audit.to_dict(),
        "source_checks": source_checks,
        "detected_issues": combined_issues,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def render_audit_markdown(payload: Mapping[str, object]) -> str:
    files = payload["files"]
    assert isinstance(files, list)
    summary = payload["summary"]
    assert isinstance(summary, Mapping)
    lines = [
        "# 自动转换审计",
        "",
        f"- passed：{summary['passed']}",
        f"- warning：{summary['warning']}",
        f"- blocked：{summary['blocked']}",
        "",
        "| 文件 | 状态 | 需要人工 | 问题类型 |",
        "|---|---|---:|---|",
    ]
    for item in files:
        assert isinstance(item, Mapping)
        issues = item.get("detected_issues", [])
        rendered = ", ".join(str(value) for value in issues) if issues else "无"
        lines.append(
            f"| `{item['source_filename']}` | `{item['automated_review_status']}` | "
            f"{str(item['human_review_required']).lower()} | {rendered} |"
        )
    return "\n".join(lines) + "\n"


def run_conversion_audit(
    *,
    source_root: Path,
    inventory_path: Path,
    conversion_report_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    conversion = json.loads(conversion_report_path.read_text(encoding="utf-8-sig"))
    inventory_by_name = {item["filename"]: item for item in inventory["files"]}
    conversion_by_name = {
        item["source_filename"]: item for item in conversion["files"]
    }
    if set(inventory_by_name) != set(conversion_by_name):
        raise ValueError("inventory and conversion report filenames differ")
    records: list[dict[str, object]] = []
    for filename in sorted(inventory_by_name):
        source = source_root / filename
        if not source.is_file():
            raise ValueError(f"source file is missing: {filename!r}")
        expected_hash = inventory_by_name[filename]["sha256"]
        if _sha256(source) != expected_hash:
            raise ValueError(f"source hash changed: {filename!r}")
        normalized = Path(conversion_by_name[filename]["normalized_path"])
        if not normalized.is_file():
            raise ValueError(f"normalized file is missing: {filename!r}")
        record = audit_material(
            source=source,
            normalized=normalized,
            conversion_record=conversion_by_name[filename],
            preparation_root=conversion_report_path.resolve().parent,
        )
        records.append(record)
        inventory_by_name[filename]["path"] = str(source.resolve())
        conversion_by_name[filename]["source_path"] = str(source.resolve())
        conversion_by_name[filename]["automated_review_status"] = record[
            "automated_review_status"
        ]
        conversion_by_name[filename]["human_review_required"] = record[
            "human_review_required"
        ]
    summary = {
        status: sum(item["automated_review_status"] == status for item in records)
        for status in ("passed", "warning", "blocked")
    }
    payload = {"summary": summary, "files": records}
    _write_json(output_json, payload)
    output_markdown.write_text(render_audit_markdown(payload), encoding="utf-8")
    inventory["source_root"] = str(source_root)
    _write_json(inventory_path, inventory)
    _write_json(conversion_report_path, conversion)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit private material conversions")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--conversion-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    payload = run_conversion_audit(
        source_root=Path(args.source_root),
        inventory_path=Path(args.inventory),
        conversion_report_path=Path(args.conversion_report),
        output_json=Path(args.output_json),
        output_markdown=Path(args.output_markdown),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0 if payload["summary"]["blocked"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
