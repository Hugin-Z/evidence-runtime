"""Gap Report JSON 序列化。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..types import EvidenceContract, EvidenceSpan, SlotStatus
from ..verifier import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
)

from .models import GapReport, SlotGapResult, _json_clone


def _fact_to_dict(fact: DetectedFact) -> dict[str, object]:
    return {
        "subject": fact.subject,
        "predicate": fact.predicate,
        "value": fact.value,
        "normalized_value": fact.normalized_value,
        "evidence_id": fact.evidence_id,
    }


def _conflict_to_dict(conflict: DetectedConflict) -> dict[str, object]:
    return {
        "conflict_type": conflict.conflict_type,
        "subject": conflict.subject,
        "predicate": conflict.predicate,
        "facts": [_fact_to_dict(item) for item in conflict.facts],
        "evidence_ids": list(conflict.evidence_ids),
        "reason": conflict.reason,
    }


def gap_report_to_dict(report: GapReport) -> dict[str, object]:
    if not isinstance(report, GapReport):
        raise ValueError("report must be GapReport")
    return {
        "contract_id": report.contract_id,
        "contract_version": report.contract_version,
        "status": report.status,
        "ready": report.ready,
        "slots": [
            {
                "slot_id": slot.slot_id,
                "slot_name": slot.slot_name,
                "required": slot.required,
                "status": slot.status.value,
                "evidence_ids": list(slot.evidence_ids),
                "field_coverage": [
                    {
                        "field_id": coverage.field_id,
                        "covered": coverage.covered,
                        "evidence_ids": list(coverage.evidence_ids),
                        "reason": coverage.reason,
                    }
                    for coverage in slot.field_coverage
                ],
                "covered_fields": list(slot.covered_fields),
                "missing_fields": list(slot.missing_fields),
                "reason_codes": list(slot.reason_codes),
                "reason": slot.reason,
                "conflicts": [_conflict_to_dict(item) for item in slot.conflicts],
                "material_request": slot.material_request,
            }
            for slot in report.slots
        ],
        "evidence": [
            {
                "evidence_id": evidence.evidence_id,
                "document_id": evidence.document_id,
                "text": evidence.text,
                "start_char": evidence.start_char,
                "end_char": evidence.end_char,
                "start_line": evidence.start_line,
                "end_line": evidence.end_line,
                "page": evidence.page,
                "discovery_sources": list(evidence.discovery_sources),
            }
            for evidence in report.evidence
        ],
        "pending": [
            {
                "slot_id": item.slot_id,
                "slot_name": item.slot_name,
                "action": item.action.value,
                "missing_fields": list(item.missing_fields),
                "existing_evidence_ids": list(item.existing_evidence_ids),
                "description": item.description,
                "priority": item.priority,
            }
            for item in report.pending
        ],
        "trace": _json_clone(report.trace, "GapReport.trace"),
    }


def gap_report_to_json(report: GapReport) -> str:
    return json.dumps(
        gap_report_to_dict(report),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"


def write_gap_report_json(report: GapReport, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as file:
        file.write(gap_report_to_json(report))
