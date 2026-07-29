"""Gap Report 的数据类型。"""

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


class PendingActionType(str, Enum):
    """The two actions the gap checker can request from a person."""

    REQUEST_MATERIAL = "request_material"
    RESOLVE_CONFLICT = "resolve_conflict"


@dataclass(frozen=True)
class PendingMaterialRequest:
    slot_id: str
    slot_name: str
    action: PendingActionType
    missing_fields: tuple[str, ...]
    existing_evidence_ids: tuple[str, ...]
    description: str
    priority: int

    def __post_init__(self) -> None:
        if not self.slot_id or not self.slot_name or not self.description:
            raise ValueError("pending material request strings must not be empty")
        if not isinstance(self.action, PendingActionType):
            raise ValueError("pending material request action is invalid")
        if type(self.priority) is not int or self.priority < 1:
            raise ValueError("pending material request priority must be a positive integer")
        object.__setattr__(self, "missing_fields", tuple(self.missing_fields))
        object.__setattr__(
            self,
            "existing_evidence_ids",
            tuple(self.existing_evidence_ids),
        )


@dataclass(frozen=True, init=False)
class EvidenceReference:
    """Serializable provenance that can only be copied from an EvidenceSpan."""

    evidence_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    page: int | None
    discovery_sources: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("EvidenceReference must be created from an EvidenceSpan")

    @classmethod
    def from_span(cls, span: EvidenceSpan) -> "EvidenceReference":
        if not isinstance(span, EvidenceSpan):
            raise ValueError(f"span must be EvidenceSpan; got {span!r}")
        instance = object.__new__(cls)
        for field_name in (
            "evidence_id",
            "document_id",
            "text",
            "start_char",
            "end_char",
            "start_line",
            "end_line",
            "page",
            "discovery_sources",
        ):
            object.__setattr__(instance, field_name, getattr(span, field_name))
        return instance


@dataclass(frozen=True, init=False)
class SlotGapResult:
    """Read-only projection of one verifier result plus its contract metadata."""

    slot_id: str
    slot_name: str
    required: bool
    status: SlotStatus
    evidence_ids: tuple[str, ...]
    field_coverage: tuple[FieldCoverage, ...]
    covered_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    reason: str
    conflicts: tuple[DetectedConflict, ...]
    material_request: str

    def __init__(self) -> None:
        raise TypeError("SlotGapResult must be projected from verification output")

    @classmethod
    def from_verification(
        cls,
        contract: EvidenceContract,
        result: SlotVerificationResult,
    ) -> "SlotGapResult":
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        if not isinstance(result, SlotVerificationResult):
            raise ValueError("result must be SlotVerificationResult")
        if result.contract_id != contract.contract_id:
            raise ValueError("slot verification contract_id does not match contract")
        slot = contract.get_slot(result.slot_id)
        instance = object.__new__(cls)
        values = {
            "slot_id": slot.slot_id,
            "slot_name": slot.name,
            "required": slot.required,
            "status": result.status,
            "evidence_ids": result.evidence_ids,
            "field_coverage": result.field_coverage,
            "covered_fields": tuple(
                item.field_id for item in result.field_coverage if item.covered
            ),
            "missing_fields": tuple(
                item.field_id for item in result.field_coverage if not item.covered
            ),
            "reason_codes": tuple(item.value for item in result.reason_codes),
            "reason": result.reason,
            "conflicts": result.conflicts,
            "material_request": slot.material_request,
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance


@dataclass(frozen=True)
class GapReport:
    contract_id: str
    contract_version: str
    status: str
    ready: bool
    slots: tuple[SlotGapResult, ...]
    evidence: tuple[EvidenceReference, ...]
    pending: tuple[PendingMaterialRequest, ...]
    trace: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_version:
            raise ValueError("GapReport contract identity must not be empty")
        if self.status not in {"complete", "incomplete", "conflicted"}:
            raise ValueError(f"invalid gap report status {self.status!r}")
        if type(self.ready) is not bool or self.ready != (self.status == "complete"):
            raise ValueError("GapReport.ready must be true exactly when status is complete")
        slots = tuple(self.slots)
        evidence = tuple(self.evidence)
        pending = tuple(self.pending)
        if any(not isinstance(item, SlotGapResult) for item in slots):
            raise ValueError("GapReport.slots must contain SlotGapResult objects")
        if any(not isinstance(item, EvidenceReference) for item in evidence):
            raise ValueError("GapReport.evidence must contain EvidenceReference objects")
        if any(not isinstance(item, PendingMaterialRequest) for item in pending):
            raise ValueError(
                "GapReport.pending must contain PendingMaterialRequest objects"
            )
        slot_ids = tuple(item.slot_id for item in slots)
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("GapReport.slots contains duplicate slot IDs")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("GapReport.evidence contains duplicate evidence IDs")
        available_evidence = set(evidence_ids)
        if any(
            not set(slot.evidence_ids) <= available_evidence for slot in slots
        ):
            raise ValueError("GapReport slot references unresolved evidence")
        available_slots = set(slot_ids)
        if any(item.slot_id not in available_slots for item in pending):
            raise ValueError("GapReport pending request references an unknown slot")
        if any(
            not set(item.existing_evidence_ids) <= available_evidence
            for item in pending
        ):
            raise ValueError("GapReport pending request references unresolved evidence")
        if not isinstance(self.trace, Mapping):
            raise ValueError("GapReport.trace must be a mapping")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "pending", pending)
        object.__setattr__(self, "trace", _json_clone(self.trace, "GapReport.trace"))


def _json_clone(value: object, field_name: str) -> object:
    """Validate strict JSON data and detach it from caller-owned containers."""

    def validate(item: object, location: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{location} is not strict JSON: non-finite number")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{location} is not strict JSON: non-string key {key!r}"
                    )
                validate(child, f"{location}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                validate(child, f"{location}[{index}]")
            return
        raise ValueError(f"{location} is not JSON serializable: {item!r}")

    validate(value, field_name)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
