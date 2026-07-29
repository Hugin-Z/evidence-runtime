"""Verifier 的结果数据类型：字段覆盖、事实、冲突、槽与契约级结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import unicodedata

from ..contract import solution_material_completeness_v1
from ..types import EvidenceContract, EvidenceSpan, SlotStatus


class VerificationReasonCode(str, Enum):
    """机器稳定的 Slot 验证理由码。"""

    NO_EVIDENCE = "NO_EVIDENCE"
    INSUFFICIENT_EVIDENCE_COUNT = "INSUFFICIENT_EVIDENCE_COUNT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    PROMOTIONAL_ONLY = "PROMOTIONAL_ONLY"
    UNVERIFIED_QUANTITATIVE_CLAIM = "UNVERIFIED_QUANTITATIVE_CLAIM"
    MISSING_SOURCE_CONTEXT = "MISSING_SOURCE_CONTEXT"
    CONFLICTING_FACTS = "CONFLICTING_FACTS"


class _FrozenJsonDict(dict):
    """保持 ``json.dumps`` 兼容的只读 dict。"""

    @staticmethod
    def _blocked(*args, **kwargs):
        raise TypeError("verification trace is immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked
    __ior__ = _blocked


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenJsonDict({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"trace value is not JSON serializable: {value!r}")


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value


def _stable_strings(
    values: object,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of strings") from exc
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must not be empty")
    seen: set[str] = set()
    for index, item in enumerate(items):
        _non_empty(item, f"{field_name}[{index}]")
        if item in seen:
            raise ValueError(f"{field_name} contains duplicate value {item!r}")
        seen.add(item)
    return items


@dataclass(frozen=True)
class FieldCoverage:
    field_id: str
    covered: bool
    evidence_ids: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.field_id, "FieldCoverage.field_id")
        if type(self.covered) is not bool:
            raise ValueError("FieldCoverage.covered must be bool")
        evidence_ids = _stable_strings(
            self.evidence_ids,
            "FieldCoverage.evidence_ids",
        )
        if self.covered and not evidence_ids:
            raise ValueError("covered FieldCoverage must reference evidence")
        if not self.covered and evidence_ids:
            raise ValueError("uncovered FieldCoverage cannot reference evidence")
        if self.reason is not None:
            _non_empty(self.reason, "FieldCoverage.reason")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class DetectedFact:
    subject: str
    predicate: str
    value: str
    normalized_value: str
    evidence_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "subject",
            "predicate",
            "value",
            "normalized_value",
            "evidence_id",
        ):
            _non_empty(getattr(self, field_name), f"DetectedFact.{field_name}")


@dataclass(frozen=True)
class DetectedConflict:
    conflict_type: str
    subject: str
    predicate: str
    facts: tuple[DetectedFact, ...]
    evidence_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.conflict_type not in {"opposite_assertion", "scalar_mismatch"}:
            raise ValueError(f"unknown conflict_type {self.conflict_type!r}")
        _non_empty(self.subject, "DetectedConflict.subject")
        _non_empty(self.predicate, "DetectedConflict.predicate")
        _non_empty(self.reason, "DetectedConflict.reason")
        facts = tuple(self.facts)
        if len(facts) < 2 or any(not isinstance(item, DetectedFact) for item in facts):
            raise ValueError("DetectedConflict.facts must contain at least two facts")
        evidence_ids = _stable_strings(
            self.evidence_ids,
            "DetectedConflict.evidence_ids",
            allow_empty=False,
        )
        if not set(evidence_ids) <= {item.evidence_id for item in facts}:
            raise ValueError("DetectedConflict evidence IDs must come from facts")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class SlotVerificationResult:
    contract_id: str
    slot_id: str
    status: SlotStatus
    evidence_ids: tuple[str, ...]
    field_coverage: tuple[FieldCoverage, ...]
    reason_codes: tuple[VerificationReasonCode, ...]
    reason: str
    conflicts: tuple[DetectedConflict, ...]
    trace: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        _non_empty(self.contract_id, "SlotVerificationResult.contract_id")
        _non_empty(self.slot_id, "SlotVerificationResult.slot_id")
        if not isinstance(self.status, SlotStatus):
            raise ValueError("SlotVerificationResult.status must be SlotStatus")
        evidence_ids = _stable_strings(
            self.evidence_ids,
            "SlotVerificationResult.evidence_ids",
        )
        coverage = tuple(self.field_coverage)
        if any(not isinstance(item, FieldCoverage) for item in coverage):
            raise ValueError("field_coverage must contain FieldCoverage")
        field_ids = [item.field_id for item in coverage]
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("field_coverage contains duplicate field IDs")
        for item in coverage:
            if not set(item.evidence_ids) <= set(evidence_ids):
                raise ValueError("FieldCoverage references evidence outside input")
        reason_codes = tuple(self.reason_codes)
        if any(not isinstance(item, VerificationReasonCode) for item in reason_codes):
            raise ValueError("reason_codes contains an invalid value")
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("reason_codes contains duplicates")
        _non_empty(self.reason, "SlotVerificationResult.reason")
        conflicts = tuple(self.conflicts)
        if any(not isinstance(item, DetectedConflict) for item in conflicts):
            raise ValueError("conflicts must contain DetectedConflict")
        if any(
            not set(conflict.evidence_ids) <= set(evidence_ids)
            for conflict in conflicts
        ):
            raise ValueError("conflict references evidence outside input")
        if self.status is SlotStatus.FILLED and any(
            not item.covered for item in coverage
        ):
            raise ValueError("filled result cannot contain an uncovered required field")
        if self.status is SlotStatus.CONFLICTING and not conflicts:
            raise ValueError("conflicting result must contain conflict details")
        if conflicts and self.status is not SlotStatus.CONFLICTING:
            raise ValueError("results with conflicts must use conflicting status")
        if not isinstance(self.trace, Mapping):
            raise ValueError("trace must be a mapping")
        frozen_trace = _freeze_json(self.trace)
        try:
            json.dumps(frozen_trace, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("trace must be JSON serializable") from exc
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "field_coverage", coverage)
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "trace", frozen_trace)


@dataclass(frozen=True)
class ContractVerificationResult:
    contract_id: str
    status: str
    slots: tuple[SlotVerificationResult, ...]

    def __post_init__(self) -> None:
        _non_empty(self.contract_id, "ContractVerificationResult.contract_id")
        if self.status not in {"complete", "incomplete", "conflicted"}:
            raise ValueError(f"invalid contract verification status {self.status!r}")
        slots = tuple(self.slots)
        if any(not isinstance(item, SlotVerificationResult) for item in slots):
            raise ValueError("ContractVerificationResult.slots contains invalid item")
        if any(item.contract_id != self.contract_id for item in slots):
            raise ValueError("slot result contract_id mismatch")
        slot_ids = [item.slot_id for item in slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("ContractVerificationResult has duplicate slots")
        object.__setattr__(self, "slots", slots)
