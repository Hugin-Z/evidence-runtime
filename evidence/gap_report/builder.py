"""Gap Report 构建器：把已完成的验证结果投影成报告。"""

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

from .models import (
    GapReport,
    PendingActionType,
    PendingMaterialRequest,
    EvidenceReference,
    SlotGapResult,
)


def _pending_for(slot: SlotGapResult) -> PendingMaterialRequest | None:
    if slot.status is SlotStatus.FILLED:
        return None
    if slot.status is SlotStatus.CONFLICTING:
        return PendingMaterialRequest(
            slot_id=slot.slot_id,
            slot_name=slot.slot_name,
            action=PendingActionType.RESOLVE_CONFLICT,
            missing_fields=slot.missing_fields,
            existing_evidence_ids=slot.evidence_ids,
            description=(
                "现有材料存在相互冲突的事实，请核实来源并提供一致版本。"
                f"{slot.material_request}"
            ),
            priority=1,
        )
    priority = 2 if slot.status is SlotStatus.MISSING else 3
    if slot.status is SlotStatus.MISSING:
        prefix = "未找到满足要求的材料。"
    elif slot.missing_fields:
        prefix = "已发现相关材料，但仍有必需字段未覆盖。"
    else:
        prefix = "已发现相关材料，但现有证据仍不足以满足该证据槽。"
    return PendingMaterialRequest(
        slot_id=slot.slot_id,
        slot_name=slot.slot_name,
        action=PendingActionType.REQUEST_MATERIAL,
        missing_fields=slot.missing_fields,
        existing_evidence_ids=slot.evidence_ids,
        description=f"{prefix}{slot.material_request}",
        priority=priority,
    )


class GapReportBuilder:
    """Build a report without changing or re-evaluating verifier output."""

    def build(
        self,
        *,
        contract: EvidenceContract,
        verification: ContractVerificationResult,
        evidence_by_slot: Mapping[str, Sequence[EvidenceSpan]],
        trace: Mapping[str, object] | None = None,
    ) -> GapReport:
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        if not isinstance(verification, ContractVerificationResult):
            raise ValueError("verification must be ContractVerificationResult")
        if verification.contract_id != contract.contract_id:
            raise ValueError("verification contract_id does not match contract")
        if not isinstance(evidence_by_slot, Mapping):
            raise ValueError("evidence_by_slot must be a mapping")
        unknown = tuple(key for key in evidence_by_slot if key not in contract.slot_ids)
        if unknown:
            raise ValueError(f"evidence_by_slot contains unknown slot keys: {unknown!r}")
        verification_ids = tuple(item.slot_id for item in verification.slots)
        if verification_ids != contract.slot_ids:
            raise ValueError(
                "verification slots must exactly match contract order; "
                f"got {verification_ids!r}"
            )

        spans_by_slot: dict[str, dict[str, EvidenceSpan]] = {}
        canonical_spans: dict[str, EvidenceSpan] = {}
        for slot in contract.slots:
            raw_spans = evidence_by_slot.get(slot.slot_id, ())
            if isinstance(raw_spans, (str, bytes)):
                raise ValueError(f"evidence for slot {slot.slot_id!r} must be a sequence")
            slot_spans: dict[str, EvidenceSpan] = {}
            for index, span in enumerate(tuple(raw_spans)):
                if not isinstance(span, EvidenceSpan):
                    raise ValueError(
                        f"evidence_by_slot[{slot.slot_id!r}][{index}] must be EvidenceSpan"
                    )
                existing = canonical_spans.get(span.evidence_id)
                if existing is not None and any(
                    getattr(existing, field_name) != getattr(span, field_name)
                    for field_name in (
                        "document_id",
                        "text",
                        "start_char",
                        "end_char",
                        "start_line",
                        "end_line",
                        "page",
                    )
                ):
                    raise ValueError(
                        f"evidence_id {span.evidence_id!r} has inconsistent provenance"
                    )
                canonical_spans.setdefault(span.evidence_id, span)
                slot_spans.setdefault(span.evidence_id, span)
            spans_by_slot[slot.slot_id] = slot_spans

        projected: list[SlotGapResult] = []
        ordered_references: list[EvidenceReference] = []
        referenced: set[str] = set()
        pending_with_order: list[tuple[int, int, PendingMaterialRequest]] = []
        for slot_index, result in enumerate(verification.slots):
            available = spans_by_slot[result.slot_id]
            unresolved = tuple(
                evidence_id
                for evidence_id in result.evidence_ids
                if evidence_id not in available
            )
            if unresolved:
                raise ValueError(
                    f"slot {result.slot_id!r} verification references unresolved evidence: "
                    f"{unresolved!r}"
                )
            item = SlotGapResult.from_verification(contract, result)
            projected.append(item)
            for evidence_id in result.evidence_ids:
                if evidence_id not in referenced:
                    referenced.add(evidence_id)
                    ordered_references.append(
                        EvidenceReference.from_span(available[evidence_id])
                    )
            request = _pending_for(item)
            if request is not None:
                pending_with_order.append((request.priority, slot_index, request))

        pending_with_order.sort(key=lambda item: (item[0], item[1]))
        return GapReport(
            contract_id=contract.contract_id,
            contract_version=contract.version,
            status=verification.status,
            ready=verification.status == "complete",
            slots=tuple(projected),
            evidence=tuple(ordered_references),
            pending=tuple(item[2] for item in pending_with_order),
            trace={} if trace is None else trace,
        )
