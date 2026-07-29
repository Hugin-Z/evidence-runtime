"""DeterministicSlotVerifier 状态机：span 校验 → 字段覆盖 → 门 → 冲突 → 状态。"""

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

from .results import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
    _freeze_json,
    _non_empty,
    _stable_strings,
)
from .rule_types import FieldEvaluator, FieldRule, SlotRule
from .predicates import (
    _CASE_SUBJECT_RE,
    _CURRENT_STATE_FUTURE_MARKERS,
    _CURRENT_STATE_PREDICATE_RE,
    _CURRENT_STATE_PROMOTIONAL_CAPABILITY_RE,
    _affected_subject,
    _basic_boundary,
    _basis_reference,
    _capability_alignment,
    _case_subject,
    _concrete_capability,
    _construction_subject,
    _contains_any,
    _current_state,
    _explicit_scope,
    _implementation_condition,
    _implementation_content,
    _implementation_status,
    _normalized,
    _objective_alignment,
    _outcome,
    _outcome_nature,
    _outcome_subject_or_basis,
    _project_relationship,
    _regex,
    _specific_problem,
    _target_state,
)
from .rules import SLOT_VERIFICATION_RULES, _rules, validate_rule_registry
from .promotional import PROMOTIONAL_PHRASES, _CONCRETE_MARKERS, _promotional_analysis
from .quantitative import (
    _QuantitativeRole,
    _quantitative_claims,
    _quantitative_clause,
    _quantitative_role,
)
from .semantic_gate import (
    _FactRecord,
    _SemanticGateResult,
    _evaluate_slot_semantic_gate,
)
from .conflicts import (
    _detect_conflicts,
    _extract_fact_records,
    _normalize_scalar,
    _normalize_topic,
    _stable_evidence_ids,
)


def _validate_evidence_spans(
    evidence_spans: Sequence[EvidenceSpan],
) -> tuple[tuple[EvidenceSpan, ...], tuple[str, ...]]:
    if isinstance(evidence_spans, (str, bytes)):
        raise ValueError("evidence_spans must be a sequence of EvidenceSpan")
    try:
        raw = tuple(evidence_spans)
    except TypeError as exc:
        raise ValueError("evidence_spans must be a sequence of EvidenceSpan") from exc
    unique: list[EvidenceSpan] = []
    seen: set[str] = set()
    missing_context: list[str] = []
    for index, span in enumerate(raw):
        if not isinstance(span, EvidenceSpan):
            raise ValueError(
                f"evidence_spans[{index}] must be EvidenceSpan; got {span!r}"
            )
        if span.evidence_id in seen:
            continue
        seen.add(span.evidence_id)
        required_metadata = {
            "source_path",
            "source_content_hash",
            "normalized_content_hash",
            "span_content_hash",
        }
        metadata = span.metadata
        if not isinstance(metadata, Mapping) or not required_metadata <= set(metadata):
            missing_context.append(span.evidence_id)
        else:
            normalized_hash = metadata["normalized_content_hash"]
            span_hash = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
            expected_id = hashlib.sha256(
                (
                    f"{span.document_id}\0{normalized_hash}\0"
                    f"{span.start_char}\0{span.end_char}"
                ).encode("utf-8")
            ).hexdigest()
            invalid_location = (
                not isinstance(metadata["source_path"], str)
                or not metadata["source_path"].strip()
                or not isinstance(normalized_hash, str)
                or len(normalized_hash) != 64
                or span.start_char < 0
                or span.end_char <= span.start_char
                or span.end_char - span.start_char != len(span.text)
                or span.start_line < 1
                or span.end_line < span.start_line
                or metadata["span_content_hash"] != span_hash
                or span.evidence_id != f"ev-{expected_id}"
            )
            if invalid_location:
                missing_context.append(span.evidence_id)
        unique.append(span)
    return tuple(unique), tuple(missing_context)


def _reason_for(
    status: SlotStatus,
    missing_fields: Sequence[str],
    conflicts: Sequence[DetectedConflict],
    *,
    no_evidence: bool,
    promotional_only: bool,
    unverified_quantitative: bool,
    insufficient_count: bool,
    semantic_gate_failures: Sequence[str],
) -> str:
    if status is SlotStatus.CONFLICTING:
        return f"检测到 {len(conflicts)} 组可定位的结构化冲突，未选择任一方。"
    if status is SlotStatus.MISSING:
        if no_evidence:
            return "没有可验证的 EvidenceSpan。"
        if missing_fields:
            return "现有证据未覆盖任何必要字段，缺少：" + "、".join(missing_fields) + "。"
        return "现有证据未形成可确定的必要字段覆盖。"
    if status is SlotStatus.WEAK:
        details: list[str] = []
        if missing_fields:
            details.append("未覆盖字段：" + "、".join(missing_fields))
        if promotional_only:
            details.append("内容主要为宣传性表达")
        if unverified_quantitative:
            details.append("存在缺少可识别支撑的量化声明")
        if insufficient_count:
            details.append("去重后的证据数量未达到最低要求")
        if semantic_gate_failures:
            details.append(
                "Slot 专属语义门槛未满足：" + "、".join(semantic_gate_failures)
            )
        return "证据相关但不足以完整满足 Slot；" + "；".join(details) + "。"
    return "全部必要字段均有可定位证据覆盖，证据数量达标，且未检测到结构化冲突。"


class DeterministicSlotVerifier:
    """固定规则、无外部调用的 EvidenceSpan-only Slot Verifier。"""

    def __init__(
        self,
        registry: Mapping[str, SlotRule] = SLOT_VERIFICATION_RULES,
    ) -> None:
        if not isinstance(registry, Mapping):
            raise ValueError("registry must be a mapping")
        validate_rule_registry(solution_material_completeness_v1, registry)
        self._registry = MappingProxyType(dict(registry))

    @property
    def registry(self) -> Mapping[str, SlotRule]:
        return self._registry

    def verify_slot(
        self,
        *,
        contract: EvidenceContract,
        slot_id: str,
        evidence_spans: Sequence[EvidenceSpan],
    ) -> SlotVerificationResult:
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        validate_rule_registry(contract, self._registry)
        slot = contract.get_slot(slot_id)
        if isinstance(evidence_spans, (str, bytes)):
            raise ValueError("evidence_spans must be a sequence of EvidenceSpan")
        try:
            evidence_spans = tuple(evidence_spans)
        except TypeError as exc:
            raise ValueError(
                "evidence_spans must be a sequence of EvidenceSpan"
            ) from exc
        unique_spans, missing_context_ids = _validate_evidence_spans(evidence_spans)
        evidence_ids = tuple(span.evidence_id for span in unique_spans)
        slot_rule = self._registry[slot.slot_id]

        coverage: list[FieldCoverage] = []
        for rule in slot_rule.field_rules:
            supporting_ids = tuple(
                span.evidence_id for span in unique_spans if rule.evaluator(span.text)
            )
            coverage.append(
                FieldCoverage(
                    field_id=rule.field_id,
                    covered=bool(supporting_ids),
                    evidence_ids=supporting_ids,
                    reason=None if supporting_ids else "未检测到可确定的字段覆盖信号",
                )
            )
        covered_fields = tuple(item.field_id for item in coverage if item.covered)
        missing_fields = tuple(item.field_id for item in coverage if not item.covered)

        promotional_phrases: list[str] = []
        promotional_only_ids: list[str] = []
        for span in unique_spans:
            phrases, only = _promotional_analysis(span.text)
            for phrase in phrases:
                if phrase not in promotional_phrases:
                    promotional_phrases.append(phrase)
            if only:
                promotional_only_ids.append(span.evidence_id)
        combined_text = "\n".join(span.text for span in unique_spans)
        weak_relevance = _contains_any(
            combined_text,
            _WEAK_RELEVANCE[slot.slot_id],
        )

        quantitative_claims = tuple(
            sorted(
                (
                    claim
                    for span in unique_spans
                    for claim in _quantitative_claims(span, slot.slot_id)
                ),
                key=lambda claim: (
                    str(claim["evidence_id"]),
                    str(claim["value"]),
                    str(claim["context"]),
                ),
            )
        )
        unverified_quantitative = any(
            claim["role"] == _QuantitativeRole.OUTCOME_CLAIM.value
            and not bool(claim["verified"])
            for claim in quantitative_claims
        )
        records = _extract_fact_records(unique_spans)
        facts = tuple(record.fact for record in records)
        conflicts = _detect_conflicts(records)
        insufficient_count = len(unique_spans) < slot.minimum_evidence
        promotional_only = bool(promotional_only_ids) and bool(missing_fields)

        if conflicts:
            status = SlotStatus.CONFLICTING
        elif not unique_spans:
            status = SlotStatus.MISSING
        elif promotional_only:
            status = SlotStatus.WEAK
        elif not covered_fields and weak_relevance:
            status = SlotStatus.WEAK
        elif not covered_fields:
            status = SlotStatus.MISSING
        elif (
            missing_fields
            or unverified_quantitative
            or insufficient_count
            or missing_context_ids
        ):
            status = SlotStatus.WEAK
        else:
            status = SlotStatus.FILLED

        preliminary_status = status
        semantic_gate = _evaluate_slot_semantic_gate(
            slot.slot_id,
            unique_spans,
            evaluate=preliminary_status is SlotStatus.FILLED,
        )
        semantic_gate_failed = semantic_gate.evaluated and not semantic_gate.passed
        if semantic_gate_failed:
            status = SlotStatus.WEAK

        reason_codes: list[VerificationReasonCode] = []
        if not unique_spans:
            reason_codes.append(VerificationReasonCode.NO_EVIDENCE)
        if insufficient_count:
            reason_codes.append(VerificationReasonCode.INSUFFICIENT_EVIDENCE_COUNT)
        if missing_fields:
            reason_codes.append(VerificationReasonCode.MISSING_REQUIRED_FIELD)
        elif semantic_gate_failed:
            reason_codes.append(VerificationReasonCode.MISSING_REQUIRED_FIELD)
        if promotional_only:
            reason_codes.append(VerificationReasonCode.PROMOTIONAL_ONLY)
        if unverified_quantitative:
            reason_codes.append(
                VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
            )
        if missing_context_ids:
            reason_codes.append(VerificationReasonCode.MISSING_SOURCE_CONTEXT)
        if conflicts:
            reason_codes.append(VerificationReasonCode.CONFLICTING_FACTS)

        decision_path = (
            "deduplicate_evidence_ids",
            "evaluate_required_field_registry",
            "evaluate_promotional_and_quantitative_quality",
            "detect_opposite_assertions_and_scalar_mismatches",
            "evaluate_objectives_and_outcomes_semantic_filled_gate",
            f"status:{status.value}",
        )
        trace = {
            "input_evidence_count": len(evidence_spans),
            "unique_evidence_count": len(unique_spans),
            "required_fields": slot.required_fields,
            "covered_fields": covered_fields,
            "missing_fields": missing_fields,
            "promotional_phrases": tuple(promotional_phrases),
            "promotional_only_evidence_ids": tuple(promotional_only_ids),
            "quantitative_claims": quantitative_claims,
            "detected_facts": tuple(
                {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "normalized_value": fact.normalized_value,
                    "evidence_id": fact.evidence_id,
                }
                for fact in facts
            ),
            "conflict_count": len(conflicts),
            "missing_source_context_evidence_ids": missing_context_ids,
            "semantic_gate": {
                "applicable": semantic_gate.applicable,
                "evaluated": semantic_gate.evaluated,
                "passed": semantic_gate.passed,
                "failed_aspects": semantic_gate.failed_aspects,
                "reason": semantic_gate.reason,
                "preliminary_status": preliminary_status.value,
            },
            "decision_path": decision_path,
        }
        reason = _reason_for(
            status,
            missing_fields,
            conflicts,
            no_evidence=not unique_spans,
            promotional_only=promotional_only,
            unverified_quantitative=unverified_quantitative,
            insufficient_count=insufficient_count,
            semantic_gate_failures=semantic_gate.failed_aspects,
        )
        return SlotVerificationResult(
            contract_id=contract.contract_id,
            slot_id=slot.slot_id,
            status=status,
            evidence_ids=evidence_ids,
            field_coverage=tuple(coverage),
            reason_codes=tuple(reason_codes),
            reason=reason,
            conflicts=conflicts,
            trace=trace,
        )

    def verify_contract(
        self,
        *,
        contract: EvidenceContract,
        evidence_by_slot: Mapping[str, Sequence[EvidenceSpan]],
    ) -> ContractVerificationResult:
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        validate_rule_registry(contract, self._registry)
        if not isinstance(evidence_by_slot, Mapping):
            raise ValueError("evidence_by_slot must be a mapping")
        unknown = tuple(key for key in evidence_by_slot if key not in contract.slot_ids)
        if unknown:
            raise ValueError(f"evidence_by_slot contains unknown slot keys: {unknown!r}")
        results = tuple(
            self.verify_slot(
                contract=contract,
                slot_id=slot.slot_id,
                evidence_spans=evidence_by_slot.get(slot.slot_id, ()),
            )
            for slot in contract.slots
        )
        required_results = tuple(
            result
            for slot, result in zip(contract.slots, results)
            if slot.required
        )
        if any(
            result.status is SlotStatus.CONFLICTING for result in required_results
        ):
            status = "conflicted"
        elif all(result.status is SlotStatus.FILLED for result in required_results):
            status = "complete"
        else:
            status = "incomplete"
        return ContractVerificationResult(
            contract_id=contract.contract_id,
            status=status,
            slots=results,
        )
from .promotional import _WEAK_RELEVANCE
