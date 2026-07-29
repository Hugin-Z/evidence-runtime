"""Deterministic Slot Verifier：结果、规则、谓词、门、冲突与状态机。

模块按职责拆成
`results` / `rule_types` / `predicates` / `rules` / `promotional` / `quantitative` /
`semantic_gate` / `conflicts` / `engine`。**纯移动**：以函数为最小移动单位，**未拆任何函数体**，
判定逻辑逐字符未改；对外符号与 import 路径（`from evidence.verifier import X`）完全不变。
"""
from .results import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
    _FrozenJsonDict,
    _freeze_json,
    _non_empty,
    _stable_strings,
)
from .rule_types import FieldEvaluator, FieldRule, SlotRule
from .rules import SLOT_VERIFICATION_RULES, _rules, validate_rule_registry
from .promotional import PROMOTIONAL_PHRASES, _promotional_analysis
from .quantitative import _QuantitativeRole, _quantitative_claims
from .semantic_gate import _SemanticGateResult, _evaluate_slot_semantic_gate
from .conflicts import _detect_conflicts, _extract_fact_records
from .engine import DeterministicSlotVerifier, _reason_for, _validate_evidence_spans

__all__ = [
    "VerificationReasonCode", "FieldCoverage", "DetectedFact", "DetectedConflict",
    "SlotVerificationResult", "ContractVerificationResult",
    "FieldRule", "SlotRule", "SLOT_VERIFICATION_RULES",
    "validate_rule_registry", "DeterministicSlotVerifier",
]
