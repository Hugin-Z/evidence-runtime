"""语义门：项目上下文、目标表达与成效角色的有界判定。"""

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

from .results import DetectedFact
from .predicates import _contains_any, _normalized, _regex
from .promotional import _promotional_analysis
from .quantitative import _QuantitativeRole, _quantitative_claims, _quantitative_role


@dataclass(frozen=True)
class _FactRecord:
    fact: DetectedFact
    topic_key: str


@dataclass(frozen=True)
class _SemanticGateResult:
    """Private, deterministic filled-eligibility result for one fixed Slot."""

    applicable: bool
    evaluated: bool
    passed: bool | None
    failed_aspects: tuple[str, ...]
    reason: str


_OBJECTIVES_GATE_ASPECTS = (
    "project_or_construction_context",
    "explicit_target_expression",
    "project_specific_target_content",
)

_OUTCOMES_GATE_ASPECTS = (
    "explicit_outcome_or_metric_role",
    "expected_or_actual_nature",
    "outcome_subject_or_evaluation_basis",
)


def _has_project_or_construction_context(text: str) -> bool:
    return _project_relationship(text) or _contains_any(
        text,
        (
            "建设任务",
            "建设项目",
            "建设目标",
            "平台建设",
            "工程建设",
            "建设对象",
        ),
    )


def _has_explicit_target_expression(text: str) -> bool:
    return _target_state(text) or _contains_any(
        text,
        (
            "建设目标",
            "总体目标",
            "具体目标",
            "目标是",
            "旨在",
            "拟实现",
            "计划达到",
        ),
    )


def _has_project_specific_target_content(text: str) -> bool:
    return any(
        evaluator(text)
        for evaluator in (
            _objective_alignment,
            _construction_subject,
            _explicit_scope,
            _concrete_capability,
        )
    )


def _has_outcome_or_metric_role(text: str) -> bool:
    role_marker = _contains_any(
        text,
        (
            "成效",
            "效果",
            "效益",
            "考核指标",
            "评价指标",
            "验收指标",
            "运行结果",
            "验收结果",
        ),
    )
    return role_marker or (_outcome(text) and _has_expected_or_actual_nature(text))


def _has_expected_or_actual_nature(text: str) -> bool:
    return (
        _outcome_nature(text)
        or _implementation_status(text)
        or _regex(text, r"(?:建成|上线|投产|投入使用|试运行|运行).{0,8}后")
    )


def _evaluate_slot_semantic_gate(
    slot_id: str,
    evidence_spans: Sequence[EvidenceSpan],
    *,
    evaluate: bool,
) -> _SemanticGateResult:
    """Evaluate only the two frozen P0 gates over the current Slot evidence.

    Signals may be supplied by different EvidenceSpans.  The helper is private,
    accepts only validated EvidenceSpan inputs from ``verify_slot``, and does
    not read files, candidates, other Slots, or external state.
    """

    if slot_id not in {"objectives", "outcomes_and_metrics"}:
        return _SemanticGateResult(
            applicable=False,
            evaluated=False,
            passed=None,
            failed_aspects=(),
            reason="Slot-specific semantic gate does not apply.",
        )
    if not evaluate:
        return _SemanticGateResult(
            applicable=True,
            evaluated=False,
            passed=None,
            failed_aspects=(),
            reason="Semantic gate was not evaluated because the preliminary status was not filled.",
        )
    texts = tuple(span.text for span in evidence_spans)
    if slot_id == "objectives":
        checks = (
            any(_has_project_or_construction_context(text) for text in texts),
            any(_has_explicit_target_expression(text) for text in texts),
            any(_has_project_specific_target_content(text) for text in texts),
        )
        aspects = _OBJECTIVES_GATE_ASPECTS
    else:
        checks = (
            any(_has_outcome_or_metric_role(text) for text in texts),
            any(_has_expected_or_actual_nature(text) for text in texts),
            any(_outcome_subject_or_basis(text) for text in texts),
        )
        aspects = _OUTCOMES_GATE_ASPECTS
    failed = tuple(aspect for aspect, passed in zip(aspects, checks) if not passed)
    return _SemanticGateResult(
        applicable=True,
        evaluated=True,
        passed=not failed,
        failed_aspects=failed,
        reason=(
            "Slot-specific semantic filled gate passed."
            if not failed
            else "Slot-specific semantic filled gate failed: " + ", ".join(failed) + "."
        ),
    )
from .predicates import _concrete_capability, _construction_subject, _explicit_scope, _implementation_status, _objective_alignment, _outcome, _outcome_nature, _outcome_subject_or_basis, _project_relationship, _target_state
