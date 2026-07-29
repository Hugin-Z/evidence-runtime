"""规则注册表：八槽规则装配与注册表自检。"""

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

from .results import _freeze_json, _non_empty, _stable_strings
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


def _rules() -> Mapping[str, SlotRule]:
    rules = (
        SlotRule(
            "background_and_basis",
            (
                FieldRule("basis_reference_or_background", _basis_reference),
                FieldRule("relationship_to_current_project", _project_relationship),
            ),
        ),
        SlotRule(
            "current_state_and_needs",
            (
                FieldRule("current_state", _current_state),
                FieldRule("specific_problem_or_need", _specific_problem),
                FieldRule("affected_subject_business_or_process", _affected_subject),
            ),
        ),
        SlotRule(
            "scope_and_subjects",
            (
                FieldRule("construction_or_service_subject", _construction_subject),
                FieldRule("business_geographic_organizational_or_system_scope", _explicit_scope),
                FieldRule("basic_boundary", _basic_boundary),
            ),
        ),
        SlotRule(
            "objectives",
            (
                FieldRule("expected_target_state", _target_state),
                FieldRule("verifiability_or_alignment_with_current_problem", _objective_alignment),
            ),
        ),
        SlotRule(
            "solution_content_and_capabilities",
            (
                FieldRule("concrete_content_module_function_or_capability", _concrete_capability),
                FieldRule("alignment_with_need_or_objective", _capability_alignment),
            ),
        ),
        SlotRule(
            "implementation_conditions_and_boundaries",
            (
                FieldRule(
                    "at_least_one_explicit_implementation_condition_or_boundary",
                    _implementation_condition,
                ),
            ),
        ),
        SlotRule(
            "reference_cases_and_proof",
            (
                FieldRule("project_customer_or_implementation_subject", _case_subject),
                FieldRule("actual_implementation_content", _implementation_content),
                FieldRule(
                    "implementation_launch_acceptance_or_usage_status",
                    _implementation_status,
                ),
            ),
        ),
        SlotRule(
            "outcomes_and_metrics",
            (
                FieldRule("expected_or_actual_outcome", _outcome),
                FieldRule("outcome_subject_or_evaluation_basis", _outcome_subject_or_basis),
                FieldRule("expected_or_actual_nature", _outcome_nature),
            ),
        ),
    )
    return MappingProxyType({item.slot_id: item for item in rules})


SLOT_VERIFICATION_RULES = _rules()


def validate_rule_registry(
    contract: EvidenceContract,
    registry: Mapping[str, SlotRule],
) -> None:
    """验证 registry 与固定 Contract 的 Slot/field 集合和声明顺序完全一致。"""

    if not isinstance(contract, EvidenceContract):
        raise ValueError("contract must be EvidenceContract")
    if not isinstance(registry, Mapping):
        raise ValueError("registry must be a mapping")
    expected_slots = contract.slot_ids
    actual_slots = tuple(registry)
    if actual_slots != expected_slots:
        raise ValueError(
            "verification registry slot mismatch: "
            f"expected={expected_slots!r}, actual={actual_slots!r}"
        )
    for slot in contract.slots:
        rule = registry.get(slot.slot_id)
        if not isinstance(rule, SlotRule) or rule.slot_id != slot.slot_id:
            raise ValueError(f"invalid SlotRule for {slot.slot_id!r}")
        actual_fields = tuple(item.field_id for item in rule.field_rules)
        if actual_fields != slot.required_fields:
            raise ValueError(
                f"verification field rule mismatch for {slot.slot_id!r}: "
                f"expected={slot.required_fields!r}, actual={actual_fields!r}"
            )


validate_rule_registry(solution_material_completeness_v1, SLOT_VERIFICATION_RULES)
