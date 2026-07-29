"""规则数据类型：FieldRule、SlotRule 与 FieldEvaluator。"""

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


FieldEvaluator = Callable[[str], bool]


@dataclass(frozen=True)
class FieldRule:
    field_id: str
    evaluator: FieldEvaluator = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _non_empty(self.field_id, "FieldRule.field_id")
        if not callable(self.evaluator):
            raise ValueError("FieldRule.evaluator must be callable")


@dataclass(frozen=True)
class SlotRule:
    slot_id: str
    field_rules: tuple[FieldRule, ...]

    def __post_init__(self) -> None:
        _non_empty(self.slot_id, "SlotRule.slot_id")
        rules = tuple(self.field_rules)
        if not rules or any(not isinstance(item, FieldRule) for item in rules):
            raise ValueError("SlotRule.field_rules must contain FieldRule")
        field_ids = [item.field_id for item in rules]
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("SlotRule contains duplicate field IDs")
        object.__setattr__(self, "field_rules", rules)
