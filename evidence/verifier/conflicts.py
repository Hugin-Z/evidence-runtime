"""冲突检测：事实抽取、归一化与两类结构化冲突判定。"""

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

from .results import DetectedConflict, DetectedFact
from .predicates import _contains_any, _normalized, _regex
from .quantitative import _quantitative_claims
from .semantic_gate import _FactRecord


def _normalize_topic(value: str) -> str:
    normalized = _normalized(value)
    normalized = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    return re.sub(r"^(?:本|该|当前)", "", normalized)


def _normalize_scalar(value: str) -> str:
    compact = re.sub(r"\s+", "", _normalized(value))
    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<unit>.*)", compact)
    if match is None:
        return compact
    try:
        number = Decimal(match.group("number"))
    except InvalidOperation:
        return compact
    normalized_number = format(number.normalize(), "f")
    return f"{normalized_number}{match.group('unit')}"


def _extract_fact_records(spans: Sequence[EvidenceSpan]) -> tuple[_FactRecord, ...]:
    records: list[_FactRecord] = []
    for span in spans:
        normalized_text = _normalized(span.text)
        for clause in _CLAUSE_SPLIT_RE.split(normalized_text):
            clause = clause.strip()
            if not clause:
                continue
            occupied: list[tuple[int, int, str]] = []
            for predicate, polarity, value, pattern in _RELATION_PATTERNS:
                for match in pattern.finditer(clause):
                    if any(
                        start <= match.start() and match.end() <= end
                        for start, end, _ in occupied
                    ):
                        continue
                    left = clause[:match.start()].strip(" ：:")
                    right = clause[match.end():].strip(" ：:")
                    topic_key = _normalize_topic(f"{left}|{right}")
                    if len(topic_key) < 2:
                        continue
                    occupied.append((match.start(), match.end(), polarity))
                    records.append(
                        _FactRecord(
                            fact=DetectedFact(
                                subject=(left or right),
                                predicate=predicate,
                                value=value,
                                normalized_value=polarity,
                                evidence_id=span.evidence_id,
                            ),
                            topic_key=topic_key,
                        )
                    )
            for match in _SCALAR_RE.finditer(clause):
                raw_key = match.group("key").strip()
                raw_value = match.group("value").strip()
                topic_key = _normalize_topic(raw_key)
                if len(topic_key) < 2:
                    continue
                records.append(
                    _FactRecord(
                        fact=DetectedFact(
                            subject=raw_key,
                            predicate="scalar_value",
                            value=raw_value,
                            normalized_value=_normalize_scalar(raw_value),
                            evidence_id=span.evidence_id,
                        ),
                        topic_key=topic_key,
                    )
                )
    return tuple(records)


def _stable_evidence_ids(facts: Sequence[DetectedFact]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for fact in facts:
        if fact.evidence_id not in seen:
            seen.add(fact.evidence_id)
            result.append(fact.evidence_id)
    return tuple(result)


def _detect_conflicts(
    records: Sequence[_FactRecord],
) -> tuple[DetectedConflict, ...]:
    groups: dict[tuple[str, str], list[DetectedFact]] = {}
    for record in records:
        groups.setdefault((record.fact.predicate, record.topic_key), []).append(record.fact)
    conflicts: list[DetectedConflict] = []
    for (predicate, _), raw_facts in groups.items():
        facts: list[DetectedFact] = []
        seen_facts: set[tuple[str, str, str]] = set()
        for fact in raw_facts:
            key = (fact.evidence_id, fact.value, fact.normalized_value)
            if key not in seen_facts:
                seen_facts.add(key)
                facts.append(fact)
        values = {fact.normalized_value for fact in facts}
        if len(values) <= 1:
            continue
        if predicate != "scalar_value" and not {"positive", "negative"} <= values:
            continue
        conflict_type = (
            "scalar_mismatch" if predicate == "scalar_value" else "opposite_assertion"
        )
        evidence_ids = _stable_evidence_ids(facts)
        conflicts.append(
            DetectedConflict(
                conflict_type=conflict_type,
                subject=facts[0].subject,
                predicate=predicate,
                facts=tuple(facts),
                evidence_ids=evidence_ids,
                reason=(
                    f"同一对象的 {predicate} 出现不一致值："
                    + "、".join(fact.value for fact in facts)
                ),
            )
        )
    return tuple(
        sorted(
            conflicts,
            key=lambda item: (
                item.conflict_type,
                _normalize_topic(item.subject),
                item.predicate,
                item.evidence_ids,
            ),
        )
    )
from .quantitative import _CLAUSE_SPLIT_RE, _RELATION_PATTERNS, _SCALAR_RE
