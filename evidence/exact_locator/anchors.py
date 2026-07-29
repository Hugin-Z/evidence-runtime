"""Anchor 抽取：从 Slot 与候选确定性构造字面搜索锚点。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from ..documents import NormalizedDocument
from ..rg_executor import (
    InvalidRipgrepOutputError,
    RipgrepExecutor,
    RipgrepLineEvent,
    RipgrepSubmatch,
)
from ..types import EvidenceContract, EvidenceSlot, EvidenceSpan, VectorCandidate
from ..vector_discovery import build_slot_query

from .constants import MAX_ANCHORS, MAX_ANCHOR_CHARS, _ANCHOR_SOURCES
from .helpers import _non_empty, _positive_int


@dataclass(frozen=True)
class ExactAnchor:
    term: str
    source: str
    priority: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.term, "ExactAnchor.term")
        if self.source not in _ANCHOR_SOURCES:
            raise ValueError(
                f"ExactAnchor.source must be one of {sorted(_ANCHOR_SOURCES)!r}; "
                f"got {self.source!r}"
            )
        if type(self.priority) is not int or self.priority < 0:
            raise ValueError("ExactAnchor.priority must be an integer >= 0")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("ExactAnchor.metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


_QUERY_SPLIT_RE = re.compile(r"[。；，,:：]+")
_QUOTED_PATTERNS = (
    re.compile(r"“([^”]{1,80})”"),
    re.compile(r'"([^"\n]{1,80})"'),
    re.compile(r"《([^》]{1,80})》"),
)
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?%")
_DATE_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)?"
)
_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9._/-]{2,}\b")
_NAMED_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·_-]{2,32}(?:政策|项目|平台|系统)"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9_.%])"
)


def extract_exact_anchors(
    slot: EvidenceSlot,
    candidates: Sequence[VectorCandidate],
    *,
    explicit_terms: Sequence[str] = (),
    max_anchors: int = MAX_ANCHORS,
) -> tuple[ExactAnchor, ...]:
    """从显式输入、Slot 正向字段、query 与 candidate 明确信号稳定提取 anchors。"""

    if not isinstance(slot, EvidenceSlot):
        raise ValueError(f"slot must be EvidenceSlot; got {slot!r}")
    max_anchors = _positive_int(max_anchors, "max_anchors")
    if isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence of VectorCandidate")
    if isinstance(explicit_terms, (str, bytes)):
        raise ValueError("explicit_terms must be a sequence of strings")
    candidates = tuple(candidates)
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, VectorCandidate):
            raise ValueError(
                f"candidates[{index}] must be VectorCandidate; got {candidate!r}"
            )
        if candidate.slot_id != slot.slot_id:
            raise ValueError(
                f"candidate {candidate.chunk_id!r} belongs to slot "
                f"{candidate.slot_id!r}, expected {slot.slot_id!r}"
            )
    try:
        explicit_terms = tuple(explicit_terms)
    except TypeError as exc:
        raise ValueError("explicit_terms must be a sequence of strings") from exc

    anchors: list[ExactAnchor] = []
    seen: set[str] = set()

    def add(term: object, source: str, priority: int, metadata=None) -> None:
        if len(anchors) >= max_anchors or not isinstance(term, str):
            return
        normalized = term.strip()
        if (
            not normalized
            or len(normalized) > MAX_ANCHOR_CHARS
            or "\n" in normalized
            or "\r" in normalized
        ):
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        anchors.append(
            ExactAnchor(
                term=normalized,
                source=source,
                priority=priority,
                metadata={} if metadata is None else metadata,
            )
        )

    for index, term in enumerate(explicit_terms):
        add(term, "caller", index, {"explicit_index": index})
    add(slot.name, "slot_name", 100, {"slot_id": slot.slot_id})
    for index, field_name in enumerate(slot.required_fields):
        add(
            field_name.replace("_", " "),
            "required_field",
            200 + index,
            {"required_field": field_name},
        )
    for index, phrase in enumerate(_QUERY_SPLIT_RE.split(build_slot_query(slot))):
        if phrase.strip() == "所需信息":
            continue
        add(phrase, "query", 300 + index, {"query_phrase_index": index})
    for candidate_index, candidate in enumerate(candidates):
        text = candidate.text
        found: list[tuple[int, str, str]] = []
        for pattern_name, pattern in (
            *(('quoted', pattern) for pattern in _QUOTED_PATTERNS),
            ("percentage", _PERCENT_RE),
            ("date", _DATE_RE),
            ("identifier", _IDENTIFIER_RE),
            ("named_entity", _NAMED_ENTITY_RE),
            ("number", _NUMBER_RE),
        ):
            for match in pattern.finditer(text):
                value = match.group(1) if pattern_name == "quoted" else match.group(0)
                found.append((match.start(), value, pattern_name))
        for local_index, (_, value, pattern_name) in enumerate(
            sorted(found, key=lambda item: (item[0], item[1]))
        ):
            add(
                value,
                "candidate",
                400 + candidate_index * 100 + local_index,
                {
                    "chunk_id": candidate.chunk_id,
                    "pattern": pattern_name,
                },
            )
    return tuple(anchors)
