"""Exact Locator 的结果数据类型。"""

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

from .anchors import ExactAnchor
from .helpers import _non_empty


@dataclass(frozen=True)
class ExactSearchAttempt:
    scope: str
    anchor_term: str
    anchor_source: str
    anchor_priority: int
    document_ids: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int
    match_count: int


@dataclass(frozen=True)
class _ExactRescueCandidate:
    """One bounded overflow span associated with one vector candidate.

    This is locator provenance, not a verification decision.  The baseline
    ``evidence_spans`` remain ``merged[:max_results]`` in their original order;
    the pipeline may inspect these private candidates only after verification.
    """

    evidence_span: EvidenceSpan
    exact_raw_rank: int
    candidate_raw_rank: int
    candidate_chunk_id: str


@dataclass(frozen=True)
class ExactLocationResult:
    slot_id: str
    anchors: tuple[ExactAnchor, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    candidate_document_ids: tuple[str, ...]
    fallback_used: bool
    search_attempts: tuple[ExactSearchAttempt, ...]
    rg_source: str
    _rescue_candidates: tuple[_ExactRescueCandidate, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )


@dataclass
class _LocatedEvidence:
    document: NormalizedDocument
    start_char: int
    end_char: int
    scope_priority: int
    anchor_priority: int
    anchor_terms: list[str]
    anchor_sources: list[str]
    search_scopes: list[str]
    discovery_sources: list[str]
