"""Span 合并与选取：确定性合并、单文档上限与 rescue 候选池。"""

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

from .constants import _MAX_RESCUE_CANDIDATES_PER_SLOT
from .models import _ExactRescueCandidate, _LocatedEvidence


def _append_unique(values: list[str], incoming: Sequence[str]) -> None:
    seen = set(values)
    for value in incoming:
        if value not in seen:
            seen.add(value)
            values.append(value)


def _merge_located(
    located: Sequence[_LocatedEvidence],
    *,
    max_span_chars: int,
) -> list[_LocatedEvidence]:
    exact: dict[tuple[str, int, int], _LocatedEvidence] = {}
    for item in sorted(
        located,
        key=lambda value: (
            value.scope_priority,
            value.anchor_priority,
            value.document.normalized_content_hash,   # 内容确定破平（原 document_id）
            value.start_char,
            value.end_char,
        ),
    ):
        key = (item.document.document_id, item.start_char, item.end_char)
        existing = exact.get(key)
        if existing is None:
            exact[key] = item
            continue
        existing.scope_priority = min(existing.scope_priority, item.scope_priority)
        existing.anchor_priority = min(existing.anchor_priority, item.anchor_priority)
        _append_unique(existing.anchor_terms, item.anchor_terms)
        _append_unique(existing.anchor_sources, item.anchor_sources)
        _append_unique(existing.search_scopes, item.search_scopes)
        _append_unique(existing.discovery_sources, item.discovery_sources)

    ordered = sorted(
        exact.values(),
        key=lambda value: (
            value.document.normalized_content_hash,   # 内容确定破平（原 document_id）
            value.start_char,
            value.end_char,
            value.scope_priority,
            value.anchor_priority,
        ),
    )
    merged: list[_LocatedEvidence] = []
    for item in ordered:
        if not merged or merged[-1].document.document_id != item.document.document_id:
            merged.append(item)
            continue
        previous = merged[-1]
        overlap = min(previous.end_char, item.end_char) - max(
            previous.start_char, item.start_char
        )
        smaller = min(
            previous.end_char - previous.start_char,
            item.end_char - item.start_char,
        )
        union_start = min(previous.start_char, item.start_char)
        union_end = max(previous.end_char, item.end_char)
        if (
            overlap > 0
            and overlap / smaller >= 0.8
            and union_end - union_start <= max_span_chars
        ):
            previous.start_char = union_start
            previous.end_char = union_end
            previous.scope_priority = min(previous.scope_priority, item.scope_priority)
            previous.anchor_priority = min(previous.anchor_priority, item.anchor_priority)
            _append_unique(previous.anchor_terms, item.anchor_terms)
            _append_unique(previous.anchor_sources, item.anchor_sources)
            _append_unique(previous.search_scopes, item.search_scopes)
            _append_unique(previous.discovery_sources, item.discovery_sources)
        else:
            merged.append(item)
    return sorted(
        merged,
        key=lambda value: (
            value.scope_priority,
            value.anchor_priority,
            value.document.normalized_content_hash,   # 内容确定破平（原 document_id）
            value.start_char,
        ),
    )


def _content_order_hash(text: str) -> str:
    """内容确定破平哈希：对 chunk 正文规范化空白后取 sha256，独立于分配的 id。"""

    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _candidate_order_key(candidate: VectorCandidate) -> tuple[object, ...]:
    """候选排序键——**内容确定**：破平不依赖 document_id / chunk_id / rank。

    先按相关性（距离），再按 chunk 正文哈希与字符位置。原键以 document_id/chunk_id
    破平，使同一正文在不同 id 下选出不同候选。
    """

    return (
        candidate.distance is None,
        0.0 if candidate.distance is None else candidate.distance,
        _content_order_hash(candidate.text),
        candidate.start_char,
        candidate.end_char,
    )


def _character_gap(span: EvidenceSpan, candidate: VectorCandidate) -> int:
    if span.end_char <= candidate.start_char:
        return candidate.start_char - span.end_char
    if candidate.end_char <= span.start_char:
        return span.start_char - candidate.end_char
    return 0


def _ranges_overlap(span: EvidenceSpan, candidate: VectorCandidate) -> bool:
    return min(span.end_char, candidate.end_char) > max(
        span.start_char,
        candidate.start_char,
    )


def _select_diverse_spans(
    ranked_spans: Sequence[tuple[int, EvidenceSpan]],
    max_results: int,
    single_doc_cap_ratio: float | None,
) -> tuple[list[tuple[int, EvidenceSpan]], list[tuple[int, EvidenceSpan]]]:
    """Exact-span 单文档占比上限；返回 (baseline, overflow)。

    ``single_doc_cap_ratio is None`` → 无上限，退化为**原行为**（``ranked_spans[:max_results]`` /
    余为 overflow），逐字节等价。否则单一 document 在 baseline 至多占
    ``ceil(ratio*max_results)`` 条；被上限挡下的先入 deferred，最后**若预算未满则由 deferred 回填**
    （其它文档无命中时不空置 → 只重分配、不减召回）。baseline 仍按 rank（内容确定序）排列，overflow
    保持 rank 序供 B1.1 rescue。破平不依赖 document_id/chunk_id（仅按 document 分组计数 + rank 序）。
    """

    ranked = list(ranked_spans)
    if single_doc_cap_ratio is None:
        return ranked[:max_results], ranked[max_results:]
    cap = max(1, math.ceil(single_doc_cap_ratio * max_results))
    baseline: list[tuple[int, EvidenceSpan]] = []
    deferred: list[tuple[int, EvidenceSpan]] = []
    per_doc: dict[str, int] = {}
    for rank, span in ranked:
        if len(baseline) < max_results and per_doc.get(span.document_id, 0) < cap:
            baseline.append((rank, span))
            per_doc[span.document_id] = per_doc.get(span.document_id, 0) + 1
        else:
            deferred.append((rank, span))
    fill = 0
    while len(baseline) < max_results and fill < len(deferred):
        baseline.append(deferred[fill])
        fill += 1
    overflow = deferred[fill:]
    baseline.sort(key=lambda item: item[0])
    return baseline, overflow


def _build_rescue_candidates(
    ranked_overflow: Sequence[tuple[int, EvidenceSpan]],
    candidates: Sequence[VectorCandidate],
) -> tuple[_ExactRescueCandidate, ...]:
    """Select at most one same-document overflow span per stable candidate.

    Candidate rank is the outer priority.  Within one candidate an overlapping
    span wins; otherwise the smaller character gap wins, followed by the
    original Exact rank and stable EvidenceSpan identity.  A selected span is
    never assigned to another candidate.  This function does not reorder or
    mutate the baseline top-k and does not call the Verifier.
    """

    if not ranked_overflow or not candidates:
        return ()
    ordered_candidates = tuple(sorted(candidates, key=_candidate_order_key))
    limit = min(len(ordered_candidates), _MAX_RESCUE_CANDIDATES_PER_SLOT)
    selected: list[_ExactRescueCandidate] = []
    selected_ids: set[str] = set()
    for candidate in ordered_candidates:
        same_document = tuple(
            (exact_rank, span)
            for exact_rank, span in ranked_overflow
            if span.document_id == candidate.document_id
            and span.evidence_id not in selected_ids
        )
        if not same_document:
            continue
        exact_rank, span = min(
            same_document,
            key=lambda item: (
                0 if _ranges_overlap(item[1], candidate) else 1,
                _character_gap(item[1], candidate),
                item[0],
                item[1].evidence_id,
            ),
        )
        selected_ids.add(span.evidence_id)
        selected.append(
            _ExactRescueCandidate(
                evidence_span=span,
                exact_raw_rank=exact_rank,
                candidate_raw_rank=candidate.rank,
                candidate_chunk_id=candidate.chunk_id,
            )
        )
        if len(selected) >= limit:
            break
    return tuple(selected)
