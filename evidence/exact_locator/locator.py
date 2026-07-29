"""ExactEvidenceLocator 主类：候选域优先，必要时全语料回退。"""

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

from .constants import CANONICAL_MANIFEST_NAME, _ANCHOR_SOURCES
from .helpers import _canonical_json, _non_empty, _positive_int
from .canonical_corpus import (
    CanonicalCorpusError,
    CanonicalCorpusManifest,
    CanonicalDocumentEntry,
    materialize_normalized_documents,
    _materialized_filename,
    _validate_document_id,
)
from .anchors import ExactAnchor, extract_exact_anchors
from .offsets import (
    _mapping_error,
    expand_match_context,
    map_rg_submatch_to_char_range,
)
from .models import (
    ExactLocationResult,
    ExactSearchAttempt,
    _ExactRescueCandidate,
    _LocatedEvidence,
)
from .span_selection import (
    _append_unique,
    _build_rescue_candidates,
    _candidate_order_key,
    _character_gap,
    _content_order_hash,
    _merge_located,
    _ranges_overlap,
    _select_diverse_spans,
)


class ExactEvidenceLocator:
    """对 candidate 文档先查，零有效命中时才按配置执行全语料 fallback。"""

    def __init__(
        self,
        canonical_root: str | Path,
        *,
        executor: RipgrepExecutor | None = None,
    ) -> None:
        self.canonical_root = Path(canonical_root)
        self.executor = executor or RipgrepExecutor()

    def _search_scope(
        self,
        *,
        scope_name: str,
        scope_priority: int,
        document_ids: tuple[str, ...],
        manifest: CanonicalCorpusManifest,
        document_by_id: Mapping[str, NormalizedDocument],
        document_id_by_path: Mapping[str, str],
        anchors: tuple[ExactAnchor, ...],
        context_lines: int,
        max_span_chars: int,
        attempts: list[ExactSearchAttempt],
    ) -> tuple[list[_LocatedEvidence], str]:
        targets = [manifest.get(document_id).canonical_path for document_id in document_ids]
        located: list[_LocatedEvidence] = []
        rg_source = getattr(self.executor, "rg_source", "unknown") or "unknown"
        for anchor in anchors:
            run = self.executor.search(
                [anchor.term],
                targets,
                context=0,
                regex=False,
                globs=(),
            )
            rg_source = run.rg_source
            match_count = sum(
                len(event.submatches) for event in run.events if event.kind == "match"
            )
            attempts.append(
                ExactSearchAttempt(
                    scope=scope_name,
                    anchor_term=anchor.term,
                    anchor_source=anchor.source,
                    anchor_priority=anchor.priority,
                    document_ids=document_ids,
                    command=run.command,
                    returncode=run.returncode,
                    match_count=match_count,
                )
            )
            for event in run.events:
                if event.kind != "match":
                    continue
                path_key = os.path.normcase(str(Path(event.path).resolve()))
                document_id = document_id_by_path.get(path_key)
                if document_id is None:
                    raise InvalidRipgrepOutputError(
                        "ripgrep returned a file outside canonical manifest",
                        command=run.command,
                        returncode=run.returncode,
                        stderr=run.stderr,
                        anchors=(anchor.term,),
                        search_scope=targets,
                    )
                document = document_by_id[document_id]
                for submatch in event.submatches:
                    if submatch.text.casefold() != anchor.term.casefold():
                        raise InvalidRipgrepOutputError(
                            "fixed-string rg submatch does not equal requested anchor",
                            command=run.command,
                            returncode=run.returncode,
                            stderr=run.stderr,
                            anchors=(anchor.term,),
                            search_scope=targets,
                        )
                    try:
                        match_start, match_end = map_rg_submatch_to_char_range(
                            document,
                            event,
                            submatch,
                        )
                    except InvalidRipgrepOutputError as exc:
                        raise InvalidRipgrepOutputError(
                            "ripgrep match could not be mapped to canonical text",
                            command=run.command,
                            returncode=run.returncode,
                            stderr=run.stderr,
                            anchors=(anchor.term,),
                            search_scope=targets,
                        ) from exc
                    span_start, span_end = expand_match_context(
                        document,
                        match_start,
                        match_end,
                        context_lines=context_lines,
                        max_chars=max_span_chars,
                    )
                    discovery_sources = (
                        ["vector", "ripgrep"]
                        if scope_name == "candidate"
                        else ["ripgrep"]
                    )
                    located.append(
                        _LocatedEvidence(
                            document=document,
                            start_char=span_start,
                            end_char=span_end,
                            scope_priority=scope_priority,
                            anchor_priority=anchor.priority,
                            anchor_terms=[anchor.term],
                            anchor_sources=[anchor.source],
                            search_scopes=[scope_name],
                            discovery_sources=discovery_sources,
                        )
                    )
        return located, rg_source

    def locate_for_slot(
        self,
        *,
        contract: EvidenceContract,
        slot_id: str,
        documents: Sequence[NormalizedDocument],
        candidates: Sequence[VectorCandidate],
        explicit_terms: Sequence[str] = (),
        context_lines: int = 2,
        max_span_chars: int = 1200,
        max_results: int = 20,
        full_corpus_fallback: bool = True,
        single_doc_cap_ratio: float | None = None,
    ) -> ExactLocationResult:
        if not isinstance(contract, EvidenceContract):
            raise ValueError(f"contract must be EvidenceContract; got {contract!r}")
        if type(full_corpus_fallback) is not bool:
            raise ValueError("full_corpus_fallback must be bool")
        context_lines = _positive_int(
            context_lines,
            "context_lines",
            allow_zero=True,
        )
        max_span_chars = _positive_int(max_span_chars, "max_span_chars")
        max_results = _positive_int(max_results, "max_results")
        slot = contract.get_slot(slot_id)
        if isinstance(documents, (str, bytes)):
            raise ValueError("documents must be a sequence of NormalizedDocument")
        try:
            documents = tuple(documents)
        except TypeError as exc:
            raise ValueError(
                "documents must be a sequence of NormalizedDocument"
            ) from exc
        manifest = materialize_normalized_documents(documents, self.canonical_root)
        document_by_id: dict[str, NormalizedDocument] = {}
        for document in documents:
            if document.document_id in document_by_id:
                raise ValueError(f"duplicate document_id {document.document_id!r}")
            document_by_id[document.document_id] = document
        candidate_document_ids: list[str] = []
        seen_documents: set[str] = set()
        candidates = tuple(candidates)
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, VectorCandidate):
                raise ValueError(
                    f"candidates[{index}] must be VectorCandidate; got {candidate!r}"
                )
            if candidate.slot_id != slot_id:
                raise ValueError(
                    f"candidate {candidate.chunk_id!r} belongs to slot "
                    f"{candidate.slot_id!r}, expected {slot_id!r}"
                )
            if candidate.document_id not in manifest.document_ids:
                raise ValueError(
                    f"candidate references unknown document_id {candidate.document_id!r}"
                )
            if candidate.document_id not in seen_documents:
                seen_documents.add(candidate.document_id)
                candidate_document_ids.append(candidate.document_id)
        anchors = extract_exact_anchors(
            slot,
            candidates,
            explicit_terms=explicit_terms,
        )
        attempts: list[ExactSearchAttempt] = []
        rg_source = getattr(self.executor, "rg_source", "unresolved") or "unresolved"
        if not anchors:
            return ExactLocationResult(
                slot_id=slot_id,
                anchors=(),
                evidence_spans=(),
                candidate_document_ids=tuple(candidate_document_ids),
                fallback_used=False,
                search_attempts=(),
                rg_source=rg_source,
            )
        document_id_by_path = {
            os.path.normcase(str(Path(entry.canonical_path).resolve())): entry.document_id
            for entry in manifest.documents
        }
        located: list[_LocatedEvidence] = []
        if candidate_document_ids:
            candidate_hits, rg_source = self._search_scope(
                scope_name="candidate",
                scope_priority=0,
                document_ids=tuple(candidate_document_ids),
                manifest=manifest,
                document_by_id=document_by_id,
                document_id_by_path=document_id_by_path,
                anchors=anchors,
                context_lines=context_lines,
                max_span_chars=max_span_chars,
                attempts=attempts,
            )
            located.extend(candidate_hits)
        fallback_used = False
        if not located and full_corpus_fallback and manifest.documents:
            fallback_used = True
            fallback_hits, rg_source = self._search_scope(
                scope_name="full_corpus",
                scope_priority=1,
                document_ids=manifest.document_ids,
                manifest=manifest,
                document_by_id=document_by_id,
                document_id_by_path=document_id_by_path,
                anchors=anchors,
                context_lines=context_lines,
                max_span_chars=max_span_chars,
                attempts=attempts,
            )
            located.extend(fallback_hits)
        merged = _merge_located(located, max_span_chars=max_span_chars)
        ranked_spans = tuple(
            (
                exact_rank,
                EvidenceSpan.from_document(
                    item.document,
                    item.start_char,
                    item.end_char,
                    discovery_sources=tuple(item.discovery_sources),
                    metadata={
                        "matched_anchors": tuple(item.anchor_terms),
                        "anchor_sources": tuple(item.anchor_sources),
                        "search_scopes": tuple(item.search_scopes),
                    },
                ),
            )
            for exact_rank, item in enumerate(merged, start=1)
        )
        baseline_ranked, overflow_ranked = _select_diverse_spans(
            ranked_spans, max_results, single_doc_cap_ratio,
        )
        evidence_spans = tuple(span for _, span in baseline_ranked)
        rescue_candidates = _build_rescue_candidates(
            overflow_ranked,
            candidates,
        )
        return ExactLocationResult(
            slot_id=slot_id,
            anchors=anchors,
            evidence_spans=evidence_spans,
            candidate_document_ids=tuple(candidate_document_ids),
            fallback_used=fallback_used,
            search_attempts=tuple(attempts),
            rg_source=rg_source,
            _rescue_candidates=rescue_candidates,
        )
