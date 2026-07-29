"""End-to-end evidence gap checker and its command-line entry.

The path combines normalized documents, a fixed contract, candidate discovery,
exact localization, deterministic verification,
then one shared GapReport rendered as JSON and Markdown.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Callable, Collection, Mapping, Sequence

from evidence.core.config import (
    DEFAULT_EMBEDDING_API_KEY_ENV,
    DEFAULT_SETTINGS,
    Settings,
    resolve_embedding_settings,
)

from .contract import load_evidence_contract
from .documents import NormalizedDocument
from .runtime_adapter import input_completeness_summary
from .exact_locator import ExactEvidenceLocator, ExactLocationResult
from .fusion import rrf_fuse
from .lexical_discovery import LexicalDiscovery
from .library import DEFAULT_LIBRARY_ROOT, FilesystemLibrary, SourceType
from .rerank import (
    RerankBatchGuard,
    RerankError,
    rerank_candidates,
    union_candidates,
)
from .gap_report import (
    GapReport,
    GapReportBuilder,
    write_gap_report_json,
    write_gap_report_markdown,
)
from .types import EvidenceContract, EvidenceSpan, SlotStatus, VectorCandidate
from .vector_discovery import VectorDiscovery, VectorIndexManifest, build_slot_query
from .verifier import (
    ContractVerificationResult,
    DeterministicSlotVerifier,
    SlotVerificationResult,
    VerificationReasonCode,
)


from .resources import contract_file

DEFAULT_CONTRACT_PATH = contract_file("solution_material_completeness_v1.json")

_MANIFEST_KEYS = frozenset({"documents"})
_DOCUMENT_KEYS = frozenset(
    {
        "document_id",
        "title",
        "source_path",
        "source_type",
        "normalized_path",
        "metadata",
    }
)
_SUPPORTED_SUFFIX_TO_TYPE = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}
_MAX_EVICTION_CANDIDATES_CHECKED = 3
_DISALLOWED_NEW_REASON_CODES = frozenset(
    {
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM,
        VerificationReasonCode.MISSING_REQUIRED_FIELD,
        VerificationReasonCode.MISSING_SOURCE_CONTEXT,
        VerificationReasonCode.PROMOTIONAL_ONLY,
        VerificationReasonCode.CONFLICTING_FACTS,
    }
)

_EVICTION_NOT_NEUTRAL = "EVICTION_NOT_NEUTRAL"
_STATUS_NOT_FILLED = "STATUS_NOT_FILLED"
_FIELD_REGRESSION = "FIELD_REGRESSION"
_FACT_REGRESSION = "FACT_REGRESSION"
_NEW_CONFLICT = "NEW_CONFLICT"
_NEW_REASON_CODE = "NEW_REASON_CODE"
_QUANTITATIVE_RISK = "QUANTITATIVE_RISK"
_SEMANTIC_GATE_REGRESSION = "SEMANTIC_GATE_REGRESSION"
_SOURCE_CONTEXT_REGRESSION = "SOURCE_CONTEXT_REGRESSION"


def _require_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def load_normalized_document_manifest(
    path: str | Path,
) -> tuple[NormalizedDocument, ...]:
    """Load the strict Markdown/TXT manifest relative to its own directory."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read document manifest {str(manifest_path)!r}") from exc
    manifest = _require_object(raw, "manifest")
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError(
            f"manifest fields must be exactly {sorted(_MANIFEST_KEYS)!r}; "
            f"got {sorted(manifest)!r}"
        )
    raw_documents = manifest["documents"]
    if not isinstance(raw_documents, list):
        raise ValueError("manifest.documents must be a JSON array")
    manifest_root = manifest_path.resolve().parent
    documents: list[NormalizedDocument] = []
    seen_ids: set[str] = set()
    for index, raw_document in enumerate(raw_documents):
        location = f"manifest.documents[{index}]"
        item = _require_object(raw_document, location)
        if set(item) != _DOCUMENT_KEYS:
            raise ValueError(
                f"{location} fields must be exactly {sorted(_DOCUMENT_KEYS)!r}; "
                f"got {sorted(item)!r}"
            )
        document_id = _require_string(item["document_id"], f"{location}.document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id {document_id!r} in manifest")
        seen_ids.add(document_id)
        title = _require_string(item["title"], f"{location}.title")
        source_path = _require_string(item["source_path"], f"{location}.source_path")
        source_type = _require_string(item["source_type"], f"{location}.source_type")
        normalized_path = _require_string(
            item["normalized_path"],
            f"{location}.normalized_path",
        )
        relative_path = Path(normalized_path)
        if relative_path.is_absolute():
            raise ValueError(f"{location}.normalized_path must be relative to the manifest")
        resolved_path = (manifest_root / relative_path).resolve()
        try:
            resolved_path.relative_to(manifest_root)
        except ValueError as exc:
            raise ValueError(
                f"{location}.normalized_path must stay within the manifest directory"
            ) from exc
        suffix = resolved_path.suffix.lower()
        expected_source_type = _SUPPORTED_SUFFIX_TO_TYPE.get(suffix)
        if expected_source_type is None:
            raise ValueError(
                f"{location}.normalized_path has unsupported extension {suffix!r}"
            )
        if source_type != expected_source_type:
            raise ValueError(
                f"{location}.source_type {source_type!r} conflicts with "
                f"extension {suffix!r}"
            )
        if not resolved_path.is_file():
            raise ValueError(
                f"{location}.normalized_path does not exist: {normalized_path!r}"
            )
        metadata = item["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}.metadata must be a JSON object")
        documents.append(
            NormalizedDocument.from_path(
                resolved_path,
                document_id=document_id,
                title=title,
                source_path=source_path,
                source_type=source_type,
                normalized_path=normalized_path.replace("\\", "/"),
                metadata=metadata,
            )
        )
    return tuple(documents)


@dataclass(frozen=True)
class EvidenceGapCheckRun:
    report: GapReport
    vector_manifest: VectorIndexManifest
    candidates_by_slot: Mapping[str, tuple[VectorCandidate, ...]]
    exact_results_by_slot: Mapping[str, ExactLocationResult]
    verification: ContractVerificationResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates_by_slot",
            MappingProxyType(dict(self.candidates_by_slot)),
        )
        object.__setattr__(
            self,
            "exact_results_by_slot",
            MappingProxyType(dict(self.exact_results_by_slot)),
        )


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        comparator = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"{name} must be an integer {comparator}; got {value!r}")
    return value


def _stable_value(value: object) -> object:
    """Convert verifier-owned structured output to a hashable stable value."""

    if isinstance(value, Enum):
        return _stable_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (item.name, _stable_value(getattr(value, item.name)))
            for item in dataclass_fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _stable_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_stable_value(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported verifier value in safety signature: {type(value).__name__}")


def _stable_items(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("verifier collection must not be a string")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("verifier collection must be iterable") from exc
    return tuple(sorted(_stable_value(item) for item in items))


def _covered_and_missing(
    result: SlotVerificationResult,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    covered = tuple(
        sorted(item.field_id for item in result.field_coverage if item.covered)
    )
    missing = tuple(
        sorted(item.field_id for item in result.field_coverage if not item.covered)
    )
    return covered, missing


def _trace_collection(
    result: SlotVerificationResult,
    key: str,
) -> tuple[object, ...]:
    return _stable_items(result.trace.get(key, ()))


def _source_context_signature(result: SlotVerificationResult) -> tuple[str, ...]:
    raw = result.trace.get("missing_source_context_evidence_ids", ())
    if isinstance(raw, (str, bytes)):
        raise TypeError("source-context verifier trace must be a sequence")
    return tuple(sorted(str(item) for item in raw))  # type: ignore[arg-type]


@dataclass(frozen=True)
class _VerificationSafetySignature:
    status: str
    covered_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    detected_facts: tuple[object, ...]
    detected_conflicts: tuple[object, ...]
    quantitative_trace: tuple[object, ...]
    semantic_gate: object
    source_context: tuple[str, ...]


def _verification_safety_signature(
    result: SlotVerificationResult,
) -> _VerificationSafetySignature:
    covered, missing = _covered_and_missing(result)
    return _VerificationSafetySignature(
        status=result.status.value,
        covered_fields=covered,
        missing_fields=missing,
        reason_codes=tuple(sorted(item.value for item in result.reason_codes)),
        detected_facts=_trace_collection(result, "detected_facts"),
        detected_conflicts=_stable_items(result.conflicts),
        quantitative_trace=_trace_collection(result, "quantitative_claims"),
        semantic_gate=_stable_value(result.trace.get("semantic_gate", {})),
        source_context=_source_context_signature(result),
    )


def _quantitative_risks(result: SlotVerificationResult) -> frozenset[object]:
    raw_claims = result.trace.get("quantitative_claims", ())
    if isinstance(raw_claims, (str, bytes)):
        raise TypeError("quantitative verifier trace must be a sequence")
    risks: set[object] = set()
    for claim in raw_claims:  # type: ignore[union-attr]
        if not isinstance(claim, Mapping):
            raise TypeError("quantitative verifier trace item must be a mapping")
        if bool(claim.get("requires_outcome_support")) and not bool(
            claim.get("verified")
        ):
            risks.add(_stable_value(claim))
    return frozenset(risks)


def _semantic_gate_regressed(
    initial: SlotVerificationResult,
    final: SlotVerificationResult,
) -> bool:
    initial_gate = initial.trace.get("semantic_gate", {})
    final_gate = final.trace.get("semantic_gate", {})
    if not isinstance(initial_gate, Mapping) or not isinstance(final_gate, Mapping):
        raise TypeError("semantic gate verifier trace must be a mapping")
    if bool(initial_gate.get("applicable")) and not bool(final_gate.get("applicable")):
        return True
    if bool(initial_gate.get("evaluated")) and not bool(final_gate.get("evaluated")):
        return True
    if initial_gate.get("passed") is True and final_gate.get("passed") is not True:
        return True
    if bool(final_gate.get("evaluated")) and final_gate.get("passed") is not True:
        return True
    initial_failed = frozenset(initial_gate.get("failed_aspects", ()))
    final_failed = frozenset(final_gate.get("failed_aspects", ()))
    return not final_failed <= initial_failed


def _post_swap_guard_failures(
    initial: SlotVerificationResult,
    final: SlotVerificationResult,
) -> tuple[str, ...]:
    failures: list[str] = []
    initial_covered, initial_missing = _covered_and_missing(initial)
    final_covered, final_missing = _covered_and_missing(final)
    if final.status is not SlotStatus.FILLED:
        failures.append(_STATUS_NOT_FILLED)
    if (
        not set(final_covered) >= set(initial_covered)
        or final_missing
        or not set(final_missing) < set(initial_missing)
    ):
        failures.append(_FIELD_REGRESSION)
    if not set(_trace_collection(initial, "detected_facts")) <= set(
        _trace_collection(final, "detected_facts")
    ):
        failures.append(_FACT_REGRESSION)
    if _stable_items(final.conflicts) != _stable_items(initial.conflicts):
        failures.append(_NEW_CONFLICT)
    new_reason_codes = set(final.reason_codes) - set(initial.reason_codes)
    if new_reason_codes & _DISALLOWED_NEW_REASON_CODES:
        failures.append(_NEW_REASON_CODE)
    if not _quantitative_risks(final) <= _quantitative_risks(initial):
        failures.append(_QUANTITATIVE_RISK)
    if _semantic_gate_regressed(initial, final):
        failures.append(_SEMANTIC_GATE_REGRESSION)
    if not set(_source_context_signature(final)) <= set(
        _source_context_signature(initial)
    ):
        failures.append(_SOURCE_CONTEXT_REGRESSION)
    return tuple(failures)


def _append_failure(failures: list[str], failure: str) -> None:
    if failure not in failures:
        failures.append(failure)


@dataclass(frozen=True)
class _SpanRescueTrace:
    initial_evidence_count: int
    initial_verification_status: str
    initial_missing_fields: tuple[str, ...]
    rescue_available: bool
    rescue_candidate_count: int
    rescue_candidate_evidence_ids: tuple[str, ...]
    rescue_candidate_exact_raw_ranks: tuple[int, ...]
    rescue_attempted: bool
    eviction_candidates_checked: int
    neutral_eviction_count: int
    swap_attempt_count: int
    rescue_applied: bool
    evicted_evidence_id: str | None
    inserted_evidence_id: str | None
    final_verification_status: str
    final_missing_fields: tuple[str, ...]
    guard_failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in dataclass_fields(self)
        }


def _verify_slot(
    verifier: object,
    *,
    contract: EvidenceContract,
    slot_id: str,
    evidence_spans: Sequence[EvidenceSpan],
) -> SlotVerificationResult:
    method = getattr(verifier, "verify_slot", None)
    if not callable(method):
        raise RuntimeError("verifier.verify_slot must be callable for span rescue")
    result = method(
        contract=contract,
        slot_id=slot_id,
        evidence_spans=evidence_spans,
    )
    if not isinstance(result, SlotVerificationResult):
        raise RuntimeError("verifier.verify_slot must return SlotVerificationResult")
    if result.slot_id != slot_id:
        raise RuntimeError("verifier.verify_slot returned a result for the wrong slot")
    return result


def _attempt_span_rescue(
    verifier: object,
    *,
    contract: EvidenceContract,
    slot_id: str,
    baseline_spans: tuple[EvidenceSpan, ...],
    rescue_candidates: Sequence[object],
    initial: SlotVerificationResult,
) -> tuple[tuple[EvidenceSpan, ...], SlotVerificationResult, _SpanRescueTrace]:
    _, initial_missing = _covered_and_missing(initial)
    rescue_candidates = tuple(rescue_candidates)
    rescue_spans: list[EvidenceSpan] = []
    rescue_ranks: list[int] = []
    for candidate in rescue_candidates:
        span = getattr(candidate, "evidence_span", None)
        exact_rank = getattr(candidate, "exact_raw_rank", None)
        if not isinstance(span, EvidenceSpan) or type(exact_rank) is not int:
            raise RuntimeError("exact locator returned an invalid rescue candidate")
        rescue_spans.append(span)
        rescue_ranks.append(exact_rank)
    rescue_available = bool(rescue_spans)
    eligible = (
        initial.status in {SlotStatus.WEAK, SlotStatus.MISSING}
        and bool(initial_missing)
        and rescue_available
    )
    checked = 0
    neutral: list[tuple[int, EvidenceSpan]] = []
    swap_attempts = 0
    applied = False
    evicted_id: str | None = None
    inserted_id: str | None = None
    final_spans = baseline_spans
    final_result = initial
    failures: list[str] = []

    if eligible:
        initial_signature = _verification_safety_signature(initial)
        first_index = len(baseline_spans) - 1
        stop_index = max(-1, first_index - _MAX_EVICTION_CANDIDATES_CHECKED)
        for index in range(first_index, stop_index, -1):
            checked += 1
            without = baseline_spans[:index] + baseline_spans[index + 1 :]
            leave_one_out = _verify_slot(
                verifier,
                contract=contract,
                slot_id=slot_id,
                evidence_spans=without,
            )
            if _verification_safety_signature(leave_one_out) == initial_signature:
                neutral.append((index, baseline_spans[index]))
            else:
                _append_failure(failures, _EVICTION_NOT_NEUTRAL)

        baseline_ids = {span.evidence_id for span in baseline_spans}
        for rescue_span in rescue_spans:
            if rescue_span.evidence_id in baseline_ids:
                continue
            for index, eviction_span in neutral:
                swap_attempts += 1
                swapped = (
                    baseline_spans[:index]
                    + (rescue_span,)
                    + baseline_spans[index + 1 :]
                )
                swapped_result = _verify_slot(
                    verifier,
                    contract=contract,
                    slot_id=slot_id,
                    evidence_spans=swapped,
                )
                guard_failures = _post_swap_guard_failures(initial, swapped_result)
                if guard_failures:
                    for failure in guard_failures:
                        _append_failure(failures, failure)
                    continue
                final_spans = swapped
                final_result = swapped_result
                applied = True
                evicted_id = eviction_span.evidence_id
                inserted_id = rescue_span.evidence_id
                break
            if applied:
                break

    _, final_missing = _covered_and_missing(final_result)
    trace = _SpanRescueTrace(
        initial_evidence_count=len(baseline_spans),
        initial_verification_status=initial.status.value,
        initial_missing_fields=initial_missing,
        rescue_available=rescue_available,
        rescue_candidate_count=len(rescue_spans),
        rescue_candidate_evidence_ids=tuple(
            span.evidence_id for span in rescue_spans
        ),
        rescue_candidate_exact_raw_ranks=tuple(rescue_ranks),
        rescue_attempted=eligible,
        eviction_candidates_checked=checked,
        neutral_eviction_count=len(neutral),
        swap_attempt_count=swap_attempts,
        rescue_applied=applied,
        evicted_evidence_id=evicted_id,
        inserted_evidence_id=inserted_id,
        final_verification_status=final_result.status.value,
        final_missing_fields=final_missing,
        guard_failure_reasons=tuple(failures),
    )
    return final_spans, final_result, trace


class EvidenceGapChecker:
    """Injectable orchestration without recovery or hidden downgrade paths."""

    def __init__(
        self,
        *,
        vector_discovery: object,
        exact_locator: object,
        verifier: object | None = None,
        report_builder: object | None = None,
        lexical_discovery: object | None = None,
        reranker: object | None = None,
        rrf_k: int = 60,
        single_doc_cap_ratio: float | None = None,
    ) -> None:
        dependencies = {
            "vector_discovery.index_documents": getattr(
                vector_discovery, "index_documents", None
            ),
            "vector_discovery.discover_contract": getattr(
                vector_discovery, "discover_contract", None
            ),
            "exact_locator.locate_for_slot": getattr(
                exact_locator, "locate_for_slot", None
            ),
        }
        for name, dependency in dependencies.items():
            if not callable(dependency):
                raise ValueError(f"{name} must be callable")
        verifier = DeterministicSlotVerifier() if verifier is None else verifier
        report_builder = GapReportBuilder() if report_builder is None else report_builder
        if not callable(getattr(verifier, "verify_contract", None)):
            raise ValueError("verifier.verify_contract must be callable")
        if not callable(getattr(report_builder, "build", None)):
            raise ValueError("report_builder.build must be callable")
        if lexical_discovery is not None and not callable(
            getattr(lexical_discovery, "discover_contract", None)
        ):
            raise ValueError("lexical_discovery.discover_contract must be callable")
        self.vector_discovery = vector_discovery
        self.exact_locator = exact_locator
        self.verifier = verifier
        self.report_builder = report_builder
        # 可选候选多样性与精排组件；默认行为由 Settings 决定。
        if reranker is not None and not callable(getattr(reranker, "scores", None)):
            raise ValueError("reranker.scores must be callable")
        self.lexical_discovery = lexical_discovery
        self.reranker = reranker
        self._rerank_guard = RerankBatchGuard()
        self.rrf_k = rrf_k
        self.single_doc_cap_ratio = single_doc_cap_ratio

    def run(
        self,
        *,
        contract: EvidenceContract,
        documents: Sequence[NormalizedDocument],
        max_chars: int,
        top_k_per_slot: int,
        context_lines: int,
        max_span_chars: int,
        max_results_per_slot: int,
        allowed_document_ids: Collection[str] | None = None,
    ) -> EvidenceGapCheckRun:
        """``allowed_document_ids`` 提供材料库元数据过滤；``None`` 表示不过滤。

        给定时索引仍建在全量 ``documents`` 上（指纹不变、
        一份索引服务任意过滤组合），只在候选层收窄来源。"""
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        if isinstance(documents, (str, bytes)):
            raise ValueError("documents must be a sequence of NormalizedDocument")
        documents = tuple(documents)
        if any(not isinstance(item, NormalizedDocument) for item in documents):
            raise ValueError("documents must contain only NormalizedDocument objects")
        document_ids = tuple(item.document_id for item in documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("documents contain duplicate document IDs")
        max_chars = _positive_int(max_chars, "max_chars")
        top_k_per_slot = _positive_int(top_k_per_slot, "top_k_per_slot")
        context_lines = _positive_int(context_lines, "context_lines", allow_zero=True)
        max_span_chars = _positive_int(max_span_chars, "max_span_chars")
        max_results_per_slot = _positive_int(
            max_results_per_slot,
            "max_results_per_slot",
        )

        manifest = self.vector_discovery.index_documents(
            documents,
            max_chars=max_chars,
        )
        fingerprint = getattr(manifest, "fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise RuntimeError("vector discovery returned a manifest without fingerprint")
        if allowed_document_ids is not None:
            allowed_document_ids = frozenset(allowed_document_ids)
            unknown = allowed_document_ids - set(document_ids)
            if unknown:
                raise ValueError(
                    f"allowed_document_ids contains unknown documents: {sorted(unknown)!r}"
                )
        discovered = self.vector_discovery.discover_contract(
            contract,
            top_k_per_slot=top_k_per_slot,
            manifest=manifest,
            allowed_document_ids=allowed_document_ids,
        )
        if not isinstance(discovered, Mapping):
            raise RuntimeError("discover_contract must return a slot mapping")
        if set(discovered) != set(contract.slot_ids):
            raise RuntimeError(
                "discover_contract keys must exactly match contract slots; "
                f"got {tuple(discovered)!r}"
            )

        # 可选词法候选通道。
        rerank_stats_by_slot: dict[str, dict] = {}
        self._rerank_guard = RerankBatchGuard()   # 每次 run 重置批次闸门
        lexical_by_slot: Mapping[str, Sequence[VectorCandidate]] = {}
        if self.lexical_discovery is not None:
            lexical_by_slot = self.lexical_discovery.discover_contract(
                contract,
                documents=documents,
                top_k_per_slot=top_k_per_slot,
                max_chars=max_chars,
            )
            if set(lexical_by_slot) != set(contract.slot_ids):
                raise RuntimeError(
                    "lexical discover_contract keys must match contract slots"
                )

        candidates_by_slot: dict[str, tuple[VectorCandidate, ...]] = {}
        exact_results: dict[str, ExactLocationResult] = {}
        evidence_by_slot: dict[str, tuple[EvidenceSpan, ...]] = {}
        for slot in contract.slots:
            raw_candidates = discovered[slot.slot_id]
            if isinstance(raw_candidates, (str, bytes)):
                raise RuntimeError(f"candidates for slot {slot.slot_id!r} are not a sequence")
            vector_candidates = tuple(raw_candidates)
            lexical_candidates = tuple(lexical_by_slot.get(slot.slot_id, ()))
            if allowed_document_ids is not None:
                # 词法通道默认关；开启时同样受 L1 过滤约束，两路口径一致。
                lexical_candidates = tuple(
                    item for item in lexical_candidates
                    if item.document_id in allowed_document_ids
                )
            if self.reranker is not None:
                # 词法只扩召回，去留交 cross-encoder 精排；这里仍是候选层，不判充分性。
                pool = union_candidates(vector_candidates, lexical_candidates)
                try:
                    candidates, rerank_stats = rerank_candidates(
                        pool,
                        query=build_slot_query(slot),
                        scorer=self.reranker,
                        top_k=top_k_per_slot,
                    )
                    self._rerank_guard.record(ok=True)
                except RerankError as exc:
                    # 单 slot 失败：记日志跳过、退回未精排的并集截断；失败率超阈值时停批。
                    self._rerank_guard.record(ok=False, detail=f"{slot.slot_id}: {exc}")
                    self._rerank_guard.check()
                    print(f"warning: rerank failed for slot {slot.slot_id!r}: {exc}",
                          file=sys.stderr)
                    candidates = pool[:top_k_per_slot]
                    rerank_stats = {"pool_size": len(pool), "elapsed_s": 0.0, "failed": True}
                rerank_stats_by_slot[slot.slot_id] = rerank_stats
            elif self.lexical_discovery is not None:
                # 可选 RRF 对照路径；产品默认使用精排路径。
                candidates = rrf_fuse(
                    [vector_candidates, lexical_candidates],
                    rrf_k=self.rrf_k,
                    top_k=top_k_per_slot,
                )
            else:
                candidates = vector_candidates
            for candidate in candidates:
                if not isinstance(candidate, VectorCandidate):
                    raise RuntimeError(
                        f"candidate for slot {slot.slot_id!r} is not VectorCandidate"
                    )
                if candidate.slot_id != slot.slot_id:
                    raise RuntimeError(
                        f"candidate {candidate.chunk_id!r} belongs to wrong slot"
                    )
            candidates_by_slot[slot.slot_id] = candidates
            exact = self.exact_locator.locate_for_slot(
                contract=contract,
                slot_id=slot.slot_id,
                documents=documents,
                candidates=candidates,
                context_lines=context_lines,
                max_span_chars=max_span_chars,
                max_results=max_results_per_slot,
                full_corpus_fallback=True,
                single_doc_cap_ratio=self.single_doc_cap_ratio,
            )
            if not isinstance(exact, ExactLocationResult):
                raise RuntimeError("locate_for_slot must return ExactLocationResult")
            if exact.slot_id != slot.slot_id:
                raise RuntimeError("exact locator returned a result for the wrong slot")
            spans = tuple(exact.evidence_spans)
            if any(not isinstance(span, EvidenceSpan) for span in spans):
                raise RuntimeError(
                    f"exact locator returned non-EvidenceSpan input for slot {slot.slot_id!r}"
                )
            exact_results[slot.slot_id] = exact
            evidence_by_slot[slot.slot_id] = spans

        initial_verification = self.verifier.verify_contract(
            contract=contract,
            evidence_by_slot=evidence_by_slot,
        )
        if not isinstance(initial_verification, ContractVerificationResult):
            raise RuntimeError("verifier must return ContractVerificationResult")
        if tuple(item.slot_id for item in initial_verification.slots) != contract.slot_ids:
            raise RuntimeError("verifier result slots must exactly match contract order")
        initial_by_slot = {
            item.slot_id: item for item in initial_verification.slots
        }
        rescue_trace_by_slot: dict[str, _SpanRescueTrace] = {}
        rescued = False
        for slot in contract.slots:
            final_spans, final_slot_result, rescue_trace = _attempt_span_rescue(
                self.verifier,
                contract=contract,
                slot_id=slot.slot_id,
                baseline_spans=evidence_by_slot[slot.slot_id],
                rescue_candidates=exact_results[slot.slot_id]._rescue_candidates,
                initial=initial_by_slot[slot.slot_id],
            )
            evidence_by_slot[slot.slot_id] = final_spans
            rescue_trace_by_slot[slot.slot_id] = rescue_trace
            rescued = rescued or rescue_trace.rescue_applied
            if final_slot_result.slot_id != slot.slot_id:
                raise RuntimeError("span rescue returned a result for the wrong slot")

        verification = initial_verification
        if rescued:
            verification = self.verifier.verify_contract(
                contract=contract,
                evidence_by_slot=evidence_by_slot,
            )
            if not isinstance(verification, ContractVerificationResult):
                raise RuntimeError("verifier must return ContractVerificationResult")
            if tuple(item.slot_id for item in verification.slots) != contract.slot_ids:
                raise RuntimeError("verifier result slots must exactly match contract order")
        verification_by_slot = {item.slot_id: item for item in verification.slots}
        trace: dict[str, object] = {
            "contract_id": contract.contract_id,
            "document_ids": sorted(document_ids),
            "document_count": len(documents),
            "vector_index_fingerprint": fingerprint,
            "max_chars": max_chars,
            "top_k_per_slot": top_k_per_slot,
            "context_lines": context_lines,
            "max_span_chars": max_span_chars,
            "max_results_per_slot": max_results_per_slot,
            "per_slot": {
                slot.slot_id: {
                    "vector_candidate_count": len(candidates_by_slot[slot.slot_id]),
                    "anchor_count": len(exact_results[slot.slot_id].anchors),
                    "exact_span_count": len(exact_results[slot.slot_id].evidence_spans),
                    "fallback_used": exact_results[slot.slot_id].fallback_used,
                    "verification_status": verification_by_slot[slot.slot_id].status.value,
                    **rescue_trace_by_slot[slot.slot_id].to_dict(),
                }
                for slot in contract.slots
            },
            # 输入完整性：记录无文本页、文本框、图表、页眉页脚与批注等未提取载体。
            # 只读文档 metadata，不参与判定；让"材料够不够"的结论带上"我可能没看全"的自知。
            "input_completeness": input_completeness_summary(documents),
        }
        report = self.report_builder.build(
            contract=contract,
            verification=verification,
            evidence_by_slot=evidence_by_slot,
            trace=trace,
        )
        if not isinstance(report, GapReport):
            raise RuntimeError("report builder must return GapReport")
        return EvidenceGapCheckRun(
            report=report,
            vector_manifest=manifest,
            candidates_by_slot=candidates_by_slot,
            exact_results_by_slot=exact_results,
            verification=verification,
        )


def build_default_gap_checker(
    settings: Settings = DEFAULT_SETTINGS,
) -> EvidenceGapChecker:
    """Wire current embedding settings without starting or downloading a service."""

    if not isinstance(settings, Settings):
        raise ValueError("settings must be Settings")
    canonical_root = Path(settings.vector_db_path) / "m1-canonical"
    return EvidenceGapChecker(
        vector_discovery=VectorDiscovery(settings),
        exact_locator=ExactEvidenceLocator(canonical_root),
        # exact-span 单文档上限默认开启；纯 RRF 路径仅保留作对照，产品默认使用精排路径。
        lexical_discovery=None,
        rrf_k=settings.rrf_k,
        single_doc_cap_ratio=settings.exact_span_single_doc_cap_ratio,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Evidence Runtime material-readiness and gap checker"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="normalized document manifest JSON (预规范化 MD/TXT)")
    source.add_argument(
        "--input",
        help="原始文件或目录（PDF/DOCX/MD/TXT）；规范化后进入主链",
    )
    source.add_argument(
        "--project",
        help="从项目材料库读取当前版本材料；索引写入该项目自己的 index/ 目录",
    )
    source.add_argument(
        "--spec",
        help="ProjectSpec JSON：解析任务、定位项目库并确定性选择契约；"
             "--project/--input 仍可单独使用",
    )
    parser.add_argument(
        "--library-root",
        default=DEFAULT_LIBRARY_ROOT,
        help=f"材料库根目录（--project / --spec 模式；默认 {DEFAULT_LIBRARY_ROOT}）",
    )
    parser.add_argument(
        "--source-types",
        help="仅 --project 模式：按来源收窄候选，逗号分隔"
             f"（{'/'.join(item.value for item in SourceType)}）；不传 = 不过滤",
    )
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="fixed Evidence Contract JSON",
    )
    parser.add_argument("--output-dir", required=True, help="directory for both reports")
    parser.add_argument(
        "--brief",
        metavar="DOC_TYPE",
        help="额外产出 GenerationBrief（对接 solution-drafter；传 doc_type，如 解决方案建议书 / "
             "proposal-document）；写 generation-brief.json + .audit.json（Engine model-free）",
    )
    parser.add_argument("--project-id", help="GenerationBrief meta.project_id（可选）")
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--top-k-per-slot", type=int, default=8)
    parser.add_argument("--context-lines", type=int, default=2)
    parser.add_argument("--max-span-chars", type=int, default=1200)
    parser.add_argument("--max-results-per-slot", type=int, default=20)
    parser.add_argument(
        "--fake-embedder", action="store_true",
        help="用确定性 fake embedder（无 embedding 服务即可跑；demo/CI 用）。"
             "**只换 L2 embedder，判定/闸门不变；检索质量不真实，产物非准确率证据**",
    )
    embedding = parser.add_argument_group(
        "embedding",
        "真实检索配置；优先级：默认值 < JSON < EVIDENCE_* 环境变量 < CLI",
    )
    embedding.add_argument(
        "--embedding-config",
        metavar="JSON",
        help="embedding JSON 配置；不得在文件中保存 API key",
    )
    embedding.add_argument(
        "--embedding-backend",
        choices=("api", "local"),
        help="api=OpenAI-compatible endpoint；local=sentence-transformers",
    )
    embedding.add_argument(
        "--embedding-api-base",
        help="OpenAI-compatible base URL（含 /v1）",
    )
    embedding.add_argument(
        "--embedding-api-model",
        help="API embedding model id；会写入索引一致性元数据",
    )
    embedding.add_argument(
        "--embedding-api-timeout",
        type=float,
        help="单次 embedding API 超时秒数（>0）",
    )
    embedding.add_argument(
        "--embedding-local-model",
        dest="embedding_model",
        help="local backend 的 sentence-transformers model id",
    )
    embedding.add_argument(
        "--embedding-api-key-env",
        metavar="ENV_NAME",
        help=f"保存 API key 的环境变量名（默认 {DEFAULT_EMBEDDING_API_KEY_ENV}）",
    )
    return parser


def _redact_error(error: Exception, settings: Settings) -> str:
    message = str(error)
    secret = settings.embedding_api_key
    if secret:
        message = message.replace(secret, "***")
    return f"{type(error).__name__}: {message}"


def main(
    argv: Sequence[str] | None = None,
    *,
    checker_factory: Callable[[Settings], EvidenceGapChecker] | None = None,
) -> int:
    parser = build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    settings = DEFAULT_SETTINGS
    library = None
    allowed_document_ids: Collection[str] | None = None
    try:
        settings = resolve_embedding_settings(
            defaults=DEFAULT_SETTINGS,
            config_path=args.embedding_config,
            cli_overrides={
                "embedding_backend": args.embedding_backend,
                "embedding_api_base": args.embedding_api_base,
                "embedding_api_model": args.embedding_api_model,
                "embedding_api_timeout": args.embedding_api_timeout,
                "embedding_model": args.embedding_model,
            },
            api_key_env=args.embedding_api_key_env,
        )
        # 云端 embedding 隐私警告：材料会发往配置的远端服务。
        if not getattr(args, "fake_embedder", False):
            from .privacy import remote_embedding_warning
            _warn = remote_embedding_warning(settings)
            if _warn:
                print(_warn, file=sys.stderr)

        # L0 spec 模式是产品入口；spec 是“用户任务 → 系统契约”的翻译层，
        #    解析后 = 定位项目库 + 确定性选契约 + 逗号来源过滤，随后与 --project 逐字节同路。
        contract_path = args.contract
        project_id = args.project
        source_types_arg = args.source_types
        if args.spec:
            from .intake import load_project_spec, select_contract
            spec = load_project_spec(args.spec)
            binding = select_contract(spec.doc_type)          # 确定性查表，不是 agent 猜
            contract_path = str(binding.contract_path)
            project_id = spec.project_id
            if spec.source_types is not None:
                source_types_arg = ",".join(spec.source_types)

        contract = load_evidence_contract(contract_path)
        if args.manifest:
            documents = load_normalized_document_manifest(args.manifest)
        elif project_id:
            # —— L1 项目模式：材料与索引都归该项目所有（vector_db_path 指向项目 index/）——
            library = FilesystemLibrary(args.library_root, project_id)
            documents = library.documents()
            if not documents:
                raise ValueError(
                    f"project {project_id!r} has no current materials; run `library add` first"
                )
            settings = replace(settings, vector_db_path=str(library.index_root()))
            if source_types_arg:
                wanted = [item.strip() for item in source_types_arg.split(",") if item.strip()]
                allowed = library.documents(source_types=wanted)
                allowed_document_ids = frozenset(item.document_id for item in allowed)
                if not allowed_document_ids:
                    raise ValueError(
                        f"no materials match source types {source_types_arg!r}"
                    )
        else:
            from .runtime_adapter import adapt_path
            documents = adapt_path(args.input)
        if checker_factory is not None:
            factory = checker_factory
        elif getattr(args, "fake_embedder", False):
            from .fake_embedder import build_fake_gap_checker
            factory = build_fake_gap_checker
        else:
            factory = build_default_gap_checker
        checker = factory(settings)
        run = checker.run(
            contract=contract,
            documents=documents,
            max_chars=args.max_chars,
            top_k_per_slot=args.top_k_per_slot,
            context_lines=args.context_lines,
            max_span_chars=args.max_span_chars,
            max_results_per_slot=args.max_results_per_slot,
            allowed_document_ids=allowed_document_ids,
        )
        if library is not None:
            library.mark_indexed()   # 索引已按当前材料集建好，登记指纹
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "gap-report.json"
        markdown_path = output_dir / "gap-report.md"
        write_gap_report_json(run.report, json_path)
        write_gap_report_markdown(run.report, markdown_path)
        brief_path = None
        if args.brief:
            from .brief import (
                assert_outward_clean,
                build_generation_brief,
                load_crosswalk,
            )
            crosswalk = load_crosswalk(args.brief, contract=contract)
            brief, audit = build_generation_brief(
                report=run.report,
                documents=documents,
                crosswalk=crosswalk,
                contract=contract,
                # brief.meta.project_id 应继承已解析的 project_id（--spec 的 spec.project_id
                # 或 --project），显式 --project-id 才覆盖。此前只取 args.project_id → --spec 下恒为 None。
                project_id=args.project_id or project_id,
            )
            assert_outward_clean(brief)   # 手段无关红线自检（违反即抛，不落盘脏 brief）
            brief_path = output_dir / "generation-brief.json"
            audit_path = output_dir / "generation-brief.audit.json"
            brief_path.write_text(
                json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception as exc:
        print(f"error: {_redact_error(exc, settings)}", file=sys.stderr)
        return 2

    counts = {
        status.value: sum(1 for item in run.report.slots if item.status is status)
        for status in SlotStatus
    }
    print(f"status: {run.report.status}")
    print("slots: " + " ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"pending: {len(run.report.pending)}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")
    if brief_path is not None:
        print(f"brief: {brief_path}")
    return 0 if run.report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
