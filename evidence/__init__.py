"""契约驱动证据层。

本包导出固定 Contract、规范化文档、无损 Chunk、EvidenceSpan 与 Verifier / Gap Report。
"""

from .types import (
    SlotStatus, EvidenceSlot,
    EvidenceContract, EvidenceSpan, VectorCandidate,
)
from .contract import (
    load_evidence_contract,
    solution_material_completeness_v1,
)
from .documents import (
    LOSSLESS_CHUNKER_VERSION,
    LosslessChunk,
    NormalizedDocument,
    chunk_document,
    normalize_text,
)
from .vector_discovery import (
    INDEX_SCHEMA_VERSION,
    VectorDiscovery,
    VectorIndexManifest,
    build_index_fingerprint,
    build_slot_query,
)
from .rg_executor import (
    InvalidRipgrepOutputError,
    RipgrepError,
    RipgrepExecutionError,
    RipgrepExecutor,
    RipgrepLineEvent,
    RipgrepNotFoundError,
    RipgrepRunResult,
    RipgrepSubmatch,
    check_rg_proc,
    parse_rg_json_events,
    resolve_rg_path,
)
from .exact_locator import (
    CANONICAL_MANIFEST_NAME,
    CanonicalCorpusError,
    CanonicalCorpusManifest,
    CanonicalDocumentEntry,
    ExactAnchor,
    ExactEvidenceLocator,
    ExactLocationResult,
    ExactSearchAttempt,
    expand_match_context,
    extract_exact_anchors,
    map_rg_submatch_to_char_range,
    materialize_normalized_documents,
)
from .verifier import (
    SLOT_VERIFICATION_RULES,
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    DeterministicSlotVerifier,
    FieldCoverage,
    FieldRule,
    SlotRule,
    SlotVerificationResult,
    VerificationReasonCode,
    validate_rule_registry,
)
from .gap_report import (
    EvidenceReference,
    GapReport,
    GapReportBuilder,
    PendingActionType,
    PendingMaterialRequest,
    SlotGapResult,
    gap_report_to_dict,
    gap_report_to_json,
    render_gap_report_markdown,
    write_gap_report_json,
    write_gap_report_markdown,
)
from .gating import (
    ContractGate,
    SlotGate,
    aggregate_section_gate,
    contract_gate,
    gate_for_slot_status,
)
from .lexical_discovery import LEXICAL_DISCOVERY_VERSION, LexicalDiscovery
from .fusion import rrf_fuse
from .brief import (
    BRIEF_VERSION,
    CrosswalkError,
    assert_outward_clean,
    build_generation_brief,
    load_crosswalk,
)
_CLAIM_AUDIT_EXPORTS = frozenset({
    "AUDIT_BRIEF_VERSION",
    "AUDIT_SHEET_SCHEMA_VERSION",
    "RECHECK_VERSION",
    "VERDICTS",
    "AuditSheetError",
    "RecheckCode",
    "build_audit_brief",
    "recheck_audit_sheet",
    "validate_audit_sheet",
})


def __getattr__(name: str):
    """延迟暴露 Claim Audit API，避免 ``python -m evidence.claim_audit`` 的 runpy 预导入告警。"""

    if name in _CLAIM_AUDIT_EXPORTS:
        from . import claim_audit
        return getattr(claim_audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "load_evidence_contract",
    "solution_material_completeness_v1",
    "SlotStatus", "EvidenceSlot", "EvidenceContract", "EvidenceSpan",
    "VectorCandidate", "VectorDiscovery", "VectorIndexManifest",
    "INDEX_SCHEMA_VERSION", "build_index_fingerprint", "build_slot_query",
    "NormalizedDocument", "LosslessChunk", "LOSSLESS_CHUNKER_VERSION",
    "chunk_document", "normalize_text",
    "RipgrepError", "RipgrepNotFoundError", "RipgrepExecutionError",
    "InvalidRipgrepOutputError", "RipgrepSubmatch", "RipgrepLineEvent",
    "RipgrepRunResult", "RipgrepExecutor", "resolve_rg_path",
    "check_rg_proc", "parse_rg_json_events", "CANONICAL_MANIFEST_NAME",
    "CanonicalCorpusError", "CanonicalDocumentEntry",
    "CanonicalCorpusManifest", "ExactAnchor", "ExactSearchAttempt",
    "ExactLocationResult", "ExactEvidenceLocator",
    "materialize_normalized_documents", "extract_exact_anchors",
    "map_rg_submatch_to_char_range", "expand_match_context",
    "VerificationReasonCode", "FieldCoverage", "DetectedFact",
    "DetectedConflict", "SlotVerificationResult", "ContractVerificationResult",
    "FieldRule", "SlotRule", "SLOT_VERIFICATION_RULES",
    "validate_rule_registry", "DeterministicSlotVerifier",
    "PendingActionType", "PendingMaterialRequest", "EvidenceReference",
    "SlotGapResult", "GapReport", "GapReportBuilder", "gap_report_to_dict",
    "gap_report_to_json", "render_gap_report_markdown",
    "write_gap_report_json", "write_gap_report_markdown",
    "SlotGate", "ContractGate", "gate_for_slot_status",
    "aggregate_section_gate", "contract_gate",
    "LexicalDiscovery", "LEXICAL_DISCOVERY_VERSION", "rrf_fuse",
    "BRIEF_VERSION", "CrosswalkError", "load_crosswalk",
    "build_generation_brief", "assert_outward_clean",
    "AUDIT_BRIEF_VERSION", "AUDIT_SHEET_SCHEMA_VERSION", "RECHECK_VERSION",
    "VERDICTS", "AuditSheetError", "RecheckCode",
    "build_audit_brief", "validate_audit_sheet", "recheck_audit_sheet",
]
