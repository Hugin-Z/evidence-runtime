"""Exact Evidence Locator：物化、锚点、偏移、选取与主类。

模块按职责拆成
`constants` / `helpers` / `canonical_corpus` / `anchors` / `offsets` / `models` /
`span_selection` / `locator`。**纯移动**：函数体逐字符未改，对外符号与 import 路径
（`from evidence.exact_locator import X`）完全不变——含测试反向依赖的私有符号
`_build_rescue_candidates` / `_select_diverse_spans`。
"""
from .constants import (
    CANONICAL_MANIFEST_NAME,
    MAX_ANCHORS,
    MAX_ANCHOR_CHARS,
    _ANCHOR_SOURCES,
    _MAX_RESCUE_CANDIDATES_PER_SLOT,
)
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
from .locator import ExactEvidenceLocator

__all__ = [
    "CANONICAL_MANIFEST_NAME", "CanonicalCorpusError", "CanonicalDocumentEntry",
    "CanonicalCorpusManifest", "ExactAnchor", "ExactSearchAttempt",
    "ExactLocationResult", "ExactEvidenceLocator",
    "materialize_normalized_documents", "extract_exact_anchors",
    "map_rg_submatch_to_char_range", "expand_match_context",
]
