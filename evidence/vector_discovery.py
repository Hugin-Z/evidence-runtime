"""按 Evidence Slot 执行可追溯的向量语义候选发现。

本模块复用 ``core.embeddings`` 的唯一 embedding 后端、``core.config.Settings``
的 Chroma 路径，以及从 ``core.retrievers.vector.VectorRetriever.search`` 抽出的
``query_vector_collection``。索引只接受 canonical ``NormalizedDocument`` /
``LosslessChunk``，不会调用旧 ``core.ingest`` chunker。

``VectorCandidate`` 只是 discovery 结果，不是 ``EvidenceSpan``，不得绕过后续
Exact Locator 和 Verifier。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Collection, Mapping, Sequence

from evidence.core import embeddings
from evidence.core.config import DEFAULT_SETTINGS, Settings
from evidence.core.embeddings import Embedder
from evidence.core.retrievers.vector import query_vector_collection

from .documents import (
    LOSSLESS_CHUNKER_VERSION,
    LosslessChunk,
    NormalizedDocument,
    chunk_document,
)
from .types import EvidenceContract, EvidenceSlot, VectorCandidate


INDEX_SCHEMA_VERSION = "m1-vector-index-v1"
COLLECTION_PREFIX = "m1-vector-"
MANIFEST_DIRECTORY = "m1-vector-manifests"
_DIVERSITY_OVERSAMPLE_FACTOR = 4
_DIVERSITY_OVERSAMPLE_CAP = 32
_HIGH_OVERLAP_RATIO = 0.8


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1; got {value!r}")
    return value


def _embedding_dimension(value: object) -> int | None:
    if value is None:
        return None
    return _positive_int(value, "embedding_dimension")


def _raw_query_k(top_k: int) -> int:
    """Return the bounded raw depth used before diversity selection.

    Public ``top_k`` remains the final result limit.  Oversampling is capped at
    32 candidates; a caller explicitly requesting more than 32 still receives
    at least its requested depth rather than being silently truncated.
    """

    top_k = _positive_int(top_k, "top_k")
    return max(
        top_k,
        min(top_k * _DIVERSITY_OVERSAMPLE_FACTOR, _DIVERSITY_OVERSAMPLE_CAP),
    )


def _normalized_candidate_text(text: str) -> str:
    """Normalize only line breaks and consecutive whitespace for exact equality."""

    return " ".join(text.split())


def _content_order_hash(text: str) -> str:
    """内容确定的破平哈希：对 chunk 正文规范化后取 sha256，独立于分配的 id。"""

    return hashlib.sha256(_normalized_candidate_text(text).encode("utf-8")).hexdigest()


def _candidate_order_key(candidate: VectorCandidate) -> tuple[object, ...]:
    """候选排序键——**内容确定**：破平不依赖分配的 document_id / chunk_id / rank。

    先按相关性（向量距离，越小越前），距离相同再按 **chunk 正文规范化哈希**、最后按
    字符位置确定次序。刻意**不含** document_id / chunk_id / rank——它们是与内容无关的
    分配值；以 id 破平会使同一正文在不同 document_id 下选出不同候选、结果随 id 漂移。
    距离本身与 Chroma 的 rank 单调一致，故非平局排序不变，
    仅平局项改为内容确定。
    """

    return (
        candidate.distance is None,
        math.inf if candidate.distance is None else candidate.distance,
        _content_order_hash(candidate.text),
        candidate.start_char,
        candidate.end_char,
    )


def _range_overlap_ratio(first: VectorCandidate, second: VectorCandidate) -> float:
    intersection = min(first.end_char, second.end_char) - max(
        first.start_char,
        second.start_char,
    )
    if intersection <= 0:
        return 0.0
    return intersection / min(
        first.end_char - first.start_char,
        second.end_char - second.start_char,
    )


def _select_diverse_candidates(
    candidates: Sequence[VectorCandidate],
    *,
    top_k: int,
) -> tuple[VectorCandidate, ...]:
    """Select a stable, bounded candidate set without changing raw scores.

    The input rank is the canonical raw Vector rank.  Selection first removes
    exact coordinate duplicates, same-document 80%-overlap duplicates, and
    same-document whitespace-normalized text duplicates.  It then takes the
    best non-repeated representative available from each document in first-rank
    document order before filling unused positions in raw-rank order.  If a
    document contains only cross-document text duplicates, its best candidate
    remains as provenance; later repeated content from an already represented
    document may be suppressed.
    """

    top_k = _positive_int(top_k, "top_k")
    if isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence of VectorCandidate objects")
    raw_candidates = tuple(candidates)
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, VectorCandidate):
            raise ValueError(
                f"candidates[{index}] must be VectorCandidate; got {candidate!r}"
            )
    ordered = tuple(sorted(raw_candidates, key=_candidate_order_key))

    deduplicated: list[VectorCandidate] = []
    seen_coordinates: set[tuple[str, int, int]] = set()
    normalized_by_document: dict[str, set[str]] = {}
    for candidate in ordered:
        coordinate = (
            candidate.document_id,
            candidate.start_char,
            candidate.end_char,
        )
        if coordinate in seen_coordinates:
            continue
        normalized_text = _normalized_candidate_text(candidate.text)
        document_texts = normalized_by_document.setdefault(
            candidate.document_id,
            set(),
        )
        if normalized_text in document_texts:
            continue
        if any(
            existing.document_id == candidate.document_id
            and _range_overlap_ratio(existing, candidate) >= _HIGH_OVERLAP_RATIO
            for existing in deduplicated
        ):
            continue
        seen_coordinates.add(coordinate)
        document_texts.add(normalized_text)
        deduplicated.append(candidate)

    selected: list[VectorCandidate] = []
    selected_keys: set[tuple[str, str, int, int, int]] = set()
    represented_documents: set[str] = set()
    selected_texts: set[str] = set()

    def append(candidate: VectorCandidate) -> None:
        selected.append(candidate)
        selected_keys.add(
            (
                candidate.document_id,
                candidate.chunk_id,
                candidate.start_char,
                candidate.end_char,
                candidate.rank,
            )
        )
        represented_documents.add(candidate.document_id)
        selected_texts.add(_normalized_candidate_text(candidate.text))

    candidates_by_document: dict[str, list[VectorCandidate]] = {}
    for candidate in deduplicated:
        candidates_by_document.setdefault(candidate.document_id, []).append(candidate)

    for document_candidates in candidates_by_document.values():
        representative = next(
            (
                candidate
                for candidate in document_candidates
                if _normalized_candidate_text(candidate.text) not in selected_texts
            ),
            document_candidates[0],
        )
        append(representative)
        if len(selected) >= top_k:
            return tuple(selected)

    for candidate in deduplicated:
        key = (
            candidate.document_id,
            candidate.chunk_id,
            candidate.start_char,
            candidate.end_char,
            candidate.rank,
        )
        if key in selected_keys:
            continue
        normalized_text = _normalized_candidate_text(candidate.text)
        if normalized_text in selected_texts:
            continue
        append(candidate)
        if len(selected) >= top_k:
            break
    return tuple(selected)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_documents(
    documents: Sequence[NormalizedDocument],
) -> tuple[NormalizedDocument, ...]:
    if isinstance(documents, (str, bytes)):
        raise ValueError("documents must be a sequence of NormalizedDocument objects")
    try:
        result = tuple(documents)
    except TypeError as exc:
        raise ValueError(
            "documents must be a sequence of NormalizedDocument objects"
        ) from exc
    for index, document in enumerate(result):
        if not isinstance(document, NormalizedDocument):
            raise ValueError(
                f"documents[{index}] must be NormalizedDocument; got {document!r}"
            )
    document_ids = [document.document_id for document in result]
    duplicates = sorted(
        {document_id for document_id in document_ids if document_ids.count(document_id) > 1}
    )
    if duplicates:
        raise ValueError(f"documents contain duplicate document_id values: {duplicates!r}")
    return tuple(sorted(result, key=lambda document: document.document_id))


def _identity_payload(
    ordered_documents: Sequence[NormalizedDocument],
    *,
    max_chars: int,
    embedding_backend: str,
    embedding_model: str,
    embedding_dimension: int | None,
    chunker_version: str,
    index_schema_version: str,
) -> dict[str, object]:
    _positive_int(max_chars, "max_chars")
    backend = _non_empty(embedding_backend, "embedding_backend")
    model = _non_empty(embedding_model, "embedding_model")
    if backend.strip().lower() == "unknown" or model.strip().lower() == "unknown":
        raise ValueError("embedding backend and model must be explicit, not 'unknown'")
    dimension = _embedding_dimension(embedding_dimension)
    _non_empty(chunker_version, "chunker_version")
    _non_empty(index_schema_version, "index_schema_version")
    return {
        "documents": [
            {
                "document_id": document.document_id,
                "normalized_content_hash": document.normalized_content_hash,
            }
            for document in ordered_documents
        ],
        "chunker_version": chunker_version,
        "max_chars": max_chars,
        "embedding_backend": backend,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "index_schema_version": index_schema_version,
    }


def build_index_fingerprint(
    documents: Sequence[NormalizedDocument],
    *,
    max_chars: int,
    embedding_backend: str,
    embedding_model: str,
    embedding_dimension: int | None = None,
    chunker_version: str = LOSSLESS_CHUNKER_VERSION,
    index_schema_version: str = INDEX_SCHEMA_VERSION,
) -> str:
    """返回与输入顺序无关、与语料和向量配置绑定的 SHA-256 identity。"""

    ordered = _ordered_documents(documents)
    payload = _identity_payload(
        ordered,
        max_chars=max_chars,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunker_version=chunker_version,
        index_schema_version=index_schema_version,
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VectorIndexManifest:
    """可序列化的 collection identity 与完整建库配置。"""

    fingerprint: str
    collection_name: str
    index_schema_version: str
    ordered_documents: tuple[tuple[str, str], ...]
    chunker_version: str
    max_chars: int
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int | None
    chunk_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fingerprint, str)
            or len(self.fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in self.fingerprint)
        ):
            raise ValueError(
                f"fingerprint must be a lowercase SHA-256 digest; got {self.fingerprint!r}"
            )
        _non_empty(self.collection_name, "collection_name")
        _non_empty(self.index_schema_version, "index_schema_version")
        _non_empty(self.chunker_version, "chunker_version")
        _positive_int(self.max_chars, "max_chars")
        _non_empty(self.embedding_backend, "embedding_backend")
        _non_empty(self.embedding_model, "embedding_model")
        _embedding_dimension(self.embedding_dimension)
        if type(self.chunk_count) is not int or self.chunk_count < 0:
            raise ValueError(
                f"chunk_count must be an integer >= 0; got {self.chunk_count!r}"
            )
        pairs = tuple(tuple(pair) for pair in self.ordered_documents)
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError(
                    f"ordered_documents[{index}] must contain document_id and hash"
                )
            _non_empty(pair[0], f"ordered_documents[{index}].document_id")
            if (
                len(pair[1]) != 64
                or any(ch not in "0123456789abcdef" for ch in pair[1])
            ):
                raise ValueError(
                    f"ordered_documents[{index}] has invalid normalized hash {pair[1]!r}"
                )
        if tuple(sorted(pairs)) != pairs:
            raise ValueError("ordered_documents must be sorted by document_id")
        if len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError("ordered_documents must not contain duplicate document IDs")
        object.__setattr__(self, "ordered_documents", pairs)

    def to_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "collection_name": self.collection_name,
            "index_schema_version": self.index_schema_version,
            "documents": [
                {
                    "document_id": document_id,
                    "normalized_content_hash": content_hash,
                }
                for document_id, content_hash in self.ordered_documents
            ],
            "chunker_version": self.chunker_version,
            "max_chars": self.max_chars,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "chunk_count": self.chunk_count,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VectorIndexManifest":
        if not isinstance(value, Mapping):
            raise ValueError(f"manifest must be a mapping; got {value!r}")
        expected = {
            "fingerprint",
            "collection_name",
            "index_schema_version",
            "documents",
            "chunker_version",
            "max_chars",
            "embedding_backend",
            "embedding_model",
            "embedding_dimension",
            "chunk_count",
        }
        if set(value) != expected:
            raise ValueError(
                f"manifest fields must be exactly {sorted(expected)!r}; got {sorted(value)!r}"
            )
        raw_documents = value["documents"]
        if isinstance(raw_documents, (str, bytes)):
            raise ValueError("manifest.documents must be a sequence")
        try:
            documents = tuple(raw_documents)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("manifest.documents must be a sequence") from exc
        pairs: list[tuple[str, str]] = []
        for index, document in enumerate(documents):
            if not isinstance(document, Mapping) or set(document) != {
                "document_id",
                "normalized_content_hash",
            }:
                raise ValueError(f"manifest.documents[{index}] has invalid fields")
            pairs.append(
                (
                    document["document_id"],  # type: ignore[arg-type]
                    document["normalized_content_hash"],  # type: ignore[arg-type]
                )
            )
        return cls(
            fingerprint=value["fingerprint"],  # type: ignore[arg-type]
            collection_name=value["collection_name"],  # type: ignore[arg-type]
            index_schema_version=value["index_schema_version"],  # type: ignore[arg-type]
            ordered_documents=tuple(pairs),
            chunker_version=value["chunker_version"],  # type: ignore[arg-type]
            max_chars=value["max_chars"],  # type: ignore[arg-type]
            embedding_backend=value["embedding_backend"],  # type: ignore[arg-type]
            embedding_model=value["embedding_model"],  # type: ignore[arg-type]
            embedding_dimension=value["embedding_dimension"],  # type: ignore[arg-type]
            chunk_count=value["chunk_count"],  # type: ignore[arg-type]
        )


def build_slot_query(slot: EvidenceSlot) -> str:
    """用 Slot 的正向定义确定性构造单条查询，不调用模型或使用 weak conditions。"""

    if not isinstance(slot, EvidenceSlot):
        raise ValueError(f"slot must be EvidenceSlot; got {slot!r}")
    name = _non_empty(slot.name, "EvidenceSlot.name").strip()
    description = _non_empty(slot.description, "EvidenceSlot.description").strip()
    if not slot.required_fields:
        raise ValueError(f"EvidenceSlot[{slot.slot_id}] must declare required_fields")
    fields = "；".join(field.replace("_", " ") for field in slot.required_fields)
    query = f"{name}。{description}。所需信息：{fields}"
    return _non_empty(query, f"query for slot {slot.slot_id!r}")


def _validate_vectors(
    vectors: object,
    expected_count: int,
) -> tuple[list[list[float]], int | None]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise RuntimeError("embedding backend must return a sequence of vectors")
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"embedding backend returned {len(vectors)} vectors for {expected_count} chunks"
        )
    if expected_count == 0:
        return [], None
    normalized: list[list[float]] = []
    dimension: int | None = None
    for index, vector in enumerate(vectors):
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            raise RuntimeError(f"embedding vector {index} must be a non-empty sequence")
        converted: list[float] = []
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, Real):
                raise RuntimeError(
                    f"embedding vector {index} contains non-numeric component {component!r}"
                )
            number = float(component)
            if not math.isfinite(number):
                raise RuntimeError(
                    f"embedding vector {index} contains non-finite component {component!r}"
                )
            converted.append(number)
        if dimension is None:
            dimension = len(converted)
        elif len(converted) != dimension:
            raise RuntimeError(
                f"embedding vector dimension mismatch: expected {dimension}, got {len(converted)}"
            )
        normalized.append(converted)
    return normalized, dimension


def _collection_names(client) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            names.add(item)
        else:
            name = getattr(item, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


class VectorDiscovery:
    """索引 canonical chunks，并按固定 Contract 的每个 Slot 发现语义候选。"""

    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        *,
        embedder: Embedder | None = None,
        embedding_backend: str | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise ValueError(f"settings must be Settings; got {settings!r}")
        self.settings = settings
        self._embedder = embedder
        if embedding_backend is None:
            if embedder is None:
                embedding_backend = settings.embedding_backend
            else:
                cls = type(embedder)
                embedding_backend = f"injected:{cls.__module__}.{cls.__qualname__}"
        self.embedding_backend = _non_empty(
            embedding_backend,
            "embedding_backend",
        )
        if self.embedding_backend.strip().lower() == "unknown":
            raise ValueError("embedding_backend must be explicit, not 'unknown'")
        self._manifest: VectorIndexManifest | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = embeddings.load_embedder(self.settings)
        model_id = getattr(self._embedder, "model_id", None)
        _non_empty(model_id, "embedder.model_id")
        if model_id.strip().lower() == "unknown":
            raise ValueError("embedder.model_id must be explicit, not 'unknown'")
        return self._embedder

    @property
    def manifest(self) -> VectorIndexManifest | None:
        return self._manifest

    def _manifest_path(self, fingerprint: str) -> Path:
        return (
            Path(self.settings.vector_db_path)
            / MANIFEST_DIRECTORY
            / f"{fingerprint}.json"
        )

    def _write_manifest(self, manifest: VectorIndexManifest) -> None:
        path = self._manifest_path(manifest.fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
        temporary.replace(path)

    def _validate_stored_manifest(self, manifest: VectorIndexManifest) -> None:
        path = self._manifest_path(manifest.fingerprint)
        if not path.is_file():
            raise RuntimeError(f"vector index manifest does not exist: {str(path)!r}")
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"vector index manifest is unreadable: {str(path)!r}") from exc
        if stored != manifest.to_dict():
            raise RuntimeError(
                f"vector index manifest does not match requested fingerprint {manifest.fingerprint}"
            )

    def index_documents(
        self,
        documents: Sequence[NormalizedDocument],
        *,
        max_chars: int,
    ) -> VectorIndexManifest:
        """重建并返回由完整 identity 隔离的 Chroma collection。"""

        max_chars = _positive_int(max_chars, "max_chars")
        ordered = _ordered_documents(documents)
        document_by_id = {document.document_id: document for document in ordered}
        chunks: list[LosslessChunk] = []
        for document in ordered:
            chunks.extend(chunk_document(document, max_chars=max_chars))
        # 插入 Chroma 的顺序影响 HNSW 近似图 → 影响查询近邻结果。按**内容确定**键排序
        # （chunk 正文规范化哈希 + 字符位置），使同一正文在不同 document_id 下建出同一图，
        # 结果不再随 id 漂移；fingerprint 仍按 document_id 排序，见 _ordered_documents。
        chunks.sort(key=lambda chunk: (_content_order_hash(chunk.text),
                                       chunk.start_char, chunk.end_char))
        for chunk in chunks:
            document = document_by_id[chunk.document_id]
            if chunk.text != document.slice(chunk.start_char, chunk.end_char):
                raise RuntimeError(
                    f"canonical chunk {chunk.chunk_id!r} does not equal its document slice"
                )
            if len(chunk.text) > max_chars:
                raise RuntimeError(
                    f"canonical chunk {chunk.chunk_id!r} exceeds max_chars={max_chars}"
                )

        embedder = self._get_embedder()
        if chunks:
            raw_vectors = embedder.encode([chunk.text for chunk in chunks])
        else:
            raw_vectors = []
        vectors, dimension = _validate_vectors(raw_vectors, len(chunks))
        fingerprint = build_index_fingerprint(
            ordered,
            max_chars=max_chars,
            embedding_backend=self.embedding_backend,
            embedding_model=embedder.model_id,
            embedding_dimension=dimension,
        )
        manifest = VectorIndexManifest(
            fingerprint=fingerprint,
            collection_name=f"{COLLECTION_PREFIX}{fingerprint}",
            index_schema_version=INDEX_SCHEMA_VERSION,
            ordered_documents=tuple(
                (document.document_id, document.normalized_content_hash)
                for document in ordered
            ),
            chunker_version=LOSSLESS_CHUNKER_VERSION,
            max_chars=max_chars,
            embedding_backend=self.embedding_backend,
            embedding_model=embedder.model_id,
            embedding_dimension=dimension,
            chunk_count=len(chunks),
        )

        import chromadb

        client = chromadb.PersistentClient(path=self.settings.vector_db_path)
        if manifest.collection_name in _collection_names(client):
            client.delete_collection(manifest.collection_name)
        collection_metadata: dict[str, object] = {
            "hnsw:space": "cosine",
            "embedding_model": manifest.embedding_model,
            "embedding_backend": manifest.embedding_backend,
            "index_fingerprint": manifest.fingerprint,
            "index_schema_version": manifest.index_schema_version,
            "chunker_version": manifest.chunker_version,
            "max_chars": manifest.max_chars,
            "manifest_sha256": manifest.manifest_hash(),
        }
        if dimension is not None:
            collection_metadata["embedding_dimension"] = dimension
        collection = client.create_collection(
            manifest.collection_name,
            metadata=collection_metadata,
        )
        if chunks:
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content_hash": chunk.content_hash,
                        "chunker_version": chunk.chunker_version,
                        "normalized_content_hash": document_by_id[
                            chunk.document_id
                        ].normalized_content_hash,
                    }
                    for chunk in chunks
                ],
            )
        if collection.count() != manifest.chunk_count:
            raise RuntimeError(
                f"vector index count mismatch after build: expected {manifest.chunk_count}, "
                f"got {collection.count()}"
            )
        self._write_manifest(manifest)
        self._manifest = manifest
        return manifest

    @staticmethod
    def _required_collection_metadata(
        manifest: VectorIndexManifest,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "embedding_model": manifest.embedding_model,
            "embedding_backend": manifest.embedding_backend,
            "index_fingerprint": manifest.fingerprint,
            "index_schema_version": manifest.index_schema_version,
            "chunker_version": manifest.chunker_version,
            "max_chars": manifest.max_chars,
            "manifest_sha256": manifest.manifest_hash(),
        }
        if manifest.embedding_dimension is not None:
            result["embedding_dimension"] = manifest.embedding_dimension
        return result

    @staticmethod
    def _result_group(result: Mapping[str, object], key: str) -> list[object]:
        value = result.get(key)
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
            raise RuntimeError(f"vector query result has invalid {key!r} group")
        return value[0]

    @staticmethod
    def _validate_candidate_record(
        raw_id: object,
        text: object,
        metadata: object,
        distance: object,
        manifest: VectorIndexManifest,
    ) -> tuple[str, str, dict[str, object], float | None]:
        if not isinstance(raw_id, str) or not raw_id:
            raise RuntimeError(f"vector query returned invalid chunk id {raw_id!r}")
        if not isinstance(text, str) or not text:
            raise RuntimeError(f"vector query returned empty text for chunk {raw_id!r}")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"vector query returned invalid metadata for chunk {raw_id!r}")
        record = dict(metadata)
        required = {
            "chunk_id",
            "document_id",
            "start_char",
            "end_char",
            "start_line",
            "end_line",
            "content_hash",
            "chunker_version",
            "normalized_content_hash",
        }
        missing = sorted(required - set(record))
        if missing:
            raise RuntimeError(
                f"vector candidate {raw_id!r} is missing metadata fields: {missing!r}"
            )
        if record["chunk_id"] != raw_id:
            raise RuntimeError(
                f"vector candidate ID mismatch: query id {raw_id!r}, "
                f"metadata id {record['chunk_id']!r}"
            )
        document_id = record["document_id"]
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError(f"vector candidate {raw_id!r} has invalid document_id")
        document_hashes = dict(manifest.ordered_documents)
        if document_id not in document_hashes:
            raise RuntimeError(
                f"vector candidate {raw_id!r} references unknown document {document_id!r}"
            )
        if record["normalized_content_hash"] != document_hashes[document_id]:
            raise RuntimeError(
                f"vector candidate {raw_id!r} has stale normalized_content_hash"
            )
        if record["chunker_version"] != manifest.chunker_version:
            raise RuntimeError(f"vector candidate {raw_id!r} has stale chunker_version")
        for key in ("start_char", "end_char", "start_line", "end_line"):
            if type(record[key]) is not int:
                raise RuntimeError(
                    f"vector candidate {raw_id!r} metadata {key!r} must be an integer"
                )
        if record["start_char"] < 0 or record["end_char"] <= record["start_char"]:
            raise RuntimeError(f"vector candidate {raw_id!r} has invalid character range")
        if record["end_char"] - record["start_char"] != len(text):
            raise RuntimeError(
                f"vector candidate {raw_id!r} text does not match its character range"
            )
        if record["start_line"] < 1 or record["end_line"] < record["start_line"]:
            raise RuntimeError(f"vector candidate {raw_id!r} has invalid line range")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record["content_hash"] != content_hash:
            raise RuntimeError(f"vector candidate {raw_id!r} text hash does not match metadata")
        if distance is None:
            parsed_distance = None
        else:
            if isinstance(distance, bool) or not isinstance(distance, Real):
                raise RuntimeError(
                    f"vector candidate {raw_id!r} has invalid distance {distance!r}"
                )
            parsed_distance = float(distance)
            if not math.isfinite(parsed_distance):
                raise RuntimeError(
                    f"vector candidate {raw_id!r} has non-finite distance {distance!r}"
                )
        return raw_id, text, record, parsed_distance

    def discover_for_slot(
        self,
        contract: EvidenceContract,
        slot_id: str,
        *,
        top_k: int,
        manifest: VectorIndexManifest | None = None,
        allowed_document_ids: Collection[str] | None = None,
    ) -> tuple[VectorCandidate, ...]:
        """查询一个 Slot，并按原始 rank 返回确定性多样候选。

        Chroma 的内部请求深度由 :func:`_raw_query_k` 有界放大；公开
        ``top_k`` 仍是最终 tuple 的严格上限。候选 ``rank`` 和 ``distance``
        均保留多样性选择前的规范化原始排序值。

        ``allowed_document_ids`` 是 L1 元数据过滤的落点；``None`` 表示不进行过滤。
        给定集合时，在**扩大召回之后、
        top_k 截断之前**丢弃不在集合内的记录——这样过滤只是收窄来源范围，不会白白
        浪费 top_k 名额。过滤条件由 L1 从 ``source_type`` 解析而来，**L2 不认识
        source_type 语义**，索引与指纹也不受影响（同一份索引可服务任意过滤组合）。
        """

        if not isinstance(contract, EvidenceContract):
            raise ValueError(f"contract must be EvidenceContract; got {contract!r}")
        top_k = _positive_int(top_k, "top_k")
        slot = contract.get_slot(slot_id)
        query = build_slot_query(slot)
        active_manifest = manifest if manifest is not None else self._manifest
        if not isinstance(active_manifest, VectorIndexManifest):
            raise RuntimeError("no VectorIndexManifest is active; index documents first")
        if active_manifest.embedding_backend != self.embedding_backend:
            raise RuntimeError(
                "embedding backend mismatch: index used "
                f"{active_manifest.embedding_backend!r}, query uses {self.embedding_backend!r}"
            )
        self._validate_stored_manifest(active_manifest)
        result = query_vector_collection(
            self.settings,
            self._get_embedder(),
            active_manifest.collection_name,
            query,
            k=_raw_query_k(top_k),
            require_collection=True,
            required_metadata=self._required_collection_metadata(active_manifest),
            expected_count=active_manifest.chunk_count,
        )
        if result is None:
            return ()
        if not isinstance(result, Mapping):
            raise RuntimeError(f"vector query result must be a mapping; got {result!r}")
        ids = self._result_group(result, "ids")
        documents = self._result_group(result, "documents")
        metadatas = self._result_group(result, "metadatas")
        distances = self._result_group(result, "distances")
        lengths = {len(ids), len(documents), len(metadatas), len(distances)}
        if len(lengths) != 1:
            raise RuntimeError(
                "vector query result arrays have inconsistent lengths: "
                f"ids={len(ids)}, documents={len(documents)}, "
                f"metadatas={len(metadatas)}, distances={len(distances)}"
            )
        records = [
            self._validate_candidate_record(raw_id, text, metadata, distance, active_manifest)
            for raw_id, text, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ]
        if allowed_document_ids is not None:
            allowed = set(allowed_document_ids)
            records = [record for record in records if record[2]["document_id"] in allowed]
        # rank 的赋值序也必须**内容确定**：距离相同的记录改按 chunk 正文规范化哈希与
        # 字符位置破平，不再用 document_id / chunk_id(raw_id)。原键以 id 破平，会把
        # 分配的 id 顺序带入 rank；虽然下游选择在 _candidate_order_key（内容确定序）上
        # 进行、rank 不入最终结果，但为彻底消除"id 参与任何排序决策"的尾巴，一并改掉。
        records.sort(
            key=lambda record: (
                record[3] is None,
                math.inf if record[3] is None else record[3],
                _content_order_hash(record[1]),
                record[2]["start_char"],
                record[2]["end_char"],
            )
        )
        raw_candidates = tuple(
            VectorCandidate(
                slot_id=slot.slot_id,
                query=query,
                document_id=record[2]["document_id"],  # type: ignore[arg-type]
                chunk_id=record[0],
                text=record[1],
                start_char=record[2]["start_char"],  # type: ignore[arg-type]
                end_char=record[2]["end_char"],  # type: ignore[arg-type]
                start_line=record[2]["start_line"],  # type: ignore[arg-type]
                end_line=record[2]["end_line"],  # type: ignore[arg-type]
                rank=rank,
                distance=record[3],
                discovery_source="vector",
                metadata=record[2],
            )
            for rank, record in enumerate(records, start=1)
        )
        return _select_diverse_candidates(raw_candidates, top_k=top_k)

    def discover_contract(
        self,
        contract: EvidenceContract,
        *,
        top_k_per_slot: int,
        manifest: VectorIndexManifest | None = None,
        allowed_document_ids: Collection[str] | None = None,
    ) -> Mapping[str, tuple[VectorCandidate, ...]]:
        """按 Contract 声明顺序执行所有 Slot；空结果也保留 key。

        ``allowed_document_ids`` 原样透传给 :meth:`discover_for_slot`（默认 ``None`` = 不过滤）。
        """

        if not isinstance(contract, EvidenceContract):
            raise ValueError(f"contract must be EvidenceContract; got {contract!r}")
        _positive_int(top_k_per_slot, "top_k_per_slot")
        results: dict[str, tuple[VectorCandidate, ...]] = {}
        for slot in contract.slots:
            results[slot.slot_id] = self.discover_for_slot(
                contract,
                slot.slot_id,
                top_k=top_k_per_slot,
                manifest=manifest,
                allowed_document_ids=allowed_document_ids,
            )
        return MappingProxyType(results)
