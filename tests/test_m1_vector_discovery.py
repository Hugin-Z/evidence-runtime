"""M1 Step 3 离线测试：canonical chunk → 指纹索引 → per-slot candidates。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evidence.core.config import Settings
from evidence import (
    EvidenceSpan,
    EvidenceSlot,
    NormalizedDocument,
    VectorCandidate,
    VectorDiscovery,
    VectorIndexManifest,
    build_index_fingerprint,
    build_slot_query,
    solution_material_completeness_v1,
)
from evidence.documents import LOSSLESS_CHUNKER_VERSION
from evidence.vector_discovery import (
    INDEX_SCHEMA_VERSION,
    MANIFEST_DIRECTORY,
    _DIVERSITY_OVERSAMPLE_CAP,
    _HIGH_OVERLAP_RATIO,
    _content_order_hash,
    _raw_query_k,
    _select_diverse_candidates,
)


class FakeSemanticEmbedder:
    """TEST-ONLY 确定性 fake；不访问网络，也不代表真实语义质量。"""

    model_id = "fake-m1-semantic-v1"
    _groups = (
        ("背景", "依据", "政策", "background", "basis"),
        ("现状", "需求", "问题", "current", "need"),
        ("范围", "对象", "scope", "subject"),
        ("目标", "objective", "target"),
        ("能力", "建设内容", "function", "capability"),
        ("实施", "条件", "边界", "implementation", "boundary"),
        ("案例", "项目", "客户", "case", "customer"),
        ("成效", "指标", "outcome", "metric"),
    )

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vector = [
                float(sum(lowered.count(term) for term in group))
                for group in self._groups
            ]
            vector.append(1.0)
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors


class RaisingEmbedder:
    model_id = FakeSemanticEmbedder.model_id

    def encode(self, texts):
        raise RuntimeError("fake embedding backend failed")


def _document(
    root: Path,
    name: str,
    text: str,
    *,
    document_id: str,
    source_path: str | None = None,
) -> NormalizedDocument:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return NormalizedDocument.from_path(
        path,
        document_id=document_id,
        source_path=source_path or f"materials/{name}",
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(vector_db_path=str(tmp_path / "chroma"))


def _indexed(tmp_path: Path, *, max_chars: int = 24):
    documents = (
        _document(
            tmp_path / "source",
            "basis.md",
            "政策依据明确本项目建设背景。\n现状问题需要解决。",
            document_id="doc-basis",
        ),
        _document(
            tmp_path / "source",
            "delivery.md",
            "实施条件包括系统对接。\n项目案例已完成验收。\n建设成效有指标。",
            document_id="doc-delivery",
        ),
    )
    discovery = VectorDiscovery(
        _settings(tmp_path),
        embedder=FakeSemanticEmbedder(),
        embedding_backend="fake",
    )
    manifest = discovery.index_documents(documents, max_chars=max_chars)
    return discovery, manifest, documents


def _candidate(
    *,
    document_id: str,
    rank: int,
    start_char: int,
    text: str,
    chunk_id: str | None = None,
    distance: float | None = None,
) -> VectorCandidate:
    return VectorCandidate(
        slot_id="test-slot",
        query="test query",
        document_id=document_id,
        chunk_id=chunk_id or f"{document_id}-{rank}-{start_char}",
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        start_line=1,
        end_line=1,
        rank=rank,
        distance=float(rank) / 100 if distance is None else distance,
        discovery_source="vector",
        metadata={"raw": rank},
    )


@pytest.mark.parametrize("slot", solution_material_completeness_v1.slots)
def test_all_fixed_slots_build_non_empty_queries(slot):
    assert build_slot_query(slot).strip()


def test_slot_query_is_stable_and_contract_order_is_stable():
    first = tuple(build_slot_query(slot) for slot in solution_material_completeness_v1.slots)
    second = tuple(build_slot_query(slot) for slot in solution_material_completeness_v1.slots)
    assert first == second
    assert solution_material_completeness_v1.slot_ids == tuple(
        slot.slot_id for slot in solution_material_completeness_v1.slots
    )


def test_slot_query_expresses_required_fields_without_weak_conditions():
    for slot in solution_material_completeness_v1.slots:
        query = build_slot_query(slot)
        for field in slot.required_fields:
            assert field.replace("_", " ") in query
        assert all(condition not in query for condition in slot.weak_conditions)


def test_slot_query_rejects_non_slot_and_forged_empty_slot():
    with pytest.raises(ValueError, match="EvidenceSlot"):
        build_slot_query(object())  # type: ignore[arg-type]
    empty = object.__new__(EvidenceSlot)
    object.__setattr__(empty, "slot_id", "forged")
    object.__setattr__(empty, "name", "")
    with pytest.raises(ValueError, match="name"):
        build_slot_query(empty)


def _fingerprint(document, **overrides):
    values = {
        "max_chars": 40,
        "embedding_backend": "fake",
        "embedding_model": "fake-m1-semantic-v1",
        "embedding_dimension": 9,
        "chunker_version": LOSSLESS_CHUNKER_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
    }
    values.update(overrides)
    return build_index_fingerprint([document], **values)


def test_fingerprint_is_stable_for_same_document_and_config(tmp_path):
    document = _document(tmp_path, "same.md", "同一正文", document_id="doc-same")
    assert _fingerprint(document) == _fingerprint(document)


def test_fingerprint_changes_when_text_changes(tmp_path):
    first = _document(tmp_path / "one", "x.md", "正文一", document_id="doc-fixed")
    second = _document(tmp_path / "two", "x.md", "正文二", document_id="doc-fixed")
    assert _fingerprint(first) != _fingerprint(second)


@pytest.mark.parametrize(
    "override",
    (
        {"max_chars": 41},
        {"chunker_version": "m1-lossless-v2"},
        {"embedding_model": "different-model"},
        {"embedding_backend": "different-backend"},
        {"embedding_dimension": 10},
    ),
)
def test_fingerprint_changes_for_any_vector_identity_setting(tmp_path, override):
    document = _document(tmp_path, "config.md", "配置正文", document_id="doc-config")
    assert _fingerprint(document) != _fingerprint(document, **override)


def test_fingerprint_sorts_documents_by_id(tmp_path):
    first = _document(tmp_path, "a.md", "甲", document_id="doc-a")
    second = _document(tmp_path, "b.md", "乙", document_id="doc-b")
    values = {
        "max_chars": 10,
        "embedding_backend": "fake",
        "embedding_model": "model",
        "embedding_dimension": 2,
    }
    assert build_index_fingerprint([first, second], **values) == build_index_fingerprint(
        [second, first], **values
    )


def test_fingerprint_ignores_absolute_machine_path(tmp_path):
    first = _document(
        tmp_path / "machine-a",
        "same.md",
        "相同内容",
        document_id="doc-portable",
        source_path="machine-a/same.md",
    )
    second = _document(
        tmp_path / "machine-b",
        "same.md",
        "相同内容",
        document_id="doc-portable",
        source_path="machine-b/same.md",
    )
    assert _fingerprint(first) == _fingerprint(second)


def test_fingerprint_rejects_duplicate_document_ids_and_unknown_model(tmp_path):
    first = _document(tmp_path, "a.md", "甲", document_id="duplicate")
    second = _document(tmp_path, "b.md", "乙", document_id="duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        build_index_fingerprint(
            [first, second],
            max_chars=10,
            embedding_backend="fake",
            embedding_model="model",
        )
    with pytest.raises(ValueError, match="unknown"):
        _fingerprint(first, embedding_model="unknown")


def test_index_uses_complete_lossless_chunk_metadata_and_writes_manifest(tmp_path):
    discovery, manifest, documents = _indexed(tmp_path, max_chars=13)
    import chromadb

    client = chromadb.PersistentClient(path=discovery.settings.vector_db_path)
    collection = client.get_collection(manifest.collection_name)
    stored = collection.get(include=["documents", "metadatas"])
    by_id = {document.document_id: document for document in documents}
    assert collection.count() == manifest.chunk_count > 2
    assert manifest.collection_name == f"m1-vector-{manifest.fingerprint}"
    assert manifest.embedding_dimension == 9
    for raw_id, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    ):
        assert metadata["chunk_id"] == raw_id
        assert {
            "chunk_id",
            "document_id",
            "start_char",
            "end_char",
            "start_line",
            "end_line",
            "content_hash",
            "chunker_version",
            "normalized_content_hash",
        } <= set(metadata)
        document = by_id[metadata["document_id"]]
        assert text == document.slice(metadata["start_char"], metadata["end_char"])
        assert metadata["normalized_content_hash"] == document.normalized_content_hash
        assert metadata["chunker_version"] == LOSSLESS_CHUNKER_VERSION
    manifest_path = (
        Path(discovery.settings.vector_db_path)
        / MANIFEST_DIRECTORY
        / f"{manifest.fingerprint}.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest.to_dict()
    assert VectorIndexManifest.from_dict(manifest.to_dict()) == manifest
    assert collection.metadata["manifest_sha256"] == manifest.manifest_hash()


def test_index_preserves_chinese_and_cross_line_chunk_text(tmp_path):
    text = "中文第一行\n第二行内容\n末行"
    document = _document(tmp_path, "zh.md", text, document_id="doc-zh")
    discovery = VectorDiscovery(
        _settings(tmp_path), embedder=FakeSemanticEmbedder(), embedding_backend="fake"
    )
    manifest = discovery.index_documents([document], max_chars=8)
    import chromadb

    collection = chromadb.PersistentClient(
        path=discovery.settings.vector_db_path
    ).get_collection(manifest.collection_name)
    stored = collection.get(include=["documents", "metadatas"])
    reconstructed = "".join(
        text
        for _, text in sorted(
            zip(stored["metadatas"], stored["documents"]),
            key=lambda item: item[0]["start_char"],
        )
    )
    assert reconstructed == document.text


def test_single_slot_discovery_returns_candidates_not_evidence(tmp_path):
    discovery, _, documents = _indexed(tmp_path)
    slot_id = "background_and_basis"
    candidates = discovery.discover_for_slot(
        solution_material_completeness_v1,
        slot_id,
        top_k=3,
    )
    assert candidates
    assert all(isinstance(candidate, VectorCandidate) for candidate in candidates)
    assert all(not isinstance(candidate, EvidenceSpan) for candidate in candidates)
    assert all(candidate.slot_id == slot_id for candidate in candidates)
    assert all(candidate.query == build_slot_query(
        solution_material_completeness_v1.get_slot(slot_id)
    ) for candidate in candidates)
    assert all(candidate.discovery_source == "vector" for candidate in candidates)
    assert all(isinstance(candidate.distance, float) for candidate in candidates)
    assert {candidate.document_id for candidate in candidates} <= {
        document.document_id for document in documents
    }


def test_discovery_top_k_dedup_and_rank_start_at_one(tmp_path):
    discovery, _, _ = _indexed(tmp_path, max_chars=10)
    candidates = discovery.discover_for_slot(
        solution_material_completeness_v1,
        "implementation_conditions_and_boundaries",
        top_k=2,
    )
    assert len(candidates) == 2
    assert candidates[0].rank == 1
    assert all(candidate.rank >= 1 for candidate in candidates)
    assert len({candidate.chunk_id for candidate in candidates}) == 2
    assert len({candidate.rank for candidate in candidates}) == 2


def test_duplicate_results_are_deduplicated_with_stable_tie_order(tmp_path, monkeypatch):
    discovery, manifest, _ = _indexed(tmp_path, max_chars=10)
    import chromadb

    collection = chromadb.PersistentClient(
        path=discovery.settings.vector_db_path
    ).get_collection(manifest.collection_name)
    stored = collection.get(include=["documents", "metadatas"])
    pairs = sorted(
        zip(stored["ids"], stored["documents"], stored["metadatas"]),
        key=lambda item: item[0],
    )[:2]
    first, second = pairs

    def duplicate_query(*args, **kwargs):
        return {
            "ids": [[first[0], first[0], second[0]]],
            "documents": [[first[1], first[1], second[1]]],
            "metadatas": [[first[2], first[2], second[2]]],
            "distances": [[0.4, 0.1, 0.1]],
        }

    monkeypatch.setattr(
        "evidence.vector_discovery.query_vector_collection",
        duplicate_query,
    )
    candidates = discovery.discover_for_slot(
        solution_material_completeness_v1,
        "background_and_basis",
        top_k=3,
    )
    # 平局（同距离）破平改为**内容确定**：按 chunk 正文规范化哈希、再按字符位置，
    # 不再按 document_id / chunk_id。同一正文无论分配何种 id 都得到同一次序。
    expected = sorted(
        (first, second),
        key=lambda item: (
            _content_order_hash(item[1]),
            item[2]["start_char"],
            item[2]["end_char"],
        ),
    )
    assert [candidate.chunk_id for candidate in candidates] == [
        item[0] for item in expected
    ]
    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert [candidate.distance for candidate in candidates] == [0.1, 0.1]


def test_contract_discovery_executes_all_slots_in_declared_order(tmp_path):
    discovery, _, _ = _indexed(tmp_path)
    results = discovery.discover_contract(
        solution_material_completeness_v1,
        top_k_per_slot=1,
    )
    assert tuple(results) == solution_material_completeness_v1.slot_ids
    assert all(len(results[slot_id]) == 1 for slot_id in results)


def test_one_empty_slot_does_not_skip_later_contract_slots(tmp_path, monkeypatch):
    discovery, manifest, _ = _indexed(tmp_path)
    import chromadb

    collection = chromadb.PersistentClient(
        path=discovery.settings.vector_db_path
    ).get_collection(manifest.collection_name)
    stored = collection.get(include=["documents", "metadatas"])
    raw = {
        "ids": [[stored["ids"][0]]],
        "documents": [[stored["documents"][0]]],
        "metadatas": [[stored["metadatas"][0]]],
        "distances": [[0.2]],
    }
    calls = []

    def first_empty(*args, **kwargs):
        calls.append(args[3])
        return None if len(calls) == 1 else raw

    monkeypatch.setattr(
        "evidence.vector_discovery.query_vector_collection",
        first_empty,
    )
    results = discovery.discover_contract(
        solution_material_completeness_v1,
        top_k_per_slot=1,
    )
    assert len(calls) == len(solution_material_completeness_v1.slots)
    assert results[solution_material_completeness_v1.slot_ids[0]] == ()
    assert all(results[slot_id] for slot_id in solution_material_completeness_v1.slot_ids[1:])


def test_empty_index_returns_empty_tuple_and_keeps_all_contract_keys(tmp_path):
    empty = _document(tmp_path, "empty.md", "", document_id="doc-empty")
    discovery = VectorDiscovery(
        _settings(tmp_path), embedder=FakeSemanticEmbedder(), embedding_backend="fake"
    )
    manifest = discovery.index_documents([empty], max_chars=20)
    assert manifest.chunk_count == 0
    assert discovery.discover_for_slot(
        solution_material_completeness_v1,
        "background_and_basis",
        top_k=2,
    ) == ()
    results = discovery.discover_contract(
        solution_material_completeness_v1,
        top_k_per_slot=2,
    )
    assert tuple(results) == solution_material_completeness_v1.slot_ids
    assert all(value == () for value in results.values())


def test_embedding_backend_error_is_not_converted_to_empty_result(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path)
    failing = VectorDiscovery(
        discovery.settings,
        embedder=RaisingEmbedder(),
        embedding_backend=manifest.embedding_backend,
    )
    with pytest.raises(RuntimeError, match="fake embedding backend failed"):
        failing.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
            manifest=manifest,
        )


def test_embedding_backend_identity_mismatch_fails_before_query(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path)
    mismatched = VectorDiscovery(
        discovery.settings,
        embedder=FakeSemanticEmbedder(),
        embedding_backend="different-backend",
    )
    with pytest.raises(RuntimeError, match="embedding backend mismatch"):
        mismatched.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
            manifest=manifest,
        )


def test_missing_collection_fails_clearly(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path)
    import chromadb

    chromadb.PersistentClient(path=discovery.settings.vector_db_path).delete_collection(
        manifest.collection_name
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
        )


def test_incomplete_candidate_metadata_fails_clearly(tmp_path):
    document = _document(tmp_path, "one.md", "政策背景", document_id="doc-one")
    discovery = VectorDiscovery(
        _settings(tmp_path), embedder=FakeSemanticEmbedder(), embedding_backend="fake"
    )
    manifest = discovery.index_documents([document], max_chars=50)
    import chromadb

    client = chromadb.PersistentClient(path=discovery.settings.vector_db_path)
    collection = client.get_collection(manifest.collection_name)
    raw_id = collection.get(include=[])["ids"][0]
    collection_metadata = dict(collection.metadata or {})
    client.delete_collection(manifest.collection_name)
    collection = client.create_collection(
        manifest.collection_name,
        metadata=collection_metadata,
    )
    collection.add(
        ids=[raw_id],
        embeddings=FakeSemanticEmbedder().encode([document.text]),
        documents=[document.text],
        metadatas=[{"document_id": "doc-one"}],
    )
    with pytest.raises(RuntimeError, match="missing metadata fields"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
        )


def test_incomplete_collection_metadata_fails_clearly(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path)
    import chromadb

    client = chromadb.PersistentClient(path=discovery.settings.vector_db_path)
    client.delete_collection(manifest.collection_name)
    client.create_collection(
        manifest.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": manifest.embedding_model},
    )
    with pytest.raises(RuntimeError, match="missing required metadata"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
        )


def test_partial_or_stale_collection_count_fails_clearly(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path, max_chars=10)
    import chromadb

    collection = chromadb.PersistentClient(
        path=discovery.settings.vector_db_path
    ).get_collection(manifest.collection_name)
    collection.delete(ids=[collection.get(include=[])["ids"][0]])
    with pytest.raises(RuntimeError, match="count mismatch"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
        )


def test_corrupt_manifest_fails_before_query(tmp_path):
    discovery, manifest, _ = _indexed(tmp_path)
    path = (
        Path(discovery.settings.vector_db_path)
        / MANIFEST_DIRECTORY
        / f"{manifest.fingerprint}.json"
    )
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=1,
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_discovery_rejects_invalid_top_k(tmp_path, top_k):
    discovery, _, _ = _indexed(tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "background_and_basis",
            top_k=top_k,
        )


def test_unknown_slot_fails_clearly(tmp_path):
    discovery, _, _ = _indexed(tmp_path)
    with pytest.raises(KeyError, match="unknown slot_id"):
        discovery.discover_for_slot(
            solution_material_completeness_v1,
            "not-a-slot",
            top_k=1,
        )


def test_diversity_keeps_all_legal_candidates_when_fewer_than_top_k():
    candidates = (
        _candidate(document_id="doc-a", rank=1, start_char=0, text="alpha"),
        _candidate(document_id="doc-a", rank=2, start_char=5, text="bravo"),
    )
    assert _select_diverse_candidates(candidates, top_k=5) == candidates


def test_diversity_single_document_keeps_multiple_non_duplicate_candidates():
    candidates = tuple(
        _candidate(
            document_id="doc-a",
            rank=rank,
            start_char=(rank - 1) * 5,
            text=f"item{rank}",
        )
        for rank in range(1, 5)
    )
    selected = _select_diverse_candidates(candidates, top_k=3)
    assert [item.rank for item in selected] == [1, 2, 3]


def test_diversity_removes_duplicate_chunk_identity_and_duplicate_offset():
    best = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="abcdefgh",
        chunk_id="same-chunk",
    )
    repeated_chunk = _candidate(
        document_id="doc-a",
        rank=3,
        start_char=0,
        text="abcdefgh",
        chunk_id="same-chunk",
    )
    repeated_offset = _candidate(
        document_id="doc-a",
        rank=2,
        start_char=0,
        text="abcdefgh",
        chunk_id="different-id",
    )
    assert _select_diverse_candidates(
        (repeated_chunk, repeated_offset, best),
        top_k=3,
    ) == (best,)


def test_diversity_suppresses_same_document_high_overlap_at_fixed_threshold():
    best = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="abcdefghij",
    )
    overlapping = _candidate(
        document_id="doc-a",
        rank=2,
        start_char=2,
        text="cdefghijkl",
    )
    assert _HIGH_OVERLAP_RATIO == 0.8
    assert _select_diverse_candidates((best, overlapping), top_k=2) == (best,)


def test_diversity_keeps_adjacent_non_overlapping_ranges():
    first = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="abcdefghij",
    )
    adjacent = _candidate(
        document_id="doc-a",
        rank=2,
        start_char=10,
        text="klmnopqrst",
    )
    assert _select_diverse_candidates((first, adjacent), top_k=2) == (
        first,
        adjacent,
    )


def test_diversity_whitespace_text_dedup_is_stable_and_preserves_document_provenance():
    first = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="alpha  beta",
    )
    same_document_copy = _candidate(
        document_id="doc-a",
        rank=2,
        start_char=20,
        text="alpha\nbeta",
    )
    other_document_copy = _candidate(
        document_id="doc-b",
        rank=3,
        start_char=0,
        text="alpha beta",
    )
    selected = _select_diverse_candidates(
        (same_document_copy, other_document_copy, first),
        top_k=3,
    )
    assert selected == (first, other_document_copy)
    assert {item.document_id for item in selected} == {"doc-a", "doc-b"}


def test_diversity_never_exceeds_final_top_k():
    candidates = tuple(
        _candidate(
            document_id=f"doc-{rank}",
            rank=rank,
            start_char=0,
            text=f"candidate-{rank}",
        )
        for rank in range(1, 10)
    )
    assert len(_select_diverse_candidates(candidates, top_k=4)) == 4


def test_diversity_selects_best_candidate_per_document_before_rank_fill():
    candidates = (
        _candidate(document_id="doc-a", rank=1, start_char=0, text="a-one"),
        _candidate(document_id="doc-a", rank=2, start_char=5, text="a-two"),
        _candidate(document_id="doc-a", rank=3, start_char=10, text="a-three"),
        _candidate(document_id="doc-b", rank=4, start_char=0, text="b-one"),
        _candidate(document_id="doc-b", rank=5, start_char=5, text="b-two"),
        _candidate(document_id="doc-c", rank=6, start_char=0, text="c-one"),
    )
    selected = _select_diverse_candidates(candidates, top_k=4)
    assert [item.rank for item in selected] == [1, 4, 6, 2]


def test_diversity_top_k_below_document_count_uses_highest_ranked_documents():
    candidates = tuple(
        _candidate(
            document_id=f"doc-{letter}",
            rank=rank,
            start_char=0,
            text=f"text-{letter}",
        )
        for rank, letter in enumerate("abcd", start=1)
    )
    selected = _select_diverse_candidates(candidates, top_k=2)
    assert [item.document_id for item in selected] == ["doc-a", "doc-b"]


def test_diversity_prevents_one_document_from_consuming_every_position():
    candidates = tuple(
        _candidate(
            document_id="doc-a",
            rank=rank,
            start_char=rank * 10,
            text=f"a-{rank:02d}",
        )
        for rank in range(1, 6)
    ) + (
        _candidate(document_id="doc-b", rank=6, start_char=0, text="b-unique"),
    )
    selected = _select_diverse_candidates(candidates, top_k=3)
    assert [item.rank for item in selected] == [1, 6, 2]


def test_diversity_does_not_force_low_quality_duplicate_from_represented_document():
    first = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="same content",
    )
    low_quality_duplicate = _candidate(
        document_id="doc-a",
        rank=2,
        start_char=20,
        text="same\ncontent",
    )
    complementary = _candidate(
        document_id="doc-b",
        rank=3,
        start_char=0,
        text="complementary",
    )
    assert _select_diverse_candidates(
        (first, low_quality_duplicate, complementary),
        top_k=2,
    ) == (first, complementary)


def test_diversity_uses_unique_content_instead_of_other_document_duplicate():
    first = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="shared content",
    )
    other_document_duplicate = _candidate(
        document_id="doc-b",
        rank=2,
        start_char=0,
        text="shared\ncontent",
    )
    other_document_unique = _candidate(
        document_id="doc-b",
        rank=3,
        start_char=30,
        text="complementary detail",
    )
    selected = _select_diverse_candidates(
        (first, other_document_duplicate, other_document_unique),
        top_k=2,
    )
    assert selected == (first, other_document_unique)
    assert {item.document_id for item in selected} == {"doc-a", "doc-b"}


def test_diversity_stable_ties_are_content_deterministic_and_diversity_first():
    # 平局（同距离）时，次序由**内容确定**键（正文规范化哈希→字符位置）决定，不再按
    # document_id / chunk_id；且每文档先出一个代表（diversity）。故对输入顺序不敏感，
    # 也对分配的 id 不敏感。此处不硬编码 sha256 派生的具体排列（那是在测 sha256，
    # 不是测本模块逻辑），而断言真正的不变量：确定性、完整性、diversity 优先。
    candidates = (
        _candidate(
            document_id="doc-b",
            rank=1,
            start_char=0,
            text="bravo",
            distance=0.1,
        ),
        _candidate(
            document_id="doc-a",
            rank=1,
            start_char=5,
            text="later",
            distance=0.1,
        ),
        _candidate(
            document_id="doc-a",
            rank=1,
            start_char=0,
            text="alpha",
            distance=0.1,
        ),
    )
    first = _select_diverse_candidates(candidates, top_k=3)
    second = _select_diverse_candidates(tuple(reversed(candidates)), top_k=3)
    # 输入洗牌后逐字段一致 → 内容确定、与输入顺序/分配 id 无关。
    assert first == second
    # 完整性：三条平局候选全部保留，无丢失。
    assert {(item.document_id, item.start_char) for item in first} == {
        ("doc-a", 0),
        ("doc-a", 5),
        ("doc-b", 0),
    }
    # diversity：前两位来自不同文档（每文档先出一个代表）。
    assert first[0].document_id != first[1].document_id
    # 次序完全由内容确定键复现（证明破平不掺入 document_id / chunk_id / rank）。
    expected_order = sorted(
        candidates,
        key=lambda c: (_content_order_hash(c.text), c.start_char, c.end_char),
    )
    representative_first = sorted(
        {c.document_id for c in candidates},
        key=lambda doc: next(
            i for i, c in enumerate(expected_order) if c.document_id == doc
        ),
    )
    assert [item.document_id for item in first[: len(representative_first)]] == (
        representative_first
    )


def test_diversity_preserves_raw_rank_distance_and_does_not_mutate_input():
    best = _candidate(
        document_id="doc-a",
        rank=1,
        start_char=0,
        text="alpha",
        distance=0.01,
    )
    complementary = _candidate(
        document_id="doc-b",
        rank=7,
        start_char=0,
        text="bravo",
        distance=0.42,
    )
    raw = [complementary, best]
    before = list(raw)
    selected = _select_diverse_candidates(raw, top_k=2)
    assert raw == before
    assert selected == (best, complementary)
    assert selected[1] is complementary
    assert (selected[1].rank, selected[1].distance) == (7, 0.42)


def test_oversampling_rule_is_bounded_and_never_below_public_top_k():
    assert _raw_query_k(1) == 4
    assert _raw_query_k(8) == _DIVERSITY_OVERSAMPLE_CAP == 32
    assert _raw_query_k(20) == 32
    assert _raw_query_k(40) == 40


def test_discovery_queries_oversampled_depth_but_returns_public_top_k(
    tmp_path,
    monkeypatch,
):
    discovery, manifest, _ = _indexed(tmp_path, max_chars=10)
    import chromadb

    collection = chromadb.PersistentClient(
        path=discovery.settings.vector_db_path
    ).get_collection(manifest.collection_name)
    stored = collection.get(include=["documents", "metadatas"])
    indexes = list(range(min(4, len(stored["ids"]))))
    captured = []

    def raw_query(*args, **kwargs):
        captured.append(kwargs["k"])
        return {
            "ids": [[stored["ids"][index] for index in indexes]],
            "documents": [[stored["documents"][index] for index in indexes]],
            "metadatas": [[stored["metadatas"][index] for index in indexes]],
            "distances": [[0.1 + index / 100 for index in indexes]],
        }

    monkeypatch.setattr(
        "evidence.vector_discovery.query_vector_collection",
        raw_query,
    )
    candidates = discovery.discover_for_slot(
        solution_material_completeness_v1,
        "background_and_basis",
        top_k=1,
    )
    assert captured == [4]
    assert len(candidates) == 1


def test_discovery_empty_oversampled_query_keeps_empty_behavior(tmp_path, monkeypatch):
    discovery, _, _ = _indexed(tmp_path)
    captured = []

    def empty_query(*args, **kwargs):
        captured.append(kwargs["k"])
        return None

    monkeypatch.setattr(
        "evidence.vector_discovery.query_vector_collection",
        empty_query,
    )
    assert discovery.discover_for_slot(
        solution_material_completeness_v1,
        "background_and_basis",
        top_k=2,
    ) == ()
    assert captured == [8]


# ── P-A：L1 元数据过滤在 L2 的落点（默认关）──────────────────────────────
def test_allowed_document_ids_defaults_to_no_filter(tmp_path):
    """默认 ``None`` = 不过滤：与显式不传参逐字一致（P-A 前后行为不变）。"""

    discovery, _, _ = _indexed(tmp_path)
    baseline = discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=4,
    )
    explicit_none = discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=4,
        allowed_document_ids=None,
    )
    assert explicit_none == baseline
    assert {c.document_id for c in baseline} == {"doc-basis", "doc-delivery"}


def test_allowed_document_ids_narrows_candidates_to_that_document(tmp_path):
    discovery, _, _ = _indexed(tmp_path)
    candidates = discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=4,
        allowed_document_ids={"doc-basis"},
    )
    assert candidates
    assert {c.document_id for c in candidates} == {"doc-basis"}


def test_allowed_document_ids_filters_before_top_k_truncation(tmp_path):
    """过滤发生在扩召回之后、top_k 截断之前——收窄来源而不浪费 top_k 名额。"""

    discovery, _, _ = _indexed(tmp_path, max_chars=10)
    unfiltered = discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=2,
    )
    filtered = discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=2,
        allowed_document_ids={"doc-delivery"},
    )
    assert {c.document_id for c in filtered} == {"doc-delivery"}
    # 不过滤时 top-2 里只有 1 条来自 doc-delivery；过滤后腾出的名额被同文档下一条补上，
    # 得到 2 条。若过滤发生在截断之后，这里只会剩 1 条。
    delivery_in_unfiltered = [c for c in unfiltered if c.document_id == "doc-delivery"]
    assert len(delivery_in_unfiltered) == 1
    assert len(filtered) == 2


def test_allowed_document_ids_empty_set_yields_no_candidates(tmp_path):
    discovery, _, _ = _indexed(tmp_path)
    assert discovery.discover_for_slot(
        solution_material_completeness_v1, "background_and_basis", top_k=4,
        allowed_document_ids=set(),
    ) == ()


def test_discover_contract_forwards_allowed_document_ids(tmp_path):
    discovery, _, _ = _indexed(tmp_path)
    results = discovery.discover_contract(
        solution_material_completeness_v1, top_k_per_slot=4,
        allowed_document_ids={"doc-basis"},
    )
    assert set(results) == set(solution_material_completeness_v1.slot_ids)
    seen = {c.document_id for candidates in results.values() for c in candidates}
    assert seen <= {"doc-basis"}
