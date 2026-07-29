"""M5-Rerank 单测：候选并集去重 + 精排定序（fake scorer 确定性）+ 破平内容确定 +
P4-A 多样性不回退 + 失败率闸门。真 rerank 需 llama-server，故单测一律用 fake。"""

from __future__ import annotations

import pytest

from evidence.rerank import (
    MAX_FAILURE_RATIO,
    RERANK_VERSION,
    RerankBatchGuard,
    RerankError,
    rerank_candidates,
    union_candidates,
)
from evidence.types import VectorCandidate


def _cand(text: str, *, doc: str, start: int = 0, rank: int = 1,
          distance: float | None = 0.5, source: str = "vector") -> VectorCandidate:
    return VectorCandidate(
        slot_id="background_and_basis", query="q", document_id=doc,
        chunk_id=f"{doc}-{start}", text=text, start_char=start,
        end_char=start + len(text), start_line=1, end_line=1,
        rank=rank, distance=distance, discovery_source=source,
    )


class FakeScorer:
    """按 {text: score} 打分；记录调用，便于断言"只排序不判定"。"""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls: list[tuple[str, int]] = []

    def scores(self, query, documents):
        self.calls.append((query, len(documents)))
        return [self.table[d] for d in documents], 0.123


# ── 并集去重 ──────────────────────────────────────────────────────────────
def test_union_dedups_by_content_not_id():
    # 同正文同偏移、不同 document_id/chunk_id → 视为同一候选（内容确定去重）。
    a = _cand("同一段正文", doc="doc-a", start=0)
    b = _cand("同一段正文", doc="doc-b", start=0)
    out = union_candidates([a], [b])
    assert len(out) == 1
    assert out[0] is a                      # 先见者保留（向量在前 → 保住 distance）


def test_union_keeps_distinct_and_preserves_order():
    a = _cand("向量候选", doc="doc-a", start=0)
    b = _cand("词法候选", doc="doc-b", start=10, distance=None, source="lexical")
    out = union_candidates([a], [b])
    assert [c.text for c in out] == ["向量候选", "词法候选"]
    assert out[1].discovery_source == "lexical"


def test_union_rejects_non_candidate():
    with pytest.raises(ValueError):
        union_candidates(["not a candidate"])


# ── 精排定序 ──────────────────────────────────────────────────────────────
def test_rerank_orders_by_score_desc():
    low = _cand("低分", doc="doc-a", start=0)
    high = _cand("高分", doc="doc-a", start=10)
    mid = _cand("中分", doc="doc-a", start=20)
    scorer = FakeScorer({"低分": 0.1, "高分": 9.9, "中分": 5.0})
    out, stats = rerank_candidates([low, high, mid], query="q", scorer=scorer, top_k=3)
    assert [c.text for c in out] == ["高分", "中分", "低分"]
    assert [c.rank for c in out] == [1, 2, 3]
    assert stats["pool_size"] == 3 and stats["elapsed_s"] == 0.123


def test_rerank_records_score_in_metadata_only():
    c = _cand("正文", doc="doc-a")
    out, _ = rerank_candidates([c], query="q", scorer=FakeScorer({"正文": 3.0}), top_k=1)
    # 分数只作元数据（可观测），不改变候选身份/正文——rerank 不判充分性。
    assert out[0].metadata["rerank_score"] == 3.0
    assert out[0].metadata["rerank_version"] == RERANK_VERSION
    assert out[0].text == "正文" and out[0].document_id == "doc-a"


def test_rerank_truncates_to_top_k():
    cands = [_cand(f"c{i}", doc="doc-a", start=i * 10) for i in range(5)]
    scorer = FakeScorer({f"c{i}": float(i) for i in range(5)})
    out, _ = rerank_candidates(cands, query="q", scorer=scorer, top_k=2)
    assert [c.text for c in out] == ["c4", "c3"]


def test_rerank_empty_pool_is_noop():
    out, stats = rerank_candidates([], query="q", scorer=FakeScorer({}), top_k=3)
    assert out == () and stats["pool_size"] == 0


def test_rerank_rejects_bad_top_k():
    with pytest.raises(ValueError):
        rerank_candidates([_cand("x", doc="d")], query="q",
                          scorer=FakeScorer({"x": 1.0}), top_k=0)


def test_rerank_detects_score_count_mismatch():
    class BadScorer:
        def scores(self, query, documents):
            return [1.0], 0.0
    with pytest.raises(RerankError):
        rerank_candidates([_cand("a", doc="d", start=0), _cand("b", doc="d", start=5)],
                          query="q", scorer=BadScorer(), top_k=2)


# ── 破平内容确定（不依赖 id）──────────────────────────────────────────────
def test_tie_break_is_content_deterministic_not_id():
    """同分候选：换 document_id / chunk_id 不改变次序（承接 M1b determinism）。"""
    scorer = FakeScorer({"甲段正文": 5.0, "乙段正文": 5.0})
    a1 = _cand("甲段正文", doc="doc-a", start=0)
    b1 = _cand("乙段正文", doc="doc-b", start=0)
    first, _ = rerank_candidates([a1, b1], query="q", scorer=scorer, top_k=2)
    # 换一套 id（内容、偏移不变）
    a2 = _cand("甲段正文", doc="zzz-9", start=0)
    b2 = _cand("乙段正文", doc="aaa-0", start=0)
    second, _ = rerank_candidates([a2, b2], query="q", scorer=scorer, top_k=2)
    assert [c.text for c in first] == [c.text for c in second]


def test_tie_break_stable_under_input_shuffle():
    scorer = FakeScorer({"x1": 1.0, "x2": 1.0, "x3": 1.0})
    c1 = _cand("x1", doc="d", start=0)
    c2 = _cand("x2", doc="d", start=10)
    c3 = _cand("x3", doc="d", start=20)
    a, _ = rerank_candidates([c1, c2, c3], query="q", scorer=scorer, top_k=3)
    b, _ = rerank_candidates([c3, c1, c2], query="q", scorer=scorer, top_k=3)
    assert [c.text for c in a] == [c.text for c in b]


# ── P4-A 跨文档多样性不回退 ───────────────────────────────────────────────
def test_document_diversity_preserved_after_rerank():
    """单一文档霸占高分时，仍每文档先出一个代表（P4-A 等价保持）。"""
    hi1 = _cand("a-hi1", doc="doc-a", start=0)
    hi2 = _cand("a-hi2", doc="doc-a", start=10)
    lo = _cand("b-lo", doc="doc-b", start=0)
    scorer = FakeScorer({"a-hi1": 9.0, "a-hi2": 8.0, "b-lo": 0.1})
    out, _ = rerank_candidates([hi1, hi2, lo], query="q", scorer=scorer, top_k=2)
    # 若无多样性保持，top-2 会是 doc-a 的两条；保持后第二位让给 doc-b 代表。
    assert [c.document_id for c in out] == ["doc-a", "doc-b"]


# ── 失败率闸门 ────────────────────────────────────────────────────────────
def test_batch_guard_passes_under_threshold():
    guard = RerankBatchGuard()
    for _ in range(100):
        guard.record(ok=True)
    guard.record(ok=False, detail="one failure")   # 1/101 < 2%
    guard.check()                                  # 不抛


def test_batch_guard_stops_batch_over_threshold():
    guard = RerankBatchGuard()
    for _ in range(10):
        guard.record(ok=True)
    guard.record(ok=False, detail="boom")          # 1/11 ≈ 9% > 2%
    with pytest.raises(RerankError, match="停批"):
        guard.check()
    assert guard.failure_ratio > MAX_FAILURE_RATIO


def test_batch_guard_empty_is_safe():
    RerankBatchGuard().check()


# ── 端到端接线：确保 checker 的 rerank 分支真被执行（防"分支从未被测"类漏网）──
def test_checker_rerank_branch_is_exercised_with_fake_reranker(tmp_path):
    """用 gap_checker 既有 fake 装置跑一遍 reranker 分支：验证接线可用、
    query 由 build_slot_query 提供、每 slot 都调用了打分器。"""
    import sys
    sys.path.insert(0, "tests")
    from test_m1_gap_checker import _FakeVector, _FakeLocator, _InspectingVerifier, _RecordingBuilder
    from evidence.contract import solution_material_completeness_v1 as contract
    from evidence.documents import NormalizedDocument
    from evidence.gap_checker import EvidenceGapChecker

    path = tmp_path / "m.md"
    path.write_text("建设内容包括数据平台模块。\n系统提供数据汇聚能力。", encoding="utf-8")
    document = NormalizedDocument.from_path(
        path, document_id="doc-1", source_path="materials/m.md",
        normalized_path="materials/m.md")

    class CountingScorer:
        def __init__(self):
            self.queries = []
        def scores(self, query, documents):
            self.queries.append(query)
            # 逆序打分：确保排序确实由打分器驱动（非原序透传）
            return [float(len(documents) - i) for i in range(len(documents))], 0.01

    events = []
    scorer = CountingScorer()
    checker = EvidenceGapChecker(
        vector_discovery=_FakeVector(events, include_first_candidate=True),
        exact_locator=_FakeLocator(events),
        verifier=_InspectingVerifier(events, weak_first=True),
        report_builder=_RecordingBuilder(events),
        reranker=scorer,
    )
    run = checker.run(
        contract=contract, documents=[document], max_chars=200, top_k_per_slot=3,
        context_lines=1, max_span_chars=400, max_results_per_slot=5,
    )
    assert run.report is not None
    # fake vector 只给 slot[0] 产候选；空池 slot 会提前返回、不调打分器（预期行为）。
    # 关键断言：有候选的 slot 确实走了 rerank 分支，且 query 来自 build_slot_query。
    assert len(scorer.queries) == 1
    assert contract.slots[0].name in scorer.queries[0]


def test_checker_rejects_reranker_without_scores_method():
    from evidence.gap_checker import EvidenceGapChecker
    with pytest.raises(ValueError, match="reranker.scores"):
        EvidenceGapChecker(
            vector_discovery=type("V", (), {"index_documents": lambda *a, **k: None,
                                            "discover_contract": lambda *a, **k: {}})(),
            exact_locator=type("L", (), {"locate_for_slot": lambda *a, **k: None})(),
            reranker=object(),
        )
