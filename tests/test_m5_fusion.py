"""M5 RRF 融合单测：名次算术、内容确定破平、跨路去重、单路退化、文档多样性、top_k。"""

from __future__ import annotations

from evidence.fusion import rrf_fuse
from evidence.types import VectorCandidate


def _cand(doc: str, text: str, start: int, rank: int, source: str, distance):
    return VectorCandidate(
        slot_id="s", query="q", document_id=doc, chunk_id=f"{doc}-{start}",
        text=text, start_char=start, end_char=start + len(text),
        start_line=1, end_line=1, rank=rank, distance=distance,
        discovery_source=source,
    )


def test_single_path_degenerates_to_input_order():
    vec = [
        _cand("d1", "alpha", 0, 1, "vector", 0.1),
        _cand("d2", "bravo", 0, 2, "vector", 0.2),
        _cand("d3", "charlie", 0, 3, "vector", 0.3),
    ]
    fused = rrf_fuse([vec], rrf_k=60, top_k=8)
    assert [c.text for c in fused] == ["alpha", "bravo", "charlie"]
    assert [c.rank for c in fused] == [1, 2, 3]


def test_cross_path_overlap_boosts_and_dedups():
    # 同一内容（同 text+坐标）在两路都命中 → 合并一条、分数累加 → 排到最前。
    shared_v = _cand("d1", "shared", 0, 3, "vector", 0.9)   # 向量里排第 3
    shared_l = _cand("d1", "shared", 0, 1, "lexical", None)  # 词法里排第 1
    only_v = _cand("d2", "onlyv", 0, 1, "vector", 0.1)       # 向量里排第 1（单路）
    fused = rrf_fuse([[only_v, shared_v], [shared_l]], rrf_k=60, top_k=8)
    texts = [c.text for c in fused]
    assert texts.count("shared") == 1, "跨路同内容去重为一条"
    # shared: 1/(60+2)+1/(60+1) ；onlyv: 1/(60+1) → shared 分更高 → 居前
    assert texts[0] == "shared"
    shared_out = next(c for c in fused if c.text == "shared")
    assert set(shared_out.metadata["rrf_sources"]) == {"vector", "lexical"}


def test_tie_break_is_content_deterministic():
    # 两条不同内容、各自单路 rank 1 → 同 RRF 分；破平按内容哈希+位置，稳定且与输入顺序无关。
    a = _cand("d1", "aaa", 0, 1, "vector", 0.1)
    b = _cand("d2", "bbb", 0, 1, "lexical", None)
    forward = [c.text for c in rrf_fuse([[a], [b]], rrf_k=60, top_k=8)]
    reverse = [c.text for c in rrf_fuse([[b], [a]], rrf_k=60, top_k=8)]
    assert forward == reverse, "破平与输入顺序无关（内容确定）"


def test_document_diversity_preserved_after_fusion():
    # d1 占据高名次的多条；d2 只有一条较低名次。融合后 d2 代表应被多样化提前（P4-A 等价）。
    vec = [
        _cand("d1", "d1-a", 0, 1, "vector", 0.1),
        _cand("d1", "d1-b", 10, 2, "vector", 0.2),
        _cand("d1", "d1-c", 20, 3, "vector", 0.3),
        _cand("d2", "d2-a", 0, 4, "vector", 0.4),
    ]
    fused = rrf_fuse([vec], rrf_k=60, top_k=2)
    docs = [c.document_id for c in fused]
    assert docs == ["d1", "d2"], "每文档先出一个代表（多样性），非 d1 独占 top-2"


def test_top_k_caps_output():
    vec = [_cand(f"d{i}", f"t{i}", 0, i + 1, "vector", 0.1 * i) for i in range(10)]
    assert len(rrf_fuse([vec], rrf_k=60, top_k=3)) == 3
