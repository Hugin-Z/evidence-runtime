"""RRF 融合层（core/fusion.py）测试 —— 纯确定性，手算钉死期望值。

RRF：fused_score = Σ_每路 1/(rrf_k + rank)，rank 从 1 起。默认 rrf_k=60。
只断言机械可验证的事实（顺序、分数、path、合并与否）；分数用 pytest.approx 手算比对。
"""
from __future__ import annotations

import pytest

from evidence.core.types import Snippet, RetrievalPath
from evidence.core.fusion import fuse


def _snip(source_path: str, path: RetrievalPath, start: int = 1, end: int = 1) -> Snippet:
    """构造输入片段。score 由 RRF 忽略（只吃名次），这里置 0.0。"""
    return Snippet(source_path=source_path, text="t", score=0.0,
                   path=path, start_line=start, end_line=end)


# ── 两路无交集：按 RRF 分交错合并 ────────────────────────────────
def test_two_paths_disjoint_interleave_by_rrf():
    V = [_snip("a.md", RetrievalPath.VECTOR), _snip("c.md", RetrievalPath.VECTOR)]
    L = [_snip("b.md", RetrievalPath.LEXICAL), _snip("d.md", RetrievalPath.LEXICAL)]
    out = fuse([V, L], rrf_k=60, top_k=10)
    # rank1 两条同分 1/61（a<b），rank2 两条同分 1/62（c<d）→ a, b, c, d
    assert [s.source_path for s in out] == ["a.md", "b.md", "c.md", "d.md"]
    assert out[0].score == pytest.approx(1 / 61)
    assert out[1].score == pytest.approx(1 / 61)
    assert out[2].score == pytest.approx(1 / 62)
    assert out[3].score == pytest.approx(1 / 62)
    assert out[0].path is RetrievalPath.VECTOR
    assert out[1].path is RetrievalPath.LEXICAL


# ── 两路有重复：合并为一条、path=BOTH、分数累加、排位上升 ────────
def test_duplicate_merged_to_both_and_rank_rises():
    X_v = _snip("x.md", RetrievalPath.VECTOR)   # vector rank1
    Y = _snip("y.md", RetrievalPath.VECTOR)     # vector rank2
    X_l = _snip("x.md", RetrievalPath.LEXICAL)  # lexical rank1（与 X_v 同键）
    Z = _snip("z.md", RetrievalPath.LEXICAL)    # lexical rank2
    out = fuse([[X_v, Y], [X_l, Z]], rrf_k=60, top_k=10)
    assert len(out) == 3                                  # X 两路合并为一条
    top = out[0]
    assert top.source_path == "x.md"
    assert top.path is RetrievalPath.BOTH
    assert top.score == pytest.approx(1 / 61 + 1 / 61)    # 两路 rank1 贡献之和
    assert top.meta["rrf_paths"] == ["lexical", "vector"]
    assert top.meta["rrf_k"] == 60
    # 排位上升：X 单路各 rank1（1/61），合并后（2/61）严格高于任何单路条目（≤1/62）
    assert out[0].score > out[1].score
    assert {s.source_path for s in out} == {"x.md", "y.md", "z.md"}


# ── 单路输入：退化为原序（名次序，不是字母序）──────────────────
def test_single_path_preserves_order():
    items = [_snip("m.md", RetrievalPath.VECTOR),
             _snip("a.md", RetrievalPath.VECTOR),
             _snip("z.md", RetrievalPath.VECTOR)]
    out = fuse([items], rrf_k=60, top_k=10)
    assert [s.source_path for s in out] == ["m.md", "a.md", "z.md"]   # 原序，非字母序
    assert all(s.path is RetrievalPath.VECTOR for s in out)
    assert out[0].score > out[1].score > out[2].score                # 1/61 > 1/62 > 1/63


# ── 空输入 / 一路为空：不炸 ─────────────────────────────────────
def test_empty_inputs_do_not_crash():
    assert fuse([], rrf_k=60, top_k=10) == []
    assert fuse([[], []], rrf_k=60, top_k=10) == []
    out = fuse([[_snip("a.md", RetrievalPath.VECTOR),
                 _snip("b.md", RetrievalPath.VECTOR)], []], rrf_k=60, top_k=10)
    assert [s.source_path for s in out] == ["a.md", "b.md"]
    assert all(s.path is RetrievalPath.VECTOR for s in out)


# ── top_k 截断 ───────────────────────────────────────────────────
def test_top_k_truncates():
    items = [_snip(f"{i}.md", RetrievalPath.VECTOR) for i in range(4)]
    out = fuse([items], rrf_k=60, top_k=2)
    assert len(out) == 2
    assert [s.source_path for s in out] == ["0.md", "1.md"]


# ── rrf_k 参数生效：构造一个改 rrf_k 就翻转排序的例子 ────────────
def test_rrf_k_changes_ordering():
    # P：单路 rank1 → 1/(k+1)。Q：两路各 rank4 → 2/(k+4)。
    # k=1:  P=1/2=0.5   > Q=2/5=0.4    → P 在前
    # k=10: P=1/11≈.091 < Q=2/14≈.143  → Q 在前  → 翻转
    P = _snip("p.md", RetrievalPath.VECTOR)
    f = [_snip("f1.md", RetrievalPath.LEXICAL), _snip("f2.md", RetrievalPath.LEXICAL),
         _snip("f3.md", RetrievalPath.LEXICAL), _snip("q.md", RetrievalPath.LEXICAL)]
    g = [_snip("g1.md", RetrievalPath.VECTOR), _snip("g2.md", RetrievalPath.VECTOR),
         _snip("g3.md", RetrievalPath.VECTOR), _snip("q.md", RetrievalPath.VECTOR)]

    def positions(k):
        paths = [s.source_path for s in fuse([[P], f, g], rrf_k=k, top_k=100)]
        return paths.index("p.md"), paths.index("q.md")

    p1, q1 = positions(1)
    assert p1 < q1        # rrf_k=1：P 在 Q 前
    p10, q10 = positions(10)
    assert q10 < p10      # rrf_k=10：Q 在 P 前（排序翻转）


# ── 重叠但不相同的行区间：不合并（钉住已知简化）──────────────────
def test_overlapping_but_different_ranges_not_merged():
    s1 = _snip("same.md", RetrievalPath.VECTOR, start=1, end=5)
    s2 = _snip("same.md", RetrievalPath.LEXICAL, start=3, end=7)   # 重叠但键不同
    out = fuse([[s1], [s2]], rrf_k=60, top_k=10)
    assert len(out) == 2                                            # 不合并
    assert {s.path for s in out} == {RetrievalPath.VECTOR, RetrievalPath.LEXICAL}
    assert RetrievalPath.BOTH not in {s.path for s in out}


# ── E-2：同一路内重复键只计最优（最小）名次一次，不重复累加 ──────────
def test_within_path_duplicate_key_counts_best_rank_once():
    # 同一路（vector）内两条同键片段：rank1 与 rank3。RRF 每路每项至多贡献一次，
    # 只按最优 rank1（1/61）计；不得累加 rank3 的 1/63。
    dup1 = _snip("dup.md", RetrievalPath.VECTOR, start=2, end=4)   # rank1
    other = _snip("o.md", RetrievalPath.VECTOR, start=1, end=1)    # rank2
    dup2 = _snip("dup.md", RetrievalPath.VECTOR, start=2, end=4)   # rank3，与 dup1 同键
    out = fuse([[dup1, other, dup2]], rrf_k=60, top_k=10)
    dup = [s for s in out if s.source_path == "dup.md"]
    assert len(dup) == 1                                  # 只出现一条
    assert dup[0].score == pytest.approx(1 / 61)          # 仅最优名次，不含 rank3 的 1/63
    assert dup[0].path is RetrievalPath.VECTOR
    assert out[0].source_path == "dup.md"                 # 1/61 > other 的 1/62
