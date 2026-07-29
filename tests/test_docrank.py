"""tests/test_docrank.py — C-1 document 评测单位 + 候选深度（eval/bench/docrank）。

钉死：① 一 passage 多 chunk 不重复贡献 RRF 分（折叠去重）；② 长文碎片化下 document top-10
仍满 10（候选充足时）；③ N_DOC 停止 + chunk_cap 长尾保护生效；④ 折叠保序（首见定位次）。
"""
from __future__ import annotations

import pytest

from evidence.core.fusion import fuse
from evidence.core.types import Snippet, RetrievalPath
from evidence.core.config import DEFAULT_SETTINGS
from eval.bench.docrank import (fold_to_docs, docs_to_fusion_input, fused_doc_ids,
                                N_DOC, CHUNK_CAP)

RRF_K = DEFAULT_SETTINGS.rrf_k   # 60


def _snip(doc_id, path=RetrievalPath.VECTOR, score=0.5, char_start=0):
    """一个 chunk 级 Snippet（source_path 即其所属 document id）。"""
    return Snippet(source_path=doc_id, text="", score=score, path=path,
                   start_line=1, end_line=1, meta={"char_start": char_start})


def _id_of(snip):
    return snip.source_path


# ── ① 一 passage 多 chunk 不重复贡献 RRF ──────────────────────
def test_multi_chunk_same_doc_folds_once():
    # d1 出现 3 次（碎片），fold 后只应保留一次、位次为首见
    snips = [_snip("d1", char_start=0), _snip("d1", char_start=800),
             _snip("d2"), _snip("d1", char_start=1600), _snip("d3")]
    docs = fold_to_docs(snips, _id_of)
    assert docs == ["d1", "d2", "d3"]


def test_multi_chunk_no_double_rrf_contribution():
    # d1 在 vec 3 chunk、lex 1 chunk：折叠后每路只贡献一次 → 融合分 = 2/(k+1)，不是 4/(k+1)
    vec = [_snip("d1"), _snip("d1", char_start=800), _snip("d1", char_start=1600), _snip("d2")]
    lex = [_snip("d1", path=RetrievalPath.LEXICAL), _snip("d4", path=RetrievalPath.LEXICAL)]
    vec_in = docs_to_fusion_input(fold_to_docs(vec, _id_of), RetrievalPath.VECTOR)
    lex_in = docs_to_fusion_input(fold_to_docs(lex, _id_of), RetrievalPath.LEXICAL)
    fused = fuse([vec_in, lex_in], top_k=10)
    by_id = {s.source_path: s for s in fused}
    # d1：vec rank1 + lex rank1 = 两次贡献
    assert by_id["d1"].score == pytest.approx(2.0 / (RRF_K + 1))
    assert by_id["d1"].path == RetrievalPath.BOTH


def test_single_path_score_is_single_contribution():
    vec = [_snip("d1"), _snip("d1", char_start=800), _snip("d2")]
    vec_in = docs_to_fusion_input(fold_to_docs(vec, _id_of), RetrievalPath.VECTOR)
    fused = fuse([vec_in], top_k=10)
    assert fused_doc_ids(fused) == ["d1", "d2"]
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))   # 单次贡献，非 2/(k+1)


# ── ② 长文碎片化下 document top-10 仍满 10 ─────────────────────
def test_fragmented_long_doc_still_fills_doc_top10():
    # d1 被切成 30 chunk 排在最前；其后仍有 d2..d12（11 个）distinct doc
    snips = [_snip("d1", char_start=i * 800) for i in range(30)]
    snips += [_snip(f"d{j}") for j in range(2, 13)]
    docs = fold_to_docs(snips, _id_of)                      # 折叠：d1 塌成一个
    fused = fuse([docs_to_fusion_input(docs, RetrievalPath.VECTOR)], top_k=10)
    ids = fused_doc_ids(fused)
    assert len(ids) == 10
    assert len(set(ids)) == 10                              # 全不同 document
    assert ids[0] == "d1"


# ── ③ N_DOC 停止 + chunk_cap 长尾保护 ────────────────────────
def test_ndoc_stop_condition():
    snips = [_snip(f"d{i}") for i in range(50)]             # 50 个 distinct doc
    assert len(fold_to_docs(snips, _id_of, n_doc=20)) == 20        # 折够 20 即止
    assert len(fold_to_docs(snips, _id_of, n_doc=N_DOC)) == N_DOC


def test_chunk_cap_long_tail_protection():
    # 前 5 个 chunk 全是 d1（碎片），其后才有别的 doc；cap=5 → 只扫 5 个 chunk → 仅 1 个 doc
    snips = [_snip("d1", char_start=i * 800) for i in range(5)] + [_snip(f"d{i}") for i in range(2, 40)]
    assert fold_to_docs(snips, _id_of, n_doc=20, chunk_cap=5) == ["d1"]
    # cap 充足时能折出更多
    assert len(fold_to_docs(snips, _id_of, n_doc=20, chunk_cap=200)) == 20


def test_fold_preserves_first_occurrence_order():
    snips = [_snip("b"), _snip("a"), _snip("b"), _snip("c"), _snip("a")]
    assert fold_to_docs(snips, _id_of) == ["b", "a", "c"]
