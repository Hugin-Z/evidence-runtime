"""eval/metrics.py 单测 —— 每个指标 ≥3 条手算钉死用例（含边界）。纯确定性、无外部依赖。"""
from __future__ import annotations

from collections import namedtuple

import pytest

from eval.metrics import snippets_to_doc_ids, recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k

_Snip = namedtuple("_Snip", "source_path")


# ── snippets_to_doc_ids：按 source_path 去重保序 ─────────────────
def test_snippets_to_doc_ids_dedup_preserves_first_order():
    snips = [_Snip("a.md"), _Snip("a.md"), _Snip("b.md"), _Snip("c.md"), _Snip("b.md")]
    assert snippets_to_doc_ids(snips, lambda s: s.source_path) == ["a.md", "b.md", "c.md"]


def test_snippets_to_doc_ids_keeps_best_rank():
    # 同文档重复只留首次（最好）名次
    snips = [_Snip("c.md"), _Snip("a.md"), _Snip("c.md"), _Snip("b.md")]
    assert snippets_to_doc_ids(snips, lambda s: s.source_path) == ["c.md", "a.md", "b.md"]


def test_snippets_to_doc_ids_empty():
    assert snippets_to_doc_ids([], lambda s: s.source_path) == []


# ── recall@k ─────────────────────────────────────────────────────
def test_recall_all_gold_in_topk():
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], {"a", "c"}, 5) == 1.0   # 2/2


def test_recall_partial():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c", "d"}, 10) == 0.75       # 3/4


def test_recall_none_in_topk():
    assert recall_at_k(["x", "y", "a", "z", "w", "c"], {"a", "c"}, 2) == 0.0    # top2 无 gold


def test_recall_edge_empty_gold_is_one():
    assert recall_at_k(["a"], set(), 5) == 1.0                                  # 空 gold → 1.0


def test_recall_edge_empty_results_is_zero():
    assert recall_at_k([], {"a"}, 5) == 0.0


# ── precision@k（固定分母 k）─────────────────────────────────────
def test_precision_two_of_five():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "b"}, 5) == 0.4      # 2/5


def test_precision_fixed_denominator_when_fewer_results():
    assert precision_at_k(["a", "b", "c"], {"a"}, 5) == 0.2                     # 1/5（分母固定 5）


def test_precision_zero_hits():
    assert precision_at_k(["x", "y", "z"], {"a"}, 5) == 0.0


def test_precision_edge_k_zero():
    assert precision_at_k(["a"], {"a"}, 0) == 0.0


# ── MRR@k ────────────────────────────────────────────────────────
def test_mrr_first_gold_rank_two():
    assert mrr_at_k(["x", "a", "y"], {"a"}, 10) == 0.5                          # rank2 → 1/2


def test_mrr_rank_one():
    assert mrr_at_k(["a", "b"], {"a", "b"}, 10) == 1.0                          # rank1


def test_mrr_not_found_zero():
    assert mrr_at_k(["x", "y", "z"], {"a"}, 10) == 0.0


def test_mrr_edge_gold_beyond_k_is_zero():
    ranked = [f"n{i}" for i in range(10)] + ["a"]                               # a 在第 11 名
    assert mrr_at_k(ranked, {"a"}, 10) == 0.0


def test_mrr_edge_empty_gold_zero():
    assert mrr_at_k(["a"], set(), 10) == 0.0


# ── nDCG@10 双口径（B-1：linear 默认/主指标 + exponential 诊断）──────────
# 边界（口径无关）
def test_ndcg_empty_results_is_zero():
    assert ndcg_at_k([], {"a": 1}, 10) == 0.0
    assert ndcg_at_k([], {"a": 1}, 10, gain="exponential") == 0.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], {}, 10) == 0.0
    assert ndcg_at_k(["a", "b"], {}, 10, gain="exponential") == 0.0


def test_ndcg_bad_gain_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1}, 10, gain="quadratic")


def test_ndcg_binary_linear_equals_exponential():
    # 二值(0/1) 下 2^1-1==1 → 两口径恒等；单相关在第2位：DCG=1/log2(3)=0.6309298, IDCG=1
    r, q = ["x", "a", "y"], {"a": 1}
    assert ndcg_at_k(r, q, 10, gain="linear") == pytest.approx(ndcg_at_k(r, q, 10, gain="exponential"))
    assert ndcg_at_k(r, q, 10) == pytest.approx(0.630930, abs=1e-5)


# —— linear（默认/主指标）3 条手算 ——
def test_ndcg_linear_ideal_graded_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, 10) == pytest.approx(1.0)


def test_ndcg_linear_swapped_handcalc():
    # ranked=[b,a]，rel a=2,b=1，linear gain(rel)=rel：
    #   DCG  = 1/log2(2) + 2/log2(3) = 1 + 1.2618595 = 2.2618595
    #   IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.6309298 = 2.6309298
    #   nDCG = 2.2618595 / 2.6309298 = 0.859720
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(0.859720, abs=1e-4)


def test_ndcg_linear_partial_handcalc():
    # ranked=[a,b]，rel a=1,b=3（强档 b 排后），linear：
    #   DCG  = 1/log2(2) + 3/log2(3) = 1 + 1.8927893 = 2.8927893
    #   IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309298 = 3.6309298
    #   nDCG = 2.8927893 / 3.6309298 = 0.796707
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 3}, 10) == pytest.approx(0.796707, abs=1e-4)


# —— exponential（诊断）3 条手算 ——
def test_ndcg_exp_ideal_graded_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, 10, gain="exponential") == pytest.approx(1.0)


def test_ndcg_exp_swapped_handcalc():
    # ranked=[b,a]，rel a=2,b=1，exp gain(rel)=2^rel-1：
    #   DCG  = (2^1-1)/log2(2) + (2^2-1)/log2(3) = 1 + 1.8927893 = 2.8927893
    #   IDCG = (2^2-1)/log2(2) + (2^1-1)/log2(3) = 3 + 0.6309298 = 3.6309298
    #   nDCG = 2.8927893 / 3.6309298 = 0.796707
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10, gain="exponential") == pytest.approx(0.796707, abs=1e-4)


def test_ndcg_exp_diverges_from_linear_on_graded():
    # 同排序同 qrels，分级下两口径不同：linear 0.859720（对“强档排后”惩罚更轻）> exp 0.796707
    lin = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10, gain="linear")
    exp = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10, gain="exponential")
    assert lin == pytest.approx(0.859720, abs=1e-4)
    assert exp == pytest.approx(0.796707, abs=1e-4)
    assert lin > exp
