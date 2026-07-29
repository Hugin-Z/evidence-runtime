"""检索评测指标（自实现，不装 ragas —— PLAN D4）。RAGAS 口径的文档级实现。

**文档级口径**：先用 ``snippets_to_doc_ids`` 把 ``list[Snippet]`` 按 ``source_path`` 去重、
保序，映射成"文档 id 序列"（同一文档被多路命中/多次出现只保留首次名次），再在该序列上
算指标。gold 为该 query 的答案文档 id 集合。

约定：``ranked_ids`` 是**已去重**的文档 id 序列（rank 从 1 起）；``gold`` 是文档 id 集合。
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence


def snippets_to_doc_ids(snippets: Iterable, id_of: Callable[[object], str]) -> list[str]:
    """把 snippet 序列按 source_path 去重、保序，映射为文档 id 序列。

    ``id_of(snippet) -> doc_id``。同一文档重复出现（如向量、词法各命中一次，或行区间不同
    未合并）只保留**首次**出现的名次；因 snippets 已按分数降序，首次即最好名次。
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in snippets:
        did = id_of(s)
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
    return out


def recall_at_k(ranked_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    """recall@k = |gold ∩ 前 k 个文档| / |gold|。

    公式：取 ``ranked_ids[:k]`` 与 gold 的交集大小，除以 gold 总数。
    边界：gold 为空 → 返回 1.0（无需召回任何文档即视为满足，避免除零）；
          结果为空 → 0.0（交集为空）。k 会被 ``ranked_ids`` 实际长度自然截断。
    """
    g = set(gold)
    if not g:
        return 1.0
    topk = set(ranked_ids[:k])
    return len(topk & g) / len(g)


def precision_at_k(ranked_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    """precision@k = |gold ∩ 前 k 个文档| / k（**固定分母 k**）。

    公式：命中数除以 k。采用固定分母 k（标准 precision@k）：即使返回不足 k 个文档，
    空缺名次按非相关计。边界：k ≤ 0 → 0.0；gold 为空 → 0.0（无正确答案，命中数必为 0）。
    """
    if k <= 0:
        return 0.0
    g = set(gold)
    hits = sum(1 for d in ranked_ids[:k] if d in g)
    return hits / k


def mrr_at_k(ranked_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    """MRR@k = 1 / (首个 gold 文档的名次)，名次 > k 或未出现记 0。

    公式：在 ``ranked_ids[:k]`` 中找第一个属于 gold 的文档，返回其 1-based 名次的倒数；
    前 k 名内无 gold → 0.0。边界：gold 为空 → 0.0；结果为空 → 0.0。
    """
    g = set(gold)
    if not g:
        return 0.0
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int], k: int,
              gain: str = "linear") -> float:
    """nDCG@k，**双口径增益**（B-1）。

    折扣统一 ``1/log2(i+1)``（i 为 1-based 名次）。增益两种口径：
      * ``gain="linear"``（**默认，主指标，公开对照唯一口径**）：``gain(rel)=rel``——
        与 trec_eval / pytrec_eval `ndcg_cut`（BEIR 官方 nDCG@10 如 BM25 0.325 用此）一致。
      * ``gain="exponential"``（**补充诊断**，表中标为诊断值、不与公开数字并置）：
        ``gain(rel)=2^rel − 1``——放大高相关档差异（Burges 口径）。

    公式：``DCG@k = Σ_{i=1..k} gain(rel_i) / log2(i + 1)``，``rel_i`` = 排名第 i 文档在 ``qrels``
    的分级相关度（不在记 0）；``IDCG@k`` = 同式对 qrels 值降序理想排列；``nDCG=DCG/IDCG``。
    二值(0/1) 下两口径恒等（2^1−1=1）；分级(0/1/2) 下不同。

    ``qrels``：``{doc_id: rel(int ≥ 0)}``。``ranked_ids`` 须已按文档去重保序。
    边界：qrels 无正例（IDCG=0）→ 0.0；结果为空 → 0.0；未知 gain → ValueError。
    """
    if gain == "linear":
        g = lambda r: float(r)
    elif gain == "exponential":
        g = lambda r: 2.0 ** r - 1.0
    else:
        raise ValueError(f"未知 gain 口径 {gain!r}（linear|exponential）")

    def dcg(rels) -> float:
        # 枚举 i 从 0 起 → 排名 i+1 → 折扣 log2((i+1)+1) = log2(i+2)
        return sum(g(r) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    idcg = dcg(sorted(qrels.values(), reverse=True))
    if idcg == 0.0:
        return 0.0
    return dcg([qrels.get(d, 0) for d in ranked_ids]) / idcg
