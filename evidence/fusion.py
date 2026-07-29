"""候选融合：向量路 + 词法路 → RRF（Reciprocal Rank Fusion）。

复用本仓四臂验证过的 RRF 思路（``core/fusion.py``，纯名次算术、model-free）：
``fused_score(c) = Σ_每路 1/(rrf_k + rank)``，``rank`` 从 1 起，``rrf_k`` 复用 ``Settings.rrf_k``（60）。
RRF 只吃名次不吃原始分，天然免疫「向量距离 vs 词法命中数」量纲不可比。

**确定性破平：** 去重键 = ``(chunk 正文规范化哈希, start_char, end_char)``，
同分排序也按该键——**不用 document_id / chunk_id**，同一正文无论 id 如何，融合结果一致。

RRF 产出 fused 名次后，再做一次 **document-aware 选择**
（每文档先出一个代表、再按 fused 名次补齐），把跨文档多样性保持在融合层——因为 RRF 本身可能让单一
文档的多个 chunk 重新聚集。**不**复用
``vector_discovery._select_diverse_candidates``（它按 ``_candidate_order_key`` 距离重排，会丢弃 fused 名次；
词法候选 distance=None 会被排到最后 → 破坏融合）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .types import VectorCandidate
from .vector_discovery import _content_order_hash


def _content_key(candidate: VectorCandidate) -> tuple[str, int, int]:
    return (_content_order_hash(candidate.text), candidate.start_char, candidate.end_char)


def _diversify_by_document(
    ordered: Sequence[VectorCandidate], *, top_k: int
) -> tuple[VectorCandidate, ...]:
    """在**给定顺序**（fused 名次）上做文档多样化：先按坐标去重，再每文档一个代表优先、
    然后按 fused 名次补齐，最后重排 rank=1..N。保持输入顺序、不按距离重排。"""

    seen_coord: set[tuple[str, int, int]] = set()
    deduped: list[VectorCandidate] = []
    for cand in ordered:
        coord = (cand.document_id, cand.start_char, cand.end_char)
        if coord in seen_coord:
            continue
        seen_coord.add(coord)
        deduped.append(cand)

    selected: list[VectorCandidate] = []
    represented: set[str] = set()
    for cand in deduped:                        # 第一遍：每文档首个候选作代表
        if len(selected) >= top_k:
            break
        if cand.document_id not in represented:
            represented.add(cand.document_id)
            selected.append(cand)
    if len(selected) < top_k:                   # 第二遍：按 fused 名次补齐
        chosen = {id(c) for c in selected}
        for cand in deduped:
            if len(selected) >= top_k:
                break
            if id(cand) not in chosen:
                selected.append(cand)
    return tuple(replace(c, rank=i) for i, c in enumerate(selected, start=1))


def rrf_fuse(
    rankings: Sequence[Sequence[VectorCandidate]],
    *,
    rrf_k: int,
    top_k: int,
) -> tuple[VectorCandidate, ...]:
    """把「每路一个已按该路名次排好的候选列表」RRF 融合 → 文档多样化 → 截 top_k。

    输入必须是「每路独立列表」（``[[向量...],[词法...]]``）；同一路内重复内容键只按最优名次计一次。
    保留首见候选为 base（若向量先于词法，则保留带 distance 的向量候选，供下游稳定）。
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    agg: dict[tuple[str, int, int], dict] = {}
    for ranking in rankings:
        seen_in_path: set[tuple[str, int, int]] = set()
        for rank, cand in enumerate(ranking, start=1):
            if not isinstance(cand, VectorCandidate):
                raise ValueError(f"rankings must contain VectorCandidate; got {cand!r}")
            key = _content_key(cand)
            if key in seen_in_path:              # 同一路内重复键只计最优（最小）名次
                continue
            seen_in_path.add(key)
            contribution = 1.0 / (rrf_k + rank)
            entry = agg.get(key)
            if entry is None:
                agg[key] = {
                    "score": contribution,
                    "cand": cand,
                    "sources": {cand.discovery_source},
                    "key": key,
                }
            else:
                entry["score"] += contribution
                entry["sources"].add(cand.discovery_source)

    fused_sorted = sorted(
        agg.values(),
        key=lambda e: (-e["score"], e["key"][0], e["key"][1], e["key"][2]),
    )
    ranked: list[VectorCandidate] = []
    for index, entry in enumerate(fused_sorted, start=1):
        base: VectorCandidate = entry["cand"]
        metadata = dict(base.metadata)
        metadata.update({
            "rrf_score": entry["score"],
            "rrf_sources": tuple(sorted(entry["sources"])),
            "rrf_k": rrf_k,
        })
        ranked.append(replace(base, rank=index, metadata=metadata))
    return _diversify_by_document(ranked, top_k=top_k)
