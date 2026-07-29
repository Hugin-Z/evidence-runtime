"""融合层 —— RRF（Reciprocal Rank Fusion）。纯确定性、零依赖、零模型。

把多路检索结果（每路一个**已按该路自身分数排好序**的 ``list[Snippet]``）融合成
单一 ``list[Snippet]``。core model-free 铁律不变：这里只有名次算术，无模型、无 LLM。

**RRF 公式**：``fused_score(s) = Σ_每路 1/(rrf_k + rank)``，``rank`` 从 1 起（该 snippet
在那一路里的名次）。``rrf_k`` 默认 60（业界惯例，见 ``settings.rrf_k``）。RRF 只吃"名次"、
不吃各路原始分数——因此**天然免疫**"向量相似度 vs 词法命中率"这类量纲不可比问题。

**去重**：键 = ``(source_path, start_line, end_line)`` 精确匹配。同一 snippet 被**多路**命中
→ 合并为一条，``fused_score`` 累加各路贡献。同一路内若出现重复键（同 source_path/起止行），
只按其**最优（最小）名次**计一次贡献，**不重复累加**（RRF 每路每项至多贡献一次）。

**已知简化（eval 阶段观察是否成为问题）**：行区间"**重叠但不完全相同**"的两条**不合并**
（键不相等），保留为两条。极端情形下会造成同一处内容的碎片化/重复展示；本版刻意不做
区间归并，留待 eval 用真实语料判断其影响再决定是否处理。

**可观测性（A/B 依据）**：合并条目的 ``path`` 置为 ``RetrievalPath.BOTH``（两路都命中）；
单路条目保留其原 ``path``（VECTOR / LEXICAL）。下游据此统计"多少命中来自双路交集 vs 单路"。
另在 ``meta`` 记 ``rrf_paths``（贡献路径）与 ``rrf_k``，仅作溯源。

**分数取舍**：输出 ``Snippet.score`` 被 ``fused_score`` 覆盖，**不保留各路原始分数**——它们
量纲不可比，保留只会诱导下游把 RRF 分误读成相似度。需要原始分数请在融合前取。

**单路退化**：只有一路输入时，RRF 分是 ``rank`` 的单调递减函数 → 输出顺序 == 该路原序
（分数变为 ``1/(rrf_k+rank)``）。由 ``test_fusion`` 钉住。
"""

from __future__ import annotations

from dataclasses import replace

from .config import DEFAULT_SETTINGS
from .types import Snippet, RetrievalPath


def _key(s: Snippet):
    """去重键：同文件、同起止行 = 同一条证据。"""
    return (s.source_path, s.start_line, s.end_line)


def fuse(results_by_path: list[list[Snippet]], *,
         rrf_k: int = DEFAULT_SETTINGS.rrf_k,
         top_k: int = DEFAULT_SETTINGS.top_k) -> list[Snippet]:
    """把"每路一个已排序列表"融合成单一列表（RRF），按 fused_score 降序、截到 top_k。

    输入必须是"每路独立列表"（``[[路1...], [路2...], ...]``），**不接受**预先混合的
    单一列表——因为 RRF 要按"每路各自的名次"计分，混合后名次信息就丢了。
    """
    # key -> {"score": 累加分, "snip": 首见片段, "paths": 贡献路径集}
    agg: dict = {}
    for path_results in results_by_path:
        seen_in_path: set = set()  # 同一路内重复键只计最优（最小）名次一次
        for rank, snip in enumerate(path_results, start=1):
            key = _key(snip)
            if key in seen_in_path:
                continue  # 该路已在更优名次贡献过此键 → 不重复累加
            seen_in_path.add(key)
            contribution = 1.0 / (rrf_k + rank)
            entry = agg.get(key)
            if entry is None:
                agg[key] = {"score": contribution, "snip": snip, "paths": {snip.path}}
            else:
                entry["score"] += contribution
                entry["paths"].add(snip.path)

    fused: list[Snippet] = []
    for entry in agg.values():
        paths = entry["paths"]
        result_path = RetrievalPath.BOTH if len(paths) >= 2 else next(iter(paths))
        base = entry["snip"]
        fused.append(replace(
            base,
            score=entry["score"],
            path=result_path,
            meta={**base.meta,
                  "rrf_k": rrf_k,
                  "rrf_paths": sorted(p.value for p in paths)},
        ))

    # 降序按 fused_score；同分用 (source_path, start_line, end_line) 稳定次序，保证确定性。
    fused.sort(key=lambda s: (-s.score, s.source_path, s.start_line or 0, s.end_line or 0))
    return fused[:top_k]
