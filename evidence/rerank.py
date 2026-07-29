"""Cross-encoder 精排层：候选发现层，不是判断层。

**职责边界（务必分清，防越界）：**
- 本层只做「**候选去留排序**」：对 (slot-query, candidate-content) 逐对打相关分，按分定 top-k。
- 它**不做证据充分性判断**——某槽是否 `filled` / `weak` / `missing` 由
  :mod:`evidence.verifier` 的**确定性规则**判定，与本层分数**无关**。rerank 分数**不进**任何
  判定逻辑，也**不写入** EvidenceSpan 的证据语义。

**架构豁免口径：** cross-encoder 是**判别式打分模型**（对 (query, passage) 出相关分），与 embedding
**同类**——属确定性检索原语，进候选发现层、**不进判断层**。Engine 的判定层仍**零模型**。

词法通道只扩召回池，候选去留由精排处理。

**确定性破平：** 同分时按 ``(chunk 正文规范化哈希, start_char, end_char)``
排序，**不用 document_id / chunk_id**。

精排定序后复用 :func:`evidence.fusion._diversify_by_document`，每文档先出一个代表，
再按精排名次补齐。exact-span 的单文档上限在下游 :mod:`evidence.exact_locator`。

**默认参数铁律：** 不调任何阈值/参数去凑结果（预注册红线）；分数只用其**序**，不归一化、不设阈值。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .fusion import _content_key, _diversify_by_document
from .types import VectorCandidate

RERANK_VERSION = "m5-rerank-v1"
DEFAULT_URL = "http://127.0.0.1:1235/v1/rerank"
DEFAULT_MODEL = "bge-reranker-v2-m3"
TIMEOUT_S = 60
RETRIES = 2
BACKOFF_S = 1.0
MAX_DOC_CHARS = 6000            # 主动头截断，避免单对超模型 context（与 bench 第四臂同口径）
MAX_FAILURE_RATIO = 0.02        # 单批失败率超过 2% 停批（预注册要求）


class RerankError(RuntimeError):
    """精排服务不可用或响应非法。"""


class RerankScorer(Protocol):
    """打分器接口（真实 HTTP 客户端与测试用 fake 共用）。"""

    def scores(self, query: str, documents: Sequence[str]) -> tuple[list[float], float]:
        """返回 (与 documents 对齐的分数, 耗时秒)。"""
        ...


class CrossEncoderReranker:
    """HTTP ``/v1/rerank`` 客户端（独立 llama-server，Jina/Cohere 兼容；仅标准库）。

    与 ``eval/bench/reranker.py`` 同协议同口径（该客户端已在四臂 benchmark 验证）。
    分数为原始 logit（无界、非 sigmoid），**只用其序**。
    """

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 timeout: float = TIMEOUT_S, retries: int = RETRIES,
                 backoff: float = BACKOFF_S) -> None:
        self.url, self.model = url, model
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    def scores(self, query: str, documents: Sequence[str]) -> tuple[list[float], float]:
        docs = [d[:MAX_DOC_CHARS] for d in documents]
        if not docs:
            return [], 0.0
        last: Exception | None = None
        shrunk = False
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                body = json.dumps({"model": self.model, "query": query,
                                   "documents": list(docs)}).encode("utf-8")
                req = urllib.request.Request(
                    self.url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                elapsed = time.time() - t0
                results = payload.get("results", payload.get("data"))
                if not results:
                    raise RerankError(f"响应无 results/data 字段: {payload}")
                out: list[float | None] = [None] * len(docs)
                for item in results:
                    out[item["index"]] = item.get("relevance_score", item.get("score"))
                if any(s is None for s in out):
                    raise RerankError(f"分数未对齐全部文档: {out}")
                return [float(s) for s in out], elapsed  # type: ignore[arg-type]
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:
                    pass
                last = RerankError(f"HTTP {exc.code}: {detail[:200]}")
                if "too large" in detail and not shrunk:
                    docs = [d[: max(1, len(d) // 2)] for d in docs]
                    shrunk = True
                    continue
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, RerankError, KeyError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
        raise RerankError(f"rerank 失败（{self.retries + 1} 次尝试）: {last}")


def union_candidates(*rankings: Sequence[VectorCandidate]) -> tuple[VectorCandidate, ...]:
    """候选**并集**（扩召回，不在此截断）；按内容键去重，先见者保留（向量在前则保留带 distance 的）。

    去重键 = ``(正文规范化哈希, start_char, end_char)``——内容确定，不用 id。
    """

    seen: set[tuple[str, int, int]] = set()
    out: list[VectorCandidate] = []
    for ranking in rankings:
        for cand in ranking:
            if not isinstance(cand, VectorCandidate):
                raise ValueError(f"rankings must contain VectorCandidate; got {cand!r}")
            key = _content_key(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return tuple(out)


def rerank_candidates(
    candidates: Sequence[VectorCandidate],
    *,
    query: str,
    scorer: RerankScorer,
    top_k: int,
) -> tuple[tuple[VectorCandidate, ...], dict]:
    """对候选池精排 → 文档多样化 → 截 top_k。返回 (candidates, stats)。

    只排序、不判充分性。同分破平内容确定；随后按每文档代表优先的方式保持来源多样性。
    """

    from dataclasses import replace

    pool = tuple(candidates)
    stats = {"pool_size": len(pool), "elapsed_s": 0.0, "failed": False}
    if not pool:
        return (), stats
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scores, elapsed = scorer.scores(query, [c.text for c in pool])
    if len(scores) != len(pool):
        raise RerankError(f"分数数量 {len(scores)} 与候选数 {len(pool)} 不一致")
    stats["elapsed_s"] = elapsed

    # 同分破平：内容哈希 → 字符位置（内容确定，不用 id）。
    order = sorted(
        range(len(pool)),
        key=lambda i: (-scores[i],) + _content_key(pool[i]),
    )
    ranked = []
    for new_rank, idx in enumerate(order, start=1):
        base = pool[idx]
        metadata = dict(base.metadata)
        metadata.update({"rerank_score": scores[idx], "rerank_version": RERANK_VERSION})
        ranked.append(replace(base, rank=new_rank, metadata=metadata))
    return _diversify_by_document(ranked, top_k=top_k), stats


class RerankBatchGuard:
    """单批失败率闸门：失败 >``MAX_FAILURE_RATIO`` 停批（预注册要求）。"""

    def __init__(self, max_failure_ratio: float = MAX_FAILURE_RATIO) -> None:
        self.max_failure_ratio = max_failure_ratio
        self.total = 0
        self.failed = 0
        self.failures: list[str] = []

    def record(self, *, ok: bool, detail: str = "") -> None:
        self.total += 1
        if not ok:
            self.failed += 1
            self.failures.append(detail)

    @property
    def failure_ratio(self) -> float:
        return (self.failed / self.total) if self.total else 0.0

    def check(self) -> None:
        if self.total and self.failure_ratio > self.max_failure_ratio:
            raise RerankError(
                f"rerank 单批失败率 {self.failure_ratio:.1%} 超过阈值 "
                f"{self.max_failure_ratio:.1%}（{self.failed}/{self.total}）；停批。"
                f" 首个失败：{self.failures[0][:200] if self.failures else ''}"
            )
