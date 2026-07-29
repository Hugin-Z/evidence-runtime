"""Cross-encoder 精排客户端：HTTP /v1/rerank（Jina/Cohere 兼容）。

不装任何包（仅标准库 urllib）。core 一行不改；本模块只在 harness 层被第四臂调用。
服务端：llama.cpp b10012（win-cuda-12.4）加载 bge-reranker-v2-m3-Q8_0.gguf，
        `llama-server.exe --reranking --embedding --pooling rank --port 1235 -ngl 99`
        `  -b 8192 -ub 8192 -c 8192 -m <bge-reranker-v2-m3-Q8_0.gguf>`。
        **`-b/-ub` 不可省**：默认 physical batch 512，而单个 (query,doc) 对常达 ~1100 token，
        省略会 HTTP 500 `input (N tokens) is too large ... increase the physical batch size`
        批大小只决定能否受理请求，
        不影响分数。
分数为原始 logit（无界，非 sigmoid）；精排只用其**序**，不做归一化。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:1235/v1/rerank"
DEFAULT_MODEL = "bge-reranker-v2-m3"
TIMEOUT_S = 60          # 单次请求超时
RETRIES = 2             # 失败重试次数（总尝试 = RETRIES + 1）
BACKOFF_S = 1.0         # 重试退避基数（线性）


class RerankError(RuntimeError):
    pass


class Reranker:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 timeout: float = TIMEOUT_S, retries: int = RETRIES,
                 backoff: float = BACKOFF_S):
        self.url, self.model = url, model
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    def scores(self, query: str, documents: list[str]) -> tuple[list[float], float]:
        """返回 (scores_aligned_to_documents, elapsed_s)。scores[i] = 第 i 个 doc 的 relevance_score。

        反应式保险：若服务端报 "too large"（单个 (query,doc) 对超模型 8192 max），把过长 doc
        头截断再试一次（正常输入不触发；调用方已在 6000 字符处主动头截断，此处仅兜底）。
        """
        docs = list(documents)
        last = None
        shrunk = False
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                body = json.dumps({"model": self.model, "query": query,
                                   "documents": docs}).encode("utf-8")
                req = urllib.request.Request(self.url, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                dt = time.time() - t0
                results = resp.get("results", resp.get("data"))
                if not results:
                    raise RerankError(f"响应无 results/data 字段: {resp}")
                out = [None] * len(docs)
                for item in results:
                    out[item["index"]] = item.get("relevance_score", item.get("score"))
                if any(s is None for s in out):
                    raise RerankError(f"分数未对齐全部文档: {out}")
                return out, dt
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                last = RerankError(f"HTTP {e.code}: {detail[:200]}")
                if "too large" in detail and not shrunk:
                    docs = [d[: max(1, len(d) // 2)] for d in docs]  # 头截断兜底
                    shrunk = True
                    continue
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, RerankError, KeyError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
        raise RerankError(f"rerank 失败（{self.retries + 1} 次尝试）: {last}")

    def rerank(self, query: str, doc_ids: list[str], doc_texts: list[str]) -> tuple[list[str], float]:
        """按 relevance_score 降序重排 doc_ids，返回 (reranked_ids, elapsed_s)。稳定排序，同分保持原序。"""
        scores, dt = self.scores(query, doc_texts)
        order = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)
        return [doc_ids[i] for i in order], dt
