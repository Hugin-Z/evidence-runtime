"""确定性测试替身，永不进入生产路径。

生产默认使用真实 embedding 后端；这里的概念向量只让检索测试可离线复现。
"""
from __future__ import annotations

# test-only 概念词典：同一概念维度的词被映射到相近向量（retry≈backoff≈requeue…）。
_CONCEPTS = {
    "reliability": ["retry", "backoff", "requeue", "timeout", "exponential",
                    "redeliver", "strategy", "attempt", "resilience", "failed"],
    "garden": ["garden", "tomato", "basil", "soil", "plant"],
    "payment": ["invoice", "payment", "refund", "charge"],
}


class FakeConceptEmbedder:
    """**TEST-ONLY** 确定性概念 embedder。

    不代表真模型的语义质量；永不进生产路径（生产默认 API/LM Studio embedder）。
    把同概念的词投到同一维度，使"查 retry 召回只含 backoff 的文档"可机械断言。
    """

    def __init__(self, model_id: str = "fake-concept-v1"):
        self.model_id = model_id

    def encode(self, texts):
        out = []
        for t in texts:
            tl = str(t).lower()
            vec = [float(sum(tl.count(w) for w in ws)) for ws in _CONCEPTS.values()]
            vec.append(0.1)  # 常数维：避免全零向量（cosine 未定义）
            norm = sum(v * v for v in vec) ** 0.5
            out.append([v / norm for v in vec])
        return out
