"""确定性 fake embedder：仅验证管线连通、demo 与 CI，不代表检索质量。

外部读者与 CI 没有私有 embedding 服务也能把全链跑通。它按 8 个证据槽的关键词做**确定性
计数向量**（无网络、无模型、无随机），因此同输入必得同输出——适合"看管线跑起来什么样"和
锁定管线行为的回归测试。

**边界（务必如实）**：这个 embedder 的**检索质量不真实**。用它跑出的槽态**不是**准确率证据、
不是基线、不能外推。合成语料只能验证确定性管线行为。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from evidence.core.config import DEFAULT_SETTINGS, Settings
from .exact_locator import ExactEvidenceLocator
from .gap_checker import EvidenceGapChecker
from .vector_discovery import VectorDiscovery

# 8 组关键词，对齐固定契约的 8 个证据槽（中英各列，命中即计数）。
_GROUPS: tuple[tuple[str, ...], ...] = (
    ("背景", "依据", "政策", "background", "basis"),
    ("现状", "需求", "问题", "current", "need"),
    ("范围", "对象", "scope", "subject"),
    ("目标", "objective", "target"),
    ("能力", "建设内容", "功能", "capability", "function"),
    ("实施", "条件", "边界", "implementation", "boundary"),
    ("案例", "项目经验", "客户", "case", "reference"),
    ("成效", "指标", "效益", "outcome", "metric"),
)


class DeterministicKeywordEmbedder:
    """按槽关键词计数的确定性向量器。**非检索质量**，仅管线连通。"""

    model_id = "fake-keyword-embedder-p-g-v1"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            lowered = str(text).lower()
            vec = [float(sum(lowered.count(term) for term in group)) for group in _GROUPS]
            vec.append(1.0)                              # 偏置维，避免零向量
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def build_fake_gap_checker(settings: Settings = DEFAULT_SETTINGS) -> EvidenceGapChecker:
    """出厂默认配置，但 L2 用确定性 fake embedder（无服务即可跑）。

    **只换 embedder**，判定/闸门/单文档上限等一切照 `build_default_gap_checker`——因此 demo/CI
    跑的是真实管线行为，唯一不真实的是检索质量。
    """

    canonical_root = Path(settings.vector_db_path) / "m1-canonical"
    return EvidenceGapChecker(
        vector_discovery=VectorDiscovery(
            settings, embedder=DeterministicKeywordEmbedder(), embedding_backend="fake-keyword"),
        exact_locator=ExactEvidenceLocator(canonical_root),
        lexical_discovery=None,
        rrf_k=settings.rrf_k,
        single_doc_cap_ratio=settings.exact_span_single_doc_cap_ratio,
    )
