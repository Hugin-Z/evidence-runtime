"""守卫测试：验收 harness 必须跑**出厂默认配置**，不得手工拼装参数。

背景：`run_acceptance._real_checker` 曾手工构造 `EvidenceGapChecker`，
漏传默认开启的 `single_doc_cap_ratio`，导致验收配置与公开 CLI 配置不一致。
修法是让 `_real_checker` 无条件委托 `build_default_gap_checker`。

本测试把这个模式**锁死**：harness 构造的 checker 配置必须逐字段等于出厂默认。出厂默认将来任何变更，
harness 自动跟随；任何人再手工拼参数（漏一个开关）都会在这里立刻红。
"""
from __future__ import annotations

from evidence.core.config import DEFAULT_SETTINGS
from evidence.gap_checker import build_default_gap_checker
from eval.task_acceptance.run_acceptance import _real_checker


def _config(checker) -> dict:
    """checker 的**结果相关配置** —— 逐字段可比。"""

    return {
        "single_doc_cap_ratio": checker.single_doc_cap_ratio,
        "rrf_k": checker.rrf_k,
        "lexical_discovery_off": checker.lexical_discovery is None,
        "reranker_off": checker.reranker is None,
        "vector_discovery_type": type(checker.vector_discovery).__name__,
        "exact_locator_type": type(checker.exact_locator).__name__,
        "verifier_type": type(checker.verifier).__name__,
        "report_builder_type": type(checker.report_builder).__name__,
    }


def test_acceptance_harness_config_equals_factory_default():
    """harness 的 checker 配置 == 出厂默认（逐字段）。"""

    harness = _real_checker(DEFAULT_SETTINGS)
    factory = build_default_gap_checker(DEFAULT_SETTINGS)
    assert _config(harness) == _config(factory)


def test_acceptance_harness_carries_m5_cap_from_settings():
    """专锁 P-A 那个缺陷：单文档上限必须来自 settings，不是被漏成 None。

    若有人再把 harness 改回手工拼装且漏传 cap，本断言立刻红。
    """

    harness = _real_checker(DEFAULT_SETTINGS)
    assert harness.single_doc_cap_ratio == DEFAULT_SETTINGS.exact_span_single_doc_cap_ratio
    assert harness.single_doc_cap_ratio is not None


def test_acceptance_harness_keeps_rejected_channels_off():
    """M5 REJECTED 的词法融合 / M5-Rerank 精排在出厂默认里默认关，harness 必须一致。"""

    harness = _real_checker(DEFAULT_SETTINGS)
    assert harness.lexical_discovery is None
    assert harness.reranker is None
