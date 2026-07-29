"""词法路（RipgrepRetriever）测试 —— fixture-first。

语料见 tests/fixtures/lexical_corpus/README.md（R10 限制声明）：合成、确定性字面命中
语料；预期 (文件, 行号, token) 由人工读语料按字面匹配规格写定，**不由运行被测函数反喂**。

本测试只断言机械可验证的事实：哪个文件、哪一行、命中的确切 token、命中数。
**不**断言相关度排序 / "哪条是最佳答案" —— 那是语义判断，属 rerank 的职责（见 rerank.py）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from evidence.core.config import Settings
from evidence.core.types import RetrievalPath, Snippet
from evidence.core.retrievers import lexical
from evidence.core.retrievers.lexical import RipgrepRetriever

CORPUS = Path(__file__).parent / "fixtures" / "lexical_corpus"


def _settings(**kw) -> Settings:
    return Settings(corpus_dir=str(CORPUS), **kw)


def _bn(s: Snippet) -> str:
    """跨平台取文件名（rg 在 Windows 上返回混合分隔符）。"""
    return os.path.basename(s.source_path.replace("\\", "/"))


def _by_file(snips, name):
    return [s for s in snips if _bn(s) == name]


# ── 命中：文件 / 行号 / token（机械事实）─────────────────────────────

def test_unique_token_exact_file_and_line():
    out = RipgrepRetriever(_settings(), context_lines=0).search("zqxmarker")
    assert len(out) == 1
    s = out[0]
    assert _bn(s) == "alpha.md"
    assert s.path is RetrievalPath.LEXICAL
    assert s.start_line == 4 and s.end_line == 4
    assert s.meta["match_lines"] == [4]
    assert s.meta["matched_keywords"] == ["zqxmarker"]
    assert s.score == 1.0  # 1/1 个查询关键词命中（机械计数比，非相关度）


def test_multi_keyword_score_is_match_ratio():
    out = RipgrepRetriever(_settings(), context_lines=0).search("retry strategy")
    assert len(out) == 3  # alpha / beta / gamma
    alpha = _by_file(out, "alpha.md")[0]
    beta = _by_file(out, "beta.py")[0]
    gamma = _by_file(out, "gamma.md")[0]
    # alpha、beta 字面含两个关键词 -> 2/2；gamma 只含 retry -> 1/2
    assert alpha.score == 1.0
    assert sorted(alpha.meta["matched_keywords"]) == ["retry", "strategy"]
    assert beta.score == 1.0
    assert gamma.score == 0.5
    assert gamma.meta["matched_keywords"] == ["retry"]
    assert gamma.meta["keyword_hits"] == 1 and gamma.meta["keyword_total"] == 2


def test_adjacent_hits_merged_into_range():
    out = RipgrepRetriever(_settings(), context_lines=0).search("retry")
    gamma = _by_file(out, "gamma.md")
    assert len(gamma) == 1  # 第 2、3 行相邻命中合并成一条区间
    s = gamma[0]
    assert s.start_line == 2 and s.end_line == 3
    assert s.meta["match_lines"] == [2, 3]
    assert "retry" in s.text.lower()


def test_context_lines_expand_snippet_range():
    out = RipgrepRetriever(_settings(), context_lines=1).search("zqxmarker")
    s = _by_file(out, "alpha.md")[0]
    assert s.meta["match_lines"] == [4]            # 命中行仍是第 4 行
    assert s.start_line == 3 and s.end_line == 4   # 纳入上文 1 行；文件共 4 行，无下文
    assert "zqxmarker" in s.text
    assert "strategy" in s.text.lower()            # 第 3 行上下文被纳入片段文本


# ── 负例 / 降级（dirty fixture，确定性）────────────────────────────

def test_lexical_misses_synonym_only_document():
    """语义相关但字面不含查询词的文档，词法路必然漏掉 —— 双路 A/B 的证据资产。"""
    r = RipgrepRetriever(_settings(), context_lines=0)
    hit_retry = r.search("retry")
    assert _by_file(hit_retry, "synonym_miss.md") == []  # 同义改写被漏掉
    assert _by_file(hit_retry, "delta.md") == []         # 无关文档不误命中
    # 反向坐实：该文档确在语料中、且字面写了 backoff —— 漏掉是因字面不符，非文件缺失
    syn = _by_file(r.search("backoff"), "synonym_miss.md")
    assert len(syn) >= 1
    assert 1 in syn[0].meta["match_lines"]


def test_stopwords_dropped_all_stopword_query_empty():
    # the/a/an/of/to/and 都在语料里出现；若不剔停用词必有命中
    assert RipgrepRetriever(_settings()).search("the a an of to and") == []


def test_missing_corpus_returns_empty(tmp_path):
    r = RipgrepRetriever(Settings(corpus_dir=str(tmp_path / "does_not_exist")))
    assert r.search("retry") == []


def test_empty_corpus_returns_empty(tmp_path):
    assert RipgrepRetriever(Settings(corpus_dir=str(tmp_path))).search("retry") == []


def test_no_hit_query_returns_empty():
    assert RipgrepRetriever(_settings()).search("nonexistentokenxyzzy") == []


def test_respects_ignore_file_default():
    out = RipgrepRetriever(_settings(), context_lines=0).search("retry")
    assert _by_file(out, "ignored_secret.md") == []  # .ignore 规则默认跳过


def test_skips_binary_files_default(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\x00\x00 retry value \x00 more\x00")
    (tmp_path / "plain.md").write_text("retry in a normal file\n", encoding="utf-8")
    out = RipgrepRetriever(Settings(corpus_dir=str(tmp_path)), context_lines=0).search("retry")
    assert _by_file(out, "plain.md")        # 文本文件命中
    assert _by_file(out, "data.bin") == []  # 二进制默认跳过


def test_rg_missing_degrades_to_empty_with_warning(monkeypatch):
    monkeypatch.setattr(lexical.shutil, "which", lambda *_a, **_k: None)
    r = RipgrepRetriever(_settings())
    with pytest.warns(RuntimeWarning):
        out = r.search("retry")
    assert out == []


# ── 截断 ───────────────────────────────────────────────────────────

def test_lexical_top_k_truncation():
    out = RipgrepRetriever(_settings(lexical_top_k=2), context_lines=0).search("retry")
    assert len(out) == 2  # retry 命中 alpha/beta/gamma 共 3 条，截到 2


def test_k_arg_overrides_setting():
    out = RipgrepRetriever(_settings(lexical_top_k=20), context_lines=0).search("retry", k=1)
    assert len(out) == 1
