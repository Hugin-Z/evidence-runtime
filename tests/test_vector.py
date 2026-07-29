"""向量路测试 —— fixture-first。证明：
  * 语义召回：查 retry 能召回字面无 retry 的 backoff 文档（ripgrep 漏的，向量捞回）
  * scope 缝：限定到某文件时只从该文件返回（Chroma 元数据过滤）
  * 降级：空库/缺失不崩
  * 一致性铁律：建库模型 id 与查询 embedder 不符 → raise
  * LocalEmbedder 加载失败必须清晰 raise（②保留可选；APIEmbedder/后端开关/三类口径见 test_embeddings.py）

只断言机械事实，不断言相关度排序（排序留给 rerank）。
fake embedder 与语料的限制声明见 fixtures/vector_corpus/README.md。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from evidence.core import Settings
from evidence.core.types import RetrievalPath
from evidence.core.scope import Scope
from evidence.core import ingest, embeddings
from evidence.core.retrievers.vector import VectorRetriever
from evidence.core.retrievers.lexical import RipgrepRetriever

VCORPUS = Path(__file__).parent / "fixtures" / "vector_corpus" / "data"  # README 在 data/ 外，不入库


def _bn(s) -> str:
    return os.path.basename(s.source_path.replace("\\", "/"))


def _build(tmp_path, embedder, corpus=VCORPUS) -> Settings:
    settings = Settings(corpus_dir=str(corpus), vector_db_path=str(tmp_path / "chroma"))
    ingest.build_index(settings, embedder=embedder)
    return settings


# ── 语义召回：向量捞回 ripgrep 漏的同义文档 ──────────────────────
def test_semantic_recall_finds_synonym_doc(tmp_path, fake_embedder):
    settings = _build(tmp_path, fake_embedder)
    out = VectorRetriever(settings, embedder=fake_embedder).search("retry")
    names = {_bn(s) for s in out}
    assert "backoff_doc.md" in names          # 字面无 'retry'，向量靠语义召回（存在理由）
    assert "retry_doc.md" in names
    assert all(s.path is RetrievalPath.VECTOR for s in out)
    assert all(isinstance(s.score, float) for s in out)
    assert all(s.start_line is not None and s.end_line is not None for s in out)


def test_lexical_misses_what_vector_catches(tmp_path):
    # 对照（双路 A/B 证据）：ripgrep 查 'retry' 漏掉只含 backoff/requeue 的文档
    lex = RipgrepRetriever(Settings(corpus_dir=str(VCORPUS)), context_lines=0).search("retry")
    assert "backoff_doc.md" not in {_bn(s) for s in lex}


# ── scope 缝：元数据过滤限定到单文件 ─────────────────────────────
def test_scope_restricts_to_single_file(tmp_path, fake_embedder):
    settings = _build(tmp_path, fake_embedder)
    only = os.path.abspath(str(VCORPUS / "retry_doc.md"))
    out = VectorRetriever(settings, embedder=fake_embedder).search("retry", scope=Scope(files=[only]))
    names = {_bn(s) for s in out}
    assert names == {"retry_doc.md"}            # 只从 scope 内文件返回
    assert "backoff_doc.md" not in names        # 语义近也被 scope 挡掉
    assert len(out) >= 1


def test_scope_empty_list_returns_empty(tmp_path, fake_embedder):
    settings = _build(tmp_path, fake_embedder)
    out = VectorRetriever(settings, embedder=fake_embedder).search("retry", scope=Scope(files=[]))
    assert out == []


# ── 降级：空库/缺失不崩 ──────────────────────────────────────────
def test_missing_db_returns_empty(tmp_path, fake_embedder):
    settings = Settings(corpus_dir=str(VCORPUS), vector_db_path=str(tmp_path / "nope"))
    assert VectorRetriever(settings, embedder=fake_embedder).search("retry") == []


def test_empty_corpus_builds_empty_db_returns_empty(tmp_path, fake_embedder):
    empty = tmp_path / "empty"
    empty.mkdir()
    settings = _build(tmp_path, fake_embedder, corpus=empty)
    assert VectorRetriever(settings, embedder=fake_embedder).search("retry") == []


# ── 一致性铁律：建库模型 id 与查询 embedder 不符 → raise ─────────
def test_consistency_guard_raises_on_model_mismatch(tmp_path, fake_embedder):
    settings = _build(tmp_path, fake_embedder)                 # 库用 fake-concept-v1 建
    other = type(fake_embedder)(model_id="some-other-model")   # 用不同 id 查
    with pytest.raises(RuntimeError):
        VectorRetriever(settings, embedder=other).search("retry")


# ── ingest 建库：provenance 落进元数据 ───────────────────────────
def test_ingest_builds_index_with_provenance(tmp_path, fake_embedder):
    settings = _build(tmp_path, fake_embedder)
    stats = ingest.build_index(settings, embedder=fake_embedder)  # 幂等重建
    assert stats["files"] == 3
    assert stats["chunks"] == 3                 # 每个小文件 1 块
    assert stats["embedding_model"] == "fake-concept-v1"
    out = VectorRetriever(settings, embedder=fake_embedder).search("retry")
    s = next(x for x in out if _bn(x) == "retry_doc.md")
    assert s.start_line == 1 and s.end_line >= 1


# ── LocalEmbedder 加载失败必须清晰（② 保留可选；APIEmbedder/后端开关/三类口径见 test_embeddings.py）──
def test_local_embedder_load_failure_is_loud(tmp_path):
    # 本沙箱无 sentence-transformers，正好覆盖"加载失败必须清晰 raise，不静默"
    with pytest.raises(RuntimeError):
        embeddings.LocalEmbedder("BAAI/bge-m3").encode(["x"])
