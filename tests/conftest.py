"""测试夹具（test-only）。

含**仅测试注入**的 ``FakeConceptEmbedder``：手写确定性概念向量，让"向量路召回
语义近邻"成为离线可复现的机械事实。它**永不进生产代码路径**——生产默认走真 embedding
后端（默认 API/LM Studio，见 ``core/embeddings.load_embedder``）。详见 ``fixtures/vector_corpus/README.md``。

本文件顶层不导入 core 的向量模块，避免测试收集阶段建立外部服务依赖。
"""
from __future__ import annotations

from pathlib import Path

import pytest

# 实际被 ingest 的语料在 data/ 子目录；README.md 是 meta 声明，刻意放在 data/ 外，不入库。
VECTOR_CORPUS = Path(__file__).parent / "fixtures" / "vector_corpus" / "data"

from tests.fakes import FakeConceptEmbedder


@pytest.fixture
def fake_embedder():
    return FakeConceptEmbedder()


@pytest.fixture
def built_settings(tmp_path, fake_embedder, monkeypatch):
    """建好一个 tmp 向量库（用 fake embedder），返回指向它的 Settings。

    同时把 ``core.embeddings.load_embedder`` 打桩成 fake，使测试中的低层检索器与 ingest
    使用同一个确定性模型。
    """
    from evidence.core import embeddings, ingest          # 懒导入：保证 conftest 本身 import-safe
    from evidence.core.config import Settings

    monkeypatch.setattr(embeddings, "load_embedder", lambda settings=None: fake_embedder)
    db = str(tmp_path / "chroma")
    settings = Settings(corpus_dir=str(VECTOR_CORPUS), vector_db_path=db)
    ingest.build_index(settings, embedder=fake_embedder)
    return settings
