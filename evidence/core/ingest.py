"""低层 benchmark/test 建库：corpus → 切块 → 嵌入 → Chroma。

这是 build-time 步骤，调 embedding 模型的两处之一（另一处是查询）。两端必须同一个
模型（嵌入一致性铁律，见 ``core/embeddings``）。embedding 不是生成式 LLM，core 查询期
仍无生成式 LLM。

幂等：每次 ``build_index`` 重建集合（先删后建），避免半截/混模型残留。

局限（R10 诚实标注，不假装做了）：
  - 朴素切块：按行累积 + 字符预算 + 行重叠，**非 token 级、无 CJK 特化分句**。
  - 文档与代码暂用同一种切法（先简单可跑）。
  - 不做 contextual enrichment；该帮助函数只承担基础切块和索引。

正式项目材料应通过 L1 Library 和 ``evidence-gap-check`` 进入主链。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import Settings, DEFAULT_SETTINGS
from . import embeddings

COLLECTION_NAME = "corpus_vectors"

_CHUNK_CHARS = 800        # 每块字符预算（朴素"几百 token"近似）
_OVERLAP_LINES = 2        # 相邻块重叠行数
# 文本/代码扩展名允许表（朴素：只嵌文本，跳过二进制/未知格式）。
_TEXT_EXTS = {".md", ".txt", ".py", ".rst", ".json", ".yaml", ".yml",
              ".toml", ".ini", ".cfg", ".csv", ".html"}


@dataclass
class Chunk:
    source_path: str   # abspath
    text: str
    start_line: int    # 1-based
    end_line: int      # 1-based, 含
    chunk_index: int


def _iter_text_files(corpus_dir: str):
    for root, _dirs, files in os.walk(corpus_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in _TEXT_EXTS:
                yield os.path.join(root, fn)


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None  # 跳过二进制/不可读，不崩


def chunk_text(text: str, source_path: str) -> list[Chunk]:
    """朴素按行累积切块：字符预算 + 行重叠，记录起止行号（1-based）。"""
    lines = text.splitlines()
    n = len(lines)
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < n:
        start = i
        char_count = 0
        j = i
        while j < n and char_count < _CHUNK_CHARS:
            char_count += len(lines[j]) + 1
            j += 1
        body = "\n".join(lines[start:j]).strip()
        if body:
            chunks.append(Chunk(source_path, body, start + 1, j, idx))
            idx += 1
        if j >= n:
            break
        i = max(j - _OVERLAP_LINES, start + 1)  # 行重叠；start+1 保证前进
    return chunks


def build_index(settings: Settings = DEFAULT_SETTINGS, embedder=None) -> dict:
    """建库，返回 stats。

    ``embedder=None`` → 生产默认走 ``embeddings.load_embedder``（按 config 选后端，
    默认 API/LM Studio，可切本地 bge-m3）。测试经依赖注入传入确定性假 embedder（test-only，永不进生产）。
    """
    embedder = embedder or embeddings.load_embedder(settings)

    chunks: list[Chunk] = []
    corpus_dir = settings.corpus_dir
    if os.path.isdir(corpus_dir):
        for path in _iter_text_files(corpus_dir):
            text = _read_text(path)
            if text is None:
                continue
            chunks.extend(chunk_text(text, os.path.abspath(path)))

    import chromadb  # 懒加载：import core 不强依赖 chromadb
    client = chromadb.PersistentClient(path=settings.vector_db_path)
    try:
        client.delete_collection(COLLECTION_NAME)  # 幂等重建
    except Exception:
        pass
    col = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_id},
    )

    if chunks:
        vectors = embedder.encode([c.text for c in chunks])
        col.add(
            ids=[f"{c.source_path}::{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{
                "source_path": c.source_path,                 # 展示用
                "src_key": os.path.normcase(c.source_path),   # scope 过滤用规整 key
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_index": c.chunk_index,
            } for c in chunks],
        )

    return {
        "files": len({c.source_path for c in chunks}),
        "chunks": len(chunks),
        "embedding_model": embedder.model_id,
        "vector_db_path": settings.vector_db_path,
    }
