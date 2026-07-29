"""eval/bench/index.py — 分批、可续的向量建库（大语料用）。

分批嵌入 + 断点续传（已入库 chunk id 跳过）。用 **bench 层硬切切块器**（`eval.bench.chunker`，
A-1 修复：单行超上限硬切、char_start 唯一 identity），core/ 不动。新索引落**含 chunker 指纹的
collection**（`corpus_vectors_<CHUNKER_VERSION>`），与 v1 的 `corpus_vectors` 物理隔离、可并存。

**建库硬断言**：任一 chunk.text 长度 > `MAX_CHUNK_CHARS` → 立即 raise（保证无 chunk 超上限 →
给定服务端 context ≥ 上限对应 token 数，embedding 不会被服务端截断）。日志打印 `[chunk-guard]`
供跑批脚本 grep 截断风险（须 0）。纯空白片（strip=="") 不入库。
"""
from __future__ import annotations

import os
import time

from eval.bench.chunker import (chunk_text, chunk_id, normalize,
                                MAX_CHUNK_CHARS, CHUNKER_VERSION)
from evidence.core.ingest import COLLECTION_NAME as _BASE_COLLECTION, _iter_text_files, _read_text

COLLECTION_NAME = f"{_BASE_COLLECTION}_{CHUNKER_VERSION}"   # 含 chunker 指纹，与 v1 隔离


def _encode_with_retry(embedder, texts, *, retries: int = 6, backoff: float = 5.0):
    """对 embedder.encode 加重试（LM Studio 长跑偶发连接超时）。同批幂等重试；耗尽才抛。

    col.add 只在 encode 成功后执行，故重试不会重复入库。配合稳定 work-dir 的 resume，
    彻底失败也可重跑续建。
    """
    last = None
    for attempt in range(retries + 1):
        try:
            return embedder.encode(texts)
        except Exception as e:      # embeddings.encode 把连接错误包成 RuntimeError
            last = e
            if attempt < retries:
                wait = backoff * (attempt + 1)
                print(f"  [index] encode 失败（{e}），{wait:.0f}s 后重试 {attempt+1}/{retries}", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"embedding encode 连续失败 {retries + 1} 次: {last}")


def collect_chunks(corpus_dir: str) -> list:
    """收集全语料 chunk（硬切版），并跳过纯空白片。"""
    chunks = []
    if os.path.isdir(corpus_dir):
        for path in _iter_text_files(corpus_dir):
            text = _read_text(path)
            if text is None:
                continue
            for c in chunk_text(text, os.path.abspath(path)):
                if c.text.strip():         # 纯空白片不入库（不破坏 chunker 划分不变式）
                    chunks.append(c)
    return chunks


def build_index_batched(settings, embedder, *, batch_size: int = 32,
                        resume: bool = True, log_every: int = 5000) -> dict:
    """分批建库（落含指纹的 collection）。resume=True 跳过已存在 chunk id（可续）。"""
    import chromadb
    chunks = collect_chunks(settings.corpus_dir)

    # —— A-1 建库硬断言：无 chunk 超上限 ——
    over = [c for c in chunks if len(c.text) > MAX_CHUNK_CHARS]
    max_len = max((len(c.text) for c in chunks), default=0)
    print(f"  [chunk-guard] chunker={CHUNKER_VERSION} max_chunk_chars={MAX_CHUNK_CHARS} "
          f"embedded_chunks={len(chunks)} max_len={max_len} over_limit={len(over)}", flush=True)
    if over:
        raise AssertionError(f"[chunk-guard] {len(over)} 个 chunk 超上限 {MAX_CHUNK_CHARS}（最长 {max_len}）——硬切失效，停")

    client = chromadb.PersistentClient(path=settings.vector_db_path)
    if not resume:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    try:
        col = client.get_collection(COLLECTION_NAME)
        stored = (col.metadata or {}).get("embedding_model")
        if stored and stored != embedder.model_id:
            raise RuntimeError(f"resume 模型不一致：库 {stored!r} != 当前 {embedder.model_id!r}（用 resume=False 重建）")
    except RuntimeError:
        raise
    except Exception:
        col = client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine",
                                       "embedding_model": embedder.model_id,
                                       "chunker": CHUNKER_VERSION})

    existing = set(col.get(include=[])["ids"]) if resume else set()
    todo = [c for c in chunks if chunk_id(c) not in existing]
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), batch_size):
        b = todo[i:i + batch_size]
        vecs = _encode_with_retry(embedder, [c.text for c in b])
        col.add(
            ids=[chunk_id(c) for c in b], embeddings=vecs, documents=[c.text for c in b],
            metadatas=[{"source_path": c.source_path, "src_key": os.path.normcase(c.source_path),
                        "start_line": c.start_line, "end_line": c.end_line,
                        "char_start": c.char_start, "chunk_index": c.chunk_index} for c in b],
        )
        done += len(b)
        if done % log_every < batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  [index] {len(existing)+done}/{len(chunks)} chunks  ({rate:.1f} chunk/s)", flush=True)
    return {"files": len({c.source_path for c in chunks}), "chunks": len(chunks),
            "added": done, "skipped": len(chunks) - len(todo),
            "elapsed_s": round(time.time() - t0, 1), "collection": COLLECTION_NAME,
            "chunker": CHUNKER_VERSION, "max_chunk_chars": MAX_CHUNK_CHARS, "max_len": max_len,
            "over_limit": len(over),
            "embedding_model": embedder.model_id, "vector_db_path": settings.vector_db_path}
