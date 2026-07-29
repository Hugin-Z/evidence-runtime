"""eval/bench/docrank.py — C-1 结构修复：document 评测单位 + 候选深度保障（bench 层，core/ 不动）。

**问题（v1）**：RRF 在 **chunk** 粒度融合，再折叠到 document 算指标。一个 passage 被切成多
chunk 时，① 多 chunk 各自占 RRF 名次、同 doc 重复贡献；② top-10 在 chunk 层截取，折叠后
document 可能不足 10。A-1 硬切后长文碎片化，此问题被放大。

**修复**：先折叠到 document，再融合。
  * 每路按 chunk **过取**，保序折叠出 ≥ `N_DOC` 个唯一 document（首个 chunk 定该 doc 位次）；
    `chunk_cap` 防长尾（扫够 cap 个 chunk 仍不足则止）。
  * 两路以相同 `N_DOC` 深度的 **document 排名**进 RRF：给每个 doc 造一个哨兵 `Snippet`
    （`source_path=doc_id`, `start_line=end_line=0`），喂 **未改动的** `core.fusion.fuse`——
    key=(doc_id,0,0) 跨路一致 → 同 doc 合并、每路每 doc 至多贡献一次。**不改 core/fusion.py**。
  * 融合后 `source_path` 即 doc_id；document top-10 截取即评测排名。
"""
from __future__ import annotations

import os
from typing import Callable

from evidence.core.types import Snippet, RetrievalPath

N_DOC = 20            # 折叠目标深度（进 RRF 的每路 document 数）
CHUNK_CAP = 200       # 单路过取 chunk 上限（防长尾：碎片化时不无限深挖）


def bench_vector_snippets(settings, embedder, collection_name: str, query: str,
                          k: int = CHUNK_CAP) -> list[Snippet]:
    """查含指纹的 bench 向量集合，返回 chunk 级 Snippet（按相似度降序，最多 k 条）。

    复用 core.embeddings 的一致性铁律；不经 core.VectorRetriever（其 collection 名写死）。
    """
    from evidence.core import embeddings
    db_path = settings.vector_db_path
    if not os.path.isdir(db_path):
        return []
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []
    if col.count() == 0:
        return []
    stored = (col.metadata or {}).get("embedding_model")
    embeddings.assert_model_match(stored, embedder)
    qvec = embedder.encode([query])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    ids = res.get("ids") or [[]]
    if not ids or not ids[0]:
        return []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(Snippet(
            source_path=meta.get("source_path", ""), text=doc, score=1.0 - float(dist),
            path=RetrievalPath.VECTOR, start_line=meta.get("start_line"),
            end_line=meta.get("end_line"),
            meta={"distance": float(dist), "char_start": meta.get("char_start")}))
    return out


def fold_to_docs(snippets: list[Snippet], id_of: Callable[[Snippet], str],
                 *, n_doc: int = N_DOC, chunk_cap: int = CHUNK_CAP) -> list[str]:
    """把 chunk 级 Snippet 折叠成唯一 document id 列表（保序，首见定位次）。

    扫描至多 `chunk_cap` 个 chunk；折叠出 `n_doc` 个唯一 doc 即止（候选不足则少于 n_doc）。
    """
    seen: set[str] = set()
    docs: list[str] = []
    for snip in snippets[:chunk_cap]:
        did = id_of(snip)
        if did in seen:
            continue
        seen.add(did)
        docs.append(did)
        if len(docs) >= n_doc:
            break
    return docs


def docs_to_fusion_input(doc_ids: list[str], path: RetrievalPath) -> list[Snippet]:
    """把有序 doc_id 列表转成 document 级哨兵 Snippet 列表，供 core.fusion.fuse。

    key=(doc_id,0,0)：同 doc 跨路合并；start_line/end_line=0 表 document 级（无行区间）。
    score 占位（fuse 只吃名次）。
    """
    return [Snippet(source_path=d, text="", score=0.0, path=path,
                    start_line=0, end_line=0, meta={}) for d in doc_ids]


def fused_doc_ids(fused: list[Snippet]) -> list[str]:
    """融合结果（document 哨兵）→ 有序 doc_id 列表（source_path 即 doc_id）。"""
    return [s.source_path for s in fused]
