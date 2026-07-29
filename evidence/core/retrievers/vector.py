"""语义路：embedding + 向量库（Chroma）。

查询期：query --(同一个 embedding 模型)--> 向量 --> Chroma 最近邻 --> Snippet。
向量库由 ingest 在 build-time 建好（切块→嵌入→存）。embedding 不是生成式 LLM，
core 查询期仍无生成式 LLM。这条路负责"语义/同义"——ripgrep 抓不到的换说法，靠它召回。

embedder：默认 ``None`` → 生产走 ``core.embeddings.load_embedder``（按 config 选后端，
  默认 API/LM Studio，可切本地 bge-m3）。假 embedder 仅测试经依赖注入传入，**永不进生产路径**。
嵌入一致性铁律（运行时强制）：库建库用的模型 id 与查询 embedder 不符 → ``RuntimeError``
  （换后端/模型必须重跑 ingest 重建库）。
降级：库不存在/集合不存在/空库 → ``[]``（已记录的退化，与词法路缺语料对称，安静返回）；
  embedding 后端加载/调用失败 → 清晰 ``RuntimeError``（不静默崩，见 ``core/embeddings``）。
score = ``1 - cosine_distance``（**仅初排用**，最终相关度由 ``rerank`` 决定）。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Mapping

from ..config import Settings, DEFAULT_SETTINGS
from ..types import Snippet, RetrievalPath
from .. import embeddings
from ..ingest import COLLECTION_NAME

if TYPE_CHECKING:  # 仅类型，不引入运行时耦合
    from ..scope import Scope
    from ..embeddings import Embedder


def query_vector_collection(
    settings: Settings,
    embedder: "Embedder",
    collection_name: str,
    query: str,
    *,
    k: int,
    where: Mapping[str, object] | None = None,
    require_collection: bool = False,
    required_metadata: Mapping[str, object] | None = None,
    expected_count: int | None = None,
) -> Mapping[str, object] | None:
    """查询一个明确命名的 Chroma collection，返回原始查询结果。

    这是 ``VectorRetriever.search`` 与正式材料链共享的 Chroma 访问、模型一致性检查和查询入口。
    低层检索器保留“未建库返回空”的行为；正式材料链通过
    ``require_collection=True`` 将缺库、缺元数据和数量不一致升级为明确错误。
    本函数不解释 distance，也不把结果转换成 ``Snippet``。
    """

    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(
            f"collection_name must be a non-empty string; got {collection_name!r}"
        )
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string; got {query!r}")
    if type(k) is not int or k < 1:
        raise ValueError(f"k must be an integer >= 1; got {k!r}")
    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 0
    ):
        raise ValueError(
            f"expected_count must be None or an integer >= 0; got {expected_count!r}"
        )

    db_path = settings.vector_db_path
    if not os.path.isdir(db_path):
        if require_collection:
            raise RuntimeError(f"vector database does not exist: {db_path!r}")
        return None

    import chromadb  # 懒加载

    client = chromadb.PersistentClient(path=db_path)
    try:
        col = client.get_collection(collection_name)
    except Exception as exc:
        if require_collection:
            raise RuntimeError(
                f"vector collection {collection_name!r} does not exist in {db_path!r}"
            ) from exc
        return None

    collection_metadata = dict(col.metadata or {})
    stored = collection_metadata.get("embedding_model")
    if require_collection and stored is None:
        raise RuntimeError(
            f"vector collection {collection_name!r} is missing embedding_model metadata"
        )
    embeddings.assert_model_match(stored, embedder)

    for key, expected in dict(required_metadata or {}).items():
        if key not in collection_metadata:
            raise RuntimeError(
                f"vector collection {collection_name!r} is missing required metadata {key!r}"
            )
        actual = collection_metadata[key]
        if actual != expected:
            raise RuntimeError(
                f"vector collection {collection_name!r} metadata mismatch for {key!r}: "
                f"expected {expected!r}, got {actual!r}"
            )

    count = col.count()
    if expected_count is not None and count != expected_count:
        raise RuntimeError(
            f"vector collection {collection_name!r} count mismatch: "
            f"expected {expected_count}, got {count}"
        )
    if count == 0:
        return None

    vectors = embedder.encode([query])
    if len(vectors) != 1 or not vectors[0]:
        raise RuntimeError(
            f"embedding backend {embedder.model_id!r} must return one non-empty query vector"
        )
    kwargs = {
        "query_embeddings": [vectors[0]],
        "n_results": min(k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    return col.query(**kwargs)


class VectorRetriever:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS,
                 embedder: "Embedder | None" = None):
        self.settings = settings
        self._embedder = embedder  # None → 懒加载 load_embedder（生产默认 API/LM Studio）

    def _get_embedder(self) -> "Embedder":
        if self._embedder is None:
            self._embedder = embeddings.load_embedder(self.settings)
        return self._embedder

    def search(self, query: str, k: int | None = None,
               scope: "Scope | None" = None) -> list[Snippet]:
        k = k or self.settings.vector_top_k

        # scope 限定到空集 → 空
        if scope is not None and scope.files is not None and len(scope.files) == 0:
            return []

        embedder = self._get_embedder()
        where = self._scope_where(scope)
        res = query_vector_collection(
            self.settings,
            embedder,
            COLLECTION_NAME,
            query,
            k=k,
            where=where,
        )
        if res is None:
            return []
        return self._to_snippets(res)

    # —— 内部 ——

    @staticmethod
    def _scope_where(scope: "Scope | None"):
        """scope.files → Chroma 元数据过滤（只在这些文件的向量里检索）。"""
        if scope is None or scope.files is None:
            return None
        keys = [os.path.normcase(os.path.abspath(f)) for f in scope.files]
        return {"src_key": {"$in": keys}}

    @staticmethod
    def _to_snippets(res) -> list[Snippet]:
        ids = res.get("ids") or [[]]
        if not ids or not ids[0]:
            return []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        out: list[Snippet] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(Snippet(
                source_path=meta.get("source_path", ""),
                text=doc,
                score=1.0 - float(dist),  # cosine 距离→相似度（仅初排，rerank 决定终值）
                path=RetrievalPath.VECTOR,
                start_line=meta.get("start_line"),
                end_line=meta.get("end_line"),
                meta={
                    "distance": float(dist),
                    "chunk_index": meta.get("chunk_index"),
                },
            ))
        return out
