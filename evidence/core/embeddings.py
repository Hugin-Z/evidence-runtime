"""Embedding 模型加载 + 嵌入一致性铁律。

ingest（建库）与 vector（查询）共用 ``load_embedder``，保证两端用同一个 embedding
后端/模型——这是"嵌入一致性铁律"的单一入口。换后端/模型必须重建库；本模块把这条铁律
做成**运行时强制**（库里存 ``embedding_model``，查询时对不上就 raise，见 ``assert_model_match``），
不只 docstring 口头。

三类 Embedder（R10 口径，别混成一句"已支持 embedding"）：
  ① ``APIEmbedder`` —— **真做且生产默认**：POST 到 OpenAI 兼容 ``/v1/embeddings``
     （默认指向本地 LM Studio，``settings.embedding_api_*``）。生产默认就是它
     （``embedding_backend="api"``），不走假 embedder。
  ② ``LocalEmbedder`` —— **保留可选**：本地 sentence-transformers（bge-m3，``settings.embedding_model``）。
     ``embedding_backend="local"`` 时启用；本环境未验证，代码在。
  ③ 测试假 embedder —— **仅测试注入**（见 ``tests/conftest.py`` 的 FakeConceptEmbedder），
     确定性概念向量，**永不进生产代码路径**，不代表真模型语义质量。

注：embedding 不是生成式 LLM；建库在 build-time，core 查询期仍无生成式 LLM——
切 API 后端只是把 embedding 从进程内换成本地 HTTP 服务，不改"core 无生成式 LLM"这条；
代价是建库/查询依赖该 embedding 服务在线（默认 LM Studio）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence, runtime_checkable

from .config import Settings, DEFAULT_SETTINGS


@runtime_checkable
class Embedder(Protocol):
    """编码器接口（本地/API 可切换，统一这一个形状）。"""

    model_id: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class APIEmbedder:
    """① 真做且生产默认：OpenAI 兼容 embedding API（默认本地 LM Studio）。

    POST ``{base_url}/embeddings``，body ``{"model": model_id, "input": [texts]}``，
    解析 ``data[*].embedding`` 并**按 ``data[*].index`` 保序**返回（响应顺序不保证，
    按位置取会把向量错配到文本——静默污染库，故必须按 index 归位 + 校验数量）。
    批量：一次传整批 texts（建库一批一批传，别一条一条调）。
    ``model_id`` = ``settings.embedding_api_model``，即写进向量库做一致性键的标识。

    降级（非静默，绝不返回空/假向量）：连接不上 / HTTP 错误 / 响应异常 →
    清晰 ``RuntimeError``，提示"确认 LM Studio 在运行且已加载 embedding 模型"。
    """

    def __init__(self, model_id: str, base_url: str,
                 api_key: str | None = None, timeout: float = 60.0):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []  # 空输入短路，不发请求
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model_id, "input": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # 空 key → 不发 Authorization（LM Studio 可空）
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        hint = "请确认 LM Studio 在运行且已加载 embedding 模型。"
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:   # 4xx/5xx（如模型没 load）——须先于 URLError 捕获
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                body = ""
            raise RuntimeError(
                f"调用 embedding API 失败（{url}, model={self.model_id!r}）：HTTP {e.code} {body}。{hint}"
            ) from e
        except urllib.error.URLError as e:    # 连接拒绝 / 超时（LM Studio 没开）
            raise RuntimeError(
                f"连接 embedding API 失败（{url}, model={self.model_id!r}）：{e.reason}。{hint}"
            ) from e
        return self._parse(doc, len(texts), url)

    @staticmethod
    def _parse(doc, n: int, url: str) -> list[list[float]]:
        data = doc.get("data") if isinstance(doc, dict) else None
        if not isinstance(data, list) or len(data) != n:
            got = len(data) if isinstance(data, list) else type(data).__name__
            raise RuntimeError(
                f"embedding API 响应异常（{url}）：期望 {n} 个向量，实得 {got}。响应：{str(doc)[:200]}"
            )
        out: list = [None] * n
        try:
            for item in data:
                out[int(item["index"])] = [float(x) for x in item["embedding"]]
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise RuntimeError(
                f"embedding API 响应解析失败（{url}）：{e}。响应：{str(doc)[:200]}"
            ) from e
        if any(v is None for v in out):
            raise RuntimeError(f"embedding API 响应缺少部分 index（{url}）。响应：{str(doc)[:200]}")
        return out


class LocalEmbedder:
    """② 保留可选：本地 bge-m3（sentence-transformers）。

    ``embedding_backend="local"`` 时启用（生产默认是 APIEmbedder，不是它）。
    懒加载 sentence-transformers，避免 ``import core`` 即拖入 torch 等重依赖。
    加载失败 → 清晰 ``RuntimeError``（不静默吞、不返回假向量）。
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:  # 依赖缺失
                raise RuntimeError(
                    f"加载本地 embedding 模型失败：缺 sentence-transformers（{e}）。"
                    f"装：pip install sentence-transformers。模型：{self.model_id!r}"
                ) from e
            try:
                self._model = SentenceTransformer(self.model_id)
            except Exception as e:  # 权重下载/加载失败
                raise RuntimeError(
                    f"加载本地 embedding 模型 {self.model_id!r} 失败：{e}"
                ) from e
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


def load_embedder(settings: Settings = DEFAULT_SETTINGS) -> Embedder:
    """按 ``settings.embedding_backend`` 返回生产 embedder（默认 "api" = LM Studio）。

    **绝不返回假 embedder**（假 embedder 仅测试注入），**也不静默回退**：
    未知 backend → ``ValueError``（点名错值 + 合法选项）。
    一致性铁律：换后端/模型后 ``embedder.model_id`` 变，旧库查询会 raise → 须重跑 ingest。
    """
    backend = settings.embedding_backend
    if backend == "api":
        return APIEmbedder(
            model_id=settings.embedding_api_model,
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key or None,
            timeout=settings.embedding_api_timeout,
        )
    if backend == "local":
        return LocalEmbedder(settings.embedding_model)
    raise ValueError(f"未知 embedding_backend: {backend!r}（应为 'api' 或 'local'）")


def assert_model_match(stored_model_id: str | None, embedder: Embedder) -> None:
    """嵌入一致性铁律（运行时强制）。

    向量库建库用的模型 id 必须与当前查询 embedder 一致；不一致 → ``RuntimeError``。
    换后端/模型必须重建库（重跑 ingest）。``stored is None``（旧库无此元数据）放行，不臆断。
    """
    if stored_model_id is not None and stored_model_id != embedder.model_id:
        raise RuntimeError(
            f"嵌入一致性破坏：向量库用 {stored_model_id!r} 建库，"
            f"却用 {embedder.model_id!r} 查询。换后端/模型必须重建库（重跑 ingest）。"
        )
