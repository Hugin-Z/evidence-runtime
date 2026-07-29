"""APIEmbedder + load_embedder 单测（全程 mock HTTP，离线确定性）。

覆盖：请求体构造(model+input) / 按 data[].index 保序解析 / len 守卫 /
空输入短路不发请求 / 连接失败·HTTP错误·响应异常 → 清晰 RuntimeError(非静默) /
api_key 空则不发 Authorization / load_embedder 后端开关(默认 api) / 一致性守卫。

R10：这些测的是 mock 路径与开关逻辑，**不**证明真 LM Studio 端点（沙箱无 LM Studio）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from evidence.core import Settings
from evidence.core import embeddings
from evidence.core.embeddings import APIEmbedder, load_embedder


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fake(monkeypatch, canned, capture):
    def fake_urlopen(req, timeout=None):
        capture["url"] = req.full_url
        capture["body"] = json.loads(req.data.decode("utf-8"))
        capture["headers"] = list(req.header_items())
        capture["timeout"] = timeout
        capture["called"] = capture.get("called", 0) + 1
        return _FakeResp(json.dumps(canned).encode("utf-8"))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _emb(**kw):
    return APIEmbedder(
        model_id=kw.get("model_id", "text-embedding-qwen3-embedding-0.6b"),
        base_url=kw.get("base_url", "http://127.0.0.1:1234/v1"),
        api_key=kw.get("api_key"),
        timeout=kw.get("timeout", 60.0),
    )


# ── 请求构造 + 解析 ──────────────────────────────────────────────
def test_request_body_and_url(monkeypatch):
    cap = {}
    canned = {"object": "list", "data": [
        {"index": 0, "embedding": [0.1, 0.2]},
        {"index": 1, "embedding": [0.3, 0.4]},
    ], "model": "text-embedding-qwen3-embedding-0.6b"}
    _install_fake(monkeypatch, canned, cap)
    out = _emb().encode(["a", "b"])
    assert cap["body"] == {"model": "text-embedding-qwen3-embedding-0.6b", "input": ["a", "b"]}
    assert cap["url"].endswith("/v1/embeddings")
    assert out == [[0.1, 0.2], [0.3, 0.4]]


def test_input_order_preserved_despite_shuffled_indices(monkeypatch):
    cap = {}
    canned = {"data": [
        {"index": 2, "embedding": [2.0]},
        {"index": 0, "embedding": [0.0]},
        {"index": 1, "embedding": [1.0]},
    ]}
    _install_fake(monkeypatch, canned, cap)
    out = _emb().encode(["x", "y", "z"])
    assert out == [[0.0], [1.0], [2.0]]   # 按输入顺序，不是响应顺序


def test_returns_floats_and_len(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]}, cap)
    out = _emb().encode(["only"])
    assert len(out) == 1
    assert all(isinstance(v, float) for v in out[0])   # int→float 归一


# ── 降级（非静默）────────────────────────────────────────────────
def test_empty_input_short_circuits_no_http(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"data": []}, cap)
    assert _emb().encode([]) == []
    assert cap.get("called", 0) == 0    # 没发请求


def test_connection_refused_raises_clear(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="确认 LM Studio 在运行且已加载 embedding 模型"):
        _emb().encode(["a"])


def test_http_error_raises_clear(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="确认 LM Studio"):
        _emb().encode(["a"])


def test_len_mismatch_raises(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"data": [{"index": 0, "embedding": [0.1]}]}, cap)
    with pytest.raises(RuntimeError):
        _emb().encode(["a", "b"])   # 1 向量 vs 2 输入


def test_malformed_response_raises(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"oops": "no data"}, cap)
    with pytest.raises(RuntimeError):
        _emb().encode(["a"])


# ── Authorization 头 ─────────────────────────────────────────────
def test_api_key_sends_bearer(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"data": [{"index": 0, "embedding": [0.1]}]}, cap)
    _emb(api_key="sk-test").encode(["a"])
    auth = {k.lower(): v for k, v in cap["headers"]}
    assert auth.get("authorization") == "Bearer sk-test"


def test_empty_api_key_omits_authorization(monkeypatch):
    cap = {}
    _install_fake(monkeypatch, {"data": [{"index": 0, "embedding": [0.1]}]}, cap)
    _emb(api_key="").encode(["a"])
    keys = {k.lower() for k, _ in cap["headers"]}
    assert "authorization" not in keys


# ── load_embedder 后端开关（默认 = API）────────────────────────────
def test_default_backend_is_api_embedder():
    emb = load_embedder(Settings())
    assert type(emb).__name__ == "APIEmbedder"                       # ① 真做且默认
    assert emb.model_id == "text-embedding-qwen3-embedding-0.6b"     # provenance 钉在 embedding_api_model


def test_local_backend_returns_local_embedder():
    emb = load_embedder(Settings(embedding_backend="local"))
    assert type(emb).__name__ == "LocalEmbedder"                     # ② 保留可选
    assert emb.model_id == "BAAI/bge-m3"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        load_embedder(Settings(embedding_backend="bogus"))           # 无声默回退是 R10 违规 → 必 raise


def test_consistency_guard_with_api_embedder():
    a = APIEmbedder("model-A", "http://127.0.0.1:1234/v1")
    with pytest.raises(RuntimeError):
        embeddings.assert_model_match("model-B", a)                  # 铁律对 API 后端同样生效
