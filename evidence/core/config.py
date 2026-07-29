"""全局设置。

三个地基参数（embedding / 向量库 / 语料目录）定在这里。embedding 后端**可切换**：
``embedding_backend`` 选 "api"（默认，OpenAI 兼容 API，如本地 LM Studio）或 "local"
（sentence-transformers 本地模型）。一旦写库就别中途换后端/模型 —— 换了得重嵌整个库
（嵌入一致性铁律，运行时由 ``core/embeddings.assert_model_match`` 强制）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    # —— 语料 ——
    corpus_dir: str = "./corpus"           # 你那堆普通文件（md + 代码），向量库由它建出来

    # —— 向量库 ——
    vector_db_path: str = "./.chroma"      # Chroma 的本地落盘目录；这个目录就是"向量库"

    # —— Embedding 后端（可切换；生产默认走 API）——
    embedding_backend: str = "api"   # "api" → APIEmbedder（默认，OpenAI 兼容，如本地 LM Studio）；"local" → LocalEmbedder
    embedding_api_base: str = "http://127.0.0.1:1234/v1"               # OpenAI 兼容 base（含 /v1）
    embedding_api_model: str = "text-embedding-qwen3-embedding-0.6b"   # API 后端的 model 标识（写进向量库做一致性键）；2026-07-13 已对本地 LM Studio 实测连通：响应 model 同名、向量维度 1024
    embedding_api_key: str = ""           # LM Studio 可空；空 → 不发 Authorization。留位以便将来云端 API
    embedding_api_timeout: float = 60.0   # 单次请求超时（秒）；防 LM Studio 挂死时无限阻塞

    # —— 本地 embedding 模型 ——
    embedding_model: str = "BAAI/bge-m3"     # 仅当 embedding_backend=="local" 时生效（LocalEmbedder 的 bge-m3）
    # 对话/编排模型不在 core；embedding 不是生成式 LLM。

    # —— 检索行为 ——
    top_k: int = 8                  # 最终返回片段数
    vector_top_k: int = 20          # 向量路初召回数（喂给 fusion 之前）
    lexical_top_k: int = 20         # 词法路初召回数

    # —— 融合（RRF）——
    # fused_score = Σ_每路 1/(rrf_k + rank)，rank 从 1 起。rrf_k 越大越弱化名次差异；
    # 60 为业界惯例（Cormack et al. 2009）。RRF 只吃名次不吃原始分，故免疫各路量纲不可比。
    rrf_k: int = 60

    # exact-span 选取阶段单一文档在 top-k 的占比上限，防止单文档垄断。
    # 按原则设「单一来源不占多数」→ 0.5：多文档均有命中时单文档至多占一半 top-k，另一半留给其它文档
    # （其它文档无命中时不空置，放宽由主文档补满 → 只重分配、不减召回）。取值不为凑 Slot-1 结果而挑。
    exact_span_single_doc_cap_ratio: float = 0.5

DEFAULT_SETTINGS = Settings()


DEFAULT_EMBEDDING_API_KEY_ENV = "EVIDENCE_EMBEDDING_API_KEY"

_EMBEDDING_CONFIG_FIELDS = frozenset(
    {
        "embedding_backend",
        "embedding_api_base",
        "embedding_api_model",
        "embedding_api_timeout",
        "embedding_model",
        "embedding_api_key_env",
    }
)

_EMBEDDING_ENV_FIELDS = {
    "embedding_backend": "EVIDENCE_EMBEDDING_BACKEND",
    "embedding_api_base": "EVIDENCE_EMBEDDING_API_BASE",
    "embedding_api_model": "EVIDENCE_EMBEDDING_API_MODEL",
    "embedding_api_timeout": "EVIDENCE_EMBEDDING_API_TIMEOUT",
    "embedding_model": "EVIDENCE_EMBEDDING_LOCAL_MODEL",
}


def _load_embedding_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the optional JSON embedding configuration.

    The file may either contain the embedding keys directly or wrap them in one
    top-level ``{"embedding": {...}}`` object.  Secrets are deliberately not
    accepted in the file; only the *name* of the environment variable holding
    the API key may be configured.
    """

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取 embedding 配置 {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"embedding 配置不是合法 JSON（{config_path}:{exc.lineno}:{exc.colno}）"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("embedding 配置顶层必须是 JSON object")
    if "embedding" in raw:
        if set(raw) != {"embedding"} or not isinstance(raw["embedding"], dict):
            raise ValueError(
                "使用 embedding 包装层时，配置必须形如 "
                '{"embedding": {"embedding_backend": "api", ...}}'
            )
        raw = raw["embedding"]
    if "embedding_api_key" in raw:
        raise ValueError(
            "embedding 配置文件不得保存 API key；请使用 "
            f"{DEFAULT_EMBEDDING_API_KEY_ENV} 或 embedding_api_key_env"
        )
    unknown = sorted(set(raw) - _EMBEDDING_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"未知 embedding 配置项: {', '.join(unknown)}")
    return dict(raw)


def resolve_embedding_settings(
    *,
    defaults: Settings = DEFAULT_SETTINGS,
    config_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    api_key_env: str | None = None,
) -> Settings:
    """Resolve embedding settings with deterministic precedence.

    Precedence is ``defaults < JSON config < environment < CLI``.  Supported
    environment variables are ``EVIDENCE_EMBEDDING_BACKEND``,
    ``EVIDENCE_EMBEDDING_API_BASE``, ``EVIDENCE_EMBEDDING_API_MODEL``,
    ``EVIDENCE_EMBEDDING_API_TIMEOUT`` and
    ``EVIDENCE_EMBEDDING_LOCAL_MODEL``.  The API key is read from
    ``EVIDENCE_EMBEDDING_API_KEY`` by default; the variable name can be changed
    through ``embedding_api_key_env``, ``EVIDENCE_EMBEDDING_API_KEY_ENV`` or
    the CLI's ``--embedding-api-key-env``.
    """

    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    key_env_name = DEFAULT_EMBEDDING_API_KEY_ENV

    if config_path is not None:
        file_values = _load_embedding_config(config_path)
        configured_key_env = file_values.pop("embedding_api_key_env", None)
        if configured_key_env is not None:
            key_env_name = str(configured_key_env)
        values.update(file_values)

    for field_name, env_name in _EMBEDDING_ENV_FIELDS.items():
        if env_name in env and env[env_name] != "":
            values[field_name] = env[env_name]
    if env.get("EVIDENCE_EMBEDDING_API_KEY_ENV"):
        key_env_name = env["EVIDENCE_EMBEDDING_API_KEY_ENV"]

    for field_name, value in (cli_overrides or {}).items():
        if value is not None:
            if field_name not in _EMBEDDING_CONFIG_FIELDS - {"embedding_api_key_env"}:
                raise ValueError(f"未知 embedding CLI 配置项: {field_name}")
            values[field_name] = value
    if api_key_env is not None:
        key_env_name = api_key_env

    key_env_name = key_env_name.strip()
    if not key_env_name:
        raise ValueError("embedding API key 环境变量名不能为空")
    if key_env_name in env:
        values["embedding_api_key"] = env[key_env_name]

    if "embedding_api_timeout" in values:
        try:
            values["embedding_api_timeout"] = float(values["embedding_api_timeout"])
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding_api_timeout 必须是数字") from exc
        if values["embedding_api_timeout"] <= 0:
            raise ValueError("embedding_api_timeout 必须大于 0")

    settings = replace(defaults, **values)
    if settings.embedding_backend not in {"api", "local"}:
        raise ValueError("embedding_backend 必须是 'api' 或 'local'")
    if settings.embedding_backend == "api":
        if not settings.embedding_api_base.startswith(("http://", "https://")):
            raise ValueError("embedding_api_base 必须是 http:// 或 https:// URL")
        if not settings.embedding_api_model.strip():
            raise ValueError("embedding_api_model 不能为空")
    elif not settings.embedding_model.strip():
        raise ValueError("embedding_model 不能为空")
    return settings
