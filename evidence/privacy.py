"""远端 embedding 的隐私提示：轻量提示，不做网络策略。

政企场景里"材料会不会离开本机"是合规问题。本模块只回答一个确定性问题：配置的 embedding
endpoint 是不是本机？非本机 → 全部材料内容会被发到该服务，运行时给一次显著警告。
**不拦截、不改配置、不做任何网络控制**——只让使用者知情。
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from evidence.core.config import Settings

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}
SILENCE_ENV = "EVIDENCE_NO_REMOTE_WARNING"


def is_remote_endpoint(api_base: str) -> bool:
    """endpoint 是否指向**非本机**主机。解析失败按远端处理（宁可多提醒）。"""

    try:
        host = (urlparse(api_base).hostname or "").lower()
    except ValueError:
        return True
    return host not in _LOCAL_HOSTS


def remote_embedding_warning(settings: Settings) -> str | None:
    """若配置为云端/远端 embedding，返回显著警告文本；本机或已静默 → None。"""

    if os.environ.get(SILENCE_ENV):
        return None
    if getattr(settings, "embedding_backend", "") != "api":
        return None
    base = getattr(settings, "embedding_api_base", "") or ""
    if not is_remote_endpoint(base):
        return None
    host = urlparse(base).hostname or base
    return (
        "⚠ 隐私警告：embedding 服务为**非本机**端点（" + str(host) + "）。"
        "运行检索会把**全部材料内容**发送到该服务——政企/涉密材料请改用本机 embedding、"
        "或入库前脱敏、或隔离环境运行。详见 docs/SECURITY.md。"
        "（静默：设环境变量 " + SILENCE_ENV + "=1，或用 --fake-embedder 离线跑。）"
    )
