"""低层检索器的显式文件范围测试。"""
from __future__ import annotations

import os
from pathlib import Path

from evidence.core import Settings
from evidence.core.types import RetrievalPath
from evidence.core.scope import Scope
from evidence.core.retrievers.lexical import RipgrepRetriever

CORPUS = Path(__file__).parent / "fixtures" / "lexical_corpus"


def _bn(s) -> str:
    return os.path.basename(s.source_path.replace("\\", "/"))


def _names(snips) -> set[str]:
    return {_bn(s) for s in snips}


def _retriever() -> RipgrepRetriever:
    return RipgrepRetriever(Settings(corpus_dir=str(CORPUS)), context_lines=0)


# ── 精搜遵守 scope（核心证明）─────────────────────────────────────

def test_fine_search_respects_scope_single_file():
    scope = Scope(files=[str(CORPUS / "alpha.md")])
    out = _retriever().search("retry", scope=scope)
    names = _names(out)
    assert len(out) >= 1
    assert names == {"alpha.md"}        # 只来自 scope 内文件
    assert "gamma.md" not in names      # scope 外文件即使有 'retry' 也不出现
    assert "beta.py" not in names
    assert all(s.path is RetrievalPath.LEXICAL for s in out)


def test_scope_none_keeps_full_corpus_behavior():
    names = _names(_retriever().search("retry", scope=None))
    # 不限制 → 全量语料里凡含 'retry' 的文件都在
    assert {"alpha.md", "beta.py", "gamma.md"} <= names


def test_scope_empty_list_returns_empty():
    out = _retriever().search("retry", scope=Scope(files=[]))
    assert out == []  # 限定到空集 → 无可搜
