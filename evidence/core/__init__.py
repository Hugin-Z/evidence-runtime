"""Low-level retrieval primitives reused by Evidence Runtime and benchmarks.

The supported product workflow starts at ``evidence-gap-check``.  This package
contains configuration, embeddings, indexing, retrievers and RRF primitives;
it intentionally exposes no second end-to-end query CLI.
"""
from .types import Snippet, RetrievalPath
from .config import Settings, DEFAULT_SETTINGS

__all__ = [
    "Snippet", "RetrievalPath", "Settings", "DEFAULT_SETTINGS",
]
