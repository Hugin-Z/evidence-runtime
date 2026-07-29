"""规范语料物化：把 NormalizedDocument 无损落盘为 rg 可搜的 canonical corpus。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from ..documents import NormalizedDocument
from ..rg_executor import (
    InvalidRipgrepOutputError,
    RipgrepExecutor,
    RipgrepLineEvent,
    RipgrepSubmatch,
)
from ..types import EvidenceContract, EvidenceSlot, EvidenceSpan, VectorCandidate
from ..vector_discovery import build_slot_query

from .constants import CANONICAL_MANIFEST_NAME
from .helpers import _canonical_json, _non_empty, _positive_int


class CanonicalCorpusError(RuntimeError):
    """canonical corpus 无法无损落盘或回读。"""


@dataclass(frozen=True)
class CanonicalDocumentEntry:
    document_id: str
    canonical_path: str
    normalized_content_hash: str
    text_length: int
    line_count: int

    def __post_init__(self) -> None:
        _non_empty(self.document_id, "CanonicalDocumentEntry.document_id")
        _non_empty(self.canonical_path, "CanonicalDocumentEntry.canonical_path")
        if (
            not isinstance(self.normalized_content_hash, str)
            or len(self.normalized_content_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.normalized_content_hash)
        ):
            raise ValueError("CanonicalDocumentEntry has invalid normalized_content_hash")
        if type(self.text_length) is not int or self.text_length < 0:
            raise ValueError("CanonicalDocumentEntry.text_length must be >= 0")
        if type(self.line_count) is not int or self.line_count < 1:
            raise ValueError("CanonicalDocumentEntry.line_count must be >= 1")


@dataclass(frozen=True)
class CanonicalCorpusManifest:
    corpus_id: str
    root: str
    documents: tuple[CanonicalDocumentEntry, ...]
    _by_id: Mapping[str, CanonicalDocumentEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.corpus_id, str)
            or len(self.corpus_id) != 64
            or any(ch not in "0123456789abcdef" for ch in self.corpus_id)
        ):
            raise ValueError("CanonicalCorpusManifest.corpus_id must be SHA-256")
        _non_empty(self.root, "CanonicalCorpusManifest.root")
        documents = tuple(self.documents)
        if any(not isinstance(item, CanonicalDocumentEntry) for item in documents):
            raise ValueError("CanonicalCorpusManifest.documents contains invalid entry")
        if tuple(sorted(documents, key=lambda item: item.normalized_content_hash)) != documents:
            raise ValueError("CanonicalCorpusManifest.documents must be sorted by content hash")
        if len({item.document_id for item in documents}) != len(documents):
            raise ValueError("CanonicalCorpusManifest has duplicate document IDs")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(
            self,
            "_by_id",
            MappingProxyType({item.document_id: item for item in documents}),
        )

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(item.document_id for item in self.documents)

    def get(self, document_id: str) -> CanonicalDocumentEntry:
        try:
            return self._by_id[document_id]
        except KeyError:
            raise KeyError(f"unknown canonical document_id {document_id!r}") from None

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "root": self.root,
            "documents": [
                {
                    "document_id": item.document_id,
                    "canonical_path": item.canonical_path,
                    "normalized_content_hash": item.normalized_content_hash,
                    "text_length": item.text_length,
                    "line_count": item.line_count,
                }
                for item in self.documents
            ],
        }


def _validate_document_id(document_id: str) -> None:
    _non_empty(document_id, "document_id")
    if (
        "\x00" in document_id
        or "/" in document_id
        or "\\" in document_id
        or document_id in {".", ".."}
    ):
        raise CanonicalCorpusError(
            f"document_id contains path traversal or separator characters: {document_id!r}"
        )


def _materialized_filename(content_hash: str) -> str:
    """物化文件名——**内容派生**（用规范化正文 sha256，不用 document_id）。

    早期用 sha256(document_id) 命名，使 rg 语料的文件名、遍历顺序与 span 选取随
    分配的 document_id 漂移。改用内容哈希后，同一正文无论 id 为何
    都物化成同名文件 → rg 语料与命中收集内容确定。content_hash 为已校验的 64 位十六进制。
    """

    return f"document-{content_hash}.txt"


def materialize_normalized_documents(
    documents: Sequence[NormalizedDocument],
    root: str | Path,
) -> CanonicalCorpusManifest:
    """把规范文档稳定写成 UTF-8/no-BOM/LF corpus，并回读校验 hash。"""

    if isinstance(documents, (str, bytes)):
        raise ValueError("documents must be a sequence of NormalizedDocument")
    try:
        ordered = tuple(documents)
    except TypeError as exc:
        raise ValueError("documents must be a sequence of NormalizedDocument") from exc
    for index, document in enumerate(ordered):
        if not isinstance(document, NormalizedDocument):
            raise ValueError(
                f"documents[{index}] must be NormalizedDocument; got {document!r}"
            )
        _validate_document_id(document.document_id)
        if "\r" in document.text:
            raise CanonicalCorpusError(
                f"NormalizedDocument {document.document_id!r} still contains CR characters"
            )
    ids = [document.document_id for document in ordered]
    duplicates = sorted({document_id for document_id in ids if ids.count(document_id) > 1})
    if duplicates:
        raise ValueError(f"documents contain duplicate document IDs: {duplicates!r}")
    # 物化/manifest 顺序 → 内容确定（按规范化正文哈希），使全语料 rg 目标顺序不随 id 变。
    ordered = tuple(sorted(ordered, key=lambda document: document.normalized_content_hash))
    canonical_root = Path(root).resolve()
    if canonical_root.exists() and not canonical_root.is_dir():
        raise CanonicalCorpusError(
            f"canonical corpus root is not a directory: {str(canonical_root)!r}"
        )
    canonical_root.mkdir(parents=True, exist_ok=True)
    entries: list[CanonicalDocumentEntry] = []
    for document in ordered:
        path = (canonical_root / _materialized_filename(document.normalized_content_hash)).resolve()
        if path.parent != canonical_root:
            raise CanonicalCorpusError(
                f"materialized path escaped canonical root for {document.document_id!r}"
            )
        expected_bytes = document.text.encode("utf-8")
        try:
            current = path.read_bytes() if path.exists() else None
            if current != expected_bytes:
                temporary = path.with_suffix(".txt.tmp")
                temporary.write_bytes(expected_bytes)
                temporary.replace(path)
            stored_bytes = path.read_bytes()
        except OSError as exc:
            raise CanonicalCorpusError(
                f"failed to materialize canonical document {document.document_id!r}: {exc}"
            ) from exc
        if stored_bytes.startswith(b"\xef\xbb\xbf"):
            raise CanonicalCorpusError(
                f"canonical document {document.document_id!r} unexpectedly contains UTF-8 BOM"
            )
        try:
            stored_text = stored_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalCorpusError(
                f"canonical document {document.document_id!r} is not UTF-8"
            ) from exc
        stored_hash = hashlib.sha256(stored_bytes).hexdigest()
        if stored_text != document.text or stored_hash != document.normalized_content_hash:
            raise CanonicalCorpusError(
                f"canonical document {document.document_id!r} failed exact read-back validation"
            )
        entries.append(
            CanonicalDocumentEntry(
                document_id=document.document_id,
                canonical_path=str(path),
                normalized_content_hash=document.normalized_content_hash,
                text_length=len(document.text),
                line_count=len(document.line_starts),
            )
        )
    identity_payload = {
        "documents": [
            {
                "document_id": item.document_id,
                "normalized_content_hash": item.normalized_content_hash,
                "text_length": item.text_length,
                "line_count": item.line_count,
            }
            for item in entries
        ]
    }
    corpus_id = hashlib.sha256(
        _canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()
    manifest = CanonicalCorpusManifest(
        corpus_id=corpus_id,
        root=str(canonical_root),
        documents=tuple(entries),
    )
    manifest_path = canonical_root / CANONICAL_MANIFEST_NAME
    manifest_bytes = (_canonical_json(manifest.to_dict()) + "\n").encode("utf-8")
    try:
        if not manifest_path.exists() or manifest_path.read_bytes() != manifest_bytes:
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_bytes(manifest_bytes)
            temporary.replace(manifest_path)
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalCorpusError(f"canonical manifest write/read failed: {exc}") from exc
    if stored_manifest != manifest.to_dict():
        raise CanonicalCorpusError("canonical manifest failed exact read-back validation")
    return manifest
