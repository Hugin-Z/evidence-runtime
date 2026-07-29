"""规范化文本、位置索引与无损切块。

本模块只读取 UTF-8 Markdown/TXT。规范化仅把 CRLF/CR 转成 LF，不删除、
合并或改写其他字符。所有字符偏移指向 :attr:`NormalizedDocument.text`；
PDF/DOCX 到原始页码的映射由 runtime adapter 提供。

这是产品路径的 canonical chunker；benchmark 保持独立实现，产品代码不反向依赖评测包。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


SUPPORTED_NORMALIZED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
SUPPORTED_SOURCE_TYPES = frozenset({"pdf", "docx", "md", "txt"})
LOSSLESS_CHUNKER_VERSION = "m1-lossless-v1"


def normalize_text(text: str) -> str:
    """只统一换行符，保留尾部换行、空行、空白、标点及 Markdown 原文。"""

    if not isinstance(text, str):
        raise ValueError(f"text must be str; got {text!r}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest; got {value!r}")
    return value


@dataclass(frozen=True)
class NormalizedDocument:
    """只读文档及可复用行首索引。

    字符偏移为 0-based，范围为左闭右开；行号为 1-based。文档末尾 offset
    ``len(text)`` 合法：若文本以换行结尾，它位于新增的空末行，否则位于最后
    一个非空逻辑行。空文档只有 offset 0，并将其解释为第 1 行。

    字符和行号 provenance 指向规范化文本；PDF/DOCX 到原始页码的映射
    由 runtime adapter 提供。
    """

    document_id: str
    title: str
    source_path: str
    source_type: str
    normalized_path: str | None
    text: str
    source_content_hash: str
    normalized_content_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    _line_starts: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty(self.document_id, "NormalizedDocument.document_id")
        _require_non_empty(self.title, "NormalizedDocument.title")
        _require_non_empty(self.source_path, "NormalizedDocument.source_path")
        if self.source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                "NormalizedDocument.source_type must be one of "
                f"{sorted(SUPPORTED_SOURCE_TYPES)!r}; got {self.source_type!r}"
            )
        if self.normalized_path is not None:
            _require_non_empty(self.normalized_path, "NormalizedDocument.normalized_path")
        if not isinstance(self.text, str):
            raise ValueError(f"NormalizedDocument.text must be str; got {self.text!r}")
        _validate_sha256(
            self.source_content_hash,
            "NormalizedDocument.source_content_hash",
        )
        _validate_sha256(
            self.normalized_content_hash,
            "NormalizedDocument.normalized_content_hash",
        )
        expected_normalized_hash = _sha256_bytes(self.text.encode("utf-8"))
        if self.normalized_content_hash != expected_normalized_hash:
            raise ValueError(
                "NormalizedDocument.normalized_content_hash does not match text; "
                f"expected {expected_normalized_hash!r}, got {self.normalized_content_hash!r}"
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError(
                f"NormalizedDocument.metadata must be a mapping; got {self.metadata!r}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        line_starts = (0,) + tuple(
            index + 1 for index, char in enumerate(self.text) if char == "\n"
        )
        object.__setattr__(self, "_line_starts", line_starts)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
        source_path: str | None = None,
        source_type: str | None = None,
        normalized_path: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "NormalizedDocument":
        """读取 UTF-8 Markdown/TXT，规范化换行并生成稳定 ID 与双 hash。

        自动 ID 只使用逻辑来源的文件名和规范化内容 hash，不包含绝对本机路径。
        ``source_content_hash`` 针对读入文件的原始 bytes（含 BOM/原换行），而
        ``normalized_content_hash`` 针对规范化后 UTF-8 文本。
        """

        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_NORMALIZED_SUFFIXES:
            raise ValueError(
                "normalized document path must end with one of "
                f"{sorted(SUPPORTED_NORMALIZED_SUFFIXES)!r}; got {str(file_path)!r}"
            )
        raw_bytes = file_path.read_bytes()
        try:
            decoded = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"normalized document {str(file_path)!r} is not valid UTF-8: {exc}"
            ) from exc

        text = normalize_text(decoded)
        source_hash = _sha256_bytes(raw_bytes)
        normalized_hash = _sha256_bytes(text.encode("utf-8"))
        stored_source_path = source_path if source_path is not None else file_path.name
        stored_title = title if title is not None else file_path.stem
        inferred_source_type = "md" if suffix in {".md", ".markdown"} else "txt"
        stored_source_type = source_type if source_type is not None else inferred_source_type
        stored_normalized_path = (
            normalized_path if normalized_path is not None else str(file_path)
        )

        if document_id is None:
            logical_name = PurePosixPath(stored_source_path.replace("\\", "/")).name
            logical_name = _require_non_empty(logical_name, "logical source filename")
            id_input = f"{logical_name}\0{normalized_hash}".encode("utf-8")
            stored_document_id = f"doc-{hashlib.sha256(id_input).hexdigest()}"
        else:
            stored_document_id = document_id

        return cls(
            document_id=stored_document_id,
            title=stored_title,
            source_path=stored_source_path,
            source_type=stored_source_type,
            normalized_path=stored_normalized_path,
            text=text,
            source_content_hash=source_hash,
            normalized_content_hash=normalized_hash,
            metadata={} if metadata is None else metadata,
        )

    @property
    def line_starts(self) -> tuple[int, ...]:
        """返回不可变的 0-based 行首字符偏移。"""

        return self._line_starts

    def line_number_at(self, char_offset: int) -> int:
        """把合法字符位置（含文档末尾）转换为 1-based 行号。"""

        if type(char_offset) is not int:
            raise ValueError(
                f"char_offset must be an integer; got {char_offset!r}"
            )
        if char_offset < 0 or char_offset > len(self.text):
            raise ValueError(
                f"char_offset must be between 0 and {len(self.text)}; got {char_offset}"
            )
        return bisect_right(self._line_starts, char_offset)

    def _validate_range(self, start_char: int, end_char: int) -> None:
        if type(start_char) is not int or type(end_char) is not int:
            raise ValueError(
                "character offsets must be integers; "
                f"got start_char={start_char!r}, end_char={end_char!r}"
            )
        if start_char < 0 or end_char < start_char or end_char > len(self.text):
            raise ValueError(
                "character range must satisfy 0 <= start_char <= end_char <= "
                f"{len(self.text)}; got start_char={start_char}, end_char={end_char}"
            )

    def line_range(self, start_char: int, end_char: int) -> tuple[int, int]:
        """返回左闭右开字符范围覆盖的 1-based 首尾行号。

        空范围归属其 offset 所在行；非空范围的末行由 ``end_char - 1`` 决定。
        """

        self._validate_range(start_char, end_char)
        start_line = self.line_number_at(start_char)
        end_line = (
            self.line_number_at(end_char - 1)
            if end_char > start_char
            else start_line
        )
        return start_line, end_line

    def slice(self, start_char: int, end_char: int) -> str:
        """校验后精确回读左闭右开字符范围。"""

        self._validate_range(start_char, end_char)
        return self.text[start_char:end_char]


@dataclass(frozen=True, init=False)
class LosslessChunk:
    """由 :func:`chunk_document` 创建的非空、无重叠规范文档切片。"""

    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    content_hash: str
    chunker_version: str

    def __init__(self) -> None:
        raise TypeError("LosslessChunk must be created with chunk_document()")

    @classmethod
    def _from_document(
        cls,
        document: NormalizedDocument,
        start_char: int,
        end_char: int,
    ) -> "LosslessChunk":
        text = document.slice(start_char, end_char)
        if not text:
            raise ValueError(
                "LosslessChunk cannot be empty; "
                f"got start_char={start_char}, end_char={end_char}"
            )
        start_line, end_line = document.line_range(start_char, end_char)
        content_hash = _sha256_bytes(text.encode("utf-8"))
        id_input = (
            f"{LOSSLESS_CHUNKER_VERSION}\0{document.document_id}\0"
            f"{document.normalized_content_hash}\0{start_char}\0{end_char}"
        ).encode("utf-8")

        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "chunk_id",
            f"chunk-{hashlib.sha256(id_input).hexdigest()}",
        )
        object.__setattr__(instance, "document_id", document.document_id)
        object.__setattr__(instance, "text", text)
        object.__setattr__(instance, "start_char", start_char)
        object.__setattr__(instance, "end_char", end_char)
        object.__setattr__(instance, "start_line", start_line)
        object.__setattr__(instance, "end_line", end_line)
        object.__setattr__(instance, "content_hash", content_hash)
        object.__setattr__(instance, "chunker_version", LOSSLESS_CHUNKER_VERSION)
        return instance


def _select_boundary(text: str, start_char: int, max_chars: int) -> int:
    hard_end = min(start_char + max_chars, len(text))
    if hard_end == len(text):
        return hard_end

    paragraph_at = text.rfind("\n\n", start_char, hard_end)
    if paragraph_at >= start_char:
        paragraph_end = paragraph_at + 2
        if paragraph_end > start_char:
            return paragraph_end

    newline_at = text.rfind("\n", start_char, hard_end)
    if newline_at >= start_char:
        newline_end = newline_at + 1
        if newline_end > start_char:
            return newline_end

    return hard_end


def chunk_document(
    document: NormalizedDocument,
    *,
    max_chars: int,
) -> tuple[LosslessChunk, ...]:
    """把规范化文本确定性切成无重叠、≤ ``max_chars`` 的精确划分。

    优先在段落（两个 LF）边界切分，其次在单个 LF，找不到合适边界时
    hard split。空文档返回空 tuple；非空输入绝不产生空 Chunk。
    """

    if type(max_chars) is not int or max_chars < 1:
        raise ValueError(f"max_chars must be an integer >= 1; got {max_chars!r}")
    if not isinstance(document, NormalizedDocument):
        raise ValueError(
            f"document must be NormalizedDocument; got {document!r}"
        )
    if document.text == "":
        return ()

    chunks: list[LosslessChunk] = []
    start_char = 0
    while start_char < len(document.text):
        end_char = _select_boundary(document.text, start_char, max_chars)
        if end_char <= start_char:
            raise RuntimeError(
                "lossless chunker failed to advance; "
                f"start_char={start_char}, end_char={end_char}, max_chars={max_chars}"
            )
        chunks.append(LosslessChunk._from_document(document, start_char, end_char))
        start_char = end_char

    return tuple(chunks)
