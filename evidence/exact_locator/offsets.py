"""偏移映射与上下文扩展：rg 字节偏移 → Python 字符偏移，并按上下文行裁剪。"""

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

from .helpers import _positive_int


def _mapping_error(
    message: str,
    event: RipgrepLineEvent,
    submatch: RipgrepSubmatch,
) -> InvalidRipgrepOutputError:
    return InvalidRipgrepOutputError(
        message,
        command=(),
        returncode=0,
        stderr="",
        anchors=(submatch.text,),
        search_scope=(event.path,),
    )


def map_rg_submatch_to_char_range(
    document: NormalizedDocument,
    event: RipgrepLineEvent,
    submatch: RipgrepSubmatch,
) -> tuple[int, int]:
    """把 rg 的行内 UTF-8 byte offset 严格转换为规范文本 Python 字符范围。"""

    if not isinstance(document, NormalizedDocument):
        raise ValueError(f"document must be NormalizedDocument; got {document!r}")
    if not isinstance(event, RipgrepLineEvent) or event.kind != "match":
        raise ValueError("event must be a ripgrep match line event")
    if not isinstance(submatch, RipgrepSubmatch):
        raise ValueError("submatch must be RipgrepSubmatch")
    if event.line_number > len(document.line_starts):
        raise _mapping_error("rg line number exceeds canonical document", event, submatch)
    line_start = document.line_starts[event.line_number - 1]
    line_end = (
        document.line_starts[event.line_number]
        if event.line_number < len(document.line_starts)
        else len(document.text)
    )
    canonical_line = document.slice(line_start, line_end)
    if event.line_text != canonical_line:
        raise _mapping_error(
            "rg line text does not equal canonical document line",
            event,
            submatch,
        )
    encoded = event.line_text.encode("utf-8")
    if submatch.end_byte > len(encoded):
        raise _mapping_error("rg submatch byte range exceeds line", event, submatch)
    try:
        prefix = encoded[: submatch.start_byte].decode("utf-8")
        matched = encoded[submatch.start_byte : submatch.end_byte].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _mapping_error(
            "rg submatch byte range splits a UTF-8 code point",
            event,
            submatch,
        ) from exc
    if matched != submatch.text:
        raise _mapping_error(
            "rg submatch text does not equal decoded canonical bytes",
            event,
            submatch,
        )
    start_char = line_start + len(prefix)
    end_char = start_char + len(matched)
    if document.slice(start_char, end_char) != submatch.text:
        raise _mapping_error(
            "mapped character range does not read back exact match",
            event,
            submatch,
        )
    return start_char, end_char


def expand_match_context(
    document: NormalizedDocument,
    match_start: int,
    match_end: int,
    *,
    context_lines: int,
    max_chars: int,
) -> tuple[int, int]:
    """先按完整行扩展，再在超长时围绕 match 对称裁切，始终保留完整 match。"""

    if not isinstance(document, NormalizedDocument):
        raise ValueError(f"document must be NormalizedDocument; got {document!r}")
    context_lines = _positive_int(context_lines, "context_lines", allow_zero=True)
    max_chars = _positive_int(max_chars, "max_chars")
    document._validate_range(match_start, match_end)
    if match_start >= match_end:
        raise ValueError("match range must be non-empty")
    match_length = match_end - match_start
    if match_length > max_chars:
        raise ValueError(
            f"match length {match_length} exceeds max_chars={max_chars}; cannot preserve match"
        )
    start_line = document.line_number_at(match_start)
    end_line = document.line_number_at(match_end - 1)
    first_line = max(1, start_line - context_lines)
    last_line = min(len(document.line_starts), end_line + context_lines)
    span_start = document.line_starts[first_line - 1]
    span_end = (
        document.line_starts[last_line]
        if last_line < len(document.line_starts)
        else len(document.text)
    )
    if span_end - span_start <= max_chars:
        return span_start, span_end

    remaining = max_chars - match_length
    available_left = match_start - span_start
    available_right = span_end - match_end
    left = min(available_left, remaining // 2)
    right = min(available_right, remaining - left)
    unused = remaining - left - right
    if unused:
        extra_left = min(available_left - left, unused)
        left += extra_left
        unused -= extra_left
    if unused:
        right += min(available_right - right, unused)
    cropped_start = match_start - left
    cropped_end = match_end + right
    if not (
        span_start <= cropped_start <= match_start < match_end <= cropped_end <= span_end
        and cropped_end - cropped_start <= max_chars
    ):
        raise RuntimeError("context expansion invariant failed")
    return cropped_start, cropped_end
