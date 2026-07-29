"""tests/test_chunker.py — A-1 硬切切块器（eval/bench/chunker）重构不变式 + 上限保证。

核心不变式：切块是 normalize(原文) 的**精确划分**——按 char_start 升序拼接 == normalize(原文)。
并保证：无 chunk 超 MAX_CHUNK_CHARS、同行硬切多片 identity 不撞。
"""
from __future__ import annotations

import pytest

from eval.bench.chunker import (chunk_text, chunk_id, normalize,
                                MAX_CHUNK_CHARS, CHUNKER_VERSION)


def _reassemble(chunks):
    """按 (char_start) 升序拼接 chunk.text。"""
    return "".join(c.text for c in sorted(chunks, key=lambda c: (c.start_line, c.char_start)))


# ── 重构不变式：拼接 == normalize(原文)（无丢字/重字/乱序）──────────
@pytest.mark.parametrize("text", [
    "hello world",                              # 单行短
    "line1\nline2\nline3",                       # 多行短
    "a" * (MAX_CHUNK_CHARS * 3 + 17),            # 单行远超上限（须硬切）
    "title\n" + "x" * (MAX_CHUNK_CHARS + 5),     # 标题 + 超长正文行
    "短\n" + "中" * (MAX_CHUNK_CHARS * 2),        # 中文超长行
    "para one line a\npara one line b\n\npara two",   # 含空行
    ("word " * 400).strip(),                     # 一长行由空格分隔的词
    "",                                          # 空
    "\n\n\n",                                    # 纯空行
    "trailing spaces   \n\t tabbed line \nend",  # 行内/行尾空白
])
def test_reassembly_equals_normalized(text):
    chunks = chunk_text(text, "/doc/x.txt")
    assert _reassemble(chunks) == normalize(text)


def test_no_chunk_exceeds_limit():
    text = "T\n" + "z" * (MAX_CHUNK_CHARS * 5 + 3)
    chunks = chunk_text(text, "/doc/x.txt")
    assert chunks, "超长输入应产出多个 chunk"
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_long_single_line_is_hard_split_into_many():
    n = MAX_CHUNK_CHARS * 4
    chunks = chunk_text("q" * n, "/doc/x.txt")
    # 4 倍上限的单行 → 至少 4 片（无重叠、精确划分）
    assert len(chunks) >= 4
    assert sum(len(c.text) for c in chunks) == n   # 无丢字无重字


def test_hard_split_pieces_have_unique_identity():
    # 同一行硬切出的多片 start_line 相同——identity 必须靠 char_start 区分，不得撞键
    chunks = chunk_text("k" * (MAX_CHUNK_CHARS * 3), "/doc/x.txt")
    same_line = [c for c in chunks if c.start_line == 1]
    assert len(same_line) >= 3
    ids = [chunk_id(c) for c in same_line]
    keys = [(c.source_path, c.start_line, c.end_line, c.char_start) for c in same_line]
    assert len(set(ids)) == len(ids)       # chunk_id 唯一
    assert len(set(keys)) == len(keys)     # 含 char_start 的去重键唯一


def test_char_start_monotonic_and_zero_based():
    chunks = chunk_text("aaa\n" + "b" * (MAX_CHUNK_CHARS + 2) + "\nccc", "/d/x.txt")
    starts = [c.char_start for c in chunks]
    assert starts[0] == 0
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)   # 偏移互异


def test_normalize_strips_cr_and_trailing_newline():
    assert normalize("a\r\nb\r\n") == "a\nb"
    assert normalize("x\n") == "x"


def test_version_fingerprint_stable():
    assert CHUNKER_VERSION == "hs800v1"
