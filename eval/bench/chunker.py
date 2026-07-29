"""Benchmark 专用硬切切块器。

**为什么另起一份**：`core.ingest.chunk_text` 按整行累积到字符预算，但**单行超预算不切**——
一个无内部换行的长 passage（Covid 正文最长 60,975 字符）会变成**一个超大 chunk**，送 embedding
服务时，超过模型上下文的超长 chunk 可能被静默截断。benchmark 因此使用本文件的硬切版本建库，
并把切块上限与内容不变式写入结果快照。

**修复**：
  1. 单行（unit）超 `MAX_CHUNK_CHARS` → **强制硬切**成 ≤ 上限的连续片，无 chunk 超上限。
  2. 每 chunk 带**字符偏移** `char_start`（在 normalize 原文中的 0-based 起点）：同一行硬切出的
     多片 `start_line` 相同，仅靠 (source_path,start_line,end_line) 会误合并；`char_start` 保证唯一。

**规范化定义（重构不变式的基准）**：`normalize(text)` = `"\n".join(text.splitlines())`
（按 splitlines 拆行、剥离行尾 \\r、以 \\n 重连、无尾随换行）。

**重构不变式**：切块是 normalize(原文) 的一个**精确划分**——同一文件所有 chunk 的 text 按
`char_start` 升序拼接 **== normalize(原文)**（无丢字、无重字、无乱序）。见 `tests/test_chunker.py`。
（划分含空白/纯换行片以保不变式；建库时按 `text.strip()==""` 跳过纯空白片，不入向量库。）
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_CHUNK_CHARS = 800     # 每块字符上限（与 core 同值；仅修「硬切」行为，不改块大小）
CHUNKER_VERSION = "hs800v1"   # 硬切(hard-split)/上限800/v1 —— 进 collection 名与快照指纹


@dataclass
class BenchChunk:
    source_path: str
    text: str
    start_line: int      # 1-based（起）
    end_line: int        # 1-based（止，含）
    char_start: int      # 该 chunk 文本在 normalize(原文) 中的起始字符偏移（0-based）
    chunk_index: int


def normalize(text: str) -> str:
    """规范化：`"\\n".join(text.splitlines())`——重构不变式的基准文本。"""
    return "\n".join(text.splitlines())


def chunk_text(text: str, source_path: str) -> list[BenchChunk]:
    """硬切版切块：把 normalize(原文) 划分为 ≤ 上限的连续片，记 char_start 保唯一。

    单元(unit) = 每行 + 其后的 '\\n' 分隔符（末行无）。unit 超上限 → 硬切成连续子串；
    否则按字符预算累积多行。产出顺序即 char_start 升序，拼接精确等于 normalize(原文)。
    不做行重叠（按 document 折叠评测，重叠只增同 doc 碎片）。
    """
    norm = normalize(text)
    lines = norm.split("\n")
    n = len(lines)
    chunks: list[BenchChunk] = []
    idx = 0
    pos = 0                      # 当前字符偏移（norm 内）
    buf = ""                     # 累积缓冲
    buf_start_line = 0
    buf_end_line = 0
    buf_char_start = 0

    def flush():
        nonlocal buf, idx
        if buf != "":
            chunks.append(BenchChunk(source_path, buf, buf_start_line + 1,
                                     buf_end_line + 1, buf_char_start, idx))
            idx += 1
        buf = ""

    for li in range(n):
        unit = lines[li] + ("\n" if li < n - 1 else "")
        if len(unit) > MAX_CHUNK_CHARS:
            flush()                                   # 先冲掉累积缓冲
            off = 0
            for k in range(0, len(unit), MAX_CHUNK_CHARS):
                piece = unit[k:k + MAX_CHUNK_CHARS]
                chunks.append(BenchChunk(source_path, piece, li + 1, li + 1,
                                         pos + off, idx))
                idx += 1
                off += len(piece)
            pos += len(unit)
            continue
        if buf and len(buf) + len(unit) > MAX_CHUNK_CHARS:
            flush()
        if buf == "":
            buf_start_line = li
            buf_char_start = pos
        buf += unit
        buf_end_line = li
        pos += len(unit)
    flush()
    return chunks


def chunk_id(c: BenchChunk) -> str:
    """稳定唯一 id（含 char_start，硬切多片不撞）：供 Chroma id 与去重键。"""
    return f"{c.source_path}::{c.start_line}:{c.char_start}:{c.chunk_index}"
