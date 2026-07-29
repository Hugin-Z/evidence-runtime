"""低层检索原语的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RetrievalPath(str, Enum):
    """一条检索片段来自哪条路。"""

    VECTOR = "vector"      # 语义路：embedding + 向量库
    LEXICAL = "lexical"    # 词法路：ripgrep 精确命中
    BOTH = "both"          # 融合后：同一片段被两路同时命中（fusion 合并置此，供 A/B 分路统计）


@dataclass
class Snippet:
    """一条检索结果。检索核心的最终产物是 list[Snippet]。"""

    source_path: str                      # 来源文件
    text: str                             # 命中的文本内容
    score: float                          # 相关度（rerank 后才有意义）
    path: RetrievalPath                   # 哪条路产出的（用于 A/B 与可观测）
    start_line: Optional[int] = None      # 行区间起（ripgrep 路会有）
    end_line: Optional[int] = None        # 行区间止
    meta: dict = field(default_factory=dict)
