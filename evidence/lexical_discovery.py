"""词法候选通道：与 :mod:`evidence.vector_discovery` 平级的候选发现路。

**职责区分（与 exact_locator 划清）：**
- 本模块是「**发现候选**」：对 slot 查询词做**字面匹配**，把「向量看不见但字面命中」的 chunk 纳入
  候选池，扩宽候选文档来源（多样性）。输出与向量候选**同型** :class:`evidence.types.VectorCandidate`。
- :mod:`evidence.exact_locator` 是「对**已知候选**精确定位 span」：拿到候选文档后用 rg 精确定位可回读
  的 EvidenceSpan。二者同用「字面匹配」语义、**不同阶段**（发现 vs 定位）。

**字面匹配实现（rg -F -i 的 chunk 级等价）：** 查询词 = slot name+description 的 jieba 分词（去停用词/
单字）；对每个 chunk 计**命中的不同查询词数**（casefold 子串，等价 ``rg -F -i``）。选择 chunk 级、
确定性（无子进程排序不确定性）、CJK 安全（查询用 jieba 切、chunk 按字面子串命中，避开 rg 的 CJK 行
分词问题）、无新依赖。exact_locator 的精确 span 定位仍使用 rg_executor。

**确定性破平：** 命中数相同按 chunk 正文规范化哈希 + 字符位置破平，
**不用 document_id / chunk_id**。
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .contract import EvidenceContract, EvidenceSlot
from .documents import NormalizedDocument, chunk_document
from .types import VectorCandidate
from .vector_discovery import _content_order_hash, build_slot_query

LEXICAL_DISCOVERY_VERSION = "m5-lexical-v1"

# 最基本中文功能词停用（刻意小；只求把实词留下来字面搜，不追求语言学完备）。
_STOPWORDS = frozenset({
    "的", "与", "和", "或", "及", "在", "为", "是", "等", "或者", "以及", "对", "对于",
    "所需", "信息", "以", "其", "该", "本", "了", "而", "并", "者", "之", "与其",
})


def _slot_query_terms(slot: EvidenceSlot) -> tuple[str, ...]:
    """把 slot 的 name+description jieba 切成去重、去停用词、去单字的查询词（保序）。"""

    import jieba

    text = f"{slot.name}。{slot.description}"
    seen: set[str] = set()
    terms: list[str] = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if len(tok) < 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
    return tuple(terms)


def _lexical_hits(chunk_text: str, terms: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """命中的**不同**查询词数 + 命中词（casefold 子串，等价 rg -F -i）。"""

    folded = chunk_text.casefold()
    matched = tuple(t for t in terms if t.casefold() in folded)
    return len(matched), matched


class LexicalDiscovery:
    """词法候选发现（chunk 级字面匹配 / 确定性 / model-free / 可 on-off）。"""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def discover_for_slot(
        self,
        contract: EvidenceContract,
        slot_id: str,
        *,
        documents: Sequence[NormalizedDocument],
        top_k: int,
        max_chars: int,
    ) -> tuple[VectorCandidate, ...]:
        if not isinstance(contract, EvidenceContract):
            raise ValueError("contract must be EvidenceContract")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not self.enabled:
            return ()
        slot = contract.get_slot(slot_id)
        query = build_slot_query(slot)
        terms = _slot_query_terms(slot)
        if not terms:
            return ()

        scored: list[tuple[int, str, int, int, object]] = []
        for document in documents:
            for chunk in chunk_document(document, max_chars=max_chars):
                hits, matched = _lexical_hits(chunk.text, terms)
                if hits <= 0:
                    continue
                scored.append((
                    hits,
                    _content_order_hash(chunk.text),
                    chunk.start_char,
                    chunk.end_char,
                    (chunk, matched),
                ))
        # 命中数降序；平局按内容确定键（正文哈希 → 字符位置），不用 id。
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        out: list[VectorCandidate] = []
        for rank, (hits, _hash, _s, _e, payload) in enumerate(scored[:top_k], start=1):
            chunk, matched = payload
            out.append(VectorCandidate(
                slot_id=slot.slot_id,
                query=query,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                rank=rank,
                distance=None,                 # 词法路无距离
                discovery_source="lexical",
                metadata={
                    "lexical_hits": hits,
                    "matched_terms": tuple(matched),
                    "lexical_version": LEXICAL_DISCOVERY_VERSION,
                },
            ))
        return tuple(out)

    def discover_contract(
        self,
        contract: EvidenceContract,
        *,
        documents: Sequence[NormalizedDocument],
        top_k_per_slot: int,
        max_chars: int,
    ) -> dict[str, tuple[VectorCandidate, ...]]:
        documents = tuple(documents)
        return {
            slot.slot_id: self.discover_for_slot(
                contract, slot.slot_id,
                documents=documents, top_k=top_k_per_slot, max_chars=max_chars,
            )
            for slot in contract.slots
        }
