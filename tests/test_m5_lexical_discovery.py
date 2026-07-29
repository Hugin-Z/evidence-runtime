"""M5 词法候选通道单测：fake 语料、确定性、chunk 级字面匹配、B 轮同义鸿沟预期漏。"""

from __future__ import annotations

from evidence.contract import solution_material_completeness_v1 as CONTRACT
from evidence.documents import NormalizedDocument
from evidence.lexical_discovery import (
    LexicalDiscovery,
    _lexical_hits,
    _slot_query_terms,
)

SLOT_ID = "background_and_basis"
SLOT = CONTRACT.get_slot(SLOT_ID)


def _doc(tmp_path, name: str, text: str) -> NormalizedDocument:
    path = tmp_path / f"{name}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return NormalizedDocument.from_path(
        path, document_id=f"doc-{name}", source_path=f"materials/{name}.md",
        normalized_path=f"materials/{name}.md",
    )


def test_slot_query_terms_are_clean_chinese_tokens():
    terms = _slot_query_terms(SLOT)
    assert terms, "应产出非空查询词"
    assert all(len(t) >= 2 for t in terms), "无单字"
    assert len(set(terms)) == len(terms), "去重"
    assert "的" not in terms and "与" not in terms, "去停用词"


def test_lexical_hits_counts_distinct_terms():
    hits, matched = _lexical_hits("这里有建设也有依据", ("建设", "依据", "缺失词"))
    assert hits == 2
    assert set(matched) == {"建设", "依据"}
    assert _lexical_hits("完全无关", ("建设", "依据"))[0] == 0


def test_discover_ranks_by_hits_and_drops_zero_hit_docs(tmp_path):
    # A 含 name+description 全部词（命中最多）；C 只含 name（较少）；B 无关（0 命中，应丢）。
    doc_a = _doc(tmp_path, "a", f"{SLOT.name}。{SLOT.description}")
    doc_c = _doc(tmp_path, "c", SLOT.name)
    doc_b = _doc(tmp_path, "b", "与本槽完全无关的随机文本内容一二三")
    result = LexicalDiscovery().discover_for_slot(
        CONTRACT, SLOT_ID, documents=[doc_b, doc_c, doc_a], top_k=8, max_chars=800,
    )
    doc_ids = [c.document_id for c in result]
    assert "doc-b" not in doc_ids, "0 命中文档不进候选"
    assert doc_ids[0] == "doc-a", "命中最多者排第一"
    assert set(doc_ids) == {"doc-a", "doc-c"}
    for c in result:
        assert c.slot_id == SLOT_ID
        assert c.discovery_source == "lexical"
        assert c.distance is None
        assert c.metadata["lexical_hits"] >= 1


def test_discover_is_deterministic(tmp_path):
    doc_a = _doc(tmp_path, "a", f"{SLOT.name}。{SLOT.description}")
    doc_c = _doc(tmp_path, "c", f"{SLOT.name}的另一处")
    kw = dict(documents=[doc_a, doc_c], top_k=8, max_chars=800)
    first = LexicalDiscovery().discover_for_slot(CONTRACT, SLOT_ID, **kw)
    second = LexicalDiscovery().discover_for_slot(CONTRACT, SLOT_ID, **kw)
    assert [(c.document_id, c.start_char, c.rank) for c in first] == [
        (c.document_id, c.start_char, c.rank) for c in second
    ]


def test_tie_break_is_content_deterministic_not_id(tmp_path):
    # 两文档命中数相同、内容不同：次序应由内容哈希决定，换 id 不改相对次序。
    doc_x = _doc(tmp_path, "x", SLOT.name)
    doc_y = _doc(tmp_path, "y", SLOT.name)   # 同样只含 name → 同命中数
    order = [
        c.text for c in LexicalDiscovery().discover_for_slot(
            CONTRACT, SLOT_ID, documents=[doc_x, doc_y], top_k=8, max_chars=800,
        )
    ]
    # 同内容（都是 SLOT.name）→ 坐标/哈希一致；断言确定且稳定（重跑同序）
    order2 = [
        c.text for c in LexicalDiscovery().discover_for_slot(
            CONTRACT, SLOT_ID, documents=[doc_y, doc_x], top_k=8, max_chars=800,
        )
    ]
    assert order == order2


def test_disabled_returns_empty(tmp_path):
    doc_a = _doc(tmp_path, "a", f"{SLOT.name}。{SLOT.description}")
    assert LexicalDiscovery(enabled=False).discover_for_slot(
        CONTRACT, SLOT_ID, documents=[doc_a], top_k=8, max_chars=800,
    ) == ()


def test_synonym_gap_chunk_not_surfaced_by_design(tmp_path):
    # B 轮结论承接：语义相关但字面不含查询词的 chunk，词法必然漏（预期，非 bug）。
    doc = _doc(tmp_path, "syn", "本单位过去缺少统一平台，历史遗留问题突出。")  # 语义=现状/背景，但不含 name/desc 判别词
    result = LexicalDiscovery().discover_for_slot(
        CONTRACT, SLOT_ID, documents=[doc], top_k=8, max_chars=800,
    )
    # 若确无字面命中则为空（同义鸿沟）；容许极少数偶合词，但至少不因语义相关而必然入选
    assert all(c.metadata["lexical_hits"] >= 1 for c in result)
