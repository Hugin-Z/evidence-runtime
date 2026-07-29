"""M5 exact-span 单文档占比上限单测：构造单文档垄断 → 断言腾位；cap=None 退化原行为。"""

from __future__ import annotations

from evidence.documents import NormalizedDocument
from evidence.exact_locator import _select_diverse_spans
from evidence.types import EvidenceSpan


def _doc(tmp_path, name: str) -> NormalizedDocument:
    path = tmp_path / f"{name}.md"
    path.write_text("x" * 200, encoding="utf-8", newline="\n")
    return NormalizedDocument.from_path(
        path, document_id=f"doc-{name}", source_path=f"materials/{name}.md",
        normalized_path=f"materials/{name}.md",
    )


def _span(doc: NormalizedDocument, start: int) -> EvidenceSpan:
    return EvidenceSpan.from_document(doc, start, start + 5, discovery_sources=("ripgrep",))


def _ranked(spans):
    return [(i + 1, s) for i, s in enumerate(spans)]


def test_cap_none_is_identical_to_old_behavior(tmp_path):
    a = _doc(tmp_path, "a")
    spans = [_span(a, i * 10) for i in range(6)]
    ranked = _ranked(spans)
    baseline, overflow = _select_diverse_spans(ranked, max_results=4, single_doc_cap_ratio=None)
    assert [s for _, s in baseline] == spans[:4]           # ranked[:max_results]
    assert [s for _, s in overflow] == spans[4:]           # 余为 overflow


def test_cap_admits_other_document_span_into_baseline(tmp_path):
    # docA 5 条(rank1-5) + docB 1 条(rank6)；max=4,cap=ceil(.5*4)=2。
    # 原行为 baseline=[A1,A2,A3,A4]（B6 落 overflow）；加上限后 B6 进 baseline。
    a, b = _doc(tmp_path, "a"), _doc(tmp_path, "b")
    spans = [_span(a, 0), _span(a, 10), _span(a, 20), _span(a, 30), _span(a, 40), _span(b, 0)]
    ranked = _ranked(spans)
    baseline, overflow = _select_diverse_spans(ranked, max_results=4, single_doc_cap_ratio=0.5)
    base_docs = [s.document_id for _, s in baseline]
    assert "doc-b" in base_docs, "上限腾位：次优文档证据进入 baseline"
    assert base_docs.count("doc-a") <= 3, "单文档不再独占（原为 4/4）"
    assert len(baseline) == 4, "预算填满、不减召回"
    # baseline 按 rank 排列（内容确定序）
    assert [r for r, _ in baseline] == sorted(r for r, _ in baseline)


def test_single_document_slot_unaffected_by_cap(tmp_path):
    # 只有一个文档有命中：无其它文档可腾位 → 放宽由主文档补满，等同原行为。
    a = _doc(tmp_path, "a")
    spans = [_span(a, i * 10) for i in range(6)]
    ranked = _ranked(spans)
    capped, cap_overflow = _select_diverse_spans(ranked, max_results=4, single_doc_cap_ratio=0.5)
    old, old_overflow = _select_diverse_spans(ranked, max_results=4, single_doc_cap_ratio=None)
    assert [s for _, s in capped] == [s for _, s in old]
    assert [s for _, s in cap_overflow] == [s for _, s in old_overflow]


def test_overflow_preserves_rank_order_for_rescue(tmp_path):
    a, b = _doc(tmp_path, "a"), _doc(tmp_path, "b")
    spans = [_span(a, i * 10) for i in range(5)] + [_span(b, 0)]
    ranked = _ranked(spans)
    _, overflow = _select_diverse_spans(ranked, max_results=4, single_doc_cap_ratio=0.5)
    ranks = [r for r, _ in overflow]
    assert ranks == sorted(ranks), "overflow 保持 rank 序（供 B1.1 rescue）"
