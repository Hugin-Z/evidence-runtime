"""P-E：渲染交付层（目录 + 页码 + 页眉）+ verify_docx 扩展 单元测试（补 P-D 欠账）。"""
from __future__ import annotations

from docx.oxml.ns import qn

from evidence.generation.orchestrator import OrchestrationState, SectionOutcome
from evidence.generation.task import DraftParagraph, SectionDraft
from evidence.render import verify_docx
from evidence.render.docx_render import (
    TOC_HEADING,
    _render_proposal_unchecked as render_proposal,
)
from evidence.render.verify_docx import VerifyCheck


def _state():
    d = SectionDraft("02", (DraftParagraph("本项目依据 GB 50011 标准。", ("e1",)),))
    return OrchestrationState(
        order=["02"],
        outcomes={"02": SectionOutcome("02", "项目理解", "dispatched", "accepted",
                                       gate="allow", draft=d)},
    )


_BRIEF = {"meta": {"doc_type_zh": "解决方案建议书"},
          "sections": [{"acquired": [{"status": "acquired", "ref": "e1",
                                      "source": "《材料》· 第1页"}]}]}
_PARAMS = {"project_name": "测试项目", "client_name": "测试甲方",
           "doc_type_zh": "解决方案建议书"}


def _fields(element, keyword):
    # fldSimple（PAGE 用）或 fldChar+instrText（TOC 用，P-K）
    hits = [f for f in element.iter(qn("w:fldSimple"))
            if keyword in (f.get(qn("w:instr")) or "")]
    hits += [i for i in element.iter(qn("w:instrText")) if keyword in (i.text or "")]
    return hits


def test_toc_heading_and_field_present():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS)
    texts = [p.text for p in doc.paragraphs]
    heads = [p.text for p in doc.paragraphs if p.style and p.style.name.startswith("Heading 1")]
    assert TOC_HEADING in texts                      # 目录标题在（作为普通段落）
    assert TOC_HEADING not in heads                  # P-K/■2b：但**不是** Heading（否则自我收录）
    assert _fields(doc.element.body, "TOC")          # TOC 域存在（fldChar+instrText）


def test_footer_has_page_number_field():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS)
    assert any(_fields(s.footer._element, "PAGE") for s in doc.sections)


def test_header_optional_off_by_default_on_when_asked():
    off = render_proposal(_state(), brief=_BRIEF, params=_PARAMS, include_header=False)
    assert not any(p.text.strip() for s in off.sections for p in s.header.paragraphs)
    on = render_proposal(_state(), brief=_BRIEF, params=_PARAMS, include_header=True)
    assert any("测试项目" in p.text for s in on.sections for p in s.header.paragraphs)


def test_verify_accepts_delivery_elements():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS)
    assert verify_docx(doc, _state(), params=_PARAMS).accepted


def test_verify_rejects_missing_toc():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS, include_toc=False)
    result = verify_docx(doc, _state(), params=_PARAMS)   # 期望有 TOC，但没渲染
    assert not result.accepted
    assert any(c == VerifyCheck.DELIVERY_ELEMENT_MISSING.value and "目录" in d
               for c, d in result.problems)


def test_verify_rejects_missing_page_numbers():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS, include_page_numbers=False)
    result = verify_docx(doc, _state(), params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.DELIVERY_ELEMENT_MISSING.value and "PAGE" in d
               for c, d in result.problems)


def test_verify_can_waive_delivery_elements():
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS,
                          include_toc=False, include_page_numbers=False)
    assert verify_docx(doc, _state(), params=_PARAMS,
                       expect_toc=False, expect_page_numbers=False).accepted


def test_content_fidelity_unaffected_by_delivery_elements():
    # 交付要素不改内容：正文逐字仍在
    doc = render_proposal(_state(), brief=_BRIEF, params=_PARAMS)
    texts = [p.text for p in doc.paragraphs]
    assert "本项目依据GB 50011标准。" in texts       # R3 后逐字
