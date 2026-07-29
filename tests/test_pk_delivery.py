"""P-K：交付语义拆分 + 政企 docx 质量 + 客户/内部隔离 + C5 输入完整性。

隔离靠**断言保护**（verify_docx 扫描），不是靠人工检查。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from docx.oxml import OxmlElement

from evidence.delivery import (
    DeliveryState,
    delivery_state,
    has_unfilled_placeholders,
    is_deliverable,
)
from evidence.generation.orchestrator import OrchestrationState, SectionOutcome
from evidence.generation.task import DraftParagraph, SectionDraft
from evidence.render import verify_docx
from evidence.render.docx_render import (
    SHAREABLE_WATERMARK_HEADER,
    _render_proposal_unchecked as render_proposal,
    render_to_file,
)
from evidence.render.docx_render import RenderError
from evidence.render.verify_docx import VerifyCheck

_BRIEF = {"meta": {"doc_type_zh": "解决方案建议书"},
          "sections": [{"acquired": [{"status": "acquired", "ref": "e1", "source": "《材料》"}]}]}
_PARAMS = {"project_name": "示例智慧园区项目", "client_name": "示例甲方",
           "doc_type_zh": "解决方案建议书"}
_ADMITTED = {"delivery": {"admitted": True}}


def _accepted(sid, text="本平台支持数据汇聚与展示。", refs=("e1",)):
    return SectionOutcome(sid, "章节", "dispatched", "accepted", gate="allow",
                          draft=SectionDraft(sid, (DraftParagraph(text, tuple(refs)),)))


def _accepted_with_ph(sid):
    return SectionOutcome(sid, "章节", "dispatched", "accepted", gate="limited",
                          draft=SectionDraft(sid, (
                              DraftParagraph("本平台支持数据汇聚。", ("e1",)),
                              DraftParagraph("【待补充：与需求目标的对齐说明】", ()))))


def _placeholder(sid, gate):
    # 内部 reason 含内部术语（模拟 orchestrator._placeholder 的产物）
    reason = {"block": "证据不足（缺字段：['reference_cases_and_proof']），阻断生成。",
              None: "out_of_evidence_scope：走 drafter L3 领域插件/intake。"}[gate]
    return SectionOutcome(sid, "章节", "placeholder", "placeholder", gate=gate,
                          placeholder_reason=reason)


def _clean_state():
    return OrchestrationState(order=["02"], outcomes={"02": _accepted("02")})


def _placeholder_state():
    return OrchestrationState(order=["01", "02"],
                             outcomes={"01": _placeholder("01", None), "02": _accepted("02")})


# ══ ■1 交付语义拆分 ═══════════════════════════════════════════════════════════
def test_state_split_clean_is_deliverable():
    assert delivery_state(_clean_state(), _ADMITTED) is DeliveryState.DELIVERABLE
    assert is_deliverable(_clean_state(), _ADMITTED) is True


def test_state_split_whole_section_placeholder_is_shareable():
    st = _placeholder_state()
    assert has_unfilled_placeholders(st) is True
    assert delivery_state(st, _ADMITTED) is DeliveryState.SHAREABLE_DRAFT
    assert is_deliverable(st, _ADMITTED) is False


def test_state_split_placeholder_paragraph_is_shareable():
    st = OrchestrationState(order=["02"], outcomes={"02": _accepted_with_ph("02")})
    assert delivery_state(st, _ADMITTED) is DeliveryState.SHAREABLE_DRAFT


def test_shareable_render_has_watermark_and_verifies():
    st = _placeholder_state()
    doc = render_proposal(st, brief=_BRIEF, params=_PARAMS)
    headers = [p.text for s in doc.sections for p in s.header.paragraphs]
    assert any("含待补料" in h for h in headers)          # SHAREABLE 水印
    assert verify_docx(doc, st, params=_PARAMS).accepted  # 态一致 + 无泄漏 → 过


def test_deliverable_render_clean_no_shareable_watermark():
    st = _clean_state()
    doc = render_proposal(st, brief=_BRIEF, params=_PARAMS)
    headers = "\n".join(p.text for s in doc.sections for p in s.header.paragraphs)
    assert SHAREABLE_WATERMARK_HEADER not in headers
    assert verify_docx(doc, st, params=_PARAMS).accepted


def test_verify_rejects_placeholder_doc_masquerading_as_clean():
    """含占位却被抹掉 SHAREABLE 水印（冒充干净 DELIVERABLE）→ DELIVERY_STATE_MISMATCH。"""
    st = _placeholder_state()
    doc = render_proposal(st, brief=_BRIEF, params=_PARAMS)
    # 抹掉水印：清空页眉 + 删除首行 banner 段落
    for s in doc.sections:
        for p in s.header.paragraphs:
            p.clear()
    for p in list(doc.paragraphs):
        if "含待补充材料" in p.text:
            p._p.getparent().remove(p._p)
    result = verify_docx(doc, st, params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.DELIVERY_STATE_MISMATCH.value for c, _ in result.problems)


# ══ ■2c 客户可见 vs 内部诊断隔离（断言保护）══════════════════════════════════════
def test_delivery_body_has_no_internal_terms():
    """交付/进度版正文只出客户语言占位——内部 reason（含 out_of_evidence_scope 等）不进正文。"""
    st = _placeholder_state()
    doc = render_proposal(st, brief=_BRIEF, params=_PARAMS)
    body = "\n".join(p.text for p in doc.paragraphs)
    assert "out_of_evidence_scope" not in body
    assert "drafter" not in body
    assert "reference_cases_and_proof" not in body
    assert "本节内容需另行提供的材料补充" in body           # 客户语言占位在


def test_verify_flags_internal_terms_in_non_audit():
    """审计版正文合法含内部诊断；若当作交付版回验（audit_mode=False）→ INTERNAL_TERM_LEAK。"""
    st = _placeholder_state()
    audit_doc = render_proposal(st, brief=_BRIEF, params=_PARAMS, audit_mode=True)
    # 审计版里有 【审计诊断】out_of_evidence_scope… → 交付版口径回验必判泄漏
    leaked = verify_docx(audit_doc, st, params=_PARAMS, audit_mode=False)
    assert any(c == VerifyCheck.INTERNAL_TERM_LEAK.value for c, _ in leaked.problems)
    # 审计版口径回验（audit_mode=True）跳过隔离扫描 → 过
    assert verify_docx(audit_doc, st, params=_PARAMS, audit_mode=True).accepted


# ══ ■2a/b/d docx 质量 ═════════════════════════════════════════════════════════
def test_page_is_a4():
    doc = render_proposal(_clean_state(), brief=_BRIEF, params=_PARAMS)
    s = doc.sections[0]
    assert round(s.page_width.cm, 1) == 21.0
    assert round(s.page_height.cm, 1) == 29.7


def test_toc_heading_not_heading_style():
    doc = render_proposal(_clean_state(), brief=_BRIEF, params=_PARAMS)
    heads = [p.text for p in doc.paragraphs
             if p.style and p.style.name and p.style.name.startswith("Heading 1")]
    texts = [p.text for p in doc.paragraphs]
    assert "目录" in texts and "目录" not in heads          # 目录在，但不是 Heading


def test_title_page_has_date_and_version():
    doc = render_proposal(_clean_state(), brief=_BRIEF,
                          params={**_PARAMS, "doc_version": "v2.1", "doc_date": "2026-07-28"})
    body = "\n".join(p.text for p in doc.paragraphs)
    assert "日期：2026-07-28" in body and "版本：v2.1" in body


def test_header_shows_project_name_by_default():
    doc = render_proposal(_clean_state(), brief=_BRIEF, params=_PARAMS)
    headers = "\n".join(p.text for s in doc.sections for p in s.header.paragraphs)
    assert "示例智慧园区项目" in headers


# ══ ■4 render_to_file 修复 ════════════════════════════════════════════════════
def test_render_to_file_default_still_refuses(tmp_path):
    # P-L：无 recheck（也非 draft）→ 拒
    with pytest.raises(RenderError):
        render_to_file(_clean_state(), str(tmp_path / "p.docx"), brief=_BRIEF, params=_PARAMS)


def test_render_to_file_rejects_bare_admitted(tmp_path):
    """P-L/■3：render_to_file 不再收裸 delivery_admitted；裸 {delivery:{admitted}} 报告被拒。"""
    from evidence.integrity import DeliveryGateError
    with pytest.raises(DeliveryGateError):
        render_to_file(_clean_state(), str(tmp_path / "p.docx"), brief=_BRIEF, params=_PARAMS,
                       recheck={"delivery": {"admitted": True}},
                       audit_brief={}, audit_map={})


def _full_render_gate_inputs():
    from evidence.claim_audit import build_audit_brief, recheck_audit_sheet
    from evidence.integrity import compute_input_digest
    from evidence.integrity import stamp_paragraphs

    stamped = stamp_paragraphs(
        "02", (DraftParagraph("本平台支持数据汇聚与展示。", ("e1",)),))
    st = OrchestrationState(order=["02"], outcomes={
        "02": SectionOutcome(
            "02", "章节", "dispatched", "accepted", gate="allow",
            draft=SectionDraft("02", stamped),
        )
    })
    audit_brief_input = {"meta": {}, "sections": [{
        "section_id": "02", "section_title": "章节", "gate": "allow",
        "out_of_evidence_scope": False,
        "acquired": [{"status": "acquired", "ref": "e1",
                      "content": "本平台支持数据汇聚与展示。", "source": "《材料》",
                      "supports_fields": [], "slot_id": "s1"}],
    }]}
    audit_map = {"e1": {"document_id": "d1", "start_char": 0, "end_char": 13}}
    paragraph = stamped[0]
    audit_brief = build_audit_brief(
        brief=audit_brief_input,
        generated_sections={"02": paragraph.text},
        audit_map=audit_map,
        section_paragraphs={"02": [{
            "paragraph_id": paragraph.paragraph_id, "content_hash": paragraph.content_hash,
            "text": paragraph.text, "is_placeholder": False, "span_refs": ["e1"],
        }]},
    )
    sheet = {
        "audit_sheet_schema_version": "m4-audit-sheet-v1",
        "meta": {"auditor": "test", "audited_at": "2026-07-29T00:00:00+08:00"},
        "no_claim_sections": [],
        "claims": [{
            "claim_text": paragraph.text, "section_id": "02", "verdict": "supported",
            "cited_span_refs": ["e1"], "paragraph_id": paragraph.paragraph_id,
            "paragraph_hash": paragraph.content_hash,
            "claim_unit_id": f"{paragraph.paragraph_id}#u01",
        }],
    }
    recheck = recheck_audit_sheet(
        audit_brief=audit_brief, sheet=sheet, audit_map=audit_map)
    digest = compute_input_digest(
        state_dict=st.to_dict(), brief=_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)
    recheck["input_digest"] = digest
    return st, audit_brief, audit_map, recheck


def test_public_render_proposal_has_no_naked_admission_boolean():
    """行为断言：公开入口必须复算证据，手写 PASS/admitted 不能掩盖伪造引用。"""
    from copy import deepcopy

    from evidence.integrity import DeliveryGateError
    from evidence.render import render_proposal as public_render

    st, audit_brief, audit_map, recheck = _full_render_gate_inputs()
    forged = deepcopy(recheck)
    forged["claims"][0]["cited_span_refs"] = ["MADE-UP-REF"]
    forged["claims"][0]["cited_content"] = {"MADE-UP-REF": "伪造内容"}
    with pytest.raises(DeliveryGateError):
        public_render(
            st,
            brief=_BRIEF,
            params=_PARAMS,
            recheck=forged,
            audit_brief=audit_brief,
            audit_map=audit_map,
        )


def test_render_to_file_with_full_recheck_works(tmp_path):
    st, audit_brief, audit_map, recheck = _full_render_gate_inputs()
    out = tmp_path / "p.docx"
    render_to_file(
        st,
        str(out),
        brief=_BRIEF,
        params=_PARAMS,
        recheck=recheck,
        audit_brief=audit_brief,
        audit_map=audit_map,
    )
    assert out.is_file()


# ══ ■3 C5 输入完整性 ══════════════════════════════════════════════════════════
def _mixed_pdf_bytes() -> bytes:
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R"
        b"/Resources<</Font<</F1 7 0 R>>>>>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 6 0 R>>",
        b"<</Length 46>>\nstream\nBT /F1 12 Tf 72 720 Td (Has text.) Tj ET\nendstream",
        b"<</Length 0>>\nstream\n\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = b"%PDF-1.4\n"
    offs = []
    for i, body in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


def test_c5_pdf_no_text_page_warns(tmp_path):
    from evidence.runtime_adapter import adapt_file
    p = tmp_path / "mixed.pdf"
    p.write_bytes(_mixed_pdf_bytes())
    gaps = adapt_file(p).metadata["runtime_adapter"]["extraction_gaps"]
    kinds = [g["kind"] for g in gaps]
    assert "pdf_no_text_page" in kinds
    assert 2 in next(g for g in gaps if g["kind"] == "pdf_no_text_page")["pages"]


def test_c5_docx_textbox_warns(tmp_path):
    from evidence.runtime_adapter import adapt_file
    doc = Document()
    doc.add_paragraph("正文：本项目建设内容包括统一数据底座。")
    run = doc.add_paragraph().add_run()
    drawing = OxmlElement("w:drawing")
    txbx = OxmlElement("w:txbxContent")
    ip = OxmlElement("w:p"); ir = OxmlElement("w:r"); it = OxmlElement("w:t")
    it.text = "文本框关键指标：效率提升 30%"
    ir.append(it); ip.append(ir); txbx.append(ip); drawing.append(txbx)
    run._r.append(drawing)
    p = tmp_path / "tb.docx"
    doc.save(str(p))
    kinds = [g["kind"] for g in adapt_file(p).metadata["runtime_adapter"]["extraction_gaps"]]
    assert "docx_textbox" in kinds


def test_c5_input_completeness_summary_and_markdown():
    from evidence.gap_report.markdown import _input_completeness_lines

    class _Doc:
        metadata = {"runtime_adapter": {"original_filename": "建设方案.docx",
                    "extraction_gaps": [{"kind": "docx_textbox", "count": 2,
                                         "detail": "正文含 2 个文本框，其中文字未提取。"}]}}
        document_id = "d1"
        source_path = "建设方案.docx"

    from evidence.runtime_adapter import input_completeness_summary
    summ = input_completeness_summary([_Doc()])
    assert summ["complete"] is False and summ["total_gaps"] == 1

    class _Report:
        trace = {"input_completeness": summ}
    md = "\n".join(_input_completeness_lines(_Report()))
    assert "输入完整性" in md and "部分材料未完整提取" in md and "建设方案.docx" in md
    assert "docx_textbox" in md
