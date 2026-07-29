"""交付状态不变量、Claim Audit 收紧与 unsupported 回流的回归测试。"""
from __future__ import annotations

import pytest

from evidence.claim_audit import build_audit_brief, recheck_audit_sheet
from evidence.delivery import (
    DeliveryState,
    delivery_state,
    is_deliverable,
    remediation_directives,
)
from evidence.generation.orchestrator import OrchestrationState, SectionOutcome
from evidence.generation.task import DraftParagraph, SectionDraft
from evidence.render import render_proposal, verify_docx
from evidence.render.docx_render import RenderError, _render_proposal_unchecked
from evidence.render.verify_docx import VerifyCheck


# ── 状态构造 ──────────────────────────────────────────────────────────────
def _accepted(sid, text="本项目依据 GB 50011 标准。", refs=("e1",)):
    return SectionOutcome(sid, "章", "dispatched", "accepted", gate="allow",
                          draft=SectionDraft(sid, (DraftParagraph(text, tuple(refs)),)))


def _settled_state():
    return OrchestrationState(order=["02"], outcomes={"02": _accepted("02")})


def _unsettled_state(bad_status="awaiting"):
    return OrchestrationState(
        order=["02", "03"],
        outcomes={"02": _accepted("02"),
                  "03": SectionOutcome("03", "方案架构", "dispatched", bad_status, gate="allow")},
    )


_BRIEF = {"meta": {"doc_type_zh": "解决方案建议书"},
         "sections": [{"acquired": [{"status": "acquired", "ref": "e1",
                                     "source": "《材料》· 第1页"}]}]}
_PARAMS = {"project_name": "示例项目", "client_name": "示例甲方", "doc_type_zh": "解决方案建议书"}


def _audit_brief(generated, *, section="02", gate="allow", oos=False, content="支持数据展示能力。"):
    brief = {"meta": {}, "sections": [{
        "section_id": section, "section_title": "章", "gate": None if oos else gate,
        "out_of_evidence_scope": oos,
        "acquired": [{"status": "acquired", "ref": "e1", "content": content,
                      "source": "《材料》", "supports_fields": [], "slot_id": "s1"}]}]}
    amap = {"e1": {"document_id": "d1", "start_char": 0, "end_char": len(content)}}
    return build_audit_brief(brief=brief, generated_sections=generated, audit_map=amap), amap


def _sheet(claims):
    return {"audit_sheet_schema_version": "m4-audit-sheet-v1",
            "meta": {"auditor": "test", "audited_at": "t"}, "claims": claims}


# ── 反例 1（P0-2）：未就绪却渲成占位、verify=accepted ─────────────────────
def test_p02_unsettled_verify_rejects():
    """反例：settled=False + 占位 + verify accepted=true 仍须判 UNSETTLED_SESSION。"""
    state = _unsettled_state("failed")
    assert state.settled() is False                       # failed 不算终态（修复点）
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS, draft=True)  # 只能出草稿版
    result = verify_docx(doc, state, params=_PARAMS)      # 默认 require_settled
    assert not result.accepted
    assert any(c == VerifyCheck.UNSETTLED_SESSION.value for c, _ in result.problems)


def test_p02_render_deliverable_refused_when_unsettled():
    with pytest.raises(RenderError, match="未就绪"):
        render_proposal(_unsettled_state(), brief=_BRIEF, params=_PARAMS)


def test_p02_generation_settled_excludes_failed():
    assert _unsettled_state("failed").settled() is False
    assert _settled_state().settled() is True
    # 占位（oos/block）算合法终态
    ph = OrchestrationState(order=["01"], outcomes={
        "01": SectionOutcome("01", "公司实力", "placeholder", "placeholder")})
    assert ph.settled() is True


# ── 反例 2（P0-1）：绑合法证据、语义无据、审计判 unsupported → 阻断交付 ──────
def test_p01_unsupported_blocks_delivery_end_to_end():
    """收货管引用合法（不改）；审计判 unsupported → 交付准入阻断 → 走不到交付 docx。"""
    ab, amap = _audit_brief({"02": "本平台已覆盖全国、提升效率30%。"}, content="支持数据展示能力。")
    # 审计者：正文断言远超证据（证据只说"支持数据展示"）→ unsupported
    sheet = _sheet([{"claim_text": "本平台已覆盖全国、提升效率30%。", "section_id": "02",
                     "verdict": "unsupported", "cited_span_refs": ["e1"]}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["rule_check_passed"] is True               # 引用合法：规则层不拦
    assert rep["accepted"] is False                       # 但交付被阻断
    assert rep["delivery"]["status"] == "BLOCKED"
    # 交付版渲染被拒（未获准入）
    with pytest.raises(RenderError, match="完整 RecheckReport"):
        render_proposal(_settled_state(), brief=_BRIEF, params=_PARAMS)


def test_p01_remediate_by_signoff_then_admitted():
    ab, amap = _audit_brief({"02": "本平台已覆盖全国。"}, content="支持数据展示能力。")
    sheet = _sheet([{"claim_text": "本平台已覆盖全国。", "section_id": "02",
                     "verdict": "unsupported", "cited_span_refs": ["e1"]}])
    blocked = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert blocked["accepted"] is False
    directives = remediation_directives(blocked)
    assert directives and directives[0]["options"] == ["delete", "weaken", "signoff"]
    # 处置：人工签字放行（留痕）→ 重跑复核 → 准入
    admitted = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap,
                                   signoffs={0: {"signer": "负责人", "reason": "客户书面确认"}})
    assert admitted["accepted"] is True
    assert admitted["delivery"]["applied_signoffs"][0]["reason"] == "客户书面确认"


def test_p01_remediate_by_reaudit_after_removing_claim():
    """删除处置：改稿去掉无据断言 → 重审的审计单不再含该 claim → 准入通过。"""
    ab, amap = _audit_brief({"02": "本平台支持数据展示。"}, content="支持数据展示能力。")
    fixed_sheet = _sheet([{"claim_text": "本平台支持数据展示。", "section_id": "02",
                           "verdict": "supported", "cited_span_refs": ["e1"]}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=fixed_sheet, audit_map=amap)
    assert rep["accepted"] is True


# ── 反例 3（P0-3）：跨 section / 未知 section 引用 ─────────────────────────
def test_p03_cross_section_citation_rejected():
    brief = {"meta": {}, "sections": [
        {"section_id": "02", "section_title": "a", "gate": "allow", "out_of_evidence_scope": False,
         "acquired": [{"status": "acquired", "ref": "e1", "content": "证据A。", "source": "s",
                       "supports_fields": [], "slot_id": "s1"}]},
        {"section_id": "04", "section_title": "b", "gate": "allow", "out_of_evidence_scope": False,
         "acquired": [{"status": "acquired", "ref": "e2", "content": "证据B。", "source": "s",
                       "supports_fields": [], "slot_id": "s2"}]}]}
    amap = {"e1": {"document_id": "d", "start_char": 0, "end_char": 3},
            "e2": {"document_id": "d", "start_char": 3, "end_char": 6}}
    ab = build_audit_brief(brief=brief, generated_sections={"02": "x。", "04": "y。"}, audit_map=amap)
    # section 02 的 claim 却引 e2（属 04 的证据包）
    sheet = _sheet([{"claim_text": "x。", "section_id": "02", "verdict": "supported",
                     "cited_span_refs": ["e2"]},
                    {"claim_text": "y。", "section_id": "04", "verdict": "supported",
                     "cited_span_refs": ["e2"]}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CROSS_SECTION_CITATION"
    assert rep["claims"][1]["recheck"] == "PASS"
    assert rep["accepted"] is False


def test_p03_unknown_section_rejected():
    ab, amap = _audit_brief({"02": "x。"})
    sheet = _sheet([{"claim_text": "x。", "section_id": "99", "verdict": "supported",
                     "cited_span_refs": ["e1"]}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["claims"][0]["recheck"] == "UNKNOWN_SECTION"
    assert rep["accepted"] is False


# ── 状态机 ────────────────────────────────────────────────────────────────
def test_delivery_state_machine_transitions():
    ab, amap = _audit_brief({"02": "本平台支持数据展示。"})
    good = _sheet([{"claim_text": "本平台支持数据展示。", "section_id": "02",
                    "verdict": "supported", "cited_span_refs": ["e1"]}])
    bad = _sheet([{"claim_text": "本平台已覆盖全国。", "section_id": "02",
                   "verdict": "unsupported", "cited_span_refs": ["e1"]}])
    admitted = recheck_audit_sheet(audit_brief=ab, sheet=good, audit_map=amap)
    blocked = recheck_audit_sheet(audit_brief=ab, sheet=bad, audit_map=amap)
    unsettled, settled = _unsettled_state(), _settled_state()
    assert delivery_state(unsettled, None) is DeliveryState.UNSETTLED
    assert delivery_state(settled, None) is DeliveryState.GENERATED
    assert delivery_state(settled, blocked) is DeliveryState.BLOCKED
    assert delivery_state(settled, admitted) is DeliveryState.DELIVERABLE
    assert is_deliverable(settled, admitted) is True
    assert is_deliverable(settled, blocked) is False


def test_render_deliverable_only_when_settled_and_admitted():
    state = _settled_state()
    # settled 但未准入 → 拒
    with pytest.raises(RenderError):
        render_proposal(state, brief=_BRIEF, params=_PARAMS)
    # 装配机制另由内部纯装配器验证；公开入口只能由完整 recheck 放行。
    doc = _render_proposal_unchecked(state, brief=_BRIEF, params=_PARAMS)
    assert verify_docx(doc, state, params=_PARAMS).accepted


def test_draft_version_has_watermark():
    doc = render_proposal(_unsettled_state(), brief=_BRIEF, params=_PARAMS, draft=True)
    texts = [p.text for p in doc.paragraphs]
    headers = [p.text for s in doc.sections for p in s.header.paragraphs]
    assert any("草稿" in t for t in texts)                # 首行红条水印
    assert any("草稿" in h for h in headers)              # 页眉水印
