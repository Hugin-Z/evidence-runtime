"""交付闸不可绕过：四条已知绕过的看门测试。

威胁模型：交互执行者默认能写工作区。写在文件里的许可标记**不是**许可。
四条绕过共同的根：许可是"读出来的"，必须改成"算出来的"。这四条从此是交付闸的看门测试。
"""
from __future__ import annotations

import json

import pytest

from evidence.claim_audit import RECHECK_VERSION, build_audit_brief, recheck_audit_sheet
from evidence.generation.orchestrator import (
    OrchestrationState,
    SectionDraft,
    SectionOutcome,
)
from evidence.generation.task import DraftParagraph
from evidence.integrity import (
    HASH_MISMATCH,
    INCOMPLETE_REPORT,
    LEGACY_BINDING_NOT_ALLOWED,
    DeliveryGateError,
    HashMismatchError,
    assert_deliverable_recheck,
    assert_state_hashes_consistent,
    compute_input_digest,
    paragraph_content_hash,
    stamp_paragraphs,
)

_CONTENT = "支持数据展示能力。"
_DIGEST_BRIEF = {"brief_version": "v1", "contract_gate": "partial", "meta": {"contract_id": "c1"},
                 "sections": [{"section_id": "02", "gate": "allow", "out_of_evidence_scope": False}]}


def _audit_brief(paras, *, content=_CONTENT):
    """paras: [(text, refs)]. 返回 (audit_brief[绑定模式], amap, stamped)。"""
    stamped = stamp_paragraphs("02", tuple(DraftParagraph(t, tuple(r)) for t, r in paras))
    brief = {"meta": {}, "sections": [{
        "section_id": "02", "section_title": "章", "gate": "allow", "out_of_evidence_scope": False,
        "acquired": [{"status": "acquired", "ref": "e1", "content": content,
                      "source": "《材料》", "supports_fields": [], "slot_id": "s1"}]}]}
    amap = {"e1": {"document_id": "d1", "start_char": 0, "end_char": len(content)}}
    sp = {"02": [{"paragraph_id": p.paragraph_id, "content_hash": p.content_hash, "text": p.text,
                  "is_placeholder": p.is_placeholder, "span_refs": list(p.span_refs)} for p in stamped]}
    ab = build_audit_brief(brief=brief, generated_sections={"02": "\n".join(p.text for p in stamped)},
                           audit_map=amap, section_paragraphs=sp)
    return ab, amap, stamped


def _sheet(claims):
    return {"audit_sheet_schema_version": "m4-audit-sheet-v1",
            "meta": {"auditor": "t", "audited_at": "t"}, "claims": list(claims), "no_claim_sections": []}


def _stamped_state(text="本平台支持数据展示。"):
    stamped = stamp_paragraphs("02", (DraftParagraph(text, ("e1",)),))
    oc = SectionOutcome("02", "章", "dispatched", "accepted", gate="allow",
                        draft=SectionDraft("02", stamped))
    return OrchestrationState(order=["02"], outcomes={"02": oc}).to_dict()


# ══ 绕过 1：审计单只覆盖部分正文（原子单元 + 全覆盖封死）════════════════════════════
def test_bypass1_partial_coverage_blocked():
    """正文两句，审计单只提交一句 → 未覆盖的单元存在 → INCOMPLETE_AUDIT，拒。"""
    ab, amap, stamped = _audit_brief([("本平台已覆盖全国、效率提升30%。本平台支持数据展示。", ("e1",))])
    p = stamped[0]
    # 只审第二句，跳过"覆盖全国、效率提升30%"那句
    sheet = _sheet([{"claim_text": "本平台支持数据展示。", "section_id": "02", "verdict": "supported",
                     "cited_span_refs": ["e1"], "paragraph_id": p.paragraph_id,
                     "paragraph_hash": p.content_hash, "claim_unit_id": f"{p.paragraph_id}#u02"}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["completeness"]["status"] == "INCOMPLETE_AUDIT"
    assert f"{p.paragraph_id}#u01" in rep["completeness"]["uncovered_units"]["02"]  # 漏审那句
    assert rep["accepted"] is False


def test_bypass1_claim_not_in_draft_when_quoting_absent_text():
    """审计单引用正文里根本没有的话 → CLAIM_NOT_IN_DRAFT。"""
    ab, amap, stamped = _audit_brief([("本平台支持数据展示。", ("e1",))])
    p = stamped[0]
    sheet = _sheet([{"claim_text": "本平台已覆盖全国、效率提升30%。", "section_id": "02",
                     "verdict": "supported", "cited_span_refs": ["e1"], "paragraph_id": p.paragraph_id,
                     "paragraph_hash": p.content_hash, "claim_unit_id": f"{p.paragraph_id}#u01"}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"
    assert rep["accepted"] is False


def test_bypass1_substring_cannot_cover_whole_claim_unit():
    """一句含两个断言时，只审有证据的子串不得把整句标成已覆盖。"""
    text = "本平台已覆盖全国，并支持数据展示。"
    ab, amap, stamped = _audit_brief([(text, ("e1",))])
    p = stamped[0]
    sheet = _sheet([{"claim_text": "支持数据展示", "section_id": "02", "verdict": "supported",
                     "cited_span_refs": ["e1"], "paragraph_id": p.paragraph_id,
                     "paragraph_hash": p.content_hash, "claim_unit_id": f"{p.paragraph_id}#u01"}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"
    assert f"{p.paragraph_id}#u01" in rep["completeness"]["uncovered_units"]["02"]
    assert rep["accepted"] is False


def test_bypass1_substantive_claim_with_placeholder_marker_still_requires_audit():
    """占位标记不能给同句的实质断言免审。"""
    text = "本平台已服务全国500家单位【待补充：数字待核】"
    ab, amap, stamped = _audit_brief([(text, ("e1",))])
    rep = recheck_audit_sheet(
        audit_brief=ab, sheet=_sheet([]), audit_map=amap)
    unit_id = f"{stamped[0].paragraph_id}#u01"
    assert rep["completeness"]["status"] == "INCOMPLETE_AUDIT"
    assert unit_id in rep["completeness"]["uncovered_units"]["02"]
    assert rep["accepted"] is False


# ══ 绕过 2：改正文却留旧哈希（交付路径一切现算封死）══════════════════════════════════
def test_bypass2_edited_text_stale_hash_rejected_on_load():
    """state.json 正文被改、content_hash 没改 → 加载即 HashMismatchError（正文↔哈希现算不符）。"""
    st = _stamped_state("本平台支持数据展示。")
    # 篡改：改正文，保留旧哈希（模拟 agent 直接编辑 state.json）
    st["outcomes"]["02"]["draft"]["paragraphs"][0]["text"] = "本平台已覆盖全国、效率提升30%。"
    with pytest.raises(HashMismatchError) as ei:
        OrchestrationState.from_dict(st)
    assert ei.value.code == HASH_MISMATCH
    with pytest.raises(HashMismatchError):
        assert_state_hashes_consistent(st)


def test_bypass2_digest_recomputed_from_text_not_stored_hash():
    """输入摘要从**当前 text 现算**：改正文即便留旧哈希，摘要也变（旧 recheck 失配）。"""
    base = _stamped_state("本平台支持数据展示。")
    d_base = compute_input_digest(
        state_dict=base, brief=_DIGEST_BRIEF, audit_brief={}, audit_map={})
    tampered = json.loads(json.dumps(base))
    tampered["outcomes"]["02"]["draft"]["paragraphs"][0]["text"] = "改过的正文。"
    # 存储哈希仍是旧的，但摘要现算 → 不同
    d_tampered = compute_input_digest(
        state_dict=tampered, brief=_DIGEST_BRIEF, audit_brief={}, audit_map={})
    assert d_base != d_tampered


# ══ 绕过 3：section 级 legacy 报告进交付路径（拒 LEGACY_BINDING_NOT_ALLOWED）════════════
def test_bypass3_legacy_section_report_barred():
    legacy = {"recheck_version": RECHECK_VERSION, "binding_mode": False,
              "rule_check_passed": True, "accepted": True,
              "delivery": {"admitted": True}, "input_digest": "whatever"}
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            legacy, state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
            audit_brief={}, audit_map={})
    assert ei.value.code == LEGACY_BINDING_NOT_ALLOWED


# ══ 绕过 4：裸 {delivery:{admitted:true}}（拒 INCOMPLETE_REPORT）══════════════════════
def test_bypass4_bare_admitted_report_rejected():
    for bare in ({"delivery": {"admitted": True}},
                 {"delivery": {"admitted": True}, "input_digest": "x"},
                 {"recheck_version": "wrong", "delivery": {"admitted": True}}):
        with pytest.raises(DeliveryGateError) as ei:
            assert_deliverable_recheck(
                bare, state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
                audit_brief={}, audit_map={})
        assert ei.value.code in (INCOMPLETE_REPORT, LEGACY_BINDING_NOT_ALLOWED)


def test_bypass4_accepted_false_rejected():
    rep = {"recheck_version": RECHECK_VERSION, "binding_mode": True, "rule_check_passed": True,
           "accepted": False, "delivery": {"admitted": False}, "input_digest": "x"}
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
            audit_brief={}, audit_map={})
    assert ei.value.code == INCOMPLETE_REPORT


def _honest_full_report():
    state = _stamped_state(_CONTENT)
    ab, amap, stamped = _audit_brief([(_CONTENT, ("e1",))])
    p = stamped[0]
    sheet = _sheet([{"claim_text": _CONTENT, "section_id": "02", "verdict": "supported",
                     "cited_span_refs": ["e1"], "paragraph_id": p.paragraph_id,
                     "paragraph_hash": p.content_hash, "claim_unit_id": f"{p.paragraph_id}#u01"}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    digest = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF, audit_brief=ab, audit_map=amap)
    rep["input_digest"] = digest
    return state, rep, ab, amap


def test_bypass2_cached_pass_cannot_hide_fabricated_reference():
    """claims[].recheck 只是缓存；把引用改成不存在的 ref，交付门必须现场复算并拒绝。"""
    state, rep, audit_brief, audit_map = _honest_full_report()
    rep["claims"][0]["cited_span_refs"] = ["MADE-UP-REF"]
    rep["claims"][0]["cited_content"] = {"MADE-UP-REF": "伪造内容"}
    # 保留 recheck=PASS、summary/accepted/delivery.admitted=true，模拟手写完整报告。
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep,
            state_dict=state,
            brief=_DIGEST_BRIEF,
            audit_brief=audit_brief,
            audit_map=audit_map,
        )
    assert ei.value.code == INCOMPLETE_REPORT


def test_bypass2_cached_pass_cannot_hide_tampered_evidence_content():
    state, rep, audit_brief, audit_map = _honest_full_report()
    rep["claims"][0]["cited_content"]["e1"] = "篡改后的证据"
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=audit_brief, audit_map=audit_map)
    assert ei.value.code == INCOMPLETE_REPORT
    assert "TAMPERED_CONTENT" in str(ei.value)


def test_bypass2_cached_pass_cannot_hide_jurisdiction_violation():
    from copy import deepcopy

    state, rep, audit_brief, audit_map = _honest_full_report()
    restricted_brief = deepcopy(audit_brief)
    restricted_brief["restricted_sections"] = ["02"]
    restricted_brief["sections"][0]["gate"] = "block"
    restricted_brief["sections"][0]["restricted"] = True
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=restricted_brief, audit_map=audit_map)
    assert ei.value.code == INCOMPLETE_REPORT
    assert "JURISDICTION_VIOLATION" in str(ei.value)


def test_input_digest_covers_audit_brief_evidence_and_audit_map():
    from copy import deepcopy

    state, _, audit_brief, audit_map = _honest_full_report()
    baseline = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)
    changed_evidence = deepcopy(audit_brief)
    changed_evidence["sections"][0]["evidence_pack"][0]["content"] = "替换证据"
    changed_map = deepcopy(audit_map)
    changed_map["e1"]["end_char"] += 1
    assert baseline != compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=changed_evidence, audit_map=audit_map)
    assert baseline != compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=changed_map)


def test_bypass4_forged_full_report_without_derivation_rejected():
    """版本/布尔/digest 齐全但没有 claims/completeness/summary，仍不是完整复核报告。"""
    state = _stamped_state(_CONTENT)
    ab, amap, _ = _audit_brief([(_CONTENT, ("e1",))])
    digest = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF, audit_brief=ab, audit_map=amap)
    forged = {"recheck_version": RECHECK_VERSION, "binding_mode": True,
              "rule_check_passed": True, "accepted": True,
              "delivery": {"admitted": True}, "input_digest": digest}
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            forged, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=ab, audit_map=amap)
    assert ei.value.code == INCOMPLETE_REPORT


def test_bypass4_stored_booleans_cannot_override_claims():
    """只篡改 blocking_claims/admitted，也必须按 verdict 复算回 BLOCKED。"""
    state, _, audit_brief, audit_map = _honest_full_report()
    paragraph = audit_brief["sections"][0]["paragraphs"][0]
    rep = recheck_audit_sheet(
        audit_brief=audit_brief,
        audit_map=audit_map,
        sheet=_sheet([{
            "claim_text": _CONTENT,
            "section_id": "02",
            "verdict": "unsupported",
            "cited_span_refs": ["e1"],
            "paragraph_id": paragraph["paragraph_id"],
            "paragraph_hash": paragraph["content_hash"],
            "claim_unit_id": paragraph["claim_units"][0]["unit_id"],
        }]),
    )
    rep["input_digest"] = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)
    assert rep["delivery"]["blocking_claims"] and rep["delivery"]["admitted"] is False
    rep["delivery"]["blocking_claims"] = []
    rep["delivery"]["admitted"] = True
    rep["delivery"]["status"] = "ADMITTED"
    rep["delivery"]["reason"] = "伪造为已准入"
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=audit_brief, audit_map=audit_map)
    assert ei.value.code == INCOMPLETE_REPORT
    assert "delivery" in str(ei.value)


def test_gate_uses_stricter_of_report_and_deployment_weak_policy():
    """报告自报 weak=allow 不能降低渲染端默认 block；部署方显式 allow 才可放行。"""
    state, _, audit_brief, audit_map = _honest_full_report()
    paragraph = audit_brief["sections"][0]["paragraphs"][0]
    rep = recheck_audit_sheet(
        audit_brief=audit_brief,
        audit_map=audit_map,
        weak_policy="allow",
        sheet=_sheet([{
            "claim_text": _CONTENT,
            "section_id": "02",
            "verdict": "weak",
            "cited_span_refs": ["e1"],
            "paragraph_id": paragraph["paragraph_id"],
            "paragraph_hash": paragraph["content_hash"],
            "claim_unit_id": paragraph["claim_units"][0]["unit_id"],
        }]),
    )
    rep["input_digest"] = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)
    assert rep["delivery"]["weak_policy"] == "allow"
    assert rep["delivery"]["admitted"] is True

    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=audit_brief, audit_map=audit_map)
    assert ei.value.code == INCOMPLETE_REPORT

    assert_deliverable_recheck(
        rep, state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map,
        deployment_weak_policy="allow")


# ══ 诚实路径：完整合规报告 + 现算一致 → 通过（四道闸不误伤）══════════════════════════
def test_honest_full_report_passes_gate():
    state, rep, audit_brief, audit_map = _honest_full_report()
    assert_deliverable_recheck(
        rep, state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)   # 不抛 = 通过
    OrchestrationState.from_dict(state)                                       # 加载不抛 = 哈希一致
