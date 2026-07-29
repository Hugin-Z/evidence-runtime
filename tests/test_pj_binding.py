"""P-J：稿件-审计-交付的不可变绑定（按"对手怎么绕"验，不是按"用户怎么用"验）。

早期实现只验了诚实路径，没验对抗路径。本文件把两条规避
（A：什么都不审也 ADMITTED；B：旧报告批新稿）+ 同族第三条（C：改稿不重跑）**做成 fixture**，
证明现在都被拒；同时证明诚实往复不被误伤。
"""
from __future__ import annotations

import json

import pytest

from evidence.claim_audit import build_audit_brief, recheck_audit_sheet
from evidence.generation.executor import InteractiveFileExecutor
from evidence.generation.orchestrator import (
    OrchestrationState,
    Orchestrator,
    SectionOutcome,
)
from evidence.generation.section_spec import load_section_specs
from evidence.generation.task import DraftParagraph, SectionDraft
from evidence.integrity import (
    INPUT_MISMATCH,
    STALE_AUDIT,
    StaleAuditError,
    assert_deliverable_recheck,
    compute_input_digest,
    paragraph_content_hash,
    stamp_paragraphs,
    verify_recheck_fresh,
)

_CONTENT = "支持数据展示能力。"


# ── 绑定模式 AuditBrief 构造 ────────────────────────────────────────────────
def _section_paragraphs(sid, stamped):
    return [{"paragraph_id": p.paragraph_id, "content_hash": p.content_hash,
             "text": p.text, "is_placeholder": p.is_placeholder,
             "span_refs": list(p.span_refs)} for p in stamped]


def _binding_setup(paras=(("本平台支持数据展示。", ("e1",)),), *, content=_CONTENT,
                   gate="allow", oos=False):
    """返回 (audit_brief[绑定模式], audit_map, 已盖章段落)。paras: [(text, refs)]。"""
    stamped = stamp_paragraphs("02", tuple(DraftParagraph(t, tuple(r)) for t, r in paras))
    brief = {"meta": {}, "sections": [{
        "section_id": "02", "section_title": "章", "gate": None if oos else gate,
        "out_of_evidence_scope": oos,
        "acquired": [{"status": "acquired", "ref": "e1", "content": content,
                      "source": "《材料》", "supports_fields": [], "slot_id": "s1"}]}]}
    amap = {"e1": {"document_id": "d1", "start_char": 0, "end_char": len(content)}}
    ab = build_audit_brief(brief=brief, generated_sections={"02": paras[0][0]},
                           audit_map=amap,
                           section_paragraphs={"02": _section_paragraphs("02", stamped)})
    return ab, amap, stamped


def _sheet(claims, no_claim=()):
    return {"audit_sheet_schema_version": "m4-audit-sheet-v1",
            "meta": {"auditor": "test", "audited_at": "t"},
            "claims": list(claims), "no_claim_sections": list(no_claim)}


def _bound_claim(p, *, text=None, verdict="supported", refs=("e1",), hash_override=None,
                 unit=1, unit_override=None):
    """P-L：claim 绑定到 claim_unit（``<paragraph_id>#u<NN>``）；claim_text 默认=段落逐字（单句测试
    段落=单元，逐字即子串）。``unit_override`` 指定错误 unit_id 以测拒收。"""
    uid = unit_override if unit_override is not None else f"{p.paragraph_id}#u{unit:02d}"
    return {"claim_text": text or p.text, "section_id": "02", "verdict": verdict,
            "cited_span_refs": list(refs),
            "paragraph_id": p.paragraph_id,
            "paragraph_hash": hash_override if hash_override is not None else p.content_hash,
            "claim_unit_id": uid}


# ══ 诚实路径（P-F 往复不被误伤）══════════════════════════════════════════════
def test_honest_bound_claim_admitted():
    ab, amap, stamped = _binding_setup()
    sheet = _sheet([_bound_claim(stamped[0])])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["claims"][0]["recheck"] == "PASS"
    assert rep["completeness"]["mode"] == "unit"
    assert rep["completeness"]["status"] == "COMPLETE"
    assert rep["accepted"] is True
    assert rep["delivery"]["status"] == "ADMITTED"


# ══ 规避 A：审计单 0 claim / 全塞 no_claim_sections → 必须拒 ═══════════════════
def test_evasion_a_zero_claims_not_admitted():
    ab, amap, _ = _binding_setup()
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([]), audit_map=amap)
    assert rep["completeness"]["status"] == "INCOMPLETE_AUDIT"
    assert "02" in rep["completeness"]["uncovered_units"]
    assert rep["accepted"] is False


def test_evasion_a_all_no_claim_sections_not_admitted():
    """把有事实正文的 section 整节塞进 no_claim_sections（哪怕带 reason）→ 非法免审 → 拒。"""
    ab, amap, _ = _binding_setup()
    sheet = _sheet([], no_claim=[{"section_id": "02", "reason": "懒得审"}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["completeness"]["invalid_exemptions"]
    assert rep["completeness"]["invalid_exemptions"][0]["section_id"] == "02"
    assert rep["accepted"] is False


def test_no_claim_exemption_requires_reason():
    """占位-only section 可免审，但必须带 reason；无 reason → 非法免审。"""
    ab, amap, _ = _binding_setup(paras=(("【待补充：本节缺料】", ()),))  # 纯占位段
    # 无 reason 的裸字符串免审
    rep_bad = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([], no_claim=["02"]),
                                  audit_map=amap)
    assert rep_bad["accepted"] is False
    assert rep_bad["completeness"]["invalid_exemptions"]
    # 带 reason 且无事实段落 → 合法免审
    rep_ok = recheck_audit_sheet(
        audit_brief=ab, audit_map=amap,
        sheet=_sheet([], no_claim=[{"section_id": "02", "reason": "本节整节为待补料占位"}]))
    assert rep_ok["completeness"]["status"] == "COMPLETE"
    assert rep_ok["accepted"] is True


# ══ CLAIM_NOT_IN_DRAFT：claim 不来自当前稿 → 拒（■1b）══════════════════════════
def test_claim_missing_binding_rejected():
    ab, amap, stamped = _binding_setup()
    claim = _bound_claim(stamped[0])
    claim.pop("claim_unit_id")  # P-L：绑定键
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([claim]), audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"
    assert rep["accepted"] is False


def test_claim_wrong_hash_rejected():
    """claim 带旧/错的 paragraph_hash（审的是改过前的正文）→ 拒。"""
    ab, amap, stamped = _binding_setup()
    claim = _bound_claim(stamped[0], hash_override=paragraph_content_hash("完全不同的旧正文"))
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([claim]), audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"
    assert "不一致" in rep["claims"][0]["detail"]
    assert rep["accepted"] is False


def test_claim_unknown_unit_id_rejected():
    ab, amap, stamped = _binding_setup()
    claim = _bound_claim(stamped[0], unit_override="02#p01#u99")
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([claim]), audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"


def test_claim_text_not_substring_rejected():
    """P-L：claim_text 不是绑定单元的规范化子串（引文不来自当前稿）→ 拒。"""
    ab, amap, stamped = _binding_setup()
    claim = _bound_claim(stamped[0], text="本平台已覆盖全国、效率提升30%。")  # 不在正文里
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([claim]), audit_map=amap)
    assert rep["claims"][0]["recheck"] == "CLAIM_NOT_IN_DRAFT"


# ══ 段落级覆盖（■1e）══════════════════════════════════════════════════════════
def test_uncovered_factual_paragraph_incomplete():
    """两个事实段，只审了一个 → 另一个未覆盖 → INCOMPLETE_AUDIT。"""
    ab, amap, stamped = _binding_setup(
        paras=(("本平台支持数据展示。", ("e1",)), ("本平台还支持导出。", ("e1",))))
    sheet = _sheet([_bound_claim(stamped[0])])          # 只审 p01
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["completeness"]["status"] == "INCOMPLETE_AUDIT"
    assert f"{stamped[1].paragraph_id}#u01" in rep["completeness"]["uncovered_units"]["02"]
    assert rep["accepted"] is False


def test_placeholder_paragraph_needs_no_claim():
    """占位段落无需 claim；事实段落覆盖了即完整。"""
    ab, amap, stamped = _binding_setup(
        paras=(("本平台支持数据展示。", ("e1",)), ("【待补充：案例】", ())))
    sheet = _sheet([_bound_claim(stamped[0])])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["completeness"]["status"] == "COMPLETE"
    assert rep["accepted"] is True


def test_out_of_scope_claim_exempt_from_binding():
    """oos section 的 claim 不受绑定约束（走 drafter，不由本系统证据背书）。"""
    ab, amap, stamped = _binding_setup(oos=True)
    claim = {"claim_text": "x", "section_id": "02", "verdict": "out_of_scope",
             "cited_span_refs": []}
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_sheet([claim], no_claim=[]), audit_map=amap)
    # oos claim 不判 CLAIM_NOT_IN_DRAFT；但事实段落仍需覆盖 → 此处段落被 oos claim 覆盖不了，
    # 故仍 INCOMPLETE（正确：oos 段落该走别处，不该在本引擎交付版里冒充已审）。
    assert rep["claims"][0]["recheck"] == "PASS"


# ══ P-L/■4：section 级 legacy 不得进入交付路径（原"无绑定也应通过"是绕过，改断言）══════
# 【META-4 第二次发作】P-J 曾写 test_backward_compat_section_level_without_paragraphs 断言"无绑定
# 也应 accepted=True"——这条**保护的正是绕过路径 3**（agent 产 section 级 legacy 报告即可跳过单元
# 绑定与全覆盖）。P-L 改：legacy 报告 binding_mode=False，**交付闸一律拒**（LEGACY_BINDING_NOT_ALLOWED）；
# legacy 仅供历史查看。
def test_legacy_section_mode_barred_from_delivery():
    brief = {"meta": {}, "sections": [{
        "section_id": "02", "section_title": "章", "gate": "allow",
        "out_of_evidence_scope": False,
        "acquired": [{"status": "acquired", "ref": "e1", "content": _CONTENT,
                      "source": "s", "supports_fields": [], "slot_id": "s1"}]}]}
    amap = {"e1": {"document_id": "d", "start_char": 0, "end_char": len(_CONTENT)}}
    ab = build_audit_brief(brief=brief, generated_sections={"02": "本平台支持数据展示。"},
                           audit_map=amap)                # 不传 section_paragraphs → legacy
    sheet = _sheet([{"claim_text": "本平台支持数据展示。", "section_id": "02",
                     "verdict": "supported", "cited_span_refs": ["e1"]}])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=amap)
    assert rep["completeness"]["mode"] == "section"
    assert rep["binding_mode"] is False                  # legacy
    # 交付闸：legacy 报告不得进交付路径
    from evidence.integrity import DeliveryGateError, LEGACY_BINDING_NOT_ALLOWED
    rep["input_digest"] = "x"                             # 即便补上 digest 也先被 legacy 拦
    with pytest.raises(DeliveryGateError) as ei:
        assert_deliverable_recheck(
            rep, state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
            audit_brief=ab, audit_map=amap)
    assert ei.value.code == LEGACY_BINDING_NOT_ALLOWED


# ══ 输入摘要链（■1c）══════════════════════════════════════════════════════════
def _stamped_state(text="本平台支持数据展示。"):
    stamped = stamp_paragraphs("02", (DraftParagraph(text, ("e1",)),))
    oc = SectionOutcome("02", "章", "dispatched", "accepted", gate="allow",
                        draft=SectionDraft("02", stamped))
    return OrchestrationState(order=["02"], outcomes={"02": oc}).to_dict()


_DIGEST_BRIEF = {"brief_version": "v1", "contract_gate": "partial",
                 "meta": {"contract_id": "c1"},
                 "sections": [{"section_id": "02", "gate": "allow",
                               "out_of_evidence_scope": False}]}


def test_input_digest_deterministic_and_content_sensitive():
    ab, amap, _ = _binding_setup()
    d1 = compute_input_digest(
        state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
        audit_brief=ab, audit_map=amap)
    d2 = compute_input_digest(
        state_dict=_stamped_state(), brief=_DIGEST_BRIEF,
        audit_brief=ab, audit_map=amap)
    d3 = compute_input_digest(
        state_dict=_stamped_state("改过的正文。"), brief=_DIGEST_BRIEF,
        audit_brief=ab, audit_map=amap)
    assert d1 == d2 and d1 != d3


def test_verify_recheck_fresh_ok_and_stale():
    state = _stamped_state()
    ab, amap, _ = _binding_setup()
    digest = compute_input_digest(
        state_dict=state, brief=_DIGEST_BRIEF, audit_brief=ab, audit_map=amap)
    verify_recheck_fresh(
        {"input_digest": digest}, state_dict=state, brief=_DIGEST_BRIEF,
        audit_brief=ab, audit_map=amap)  # ok
    # 改过的正文 + 旧 digest → STALE_AUDIT
    with pytest.raises(StaleAuditError) as ei:
        verify_recheck_fresh({"input_digest": digest},
                             state_dict=_stamped_state("改过的正文."), brief=_DIGEST_BRIEF,
                             audit_brief=ab, audit_map=amap)
    assert ei.value.code == STALE_AUDIT
    # 缺 digest → INPUT_MISMATCH
    with pytest.raises(StaleAuditError) as ei2:
        verify_recheck_fresh(
            {}, state_dict=state, brief=_DIGEST_BRIEF,
            audit_brief=ab, audit_map=amap)
    assert ei2.value.code == INPUT_MISMATCH


# ══ 规避 B（旧报告批新稿）：render CLI 交付版新鲜度闸 → 拒 ══════════════════════
def _full_recheck(digest):
    """用真实 Claim Audit 路径产生完整 RecheckReport；不再手写授权布尔。"""
    ab, amap, stamped = _binding_setup()
    report = recheck_audit_sheet(
        audit_brief=ab,
        sheet=_sheet([_bound_claim(stamped[0])]),
        audit_map=amap,
    )
    report["input_digest"] = digest
    return report, ab, amap


def _write_session(tmp_path, text="本平台支持数据展示。"):
    session = tmp_path / "gen"
    session.mkdir()
    (session / "state.json").write_text(
        json.dumps(_stamped_state(text), ensure_ascii=False), encoding="utf-8")
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(_DIGEST_BRIEF, ensure_ascii=False), encoding="utf-8")
    return session, brief_path


def test_evasion_b_stale_recheck_render_rejected(tmp_path):
    from evidence.render.__main__ import main as render_main

    session, brief_path = _write_session(tmp_path, "本平台支持数据展示。")
    state_dict = json.loads((session / "state.json").read_text(encoding="utf-8"))
    ab0, amap0, _ = _binding_setup()
    digest = compute_input_digest(
        state_dict=state_dict, brief=_DIGEST_BRIEF,
        audit_brief=ab0, audit_map=amap0)
    # P-L：render 只接完整 RecheckReport（裸 admitted 会被拒）；这里给合规的完整报告
    recheck, audit_brief, audit_map = _full_recheck(digest)
    recheck_path = tmp_path / "recheck.json"
    recheck_path.write_text(json.dumps(recheck, ensure_ascii=False), encoding="utf-8")
    audit_brief_path = tmp_path / "audit-brief.json"
    audit_brief_path.write_text(
        json.dumps(audit_brief, ensure_ascii=False), encoding="utf-8")
    audit_map_path = tmp_path / "audit-map.json"
    audit_map_path.write_text(
        json.dumps(audit_map, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "p.docx"

    # 诚实路径：新鲜 recheck → 交付版渲染成功（exit 0）
    rc = render_main(["--session-dir", str(session), "--brief", str(brief_path),
                      "--recheck", str(recheck_path),
                      "--audit-brief", str(audit_brief_path),
                      "--audit-map", str(audit_map_path),
                      "--out", str(out)])
    assert rc == 0 and out.is_file()

    # 规避 B：改过正文（换 state），旧 recheck 不变 → 交付版被拒（exit 4，STALE_AUDIT）
    (session / "state.json").write_text(
        json.dumps(_stamped_state("本平台已覆盖全国、提升效率30%。"), ensure_ascii=False),
        encoding="utf-8")
    out2 = tmp_path / "p2.docx"
    rc2 = render_main(["--session-dir", str(session), "--brief", str(brief_path),
                       "--recheck", str(recheck_path),
                       "--audit-brief", str(audit_brief_path),
                       "--audit-map", str(audit_map_path),
                       "--out", str(out2)])
    assert rc2 == 4
    assert not out2.exists()


def test_cli_render_rejects_handwritten_full_report_with_made_up_ref(tmp_path):
    """P0-2 看门 fixture：完整字段和 admitted=true 不能掩盖 MADE-UP-REF。"""
    from evidence.render.__main__ import main as render_main

    session, brief_path = _write_session(tmp_path)
    state_dict = json.loads((session / "state.json").read_text(encoding="utf-8"))
    audit_brief, audit_map, _ = _binding_setup()
    digest = compute_input_digest(
        state_dict=state_dict, brief=_DIGEST_BRIEF,
        audit_brief=audit_brief, audit_map=audit_map)
    recheck, _, _ = _full_recheck(digest)
    recheck["claims"][0]["cited_span_refs"] = ["MADE-UP-REF"]
    recheck["claims"][0]["cited_content"] = {"MADE-UP-REF": "伪造证据"}

    recheck_path = tmp_path / "recheck.json"
    recheck_path.write_text(json.dumps(recheck, ensure_ascii=False), encoding="utf-8")
    audit_brief_path = tmp_path / "audit-brief.json"
    audit_brief_path.write_text(
        json.dumps(audit_brief, ensure_ascii=False), encoding="utf-8")
    audit_map_path = tmp_path / "audit-map.json"
    audit_map_path.write_text(
        json.dumps(audit_map, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "must-not-exist.docx"

    rc = render_main([
        "--session-dir", str(session),
        "--brief", str(brief_path),
        "--recheck", str(recheck_path),
        "--audit-brief", str(audit_brief_path),
        "--audit-map", str(audit_map_path),
        "--out", str(out),
    ])
    assert rc != 0
    assert not out.exists()


# ══ 规避 C（改稿不重跑）：编排器 re-run 检测回稿变更 → 旧 state 失效 ═══════════════
def _brief_one():
    return {"brief_version": "x", "meta": {"doc_type_zh": "解决方案建议书"},
            "sections": [{"section_id": "02", "section_title": "项目理解", "gate": "allow",
                          "out_of_evidence_scope": False, "reason": "",
                          "acquired": [{"asset_type": "a", "asset_types": ["a"],
                                        "status": "acquired", "content": "本项目依据上级批复。",
                                        "source": "《甲方材料》", "supports_fields": ["f1"],
                                        "slot_id": "s1", "gate": "allow",
                                        "single_source": False, "ref": "e1"}],
                          "missing_fields": [], "conflicts": []}]}


def _orch(tmp_path, **kw):
    return Orchestrator(brief=_brief_one(),
                        section_specs=load_section_specs("proposal-document"),
                        session_dir=tmp_path, **kw)


def _write_response(executor, text):
    executor.response_path("02").write_text(
        json.dumps(SectionDraft("02", (DraftParagraph(text, ("e1",)),)).to_dict(),
                   ensure_ascii=False), encoding="utf-8")


def test_evasion_c_edited_response_rerun_updates_state(tmp_path):
    executor = InteractiveFileExecutor(tmp_path)
    _orch(tmp_path).run(executor)                        # awaiting（写任务包）
    _write_response(executor, "初稿：本项目依据上级批复。")
    state = _orch(tmp_path).run(executor)
    assert state.outcomes["02"].status == "accepted"
    first_para = state.outcomes["02"].draft.paragraphs[0]
    assert first_para.paragraph_id == "02#p01"           # 已盖章
    assert first_para.content_hash

    # 改稿：编辑回稿文件后**重跑同一命令**（不清状态）→ state 必须更新（P-H backlog #1 结清）
    _write_response(executor, "改稿：本项目依据上级批复并新增导出能力。")
    state2 = _orch(tmp_path).run(executor)
    assert state2.outcomes["02"].status == "accepted"
    assert "改稿" in state2.outcomes["02"].draft.paragraphs[0].text
    assert (state2.outcomes["02"].draft.paragraphs[0].content_hash
            != first_para.content_hash)                  # 内容变、哈希变


def test_unchanged_response_is_not_reprocessed(tmp_path):
    """回稿没变 → 已 accepted 的 section 跳过（保持幂等，不无谓重跑）。"""
    executor = InteractiveFileExecutor(tmp_path)
    _orch(tmp_path).run(executor)
    _write_response(executor, "本项目依据上级批复。")
    state = _orch(tmp_path).run(executor)
    digest = state.outcomes["02"].response_digest
    state2 = _orch(tmp_path).run(executor)               # 回稿未动
    assert state2.outcomes["02"].response_digest == digest


def test_reset_section_forces_rerun(tmp_path):
    executor = InteractiveFileExecutor(tmp_path)
    _orch(tmp_path).run(executor)
    _write_response(executor, "初稿。")
    _orch(tmp_path).run(executor)
    _write_response(executor, "被 reset 强制重收的新稿。")
    state = _orch(tmp_path, reset_sections=["02"]).run(executor)
    assert "被 reset" in state.outcomes["02"].draft.paragraphs[0].text
