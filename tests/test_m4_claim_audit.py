"""M4 Claim Audit 单测：AuditBrief 构造 + 审计单 schema + 四类确定性硬拒。

反例 fixture 全部确定性（不依赖 LLM / 网络）：FABRICATED_REF / TAMPERED_CONTENT /
JURISDICTION_VIOLATION / INCOMPLETE_AUDIT 各一，断言硬拒。
"""

from __future__ import annotations

import pytest

from evidence.claim_audit import (
    AUDIT_BRIEF_VERSION,
    VERDICTS,
    AuditSheetError,
    RecheckCode,
    build_audit_brief,
    recheck_audit_sheet,
    validate_audit_sheet,
)

# ── 最小 GenerationBrief / audit.json fixture（对齐 M2 结构） ──────────────
BRIEF = {
    "brief_version": "m2-brief-v1",
    "meta": {"contract_id": "solution_material_completeness_v1", "doc_type": "proposal-document",
             "doc_type_zh": "解决方案建议书", "project_id": "p1"},
    "contract_gate": "partial",
    "sections": [
        {"section_id": "02", "section_title": "项目理解", "out_of_evidence_scope": False,
         "gate": "allow", "missing_fields": [], "conflicts": [],
         "acquired": [
             {"asset_type": "行业政策", "status": "acquired", "content": "平台面向工程建设与城市治理。",
              "source": "《甲文档》· 第1页", "supports_fields": ["basis_reference_or_background"],
              "slot_id": "background_and_basis", "gate": "allow", "single_source": True, "ref": "e1"},
         ]},
        {"section_id": "03", "section_title": "方案架构", "out_of_evidence_scope": False,
         "gate": "block", "missing_fields": [], "conflicts": [],   # 禁区
         "acquired": []},
        {"section_id": "01", "section_title": "公司实力", "out_of_evidence_scope": True,
         "gate": None, "reason": "走 L3", "acquired": [], "missing_fields": [], "conflicts": []},
    ],
}
AUDIT_MAP = {
    "e1": {"evidence_id": "ev-" + "a" * 64, "document_id": "doc-x", "start_char": 0,
           "end_char": 14, "start_line": 1, "end_line": 1, "page": 1,
           "discovery_sources": ["vector", "ripgrep"]},
}
GENERATED = {
    "02": "平台面向工程建设与城市治理。",
    "03": "本方案将实现全面的三维底座能力。",
    "01": "公司具备相关资质。",
}


def _brief():
    return build_audit_brief(brief=BRIEF, generated_sections=GENERATED, audit_map=AUDIT_MAP)


def _sheet(claims, *, no_claim_sections=None, auditor="self-audit(generator)"):
    return {
        "audit_sheet_schema_version": "m4-audit-sheet-v1",
        "meta": {"auditor": auditor, "audited_at": "2026-07-21T00:00:00Z"},
        "no_claim_sections": no_claim_sections or [],
        "claims": claims,
    }


def _claim(text, sid, verdict, refs=(), **kw):
    return {"claim_text": text, "section_id": sid, "verdict": verdict,
            "cited_span_refs": list(refs), "auditor_note": "", **kw}


# ── AuditBrief ────────────────────────────────────────────────────────────
def test_audit_brief_structure_and_evidence_pack():
    ab = _brief()
    assert ab["audit_brief_version"] == AUDIT_BRIEF_VERSION
    assert ab["restricted_sections"] == ["03"]          # gate=block → 禁区
    assert ab["out_of_scope_sections"] == ["01"]
    s02 = next(s for s in ab["sections"] if s["section_id"] == "02")
    assert s02["evidence_pack"][0]["span_ref"] == "e1"
    assert s02["evidence_pack"][0]["content"] == "平台面向工程建设与城市治理。"   # 逐字
    assert s02["generated_text"] == GENERATED["02"]
    # 指令齐备：拆 claim 标准 + 四级定义 + 引用规则 + 边界声明
    assert set(ab["instructions"]["verdicts"]) == set(VERDICTS)
    for key in ("claim_extraction", "citation_rule", "auditor_agnostic", "engine_boundary"):
        assert ab["instructions"][key]


def test_audit_brief_only_covers_generated_sections():
    ab = build_audit_brief(brief=BRIEF, generated_sections={"02": "x"}, audit_map=AUDIT_MAP)
    assert [s["section_id"] for s in ab["sections"]] == ["02"]


def test_audit_brief_rejects_ref_missing_from_audit_map():
    with pytest.raises(ValueError, match="不在 audit.json"):
        build_audit_brief(brief=BRIEF, generated_sections=GENERATED, audit_map={})


# ── schema 校验 ───────────────────────────────────────────────────────────
def test_valid_sheet_passes_schema():
    validate_audit_sheet(_sheet([_claim("平台面向工程建设。", "02", "supported", ["e1"])]))


@pytest.mark.parametrize("mutate, match", (
    (lambda s: s.pop("meta"), "meta"),
    (lambda s: s["meta"].update(auditor=""), "auditor"),
    (lambda s: s.update(claims="nope"), "claims 必须是列表"),
    (lambda s: s["claims"][0].update(verdict="great"), "不在"),
    (lambda s: s["claims"][0].update(cited_span_refs=[]), "supported 必须至少引"),
    (lambda s: s["claims"][0].update(claim_text=""), "claim_text"),
))
def test_invalid_sheets_rejected(mutate, match):
    sheet = _sheet([_claim("c", "02", "supported", ["e1"])])
    mutate(sheet)
    with pytest.raises(AuditSheetError, match=match):
        validate_audit_sheet(sheet)


# ── 四类确定性硬拒 ────────────────────────────────────────────────────────
def test_hard_reject_fabricated_ref():
    ab = _brief()
    sheet = _sheet([_claim("平台面向工程建设。", "02", "supported", ["e999"])],
                   no_claim_sections=["03", "01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["claims"][0]["recheck"] == RecheckCode.FABRICATED_REF.value
    assert "e999" in rep["claims"][0]["detail"]
    assert rep["accepted"] is False and rep["summary"]["hard_reject_count"] == 1


def test_hard_reject_tampered_content():
    ab = _brief()
    # 审计者把被引证据逐字改写了
    sheet = _sheet([_claim("平台面向工程建设。", "02", "supported", ["e1"],
                           cited_content={"e1": "平台面向全国所有城市治理。"})],
                   no_claim_sections=["03", "01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["claims"][0]["recheck"] == RecheckCode.TAMPERED_CONTENT.value
    assert rep["accepted"] is False


def test_hard_reject_jurisdiction_violation_restricted_section():
    ab = _brief()
    # section 03 gate=block（禁区），却判 supported
    sheet = _sheet([_claim("本方案已实现全面三维底座。", "03", "supported", ["e1"])],
                   no_claim_sections=["02", "01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["claims"][0]["recheck"] == RecheckCode.JURISDICTION_VIOLATION.value
    assert "禁区" in rep["claims"][0]["detail"]
    assert rep["accepted"] is False


def test_jurisdiction_violation_out_of_scope_section():
    ab = _brief()
    sheet = _sheet([_claim("公司具备资质。", "01", "supported", ["e1"])],
                   no_claim_sections=["02", "03"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["claims"][0]["recheck"] == RecheckCode.JURISDICTION_VIOLATION.value
    assert "out_of_evidence_scope" in rep["claims"][0]["detail"]


def test_hard_reject_tampered_content_via_source_rederivation(tmp_path):
    """给了 documents 则从原文按 audit.json 偏移重切，与 AuditBrief 发出内容比对——
    防 AuditBrief 本身在链路中被改写（审计单侧未引 content 也能查出）。"""
    from evidence.documents import NormalizedDocument

    path = tmp_path / "doc.md"
    # 长度需 ≥ audit.json 记录的 end_char，才走到「内容不一致」而非「偏移越界」分支。
    path.write_text("真实原文在此处与发出内容完全不同，足够长。", encoding="utf-8", newline="\n")
    document = NormalizedDocument.from_path(path, document_id="doc-x", source_path="m/doc.md")
    ab = _brief()   # 其 e1 发出内容为「平台面向工程建设与城市治理。」，与原文切片不符
    sheet = _sheet([_claim("平台面向工程建设。", "02", "supported", ["e1"])],
                   no_claim_sections=["03", "01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP,
                              documents=[document])
    assert rep["claims"][0]["recheck"] == RecheckCode.TAMPERED_CONTENT.value
    assert "原文重切" in rep["claims"][0]["detail"]


def test_incomplete_audit_when_section_uncovered():
    ab = _brief()
    sheet = _sheet([_claim("平台面向工程建设。", "02", "supported", ["e1"])])  # 漏 03 / 01
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["completeness"]["status"] == "INCOMPLETE_AUDIT"
    assert rep["completeness"]["uncovered_sections"] == ["01", "03"]
    assert rep["accepted"] is False


# ── 合法通过 + 边界 ───────────────────────────────────────────────────────
def test_clean_sheet_passes_recheck():
    """全 supported/out_of_scope、无 unsupported/contradicted → 规则过 **且** 交付准入通过。"""
    ab = _brief()
    sheet = _sheet([
        _claim("平台面向工程建设与城市治理。", "02", "supported", ["e1"],
               cited_content={"e1": "平台面向工程建设与城市治理。"}),
        _claim("公司具备相关资质。", "01", "out_of_scope"),
    ], no_claim_sections=["03"])                          # 禁区 03 无 claim（真实管线里 block 不派）
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert [c["recheck"] for c in rep["claims"]] == ["PASS"] * 2
    assert rep["completeness"]["status"] == "COMPLETE"
    assert rep["rule_check_passed"] is True
    assert rep["delivery"]["status"] == "ADMITTED"
    assert rep["accepted"] is True


def test_unsupported_claim_blocks_delivery_even_when_rule_clean():
    """P-F/P0-1 正解：审计者判 unsupported 即使规则层 PASS，也**阻断交付**（accepted=false）。"""
    ab = _brief()
    sheet = _sheet([
        _claim("平台面向工程建设与城市治理。", "02", "supported", ["e1"]),
        _claim("本方案已实现全面三维底座能力。", "03", "unsupported"),   # 审计者判 unsupported
    ], no_claim_sections=["01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP)
    assert rep["rule_check_passed"] is True                # 规则层无违规
    assert rep["delivery"]["status"] == "BLOCKED"          # 但交付被阻断
    assert rep["accepted"] is False
    assert rep["delivery"]["blocking_claims"][0]["verdict"] == "unsupported"


def test_signoff_admits_blocked_delivery_with_record():
    """人工签字放行：blocking claim 带签字人+理由 → 准入通过，签字留痕入档。"""
    ab = _brief()
    sheet = _sheet([_claim("本方案已实现全面三维底座能力。", "03", "unsupported")],
                   no_claim_sections=["02", "01"])
    rep = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP,
                              signoffs={0: {"signer": "项目负责人", "reason": "客户书面确认该能力"}})
    assert rep["delivery"]["status"] == "ADMITTED"
    assert rep["accepted"] is True
    assert rep["delivery"]["applied_signoffs"][0]["signer"] == "项目负责人"
    # 签字缺理由 → 不算放行，仍阻断
    rep2 = recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP,
                               signoffs={0: {"signer": "x", "reason": ""}})
    assert rep2["accepted"] is False


def test_engine_does_not_judge_semantic_strength():
    """supported vs weak 属语义判断：Engine 不复核，规则层同样 PASS（rule_check_passed）。"""
    ab = _brief()
    reps = []
    for verdict in ("supported", "weak"):
        sheet = _sheet([_claim("平台面向工程建设与城市治理。", "02", verdict, ["e1"])],
                       no_claim_sections=["03", "01"])
        reps.append(recheck_audit_sheet(audit_brief=ab, sheet=sheet, audit_map=AUDIT_MAP))
    assert all(r["claims"][0]["recheck"] == "PASS" for r in reps)
    assert all(r["rule_check_passed"] for r in reps)       # 规则层不区分 supported/weak
    # 交付准入：supported 通过；weak 在 weak_policy=block 下被拦（须处置）
    assert reps[0]["accepted"] is True                     # supported
    assert reps[1]["delivery"]["status"] == "BLOCKED"      # weak（默认策略）
    assert reps[1]["accepted"] is False
    # weak_policy=allow → weak 可带注交付
    allow = recheck_audit_sheet(audit_brief=ab,
        sheet=_sheet([_claim("平台面向工程建设与城市治理。", "02", "weak", ["e1"])],
                     no_claim_sections=["03", "01"]),
        audit_map=AUDIT_MAP, weak_policy="allow")
    assert allow["accepted"] is True


def test_auditor_identity_does_not_affect_recheck():
    """审计者无关：自审 vs 独立 agent，复核结果逐字段一致（auditor 仅作记录）。"""
    ab = _brief()
    claims = [
        _claim("平台面向工程建设与城市治理。", "02", "supported", ["e1"]),
        _claim("本方案将实现全面的三维底座能力。", "03", "unsupported"),
        _claim("公司具备相关资质。", "01", "out_of_scope"),
    ]
    a = recheck_audit_sheet(audit_brief=ab, sheet=_sheet(claims, auditor="self-audit(generator)"),
                            audit_map=AUDIT_MAP)
    b = recheck_audit_sheet(audit_brief=ab, sheet=_sheet(claims, auditor="independent-agent-42"),
                            audit_map=AUDIT_MAP)
    assert a["auditor"] != b["auditor"]          # 仅记录不同
    for key in ("claims", "summary", "completeness", "accepted"):
        assert a[key] == b[key]                   # 复核逻辑完全一致
