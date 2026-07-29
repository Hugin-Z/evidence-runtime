"""M2 GenerationBrief 单测：手段无关、状态映射、single_source、out_of_evidence_scope。

Brief 只读 GapReport 组装；这里用真实 GapReportBuilder 造 bundle，再断言 brief 的对外契约。
"""

from __future__ import annotations

import json

import pytest

from evidence.brief import (
    FORBIDDEN_OUTWARD_KEYS,
    assert_outward_clean,
    build_generation_brief,
    load_crosswalk,
)
from evidence.contract import solution_material_completeness_v1
from evidence.documents import NormalizedDocument
from evidence.gap_report import GapReportBuilder
from evidence.types import EvidenceSpan, SlotStatus
from evidence.verifier import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
    VerificationReasonCode,
)

DOC_TYPE = "解决方案建议书"


def _doc(tmp_path, name: str, text: str) -> NormalizedDocument:
    path = tmp_path / f"{name}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return NormalizedDocument.from_path(
        path, document_id=f"doc-{name}", source_path=f"materials/{name}.md",
        normalized_path=f"materials/{name}.md",
    )


def _span(document: NormalizedDocument) -> EvidenceSpan:
    return EvidenceSpan.from_document(
        document, 0, len(document.text), discovery_sources=("vector", "ripgrep")
    )


def _bundle(tmp_path, *, overrides=None, multi_source=()):
    """造 (contract, report, documents)。overrides: {slot_id: SlotStatus}；
    multi_source: 该槽的多个 required_field 由**不同文档**覆盖（→ single_source=False）。"""

    contract = solution_material_completeness_v1
    overrides = overrides or {}
    results = []
    evidence_by_slot = {}
    documents: list[NormalizedDocument] = []

    for index, slot in enumerate(contract.slots):
        status = overrides.get(slot.slot_id, SlotStatus.FILLED)
        if status is SlotStatus.MISSING:
            evidence_by_slot[slot.slot_id] = ()
            coverage = tuple(
                FieldCoverage(field_id=f, covered=False, evidence_ids=(), reason="缺")
                for f in slot.required_fields
            )
            results.append(SlotVerificationResult(
                contract_id=contract.contract_id, slot_id=slot.slot_id, status=status,
                evidence_ids=(), field_coverage=coverage,
                reason_codes=(VerificationReasonCode.NO_EVIDENCE,),
                reason="missing", conflicts=(), trace={},
            ))
            continue

        if status is SlotStatus.CONFLICTING:
            da = _doc(tmp_path, f"{index}a", f"{slot.name}已上线。")
            db = _doc(tmp_path, f"{index}b", f"{slot.name}未上线。")
            documents.extend([da, db])
            sa, sb = _span(da), _span(db)
            spans = (sa, sb)
            facts = (
                DetectedFact("系统", "状态", "已上线", "true", sa.evidence_id),
                DetectedFact("系统", "状态", "未上线", "false", sb.evidence_id),
            )
            conflicts = (DetectedConflict(
                conflict_type="opposite_assertion", subject="系统", predicate="状态",
                facts=facts, evidence_ids=tuple(f.evidence_id for f in facts),
                reason="状态冲突",
            ),)
            coverage = tuple(
                FieldCoverage(field_id=f, covered=True, evidence_ids=(sa.evidence_id,))
                for f in slot.required_fields
            )
            reason_codes = (VerificationReasonCode.CONFLICTING_FACTS,)
        else:
            multi = slot.slot_id in multi_source and len(slot.required_fields) >= 2
            da = _doc(tmp_path, f"{index}a", f"{slot.name}的建设内容与依据说明。")
            documents.append(da)
            sa = _span(da)
            spans = [sa]
            cover_ids_per_field = {}
            for fi, f in enumerate(slot.required_fields):
                if multi and fi == 1:
                    db = _doc(tmp_path, f"{index}b", f"{slot.name}的第二来源证据。")
                    documents.append(db)
                    sb = _span(db)
                    spans.append(sb)
                    cover_ids_per_field[f] = (sb.evidence_id,)
                else:
                    cover_ids_per_field[f] = (sa.evidence_id,)
            spans = tuple(spans)
            weak = status is SlotStatus.WEAK
            coverage = tuple(
                FieldCoverage(
                    field_id=f,
                    covered=(not weak) or fi == 0,   # weak：只覆盖第一个字段
                    evidence_ids=cover_ids_per_field[f] if ((not weak) or fi == 0) else (),
                    reason=None if ((not weak) or fi == 0) else "缺",
                )
                for fi, f in enumerate(slot.required_fields)
            )
            conflicts = ()
            reason_codes = () if status is SlotStatus.FILLED else (
                VerificationReasonCode.MISSING_REQUIRED_FIELD,
            )

        evidence_by_slot[slot.slot_id] = tuple(spans)
        results.append(SlotVerificationResult(
            contract_id=contract.contract_id, slot_id=slot.slot_id, status=status,
            evidence_ids=tuple(s.evidence_id for s in spans), field_coverage=coverage,
            reason_codes=reason_codes, reason=status.value, conflicts=conflicts, trace={},
        ))

    if any(r.status is SlotStatus.CONFLICTING for r in results):
        overall = "conflicted"
    elif all(r.status is SlotStatus.FILLED for r in results):
        overall = "complete"
    else:
        overall = "incomplete"
    verification = ContractVerificationResult(
        contract_id=contract.contract_id, status=overall, slots=tuple(results)
    )
    report = GapReportBuilder().build(
        contract=contract, verification=verification,
        evidence_by_slot=evidence_by_slot, trace={"document_count": len(documents)},
    )
    return contract, report, documents


def _brief(tmp_path, **kw):
    contract, report, documents = _bundle(tmp_path, **kw)
    crosswalk = load_crosswalk(DOC_TYPE, contract=contract)
    return build_generation_brief(
        report=report, documents=documents, crosswalk=crosswalk,
        contract=contract, project_id="proj-x",
    )


def _iter_acquired(brief):
    for sec in brief["sections"]:
        yield from sec["acquired"]


# ── 手段无关红线 ──────────────────────────────────────────────────────────
def test_brief_outward_is_means_agnostic(tmp_path):
    brief, audit = _brief(tmp_path)
    assert_outward_clean(brief)  # 不抛即通过

    # 冗余 grep 式断言：整份 brief 序列化后不含任一禁用字段名 / 内部值形态。
    blob = json.dumps(brief, ensure_ascii=False)
    for key in FORBIDDEN_OUTWARD_KEYS:
        assert f'"{key}"' not in blob, f"brief 对外层出现禁用字段 {key}"
    assert "ev-" not in blob  # evidence_id 前缀不外泄
    # 审计侧另存内部溯源（证明"另存"而非"删掉"）：含 evidence_id + 偏移 + discovery。
    some = next(iter(audit.values()))
    assert some["evidence_id"].startswith("ev-")
    assert "start_char" in some and "discovery_sources" in some


def test_forbidden_keys_detected_by_assert(tmp_path):
    brief, _ = _brief(tmp_path)
    brief["sections"][1]["acquired"][0]["distance"] = 0.1  # 人为注入
    with pytest.raises(AssertionError):
        assert_outward_clean(brief)


# ── 状态映射 ──────────────────────────────────────────────────────────────
def test_filled_slot_yields_content(tmp_path):
    brief, _ = _brief(tmp_path)  # 默认全 filled
    sec02 = next(s for s in brief["sections"] if s["section_id"] == "02")
    assert sec02["gate"] == "allow"
    assert sec02["acquired"], "filled 槽应产出证据"
    for item in sec02["acquired"]:
        assert item["status"] == "acquired"
        assert item["content"]                       # 非空逐字内容
        assert item["source"].startswith("《")        # 去标识来源串
        assert item["supports_fields"]               # 来自 FieldCoverage（列该 span 覆盖的字段）
        assert isinstance(item["asset_types"], list) and item["asset_types"]


def test_missing_slot_no_fabricated_content(tmp_path):
    # 让 background_and_basis（→ section 02 行业政策）missing。
    brief, _ = _brief(tmp_path, overrides={"background_and_basis": SlotStatus.MISSING})
    sec02 = next(s for s in brief["sections"] if s["section_id"] == "02")
    bg_items = [i for i in sec02["acquired"] if i["slot_id"] == "background_and_basis"]
    assert bg_items and all(i["status"] == "待补充" for i in bg_items)
    assert all(i["content"] is None for i in bg_items)   # 不编造 content
    assert any(m["slot_id"] == "background_and_basis" for m in sec02["missing_fields"])
    assert sec02["gate"] in {"limited", "block"}          # 混合 → 非 allow


def test_conflicting_slot_routed_to_conflicts(tmp_path):
    # solution_content_and_capabilities → section 03/04
    brief, _ = _brief(
        tmp_path, overrides={"solution_content_and_capabilities": SlotStatus.CONFLICTING}
    )
    sec03 = next(s for s in brief["sections"] if s["section_id"] == "03")
    assert sec03["gate"] == "needs_resolution"
    assert sec03["conflicts"], "conflicting 槽应进 conflicts[]"
    c = sec03["conflicts"][0]
    assert c["slot_id"] == "solution_content_and_capabilities"
    assert len(c["conflicting_values"]) >= 2
    assert all(src.startswith("《") for src in c["sources"])  # 去标识


# ── single_source flag ────────────────────────────────────────────────────
def test_single_source_flag_true_when_one_document(tmp_path):
    brief, _ = _brief(tmp_path)  # 默认每槽单文档覆盖
    items = [i for i in _iter_acquired(brief) if i["status"] == "acquired"]
    assert items and all(i["single_source"] is True for i in items)


def test_single_source_flag_false_when_multiple_documents(tmp_path):
    brief, _ = _brief(
        tmp_path, multi_source={"solution_content_and_capabilities"}
    )
    sc_items = [
        i for i in _iter_acquired(brief)
        if i["slot_id"] == "solution_content_and_capabilities" and i["status"] == "acquired"
    ]
    assert sc_items and all(i["single_source"] is False for i in sc_items)


# ── out_of_evidence_scope ─────────────────────────────────────────────────
def test_out_of_evidence_scope_sections_not_filled(tmp_path):
    brief, _ = _brief(tmp_path)
    for sid in ("01", "06"):  # 公司实力 / 报价概要
        sec = next(s for s in brief["sections"] if s["section_id"] == sid)
        assert sec["out_of_evidence_scope"] is True
        assert sec["acquired"] == []
        assert sec["gate"] is None
        assert sec["reason"]


# ── crosswalk 校验 + unmapped slots + 契约门 ──────────────────────────────
def test_crosswalk_rejects_unknown_slot_id(tmp_path, monkeypatch):
    from evidence import brief as brief_mod

    bad = {
        "crosswalk_version": "x", "doc_type": "x", "doc_type_zh": "x",
        "contract_id": solution_material_completeness_v1.contract_id,
        "sections": [
            {"section_id": "01", "out_of_evidence_scope": False,
             "asset_map": [{"asset_type": "a", "slot_id": "NOT_A_SLOT"}]}
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(brief_mod, "_CROSSWALK_DIR", tmp_path)
    with pytest.raises(brief_mod.CrosswalkError):
        load_crosswalk("bad", contract=solution_material_completeness_v1)


def test_unmapped_slots_are_listed(tmp_path):
    brief, _ = _brief(tmp_path)
    unmapped = {u["slot_id"] for u in brief["unmapped_slots"]}
    # 解决方案建议书 crosswalk 不消费这些槽（§5 盘点）
    assert {"objectives", "outcomes_and_metrics", "scope_and_subjects",
            "implementation_conditions_and_boundaries"} <= unmapped


def test_contract_gate_complete_when_all_filled(tmp_path):
    brief, _ = _brief(tmp_path)
    assert brief["overall_status"] == "complete"
    assert brief["contract_gate"] == "complete"
    assert brief["meta"]["project_id"] == "proj-x"
    assert brief["meta"]["contract_id"] == solution_material_completeness_v1.contract_id
