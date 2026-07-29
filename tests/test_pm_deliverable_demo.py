"""P-M/■1：无占位 DELIVERABLE demo 回归——committed 产物断言 DELIVERABLE（对比草稿套 SHAREABLE_DRAFT）。

断言的是管线行为（材料齐 → 无占位 → 可交付），不是准确率。
"""
from __future__ import annotations

import json
from pathlib import Path

from evidence.delivery import DeliveryState, delivery_state, has_unfilled_placeholders
from evidence.generation.orchestrator import OrchestrationState

_EXP = Path("examples/demo-proposal-deliverable/expected")


def _load(p):
    return json.loads((_EXP / p).read_text(encoding="utf-8"))


def test_deliverable_gap_report_all_filled():
    rep = _load("gap/gap-report.json")
    assert rep["status"] == "complete" and rep["ready"] is True
    assert all(s["status"] == "filled" for s in rep["slots"])   # 8 槽全 filled


def test_deliverable_state_has_no_placeholders():
    state = OrchestrationState.from_dict(_load("gen/state.json"))
    assert state.settled()
    assert has_unfilled_placeholders(state) is False            # 无占位 = 可交付


def test_deliverable_recheck_admits_as_deliverable():
    state = OrchestrationState.from_dict(_load("gen/state.json"))
    recheck = _load("recheck.json")
    assert recheck["accepted"] is True and recheck["binding_mode"] is True
    assert recheck["completeness"]["status"] == "COMPLETE"
    # 交付态 = DELIVERABLE（无占位），非 SHAREABLE_DRAFT
    assert delivery_state(state, recheck) is DeliveryState.DELIVERABLE


def test_deliverable_recheck_fresh_for_committed_state():
    from evidence.claim_audit import _load_section_paragraphs, build_audit_brief
    from evidence.integrity import assert_deliverable_recheck
    state_dict = _load("gen/state.json")
    brief = _load("gap/generation-brief.json")
    audit_map = _load("gap/generation-brief.audit.json")
    generated = {
        sid: "\n".join(p["text"] for p in outcome["draft"]["paragraphs"])
        for sid, outcome in state_dict["outcomes"].items()
        if outcome["status"] == "accepted" and outcome.get("draft")
    }
    audit_brief = build_audit_brief(
        brief=brief,
        generated_sections=generated,
        audit_map=audit_map,
        section_paragraphs=_load_section_paragraphs(state_dict),
    )
    recheck = _load("recheck.json")
    assert_deliverable_recheck(
        recheck, state_dict=state_dict, brief=brief,
        audit_brief=audit_brief, audit_map=audit_map)  # 不抛 = 交付闸放行
