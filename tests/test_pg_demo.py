"""P-G：公开合成 demo 的回归测试（fake embedder，CI 无需外部服务）。

**断言的是管线行为，不是准确率**（demo 的边界见 examples/demo-proposal/README.md）。同时是
P-F 交付状态机的端到端回归：三态齐 / block 不派 / 交付准入 BLOCKED→ADMITTED / 交付版闸。
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evidence.core.config import DEFAULT_SETTINGS
from evidence.brief import build_generation_brief, load_crosswalk
from evidence.claim_audit import build_audit_brief, recheck_audit_sheet
from evidence.contract import load_evidence_contract
from evidence.fake_embedder import build_fake_gap_checker
from evidence.gap_checker import DEFAULT_CONTRACT_PATH
from evidence.generation.orchestrator import OrchestrationState
from evidence.render import render_proposal, verify_docx
from evidence.render.docx_render import RenderError
from evidence.runtime_adapter import adapt_path

_DEMO = Path("examples/demo-proposal")
_EXPECTED = _DEMO / "expected"


def _load(p):
    return json.loads((_EXPECTED / p).read_text(encoding="utf-8"))


# ── 1. committed 产物断言（不装 embedding 也能验的行为）──────────────────────
def test_committed_gap_report_has_all_three_states():
    rep = _load("gap/gap-report.json")
    counts = {}
    for s in rep["slots"]:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    assert counts.get("filled", 0) >= 1
    assert counts.get("weak", 0) >= 1
    assert counts.get("missing", 0) >= 1                 # 三态齐（缺案例 → missing）
    ref = next(s for s in rep["slots"] if s["slot_id"] == "reference_cases_and_proof")
    assert ref["status"] == "missing"


def test_committed_brief_gates_include_block_and_oos():
    brief = _load("gap/generation-brief.json")
    gates = {s["section_id"]: s.get("gate") for s in brief["sections"]}
    oos = {s["section_id"] for s in brief["sections"] if s.get("out_of_evidence_scope")}
    assert gates["05"] == "block"                        # missing 槽 → block section
    assert {"01", "06"} <= oos                           # 证据范围外
    assert gates["02"] == "allow"


def test_committed_generation_blocks_are_placeholders():
    state = OrchestrationState.from_dict(_load("gen/state.json"))
    assert state.settled()                               # 可写节全 accepted
    assert state.outcomes["05"].status == "placeholder"  # block 不派、占位
    assert state.outcomes["02"].status == "accepted"


# ── 2. 审计交付准入往复（committed 审计单，P-J 绑定模式）──────────────────────
def _section_paragraphs(state):
    """从 state 取各 accepted section 的**已盖章段落**（P-J 绑定模式所需）。"""
    out = {}
    for sid, oc in state["outcomes"].items():
        if oc["status"] != "accepted" or not oc.get("draft"):
            continue
        out[sid] = [{"paragraph_id": p["paragraph_id"], "content_hash": p["content_hash"],
                     "text": p["text"], "is_placeholder": "【待补充" in p["text"],
                     "span_refs": list(p.get("span_refs") or [])}
                    for p in oc["draft"]["paragraphs"]]
    return out


def _audit_brief_from(state_rel):
    brief = _load("gap/generation-brief.json")
    audit_map = _load("gap/generation-brief.audit.json")
    state = _load(state_rel)
    generated = {sid: "\n".join(p["text"] for p in oc["draft"]["paragraphs"])
                 for sid, oc in state["outcomes"].items()
                 if oc["status"] == "accepted" and oc.get("draft")}
    ab = build_audit_brief(brief=brief, generated_sections=generated, audit_map=audit_map,
                           section_paragraphs=_section_paragraphs(state))
    return ab, audit_map


def test_audit_sheet_v1_blocks_delivery():
    # v1 审的是 pre-disposition 稿（gen-draft1，含过度断言段落）
    ab, amap = _audit_brief_from("gen-draft1/state.json")
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_load("audit-sheet-v1-blocked.json"),
                              audit_map=amap)
    assert rep["completeness"]["mode"] == "unit"    # 绑定模式（段落级覆盖）
    assert rep["rule_check_passed"] is True              # 引用/绑定合法：规则层不拦
    assert rep["accepted"] is False                      # 但审计判 unsupported → 阻断
    assert rep["delivery"]["status"] == "BLOCKED"


def test_audit_sheet_v2_admits_delivery():
    ab, amap = _audit_brief_from("gen/state.json")       # honest 稿
    rep = recheck_audit_sheet(audit_brief=ab, sheet=_load("audit-sheet-v2-admitted.json"),
                              audit_map=amap)
    assert rep["completeness"]["mode"] == "unit"    # 绑定模式
    assert rep["completeness"]["status"] == "COMPLETE"   # 事实段落全被 claim 覆盖
    assert rep["accepted"] is True
    assert rep["delivery"]["status"] == "ADMITTED"


def test_committed_recheck_v2_has_fresh_input_digest():
    """committed recheck-v2 的 input_digest 与 committed gen state 一致（P-J 新鲜度可复现）。"""
    from evidence.integrity import compute_input_digest, verify_recheck_fresh
    recheck = _load("recheck-v2-admitted.json")
    state = _load("gen/state.json")
    brief = _load("gap/generation-brief.json")
    audit_brief, audit_map = _audit_brief_from("gen/state.json")
    assert recheck.get("input_digest") == compute_input_digest(
        state_dict=state, brief=brief, audit_brief=audit_brief, audit_map=audit_map)
    verify_recheck_fresh(
        recheck, state_dict=state, brief=brief,
        audit_brief=audit_brief, audit_map=audit_map)   # 不抛 = 新鲜


# ── 3. 交付版闸（committed state + recheck）────────────────────────────────
def test_render_deliverable_needs_admission():
    state = OrchestrationState.from_dict(_load("gen/state.json"))
    brief = _load("gap/generation-brief.json")
    params = {"project_name": "示范市智慧园区管理平台", "doc_type_zh": "解决方案建议书"}
    with pytest.raises(RenderError):                     # 无准入 → 拒
        render_proposal(state, brief=brief, params=params)
    audit_brief, audit_map = _audit_brief_from("gen/state.json")
    doc = render_proposal(
        state, brief=brief, params=params, recheck=_load("recheck-v2-admitted.json"),
        audit_brief=audit_brief, audit_map=audit_map)
    assert verify_docx(doc, state, params=params).accepted


# ── 4. live fake-embedder 重跑（CI 无服务；确认 committed 产物可复现）────────
def test_live_fake_embedder_reproduces_state_mix(tmp_path):
    contract = load_evidence_contract(str(DEFAULT_CONTRACT_PATH))
    documents = adapt_path(str(_DEMO / "materials"))
    settings = replace(DEFAULT_SETTINGS, vector_db_path=str(tmp_path / "chroma"))
    checker = build_fake_gap_checker(settings)
    run = checker.run(contract=contract, documents=documents, max_chars=800,
                      top_k_per_slot=8, context_lines=2, max_span_chars=1200,
                      max_results_per_slot=20)
    live = {s.slot_id: s.status.value for s in run.report.slots}
    committed = {s["slot_id"]: s["status"] for s in _load("gap/gap-report.json")["slots"]}
    assert live == committed                             # fake embedder 确定性 → 逐槽复现
