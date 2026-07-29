"""P-C：L5 生成编排（任务组装 / 收货校验四类 / 退回重试 / 不派发）单元测试。"""
from __future__ import annotations

import json

import pytest

from evidence.generation import (
    CallableExecutor,
    DraftParagraph,
    InteractiveFileExecutor,
    Orchestrator,
    SectionDraft,
    SectionTask,
    TaskEvidence,
    assert_task_clean,
    check_draft,
    load_section_specs,
)
from evidence.generation.orchestrator import OrchestrationError
from evidence.generation.receiving import ReceivingCheck


# ── brief fixtures（means-agnostic，手工构造）────────────────────────────
def _acquired(ref, content, source="《甲方材料》· 第1页", fields=("f1",), slot="s1"):
    return {"asset_type": "a", "asset_types": ["a"], "status": "acquired",
            "content": content, "source": source, "supports_fields": list(fields),
            "slot_id": slot, "gate": "allow", "single_source": False, "ref": ref}


def _section(sid, title, gate, *, acquired=(), missing=(), conflicts=(), oos=False, reason=""):
    return {
        "section_id": sid, "section_title": title, "gate": None if oos else gate,
        "out_of_evidence_scope": oos, "reason": reason,
        "acquired": list(acquired),
        "missing_fields": [{"asset_types": ["a"], "slot_id": "s1", "field_id": f} for f in missing],
        "conflicts": list(conflicts),
    }


def _brief(*sections):
    return {"brief_version": "x", "meta": {"doc_type_zh": "解决方案建议书"},
            "sections": list(sections)}


def _orch(*sections, **kw):
    return Orchestrator(brief=_brief(*sections),
                        section_specs=load_section_specs("proposal-document"), **kw)


# ── 1. 任务组装 ──────────────────────────────────────────────────────────
def test_build_task_loads_verbatim_evidence_and_source():
    orch = _orch()
    section = _section("02", "项目理解", "allow",
                       acquired=[_acquired("e1", "本项目依据上级批复。")])
    task = orch.build_task(section)
    assert task.gate == "allow"
    assert task.evidence[0].span_ref == "e1"
    assert task.evidence[0].content == "本项目依据上级批复。"     # 逐字
    assert "第1页" in task.evidence[0].source                    # 去标识来源
    assert task.evidence_refs() == frozenset({"e1"})
    # S3 写作要求注入
    assert task.writing["structure"]
    # S4 约束注入
    assert any("证据优先" in c for c in task.constraints)


def test_build_task_limited_requires_note_and_extra_constraint():
    orch = _orch()
    task = orch.build_task(_section("05", "类似案例", "limited",
                                    acquired=[_acquired("e1", "案例A")], missing=["f2"]))
    assert task.limited_note_required is True
    assert task.missing_fields == ("f2",)
    assert any("gate=limited" in c for c in task.constraints)


def test_assert_task_clean_rejects_forbidden_key():
    orch = _orch()
    task = orch.build_task(_section("02", "项目理解", "allow",
                                    acquired=[_acquired("e1", "x")]))
    dirty = SectionTask(**{**task.__dict__,
                           "project_params": {"chunk_id": "leak"}})
    with pytest.raises(OrchestrationError, match="泄漏内部字段"):
        assert_task_clean(dirty)


def test_task_evidence_carries_no_offsets():
    orch = _orch()
    task = orch.build_task(_section("02", "项目理解", "allow",
                                    acquired=[_acquired("e1", "x")]))
    blob = json.dumps([e.to_dict() for e in task.evidence])
    for forbidden in ("start_char", "chunk_id", "evidence_id", "distance", "embedding"):
        assert forbidden not in blob


# ── 2. 收货校验四类 ──────────────────────────────────────────────────────
def _task(gate="allow", refs=(("e1", "证据一"),), missing=(), limited=False):
    orch = _orch()
    return orch.build_task(_section(
        "03", "方案架构", gate,
        acquired=[_acquired(r, c) for r, c in refs], missing=list(missing)))


def test_receiving_accepts_valid_draft():
    task = _task()
    draft = SectionDraft("03", (DraftParagraph("本方案基于既有能力建设。", ("e1",)),))
    result = check_draft(task, draft)
    assert result.accepted, result.problem_summary()


def test_receiving_rejects_fabricated_ref():
    task = _task()
    draft = SectionDraft("03", (DraftParagraph("凭空引用。", ("e9",)),))
    result = check_draft(task, draft)
    assert not result.accepted
    assert any(c == ReceivingCheck.FABRICATED_REF.value for c, _ in result.problems)


def test_receiving_rejects_out_of_package_ref():
    # 包外引用与伪造引用是同一判定：cited ⊄ 包内
    task = _task(refs=(("e1", "证据一"),))
    draft = SectionDraft("03", (DraftParagraph("越权引用。", ("e1", "e2")),))
    result = check_draft(task, draft)
    assert not result.accepted
    assert any("e2" in d for _, d in result.problems)


def test_receiving_rejects_missing_placeholder():
    task = _task(missing=("f2",))
    draft = SectionDraft("03", (DraftParagraph("只写了有证据的部分。", ("e1",)),))
    result = check_draft(task, draft)
    assert not result.accepted
    assert any(c == ReceivingCheck.MISSING_PLACEHOLDER.value for c, _ in result.problems)


def test_receiving_accepts_when_placeholder_present():
    task = _task(missing=("f2",))
    draft = SectionDraft("03", (
        DraftParagraph("有证据的部分。", ("e1",)),
        DraftParagraph("【待补充：f2 相关内容】", ()),
    ))
    assert check_draft(task, draft).accepted


def test_receiving_rejects_unbound_fact_paragraph():
    task = _task()
    draft = SectionDraft("03", (DraftParagraph("这段陈述没有绑定任何证据。", ()),))
    result = check_draft(task, draft)
    assert not result.accepted
    assert any(c == ReceivingCheck.FORMAT_VIOLATION.value for c, _ in result.problems)


def test_receiving_rejects_placeholder_with_refs():
    task = _task()
    draft = SectionDraft("03", (DraftParagraph("【待补充：x】", ("e1",)),))
    result = check_draft(task, draft)
    assert not result.accepted
    assert any("占位" in d for _, d in result.problems)


def test_receiving_rejects_tampered_content():
    task = _task(refs=(("e1", "实施周期6个月"),))
    draft = SectionDraft("03", (DraftParagraph("见依据。", ("e1",)),))
    result = check_draft(task, draft, cited_content={"e1": "实施周期3个月"})
    assert not result.accepted
    assert any(c == ReceivingCheck.TAMPERED_CONTENT.value for c, _ in result.problems)


def test_receiving_rejects_limited_without_note():
    task = _task(gate="limited", missing=())
    draft = SectionDraft("03", (DraftParagraph("有证据段。", ("e1",)),), limited_note=None)
    result = check_draft(task, draft)
    assert not result.accepted
    assert any("limited_note" in d for _, d in result.problems)


def test_receiving_accepts_limited_with_note():
    task = _task(gate="limited")
    draft = SectionDraft("03", (DraftParagraph("有证据段。", ("e1",)),),
                         limited_note="案例证据偏单一，覆盖面不足。")
    assert check_draft(task, draft).accepted


# ── 3. 不派发（block / needs_resolution / oos）───────────────────────────
def test_block_section_not_dispatched():
    calls = []
    orch = _orch(_section("03", "方案架构", "block", missing=["f1", "f2"]))
    orch.run(CallableExecutor(lambda t: calls.append(t) or SectionDraft("03", ())))
    state = orch.run(CallableExecutor(lambda t: calls.append(t)))
    assert calls == []                              # 从未派发
    assert state.outcomes["03"].status == "placeholder"
    assert "证据不足" in state.outcomes["03"].placeholder_reason


def test_needs_resolution_section_not_dispatched():
    calls = []
    orch = _orch(_section("03", "方案架构", "needs_resolution",
                          conflicts=[{"slot_id": "s1"}]))
    state = orch.run(CallableExecutor(lambda t: calls.append(t)))
    assert calls == []
    assert state.outcomes["03"].status == "placeholder"
    assert "冲突" in state.outcomes["03"].placeholder_reason


def test_out_of_scope_section_not_dispatched():
    calls = []
    orch = _orch(_section("01", "公司实力", "allow", oos=True, reason="公司资质"))
    state = orch.run(CallableExecutor(lambda t: calls.append(t)))
    assert calls == []
    assert state.outcomes["01"].status == "placeholder"
    assert "out_of_evidence_scope" in state.outcomes["01"].placeholder_reason


# ── 4. 退回重试 ──────────────────────────────────────────────────────────
def test_reject_then_fix_is_accepted_on_retry():
    drafts = iter([
        SectionDraft("02", (DraftParagraph("无绑定陈述。", ()),)),          # 第1稿：违规
        SectionDraft("02", (DraftParagraph("有绑定陈述。", ("e1",)),)),      # 第2稿：修好
    ])
    orch = _orch(_section("02", "项目理解", "allow", acquired=[_acquired("e1", "x")]))
    state = orch.run_until_settled(CallableExecutor(lambda t: next(drafts)))
    oc = state.outcomes["02"]
    assert oc.status == "accepted"
    assert oc.attempts == 2                          # 退回一次后接受


def test_exceeds_retry_limit_marks_failed():
    bad = SectionDraft("02", (DraftParagraph("永远不绑定。", ()),))
    orch = _orch(_section("02", "项目理解", "allow", acquired=[_acquired("e1", "x")]),
                 max_retries=2)
    state = orch.run_until_settled(CallableExecutor(lambda t: bad))
    oc = state.outcomes["02"]
    assert oc.status == "failed"
    assert oc.attempts == 3                          # 1 + 2 retries
    assert oc.problems


# ── 5. 交互执行方（awaiting → 写回稿 → 收货）──────────────────────────────
def test_interactive_executor_awaits_then_accepts(tmp_path):
    orch = _orch(_section("02", "项目理解", "allow", acquired=[_acquired("e1", "本项目依据。")]),
                 session_dir=tmp_path)
    executor = InteractiveFileExecutor(tmp_path)

    state = orch.run(executor)                       # 第一次：写任务包、无回稿 → awaiting
    assert state.outcomes["02"].status == "awaiting"
    assert (tmp_path / "tasks" / "02.task.md").is_file()
    assert (tmp_path / "tasks" / "02.task.json").is_file()

    # agent 写回稿
    executor.response_path("02").write_text(json.dumps(SectionDraft(
        "02", (DraftParagraph("本项目依据上级批复。", ("e1",)),)).to_dict(),
        ensure_ascii=False), encoding="utf-8")

    state = orch.run(executor)                       # 第二次：读回稿 → 收货通过
    assert state.outcomes["02"].status == "accepted"
    assert state.settled()


def test_interactive_state_persists_across_instances(tmp_path):
    section = _section("02", "项目理解", "allow", acquired=[_acquired("e1", "x")])
    Orchestrator(brief=_brief(section), session_dir=tmp_path).run(
        InteractiveFileExecutor(tmp_path))
    # 新实例应从 state.json 恢复 awaiting
    state = Orchestrator(brief=_brief(section), session_dir=tmp_path)._load_state()
    assert state.outcomes["02"].status == "awaiting"


# ── 6. 产物组装 ──────────────────────────────────────────────────────────
def test_assemble_markdown_has_content_binding_and_placeholder():
    orch = _orch(
        _section("02", "项目理解", "allow", acquired=[_acquired("e1", "依据X。")]),
        _section("06", "报价概要", "allow", oos=True, reason="报价"),
    )
    good = SectionDraft("02", (DraftParagraph("本项目依据X。", ("e1",)),))
    state = orch.run_until_settled(CallableExecutor(lambda t: good))
    md = orch.assemble_markdown(state)
    assert "## 02 项目理解" in md
    assert "依据：e1" in md
    assert "【本节暂缺】" in md                        # oos 占位
    assert "out_of_evidence_scope" in md


def test_orchestrator_requires_sections():
    with pytest.raises(OrchestrationError):
        Orchestrator(brief={"meta": {}})
