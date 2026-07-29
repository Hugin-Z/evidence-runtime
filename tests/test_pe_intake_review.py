"""P-E：intake 提交前复核 UX（补 P-B 欠账）单元测试。"""
from __future__ import annotations

import pytest

from evidence.intake import (
    IntakeAborted,
    SectionScope,
    render_answers_summary,
    review_answers,
    run_interactive,
)
from evidence.intake.questions import QUESTIONS


def _scripted(responses):
    it = iter(responses)
    return lambda _prompt: next(it)


def _index_of(qid):
    return next(i for i, q in enumerate(QUESTIONS) if q.id == qid)


def _base():
    return {
        "doc_type": "proposal-document",
        "project_name": "某市 CIM 平台",
        "client_name": "某市住建局",
        "project_id": "acme-2026",
    }


def test_summary_lists_answered_in_order():
    text = render_answers_summary(_base())
    assert "请复核你的回答" in text
    assert "proposal-document" in text
    assert "某市住建局" in text


def test_review_confirm_returns_answers_unchanged():
    answers = _base()
    out = review_answers(dict(answers), reader=_scripted([""]), writer=lambda *_: None)
    assert out == answers


def test_review_edit_by_index_changes_value():
    idx = _index_of("project_name")
    out = review_answers(
        _base(),
        reader=_scripted([str(idx), "改后的项目名", ""]),  # 选编号 → 新值 → 提交
        writer=lambda *_: None,
    )
    assert out["project_name"] == "改后的项目名"


def test_review_abort_raises():
    with pytest.raises(IntakeAborted):
        review_answers(_base(), reader=_scripted(["q"]), writer=lambda *_: None)


def test_review_edit_scope_to_all_prunes_section_ids():
    answers = _base()
    answers["section_scope"] = SectionScope.CUSTOM.value
    answers["section_ids"] = ["03", "04"]
    idx = _index_of("section_scope")
    out = review_answers(
        answers,
        reader=_scripted([str(idx), SectionScope.ALL.value, ""]),  # scope custom→all
        writer=lambda *_: None,
    )
    assert "section_ids" not in out          # 依赖失活 → 修剪


def test_review_invalid_index_reprompts():
    calls = []

    def reader(_prompt):
        calls.append(1)
        return "99" if len(calls) == 1 else ""   # 先无效编号，再提交
    out = review_answers(_base(), reader=reader, writer=lambda *_: None)
    assert out == _base()
    assert len(calls) == 2                    # 无效输入没有崩，重问一次


def test_run_interactive_with_review_end_to_end():
    # collect 逐问（section_ids 因 scope 非 custom 不问）+ 复核确认
    collect = ["proposal-document", "某市 CIM 平台", "某市住建局", "",  # doc/name/client/industry
               "acme-2026", "", "", "", "", "", ""]                     # project_id + 其余留空
    spec = run_interactive(reader=_scripted(collect + [""]), writer=lambda *_: None)
    assert spec.doc_type == "proposal-document"
    assert spec.project_name == "某市 CIM 平台"
    assert spec.section_scope == SectionScope.ALL.value


def test_run_interactive_review_can_be_disabled():
    collect = ["proposal-document", "某市 CIM 平台", "某市住建局", "",
               "acme-2026", "", "", "", "", "", ""]
    spec = run_interactive(reader=_scripted(collect), writer=lambda *_: None, review=False)
    assert spec.project_name == "某市 CIM 平台"      # 无复核步骤
