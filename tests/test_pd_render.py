"""P-D：L6 渲染（内容保真 / 占位保真 / R3 规范化 / verify_docx 四项失败 fixture）单元测试。"""
from __future__ import annotations

import pytest

from evidence.generation.orchestrator import OrchestrationState, SectionOutcome
from evidence.generation.task import DraftParagraph, SectionDraft
from evidence.render import normalize_cjk_spacing, verify_docx
from evidence.render.docx_render import (
    LIMITED_PREFIX,
    PLACEHOLDER_PREFIX,
    SOURCE_PREFIX,
    _render_proposal_unchecked as render_proposal,
)
from evidence.render.verify_docx import VerifyCheck


# ── fixtures（合成，非私有）──────────────────────────────────────────────
def _state():
    d02 = SectionDraft("02", (
        DraftParagraph("本项目依据 GB 50011 标准编制。", ("e1",)),
        DraftParagraph("【待补充：现状数据】", ()),
    ))
    d05 = SectionDraft("05", (DraftParagraph("【待补充：类似案例】", ()),),
                       limited_note="类似案例证据缺失，全节留白。")
    return OrchestrationState(
        order=["01", "02", "05"],
        outcomes={
            "01": SectionOutcome("01", "公司实力", "placeholder", "placeholder",
                                 placeholder_reason="out_of_evidence_scope：走 drafter L3。"),
            "02": SectionOutcome("02", "项目理解", "dispatched", "accepted",
                                 gate="allow", draft=d02),
            "05": SectionOutcome("05", "类似案例", "dispatched", "accepted",
                                 gate="limited", draft=d05),
        },
    )


_BRIEF = {"meta": {"doc_type_zh": "解决方案建议书"},
          "sections": [{"acquired": [
              {"status": "acquired", "ref": "e1", "source": "《标准汇编》· 第5页"}]}]}
_PARAMS = {"project_name": "测试项目", "client_name": "测试甲方",
           "doc_type_zh": "解决方案建议书"}


def _texts(doc):
    return [p.text for p in doc.paragraphs]


# ── R3 空格规范化 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("保证金 3000 元", "保证金3000元"),
    ("采用 TLS 加密", "采用TLS加密"),
    ("北京 office 地址", "北京office地址"),
    ("hello world", "hello world"),
    ("中文 中文", "中文 中文"),
    ("第一行\n第二行", "第一行\n第二行"),
    ("中 A 中 B 中", "中A中B中"),
])
def test_r3_normalization(raw, expected):
    assert normalize_cjk_spacing(raw) == expected
    assert normalize_cjk_spacing(expected) == expected   # 幂等


# ── 内容保真 ─────────────────────────────────────────────────────────────
def test_content_fidelity_verbatim_after_r3():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    texts = _texts(doc)
    # 正文逐字（R3 规范化后）进 docx，渲染不改写
    assert "本项目依据GB 50011标准编制。" in texts     # 依据↔GB / 50011↔标准 去空格；GB 50011 内部保留
    assert normalize_cjk_spacing("本项目依据 GB 50011 标准编制。") in texts


def test_render_then_verify_accepts():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    result = verify_docx(doc, state, params=_PARAMS)
    assert result.accepted, result.diff_summary()


# ── 占位保真 + 样式区分 ──────────────────────────────────────────────────
def test_placeholder_and_limited_note_visible_and_styled():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    reds = []      # 占位（红）
    oranges = []   # 不足标注（橙）
    for p in doc.paragraphs:
        for r in p.runs:
            rgb = str(r.font.color.rgb) if (r.font.color and r.font.color.rgb) else None
            if rgb == "C00000":
                reds.append(p.text)
            elif rgb == "B06000":
                oranges.append(p.text)
    assert any("待补充" in t for t in reds)                 # 待补充占位=红
    assert any(PLACEHOLDER_PREFIX in t for t in reds)       # oos 占位=红
    assert any(LIMITED_PREFIX in t for t in oranges)        # 不足标注=橙
    assert all(any(r.font.italic for r in
                   [rr for pp in doc.paragraphs if pp.text == t for rr in pp.runs])
               for t in reds)                                # 占位斜体


# ── 手段无关：span_ref 不进 docx；来源仅审计版可见 ─────────────────────────
def test_span_ref_never_in_docx_and_source_toggle():
    state = _state()
    delivery = _texts(render_proposal(state, brief=_BRIEF, params=_PARAMS, audit_mode=False))
    audit = _texts(render_proposal(state, brief=_BRIEF, params=_PARAMS, audit_mode=True))
    assert not any("e1" in t for t in delivery)              # span_ref 永不进正文
    assert not any("span_ref" in t for t in delivery + audit)
    assert not any(t.startswith(SOURCE_PREFIX) for t in delivery)   # 交付版无来源
    assert any(t.startswith(SOURCE_PREFIX) and "《标准汇编》" in t for t in audit)  # 审计版有去标识来源


# ── 完整性 + 顺序 ────────────────────────────────────────────────────────
def test_sections_in_crosswalk_order():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    heads = [p.text for p in doc.paragraphs
             if p.style and p.style.name.startswith("Heading 1") and p.text != "目录"]
    assert heads == ["01 公司实力", "02 项目理解", "05 类似案例"]


# ── verify_docx 四项失败 fixture ─────────────────────────────────────────
def _remove_paragraph(doc, predicate):
    for p in list(doc.paragraphs):
        if predicate(p.text):
            p._element.getparent().remove(p._element)
            return True
    return False


def test_verify_rejects_content_mismatch_tampered():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    # 改字：把正文一段替换掉
    for p in doc.paragraphs:
        if p.text.startswith("本项目依据"):
            for r in p.runs:
                r.text = ""
            p.runs[0].text = "本项目依据某标准编制。"   # 篡改
            break
    result = verify_docx(doc, state, params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.CONTENT_MISMATCH.value for c, _ in result.problems)


def test_verify_rejects_section_missing():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    assert _remove_paragraph(doc, lambda t: t == "05 类似案例")
    result = verify_docx(doc, state, params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.SECTION_MISSING.value for c, _ in result.problems)


def test_verify_rejects_placeholder_lost():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    assert _remove_paragraph(doc, lambda t: "现状数据" in t)   # 吞掉 【待补充】占位
    result = verify_docx(doc, state, params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.PLACEHOLDER_LOST.value for c, _ in result.problems)


def test_verify_rejects_limited_note_swallowed():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    assert _remove_paragraph(doc, lambda t: t.startswith(LIMITED_PREFIX))
    result = verify_docx(doc, state, params=_PARAMS)
    assert not result.accepted
    assert any(c == VerifyCheck.PLACEHOLDER_LOST.value for c, _ in result.problems)


def test_verify_rejects_wrong_param():
    state = _state()
    doc = render_proposal(state, brief=_BRIEF, params=_PARAMS)
    result = verify_docx(doc, state, params={**_PARAMS, "client_name": "另一个甲方"})
    assert not result.accepted
    assert any(c == VerifyCheck.PARAM_WRONG.value for c, _ in result.problems)
