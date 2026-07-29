"""L6′ 渲染：结构化方案稿 + ProjectSpec → 正式 DOCX。

政企文档的中文字体**必须经 XML
``w:eastAsia`` 设置**（只设 ``font.name`` 对 CJK 不生效，这是 python-docx 的坑），正文宋体四号、
标题黑体，使用厘米页边距。渲染输入是 :class:`OrchestrationState`，不读取外部模板脚本。

**渲染层铁律 = 零内容改写**：section 正文逐字进 docx，渲染只管排版。**唯一例外**是 R3 中文空格
规范化（:func:`evidence.render.text_norm.normalize_cjk_spacing`，确定性纯函数，verify 侧施加同一函数
比对，故不破坏内容保真）。

**产品诚实性落到交付物**：`【待补充】` 占位与 limited 不足标注在成品里**样式区分、读者一眼可辨"缺料"**，
绝不渲染成看起来完整的正文。证据来源（去标识 source）默认**不进正文**（手段无关），审计版
（``audit_mode=True``）以每段下方来源行呈现；``span_ref`` 永不进 docx。
"""
from __future__ import annotations

from typing import Mapping

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from ..generation.orchestrator import OrchestrationState
from .text_norm import normalize_cjk_spacing

# 政企标准字号（迁自 drafter DEFAULT_STYLES；宋体正文四号、黑体标题）
_DEFAULT_STYLES = {
    "Normal":    {"east_asia": "宋体", "size": 14, "bold": False},
    "Heading 1": {"east_asia": "黑体", "size": 16, "bold": True},
    "Heading 2": {"east_asia": "黑体", "size": 15, "bold": True},
}
_TEXT_COLOR = RGBColor(0x00, 0x00, 0x00)
_PLACEHOLDER_COLOR = RGBColor(0xC0, 0x00, 0x00)   # 醒目红：一眼可辨缺料
_NOTE_COLOR = RGBColor(0xB0, 0x60, 0x00)          # 橙褐：证据不足标注
_SOURCE_COLOR = RGBColor(0x70, 0x70, 0x70)        # 灰：审计版来源行

# verify_docx 用来在成品里认出结构的稳定标记（不影响读者阅读语义）
PLACEHOLDER_PREFIX = "【本节暂缺】"
LIMITED_PREFIX = "⚠ 证据不足："
SOURCE_PREFIX = "来源："
TOC_HEADING = "目录"
AUDIT_DIAG_PREFIX = "【审计诊断】"                 # 仅审计版可见的内部诊断行前缀

# 两种草稿态各有显著水印，客户一眼可辨。
DRAFT_WATERMARK_HEADER = "【草稿 · 未获交付准入 · 不得作为交付物】"
DRAFT_BANNER = "【草稿版：本文档尚未通过交付准入，可能含未就绪/未获支持内容，不得对外交付】"
SHAREABLE_WATERMARK_HEADER = "【草稿 · 含待补料 · 仅供内部/进度参考 · 非最终交付物】"
SHAREABLE_BANNER = "【本文档含待补充材料，为进度草稿，非最终交付版；标注处需补齐后方可对外交付】"

# 客户可见文案与内部诊断隔离
# 整节占位的**客户语言**文案（按 gate 分档）；内部 placeholder_reason 只进审计版诊断行。
_CLIENT_PLACEHOLDER_OOS = "本节内容需另行提供的材料补充（不在本次证据核验范围内）。"
_CLIENT_PLACEHOLDER_BY_GATE = {
    "needs_resolution": "本节存在需澄清的信息口径，待确认后补充。",
    "block": "本节所需材料尚不完整，待补充相应证据后完善。",
}

#: 客户可见正文**绝不允许**出现的内部术语（verify_docx 对非审计版断言其不出现）。
#: 全部是英文内部标识或内部中文行话，正常政企方案正文不会出现——不会误伤合法文案。
INTERNAL_TERMS = (
    "out_of_evidence_scope", "needs_resolution", "gate=", "gate 为", "span_ref",
    "paragraph_id", "content_hash", "slot_id", "field_id", "audit_map",
    "solution_material_completeness", "reference_cases_and_proof",
    "current_state_and_needs", "solution_content_and_capabilities",
    "scope_and_subjects", "background_and_basis",
    "走 drafter", "领域插件", "阻断生成", "缺字段：[", "缺字段：['",
)


def client_placeholder_text(outcome) -> str:
    """整节占位的客户语言文案；内部 ``placeholder_reason`` 仅审计版可见。"""
    return _CLIENT_PLACEHOLDER_BY_GATE.get(outcome.gate, _CLIENT_PLACEHOLDER_OOS)


def find_internal_terms(text: str) -> list[str]:
    """返回 text 中命中的内部术语（供 verify_docx 断言隔离）。"""
    return [t for t in INTERNAL_TERMS if t in text]


def _set_run_font(run, *, east_asia="宋体", size_pt=14, bold=False, italic=False, color=None):
    """中文字体必须经 XML ``w:eastAsia`` 设置（迁自 drafter set_run_font）。"""

    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color if color is not None else _TEXT_COLOR
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")


def _apply_styles(doc: Document) -> None:
    for name, conf in _DEFAULT_STYLES.items():
        try:
            style = doc.styles[name]
        except KeyError:
            continue
        style.font.name = "Times New Roman"
        style.font.size = Pt(conf["size"])
        style.font.bold = conf["bold"]
        style.font.color.rgb = _TEXT_COLOR
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.insert(0, rfonts)
        rfonts.set(qn("w:eastAsia"), conf["east_asia"])
        rfonts.set(qn("w:ascii"), "Times New Roman")


def _set_page(doc: Document, *, cm=2.54) -> None:
    """设置 A4 尺寸（21.0 × 29.7 cm）与四周边距。"""
    from docx.enum.section import WD_ORIENT

    for section in doc.sections:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(cm)
        section.left_margin = section.right_margin = Cm(cm)


def _append_field(paragraph, instr: str, placeholder: str = "") -> None:
    """在段落末尾插入一个 Word 域（``w:fldSimple``），如 ``PAGE`` / ``TOC``。

    交付时 placeholder 占位，Word 打开时按域指令重算（页码/目录页码由 Word 填）。
    """

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = placeholder
    run.append(text)
    fld.append(run)
    paragraph._p.append(fld)


def _add_page_numbers(doc: Document) -> None:
    """页脚居中「第 X 页」（X = PAGE 域，Word 打开时填真实页码）。"""

    for section in doc.sections:
        section.footer.is_linked_to_previous = False
        footer_p = section.footer.paragraphs[0] if section.footer.paragraphs \
            else section.footer.add_paragraph()
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_p.add_run("第 ")
        _set_run_font(run, size_pt=10.5, color=_SOURCE_COLOR)
        _append_field(footer_p, "PAGE", "1")
        tail = footer_p.add_run(" 页")
        _set_run_font(tail, size_pt=10.5, color=_SOURCE_COLOR)


def _fldchar(kind: str):
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), kind)
    return fc


def _set_update_fields_on_open(doc: Document) -> None:
    """在 settings.xml 写 ``<w:updateFields w:val="true"/>``：Word 打开时自动更新所有域，
    （TOC 真页码 / 页码域），**不依赖手动"更新域"**。真页码需版式引擎才知，故渲染期只能靠此自动更新。"""
    settings = doc.settings.element
    if settings.find(qn("w:updateFields")) is None:
        uf = OxmlElement("w:updateFields")
        uf.set(qn("w:val"), "true")
        settings.append(uf)


def _add_toc(doc: Document, entries: list[tuple[str, str]]) -> None:
    """目录页：

    - 「目录」标题用**直接排版**（黑体），**不用 Heading 1 样式**——否则 ``TOC \\o "1-2"`` 更新时
      会把"目录"二字自我收录进目录。
    - TOC 用 ``fldChar begin/separate/end`` 结构缓存章节清单；begin 打 ``w:dirty="true"`` +
      settings 的 ``updateFields`` → Word 打开时自动生成带真页码目录。
    - 缓存内容每行 = ``序号 标题 <tab> 页`` 结构（右对齐制表位），更新前也是目录样式；真页码 Word 填。
    """

    title = doc.add_paragraph()
    trun = title.add_run(TOC_HEADING)
    _set_run_font(trun, east_asia="黑体", size_pt=16, bold=True)

    p = doc.add_paragraph()
    begin = _fldchar("begin")
    begin.set(qn("w:dirty"), "true")                      # 标记域为脏，Word 会提示或自动更新
    p.add_run()._r.append(begin)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r' TOC \o "1-2" \h \z \u '
    p.add_run()._r.append(instr)
    p.add_run()._r.append(_fldchar("separate"))
    for i, (sid, sec_title) in enumerate(entries):        # 缓存内容：逐章节一行「标题…页」
        cr = p.add_run(f"{sid} {sec_title}")
        _set_run_font(cr, size_pt=12)
        tab = p.add_run("\t")                             # 制表位 → 页码右对齐（Word 更新时填真页码）
        _set_run_font(tab, size_pt=12)
        pr = p.add_run("·")                               # 页码占位（Word 更新为真页码）
        _set_run_font(pr, size_pt=12, color=_SOURCE_COLOR)
        if i != len(entries) - 1:
            p.add_run()._r.append(OxmlElement("w:br"))
    p.add_run()._r.append(_fldchar("end"))
    doc.add_page_break()


def _set_doc_metadata(doc: Document, params: Mapping[str, object]) -> None:
    """按 ProjectSpec 填写 title/subject/keywords/author，不保留 python-docx 默认值。"""
    cp = doc.core_properties
    cp.title = str(params.get("doc_type_zh") or "解决方案建议书")
    cp.subject = str(params.get("project_name") or "")
    kws = [str(params.get(k)) for k in ("industry_domain", "project_number") if params.get(k)]
    cp.keywords = " ".join(kws)
    cp.author = str(params.get("client_name") or "")      # 委托方或留空——绝不留 "python-docx"
    cp.comments = ""                                       # 清掉 python-docx 默认注释


def _add_header(doc: Document, text: str) -> None:
    """可选页眉（项目名，右对齐小字）。"""

    for section in doc.sections:
        section.header.is_linked_to_previous = False
        header_p = section.header.paragraphs[0] if section.header.paragraphs \
            else section.header.add_paragraph()
        header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = header_p.add_run(text)
        _set_run_font(run, size_pt=9, color=_SOURCE_COLOR)


def _para(doc: Document, text: str, *, east_asia="宋体", size_pt=14, bold=False,
          italic=False, color=None, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    _set_run_font(run, east_asia=east_asia, size_pt=size_pt, bold=bold,
                  italic=italic, color=color)
    return p


def _ref_source_map(brief: Mapping[str, object]) -> dict[str, str]:
    """从 brief 建 span_ref → 去标识 source 映射（source 进正文的唯一形态）。"""

    out: dict[str, str] = {}
    for section in brief.get("sections", []):          # type: ignore[union-attr]
        for a in section.get("acquired", []):
            if a.get("status") == "acquired" and a.get("ref"):
                out[str(a["ref"])] = str(a.get("source") or "未知来源")
    return out


class RenderError(RuntimeError):
    """渲染交付版被拒（会话未就绪 / 未获交付准入）。"""


def _render_proposal_unchecked(
    state: OrchestrationState,
    *,
    brief: Mapping[str, object],
    params: Mapping[str, object],
    audit_mode: bool = False,
    include_toc: bool = True,
    include_page_numbers: bool = True,
    include_header: bool = True,
    draft: bool = False,
) -> Document:
    """内部文档装配器：结构化方案稿 → docx（不承担交付授权）。

    ``audit_mode``：交付版（False，默认）**只出客户语言正文**，占位/不足以客户措辞呈现、内部诊断
    一律不进正文；审计版（True）每段下方显去标识来源行 + 占位处附内部诊断行（`【审计诊断】`）。
    ``include_header`` 默认 True；非草稿版页眉显示项目名。

    本函数以 ``_`` 开头且不从 :mod:`evidence.render` 导出，只供公开单门
    :func:`render_proposal` 在完成完整 RecheckReport 校验后调用，也供渲染机制单测使用。
    ``draft=False`` 仍要求会话 settled；授权校验只存在于公开入口，不能传裸布尔绕过。
    """
    from datetime import date

    from ..delivery import has_unfilled_placeholders

    if not draft:
        if not state.settled():
            raise RenderError(
                "拒绝渲染交付版：会话未就绪（NOT settled）——非终态 section："
                f"{state.unsettled_sections()}。请先补齐再交付，或渲染草稿版（draft=True）。"
            )

    # 交付态：draft（未准入）/ shareable（准入但含占位）/ deliverable（准入且无占位）
    is_shareable = (not draft) and has_unfilled_placeholders(state)

    doc = Document()
    _apply_styles(doc)
    _set_page(doc)
    _set_doc_metadata(doc, params)
    _set_update_fields_on_open(doc)                       # Word 打开时自动更新域
    ref_source = _ref_source_map(brief)

    if draft:                                             # 未准入草稿水印：页眉 + 首行红条
        _add_header(doc, DRAFT_WATERMARK_HEADER)
        _para(doc, DRAFT_BANNER, east_asia="黑体", size_pt=12, bold=True,
              color=_PLACEHOLDER_COLOR, align=WD_ALIGN_PARAGRAPH.CENTER)
    elif is_shareable:                                    # 准入含占位：SHAREABLE_DRAFT 水印
        _add_header(doc, SHAREABLE_WATERMARK_HEADER)
        _para(doc, SHAREABLE_BANNER, east_asia="黑体", size_pt=12, bold=True,
              color=_NOTE_COLOR, align=WD_ALIGN_PARAGRAPH.CENTER)

    # —— 标题页（参数面来自 ProjectSpec）——
    doc_title = str(params.get("doc_type_zh") or "解决方案建议书")
    project_name = str(params.get("project_name") or "")
    _para(doc, project_name or doc_title, east_asia="黑体", size_pt=22, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, doc_title, east_asia="黑体", size_pt=16, bold=True,
          align=WD_ALIGN_PARAGRAPH.CENTER)
    for label, key in (("委托方", "client_name"), ("项目编号", "project_number"),
                       ("行业域", "industry_domain")):
        value = params.get(key)
        if value:
            _para(doc, f"{label}：{value}", size_pt=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    # 标题页补日期与版本。
    doc_date = str(params.get("doc_date") or date.today().isoformat())
    doc_version = str(params.get("doc_version") or "v1.0")
    _para(doc, f"日期：{doc_date}　版本：{doc_version}", size_pt=12,
          align=WD_ALIGN_PARAGRAPH.CENTER, color=_SOURCE_COLOR)
    doc.add_page_break()

    if include_toc:
        _add_toc(doc, [(sid, state.outcomes[sid].section_title) for sid in state.order])
    if include_page_numbers:
        _add_page_numbers(doc)
    if include_header and project_name and not draft and not is_shareable:  # 水印页眉优先
        _add_header(doc, project_name)

    # —— 正文：按 crosswalk 顺序逐 section ——
    for sid in state.order:
        outcome = state.outcomes[sid]
        doc.add_paragraph(f"{sid} {outcome.section_title}", style="Heading 1")

        if outcome.status == "accepted" and outcome.draft is not None:
            for para in outcome.draft.paragraphs:
                text = normalize_cjk_spacing(para.text)          # R3：唯一允许的变换
                if para.is_placeholder:
                    _para(doc, text, italic=True, color=_PLACEHOLDER_COLOR)
                else:
                    _para(doc, text)
                    if audit_mode and para.span_refs:
                        sources = list(dict.fromkeys(
                            ref_source.get(r, "未知来源") for r in para.span_refs))
                        _para(doc, SOURCE_PREFIX + "；".join(sources),
                              size_pt=10.5, italic=True, color=_SOURCE_COLOR)
            if outcome.draft.limited_note:
                _para(doc, LIMITED_PREFIX + normalize_cjk_spacing(outcome.draft.limited_note),
                      italic=True, color=_NOTE_COLOR)
        elif outcome.status == "placeholder":
            # 正文只输出客户语言占位文案；内部 placeholder_reason 仅审计版可见。
            _para(doc, PLACEHOLDER_PREFIX + client_placeholder_text(outcome),
                  italic=True, color=_PLACEHOLDER_COLOR)
            if audit_mode and outcome.placeholder_reason:
                _para(doc, AUDIT_DIAG_PREFIX + outcome.placeholder_reason,
                      size_pt=10.5, italic=True, color=_SOURCE_COLOR)
        else:
            # settled 会话不会走到这里（只有 draft 版可能）；同样只出客户语言。
            _para(doc, f"{PLACEHOLDER_PREFIX}本节尚在完善中。",
                  italic=True, color=_PLACEHOLDER_COLOR)
            if audit_mode:
                _para(doc, f"{AUDIT_DIAG_PREFIX}status={outcome.status}",
                      size_pt=10.5, italic=True, color=_SOURCE_COLOR)

    return doc


def render_proposal(
    state: OrchestrationState,
    *,
    brief: Mapping[str, object],
    params: Mapping[str, object],
    recheck: Mapping[str, object] | None = None,
    audit_brief: Mapping[str, object] | None = None,
    audit_map: Mapping[str, Mapping[str, object]] | None = None,
    deployment_weak_policy: str = "block",
    audit_mode: bool = False,
    include_toc: bool = True,
    include_page_numbers: bool = True,
    include_header: bool = True,
    draft: bool = False,
) -> Document:
    """公开渲染单门：完整 RecheckReport 校验通过后才装配非草稿 DOCX。

    不接受 ``delivery_admitted`` 等裸布尔。``draft=True`` 只出带水印草稿，不需要 recheck；
    其余调用必须同时传完整报告、AuditBrief 与 audit map，由
    :func:`evidence.integrity.assert_deliverable_recheck` 复跑证据规则并从 claims、completeness
    与 signoff 重新派生授权。``deployment_weak_policy`` 是部署方独立策略，默认 ``block``；
    与报告声明取更严者，报告不能用 ``allow`` 自行降级。
    """

    if not draft:
        if not state.settled():
            raise RenderError(
                "拒绝渲染交付版：会话未就绪（NOT settled）——非终态 section："
                f"{state.unsettled_sections()}。请先补齐再交付，或渲染草稿版（draft=True）。"
            )
        if recheck is None:
            raise RenderError(
                "拒绝渲染交付版：需传完整 RecheckReport（recheck）；"
                "不能用裸 admitted 布尔自授权。可用 draft=True 出内部草稿。")
        if audit_brief is None or audit_map is None:
            raise RenderError(
                "拒绝渲染交付版：需同时传 AuditBrief（audit_brief）与 audit map（audit_map），"
                "以便现场复算引用存在性、逐字内容和辖区。")
        from ..integrity import assert_deliverable_recheck
        assert_deliverable_recheck(
            recheck,
            state_dict=state.to_dict(),
            brief=brief,
            audit_brief=audit_brief,
            audit_map=audit_map,
            deployment_weak_policy=deployment_weak_policy,
        )

    return _render_proposal_unchecked(
        state,
        brief=brief,
        params=params,
        audit_mode=audit_mode,
        include_toc=include_toc,
        include_page_numbers=include_page_numbers,
        include_header=include_header,
        draft=draft,
    )


def render_to_file(
    state: OrchestrationState,
    path: str,
    *,
    brief: Mapping[str, object],
    params: Mapping[str, object],
    recheck: Mapping[str, object] | None = None,
    audit_brief: Mapping[str, object] | None = None,
    audit_map: Mapping[str, Mapping[str, object]] | None = None,
    deployment_weak_policy: str = "block",
    audit_mode: bool = False,
    draft: bool = False,
) -> str:
    """便捷封装：校验交付准入 → render_proposal → 保存。

    公开 API 不接收裸 ``delivery_admitted`` 布尔；写在参数里的 admitted 不是许可。
    交付版必须传**完整 RecheckReport + AuditBrief + audit map**，本函数内部经
    :func:`evidence.integrity.assert_deliverable_recheck` 做全套校验（schema/version/binding_mode/
    证据规则/claims/completeness/signoff/input_digest 现算一致），不过即抛
    :class:`DeliveryGateError`。
    ``draft=True`` 出水印草稿版，不设闸。
    """
    doc = render_proposal(
        state,
        brief=brief,
        params=params,
        recheck=recheck,
        audit_brief=audit_brief,
        audit_map=audit_map,
        deployment_weak_policy=deployment_weak_policy,
        audit_mode=audit_mode,
        draft=draft,
    )
    doc.save(path)
    return path
