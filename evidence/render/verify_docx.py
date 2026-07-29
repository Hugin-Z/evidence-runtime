"""L6′ DOCX 回验：确定性比较渲染产物与结构化方案稿。

四项确定性校验，任一不过即拒收并报差异清单：

1. **内容一致**：DOCX 各 section 正文与方案稿逐字一致（两侧使用同一空格规范化函数）。
2. **完整性**：方案稿每个 section（含占位节）都在 docx，顺序符合 crosswalk（= state.order）。
3. **占位保真**：`【待补充】` 与 limited 不足标注未在渲染中丢失/被吞。
4. **参数正确**：标题页 project_name / client_name 等来自 ProjectSpec 的字段无误。

**边界**：只验"渲染有没有改动/丢失内容"，不判文档写得好不好。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from docx import Document
from docx.oxml.ns import qn

from ..delivery import has_unfilled_placeholders
from ..generation.orchestrator import OrchestrationState
from .docx_render import (
    LIMITED_PREFIX,
    PLACEHOLDER_PREFIX,
    SHAREABLE_BANNER,
    SHAREABLE_WATERMARK_HEADER,
    TOC_HEADING,
    client_placeholder_text,
    find_internal_terms,
)
from .text_norm import normalize_cjk_spacing


class VerifyCheck(str, Enum):
    CONTENT_MISMATCH = "CONTENT_MISMATCH"        # 正文逐字对不上（改字）
    SECTION_MISSING = "SECTION_MISSING"          # 丢节 / 顺序错
    PLACEHOLDER_LOST = "PLACEHOLDER_LOST"        # 占位或不足标注被吞
    PARAM_WRONG = "PARAM_WRONG"                  # 标题页参数缺失/错误
    DELIVERY_ELEMENT_MISSING = "DELIVERY_ELEMENT_MISSING"  # 目录/页码等交付要素缺失
    UNSETTLED_SESSION = "UNSETTLED_SESSION"      # 会话未就绪却被当交付版回验
    DELIVERY_STATE_MISMATCH = "DELIVERY_STATE_MISMATCH"  # 声称的交付态与实际内容不符
    INTERNAL_TERM_LEAK = "INTERNAL_TERM_LEAK"    # 客户可见正文泄漏内部术语


@dataclass
class VerifyResult:
    accepted: bool
    problems: list[tuple[str, str]] = field(default_factory=list)

    def diff_summary(self) -> str:
        if self.accepted:
            return "docx verified: content / completeness / placeholder / params all consistent"
        return "\n".join(f"  - {code}: {detail}" for code, detail in self.problems)


def _docx_paragraph_texts(doc: Document) -> list[str]:
    return [p.text for p in doc.paragraphs]


def _docx_headings(doc: Document) -> list[str]:
    return [p.text for p in doc.paragraphs
            if (p.style and p.style.name and p.style.name.startswith("Heading 1"))]


def _has_field(element, keyword: str) -> bool:
    """element 子树里是否有含 keyword 的域——``w:fldSimple`` 的 instr，或 ``w:instrText``
    文本；TOC 可用 fldChar+instrText，PAGE 也可能是 fldSimple，两种都识别。"""

    for fld in element.iter(qn("w:fldSimple")):
        if keyword in (fld.get(qn("w:instr")) or ""):
            return True
    for instr in element.iter(qn("w:instrText")):
        if keyword in (instr.text or ""):
            return True
    return False


def _header_texts(doc: Document) -> list[str]:
    return [p.text for s in doc.sections for p in s.header.paragraphs]


def verify_docx(
    doc: Document,
    state: OrchestrationState,
    *,
    params: Mapping[str, object],
    expect_toc: bool = True,
    expect_page_numbers: bool = True,
    require_settled: bool = True,
    audit_mode: bool = False,
) -> VerifyResult:
    """对渲染产物做确定性回验（四项 + 交付要素 + 会话就绪 + 交付态一致 + 内部术语隔离）。

    ``require_settled`` 默认为真；会话非终态即判 ``UNSETTLED_SESSION``。草稿版传 False。
    ``audit_mode``：审计版正文合法含内部诊断行，故**跳过内部术语隔离扫描**；交付/进度版（False）
    则断言客户可见正文**零内部术语**。

    **交付态一致**：``require_settled`` 且非审计版时，校验"声称的交付态 ↔ 实际内容"——
    含待补料占位却无 SHAREABLE 水印（即冒充干净 DELIVERABLE）→ ``DELIVERY_STATE_MISMATCH``。
    """

    problems: list[tuple[str, str]] = []
    if require_settled and not state.settled():
        problems.append((VerifyCheck.UNSETTLED_SESSION.value,
                         f"会话未就绪，不得作为交付版回验；非终态 section：{state.unsettled_sections()}"))
    all_texts = _docx_paragraph_texts(doc)
    joined = "\n".join(all_texts)
    text_set = set(all_texts)
    header_joined = "\n".join(_header_texts(doc))

    # 含占位必须打 SHAREABLE 水印，否则是冒充干净交付版。
    if require_settled and not audit_mode:
        has_ph = has_unfilled_placeholders(state)
        has_wm = (SHAREABLE_WATERMARK_HEADER in header_joined
                  or any(SHAREABLE_BANNER in t for t in all_texts))
        if has_ph and not has_wm:
            problems.append((VerifyCheck.DELIVERY_STATE_MISMATCH.value,
                             "文档含待补料占位却无 SHAREABLE_DRAFT 水印——不得冒充干净 DELIVERABLE"))
        if not has_ph and has_wm:
            problems.append((VerifyCheck.DELIVERY_STATE_MISMATCH.value,
                             "文档无占位却打了 SHAREABLE_DRAFT 水印——交付态标注不实"))

    # 内部术语隔离：客户可见正文不得含内部术语。
    if not audit_mode:
        hits = find_internal_terms(joined)
        if hits:
            problems.append((VerifyCheck.INTERNAL_TERM_LEAK.value,
                             f"客户可见正文泄漏内部术语：{sorted(set(hits))}"))

    def present(needle: str) -> bool:
        # 段落精确命中，或作为某段落子串（占位/标注带前缀 → 子串更稳）
        return needle in text_set or any(needle in t for t in all_texts)

    # 4) 参数正确（标题页）
    for key in ("project_name", "client_name"):
        value = params.get(key)
        if value and not present(str(value)):
            problems.append((VerifyCheck.PARAM_WRONG.value,
                             f"标题页缺少参数 {key}={value!r}"))

    # 2) 完整性 + 顺序：每个 section 标题在 docx，顺序 == state.order
    headings = _docx_headings(doc)
    expected_heads = [f"{sid} {state.outcomes[sid].section_title}" for sid in state.order]
    for head in expected_heads:
        if head not in headings:
            problems.append((VerifyCheck.SECTION_MISSING.value, f"缺少 section 标题：{head!r}"))
    ordered_present = [h for h in expected_heads if h in headings]
    if [h for h in headings if h in expected_heads] != ordered_present:
        problems.append((VerifyCheck.SECTION_MISSING.value,
                         f"section 顺序与 crosswalk 不符：docx={headings!r} 期望={expected_heads!r}"))

    # 1) 内容一致 + 3) 占位保真：逐 section 逐段
    for sid in state.order:
        outcome = state.outcomes[sid]
        if outcome.status == "accepted" and outcome.draft is not None:
            for para in outcome.draft.paragraphs:
                norm = normalize_cjk_spacing(para.text)
                if not present(norm):
                    if para.is_placeholder:
                        problems.append((VerifyCheck.PLACEHOLDER_LOST.value,
                                         f"[{sid}] 占位段丢失：{norm[:40]!r}"))
                    else:
                        problems.append((VerifyCheck.CONTENT_MISMATCH.value,
                                         f"[{sid}] 正文对不上（改字/丢段）：{norm[:40]!r}"))
            if outcome.draft.limited_note:
                if not present(normalize_cjk_spacing(outcome.draft.limited_note)):
                    problems.append((VerifyCheck.PLACEHOLDER_LOST.value,
                                     f"[{sid}] limited 不足标注被吞"))
                elif LIMITED_PREFIX not in joined:
                    problems.append((VerifyCheck.PLACEHOLDER_LOST.value,
                                     f"[{sid}] limited 标注未以醒目样式呈现（缺前缀标记）"))
        elif outcome.status == "placeholder":
            # 交付版校验客户语言占位文案在位，不要求内部 placeholder_reason。
            client_text = client_placeholder_text(outcome)
            if not present(client_text):
                problems.append((VerifyCheck.PLACEHOLDER_LOST.value,
                                 f"[{sid}] 客户占位文案丢失：{client_text[:40]!r}"))

    # 5) 交付要素：目录标题、TOC 域与页脚 PAGE 域。
    if expect_toc:
        if not present(TOC_HEADING):
            problems.append((VerifyCheck.DELIVERY_ELEMENT_MISSING.value, "缺少目录标题「目录」"))
        elif not _has_field(doc.element.body, "TOC"):
            problems.append((VerifyCheck.DELIVERY_ELEMENT_MISSING.value, "缺少 TOC 目录域"))
        elif TOC_HEADING in headings:                     # 目录标题不能是 Heading，否则会自我收录
            problems.append((VerifyCheck.DELIVERY_ELEMENT_MISSING.value,
                             "目录标题「目录」不应为 Heading 样式（更新目录时会自我收录）"))
    if expect_page_numbers:
        has_page = any(_has_field(s.footer._element, "PAGE") for s in doc.sections)
        if not has_page:
            problems.append((VerifyCheck.DELIVERY_ELEMENT_MISSING.value, "页脚缺少页码 PAGE 域"))

    return VerifyResult(accepted=not problems, problems=problems)


def load_docx(path: str) -> Document:
    return Document(path)
