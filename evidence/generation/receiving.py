"""L6 收货校验：每份回稿必经确定性校验才接受。

**边界**：只验代码可判的**硬约束**——引用存在性 / 逐字一致 / 辖区 /
不足标注 / 占位规范 / 段落-引用绑定。**不判语义强弱**（写得好不好、论证有没有力）——
那归审计层 L7（Claim Audit）。收货只回答"这稿有没有违反任务包的硬约束"。

四类校验：
1. **伪造引用**（`FABRICATED_REF`）：段落引用了任务包 evidence 里不存在的 span_ref。
2. **包外引用 / 辖区**（`OUT_OF_PACKAGE`）：同 1 的另一面——回稿不得含包外 span_ref。
   （存在性与辖区在"引用集 ⊄ 包内引用集"上是同一判定，合并为 FABRICATED_REF 报出。）
3. **缺占位**（`MISSING_PLACEHOLDER`）：missing_fields 非空但回稿无任何 `【待补充` 占位。
4. **格式违规**（`FORMAT_VIOLATION`）：事实段落无引用绑定 / limited 缺不足标注 /
   占位符不符规范。
另有 **逐字改写**（`TAMPERED_CONTENT`）：回稿声称引用某 span_ref，却把其 content 改了。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .task import SectionDraft, SectionTask

_PLACEHOLDER_MARK = "【待补充"


class ReceivingCheck(str, Enum):
    FABRICATED_REF = "FABRICATED_REF"          # 引用了包内不存在的 span_ref（含包外引用）
    TAMPERED_CONTENT = "TAMPERED_CONTENT"      # 声称引用却改写了逐字内容
    MISSING_PLACEHOLDER = "MISSING_PLACEHOLDER"  # 有 missing 字段却无占位
    FORMAT_VIOLATION = "FORMAT_VIOLATION"      # 事实段无绑定 / limited 缺不足标注 / 占位不规范


@dataclass(frozen=True)
class ReceivingResult:
    """一份回稿的收货结论。``accepted`` 为真才接受。"""

    section_id: str
    accepted: bool
    problems: tuple[tuple[str, str], ...] = ()   # (ReceivingCheck 值, 明细)

    def problem_summary(self) -> str:
        if self.accepted:
            return "accepted"
        return "; ".join(f"{code}: {detail}" for code, detail in self.problems)


def check_draft(
    task: SectionTask,
    draft: SectionDraft,
    *,
    cited_content: Mapping[str, str] | None = None,
) -> ReceivingResult:
    """对回稿做确定性收货校验。

    ``cited_content``（可选）：回稿声称的 {span_ref: 引用到的原文}，用于逐字一致校验
    （回稿若原样嵌入了证据文本，须与任务包发出的 content 逐字相同）。
    """

    problems: list[tuple[str, str]] = []
    allowed_refs = task.evidence_refs()
    content_by_ref = {e.span_ref: e.content for e in task.evidence}

    if draft.section_id != task.section_id:
        problems.append((
            ReceivingCheck.FORMAT_VIOLATION.value,
            f"回稿 section_id={draft.section_id!r} 与任务包 {task.section_id!r} 不符",
        ))

    # 1)+2) 存在性 / 辖区：任何被引用的 span_ref 必须在包内
    cited = draft.cited_refs()
    out_of_package = sorted(cited - allowed_refs)
    if out_of_package:
        problems.append((
            ReceivingCheck.FABRICATED_REF.value,
            f"引用了任务包外/不存在的 span_ref：{out_of_package}",
        ))

    # 3) 逐字一致：若回稿回显了某 ref 的内容，必须与发出的 content 逐字相同
    for ref, quoted in (cited_content or {}).items():
        expected = content_by_ref.get(ref)
        if expected is None:
            problems.append((
                ReceivingCheck.FABRICATED_REF.value,
                f"cited_content 含包外 span_ref：{ref!r}",
            ))
        elif quoted != expected:
            problems.append((
                ReceivingCheck.TAMPERED_CONTENT.value,
                f"span_ref {ref!r} 的引用内容被改写（与任务包发出的逐字不一致）",
            ))

    # 4) 占位：missing_fields 非空 → 至少一处 【待补充】 占位
    if task.missing_fields:
        has_placeholder = any(_PLACEHOLDER_MARK in p.text for p in draft.paragraphs)
        if not has_placeholder:
            problems.append((
                ReceivingCheck.MISSING_PLACEHOLDER.value,
                f"存在无证据字段 {list(task.missing_fields)} 却无任何 【待补充】 占位",
            ))

    # 4) 格式：非占位的事实段落必须至少绑定一个 span_ref
    for i, para in enumerate(draft.paragraphs):
        if para.is_placeholder:
            continue
        if para.text.strip() and not para.span_refs:
            problems.append((
                ReceivingCheck.FORMAT_VIOLATION.value,
                f"段落[{i}] 是陈述段却未绑定任何 span_ref（证据优先）",
            ))

    # 4) 占位规范：占位段落不得绑定 span_ref（占位=无证据，绑了自相矛盾）
    for i, para in enumerate(draft.paragraphs):
        if para.is_placeholder and para.span_refs:
            problems.append((
                ReceivingCheck.FORMAT_VIOLATION.value,
                f"段落[{i}] 是 【待补充】 占位却绑定了 span_ref（占位不应有证据）",
            ))

    # 4) limited section 必须含不足标注
    if task.limited_note_required and not (draft.limited_note and draft.limited_note.strip()):
        problems.append((
            ReceivingCheck.FORMAT_VIOLATION.value,
            "gate=limited 的 section 必须提供 limited_note（明确标注证据不足之处）",
        ))

    # 至少要有内容（空回稿不接受）
    if not draft.paragraphs:
        problems.append((
            ReceivingCheck.FORMAT_VIOLATION.value,
            "回稿无任何段落",
        ))

    return ReceivingResult(
        section_id=task.section_id,
        accepted=not problems,
        problems=tuple(problems),
    )
