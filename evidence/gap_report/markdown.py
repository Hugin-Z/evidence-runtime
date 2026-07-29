"""Gap Report Markdown 渲染。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..types import EvidenceContract, EvidenceSpan, SlotStatus
from ..verifier import (
    ContractVerificationResult,
    DetectedConflict,
    DetectedFact,
    FieldCoverage,
    SlotVerificationResult,
)

from .models import GapReport


def _one_line(text: str) -> str:
    return " ".join(text.splitlines()).strip()


def _input_completeness_lines(report: GapReport) -> list[str]:
    """“输入完整性”节：从 trace.input_completeness 列出未提取载体。

    让"材料够不够"的结论带上"我可能没看全"的自知——无文本页/文本框/图表/页眉页脚/批注若存在，
    在此显式列出；这些载体只告警，不提取。
    """
    ic = (report.trace or {}).get("input_completeness") if hasattr(report.trace, "get") else None
    lines = ["## 输入完整性", ""]
    if not ic:
        lines.append("- （未记录输入完整性信息）")
        lines.append("")
        return lines
    if ic.get("complete"):
        lines.append("- 材料提取完整：`true`（未发现无文本页 / 文本框 / 图表 / 页眉页脚 / 批注等未提取载体）")
        lines.append("")
        return lines
    lines.append(f"- ⚠ **部分材料未完整提取**（{ic.get('total_gaps', 0)} 处）——"
                 f"以下载体未纳入证据判定，\"材料够不够\"的结论可能未看全：")
    for d in ic.get("documents", []):
        fname = d.get("filename") or d.get("document_id")
        lines.append(f"  - 《{fname}》")
        for g in d.get("gaps", []):
            lines.append(f"    - [{g.get('kind')}] {_one_line(str(g.get('detail', '')))}")
    lines.append("")
    return lines


def render_gap_report_markdown(report: GapReport) -> str:
    if not isinstance(report, GapReport):
        raise ValueError("report must be GapReport")
    status_counts = {
        status.value: sum(1 for item in report.slots if item.status is status)
        for status in SlotStatus
    }
    lines = [
        "# 材料缺口报告",
        "",
        f"- Contract：`{report.contract_id}`（版本 `{report.contract_version}`）",
        f"- 总体状态：`{report.status}`",
        f"- 可进入下一阶段：`{str(report.ready).lower()}`",
        f"- 待办数量：{len(report.pending)}",
        (
            "- Slot 统计："
            + "，".join(f"{key}={value}" for key, value in status_counts.items())
        ),
        "",
    ]
    lines.extend(_input_completeness_lines(report))
    lines.extend(["## 优先补料事项", ""])
    slot_by_id = {item.slot_id: item for item in report.slots}
    if not report.pending:
        lines.append("- 无。")
    for item in report.pending:
        fields = "、".join(f"`{value}`" for value in item.missing_fields) or "无"
        existing = "、".join(f"`{value}`" for value in item.existing_evidence_ids) or "无"
        lines.extend(
            [
                f"### P{item.priority} · {item.slot_name} (`{item.slot_id}`)",
                "",
                f"- 当前状态：`{slot_by_id[item.slot_id].status.value}`",
                f"- 动作：`{item.action.value}`",
                f"- 缺失字段：{fields}",
                f"- 已有证据：{existing}",
                f"- 说明：{item.description}",
                "",
            ]
        )

    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    lines.extend(["", "## Slot 明细", ""])
    for slot in report.slots:
        lines.extend(
            [
                f"### {slot.slot_name} (`{slot.slot_id}`)",
                "",
                f"- 状态：`{slot.status.value}`",
                f"- 必需：`{str(slot.required).lower()}`",
                f"- 判定：{slot.reason}",
            ]
        )
        missing = "、".join(f"`{value}`" for value in slot.missing_fields) or "无"
        covered = "、".join(f"`{value}`" for value in slot.covered_fields) or "无"
        lines.append(f"- 已覆盖字段：{covered}")
        lines.append(f"- 缺失字段：{missing}")
        if slot.evidence_ids:
            lines.append("- 证据：")
            for evidence_id in slot.evidence_ids:
                evidence = evidence_by_id[evidence_id]
                page = f"，页 {evidence.page}" if evidence.page is not None else ""
                lines.append(
                    "  - "
                    f"`{evidence.evidence_id}` · `{evidence.document_id}` · "
                    f"行 {evidence.start_line}-{evidence.end_line}{page}："
                    f"{_one_line(evidence.text)}"
                )
        else:
            lines.append("- 证据：无")
        if slot.conflicts:
            lines.append("- 冲突：")
            for conflict in slot.conflicts:
                lines.append(f"  - {conflict.reason}")
                for fact in conflict.facts:
                    lines.append(
                        f"    - `{fact.evidence_id}`：{fact.subject} / "
                        f"{fact.predicate} / {fact.value}"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_gap_report_markdown(report: GapReport, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as file:
        file.write(render_gap_report_markdown(report))


# Readable symmetry for callers that prefer the JSON helper naming pattern.
gap_report_to_markdown = render_gap_report_markdown
