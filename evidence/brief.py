"""GenerationBrief：Engine 侧 model-free 的受约束生成指令包。

把 :class:`evidence.gap_report.GapReport` + 文档 + crosswalk 组装成一份**对外数据契约**，
交给外部 writer agent 或人执行生成。本模块**只读** GapReport，不改 verifier /
gap_report / contract / adapter / 检索层，也**不调用任何 LLM**。

对外层（`generation-brief.json`）= 「填好的 assets.json」+ 一层 gate：
- ``acquired[]`` 与 drafter ``assets.json`` 的 ``acquired[]`` 对齐（asset_type / status /
  content / source 一一对应），额外带 gate / supports_field / single_source / slot_id / ref。
- **手段无关红线**：对外层的 evidence 只带 content（逐字）+ 去标识 source（文档标题 + 页 / 标题
  层级），**绝不带** distance / chunk_id / embedding / query / hash / char-offset /
  discovery_source / evidence_id。内部溯源（含以上字段）另存 `generation-brief.audit.json`
  （供 Claim Audit），不进对外层。

gate 由 :mod:`evidence.gating` 确定性导出（读 verifier 状态，不反向影响它）。
crosswalk 为 `evidence/contracts/crosswalk/<doc_type>.json` 包内配置（人工建、可审、可扩），把 drafter 的
section.asset_needs 映射到 Engine 8 槽；drafter 有而 Engine 无对应者标 out_of_evidence_scope。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .gap_report import EvidenceReference, GapReport, SlotGapResult
from .gating import (
    ContractGate,
    SlotGate,
    aggregate_section_gate,
    contract_gate,
    gate_for_slot_status,
)
from .runtime_adapter import provenance_for_range
from .types import EvidenceContract, SlotStatus

BRIEF_VERSION = "m2-brief-v1"

# 对外层绝不允许出现的手段特有 / 内部字段名（手段无关红线；用于自检与测试断言）。
FORBIDDEN_OUTWARD_KEYS = frozenset({
    "distance", "chunk_id", "embedding", "query", "hash",
    "char_offset", "start_char", "end_char", "start_line", "end_line",
    "discovery_sources", "discovery_source", "evidence_id",
    "normalized_content_hash", "span_content_hash", "source_content_hash",
    "vector", "fingerprint", "score", "rank",
})

from .resources import crosswalk_dir as _crosswalk_dir

_CROSSWALK_DIR = _crosswalk_dir()


class CrosswalkError(ValueError):
    """crosswalk 配置无效（结构错误 / slot_id 不在契约）。"""


# ── crosswalk 加载 + 校验 ────────────────────────────────────────────────
def load_crosswalk(doc_type: str, *, contract: EvidenceContract) -> dict:
    """按 doc_type 加载 `evidence/contracts/crosswalk/<doc_type>.json` 并对契约校验。

    doc_type 可为中文名（如 "解决方案建议书"）或 outline 的英文 doc_type（如 proposal-document）。
    """

    path = _resolve_crosswalk_path(doc_type)
    if path is None:
        available = sorted(p.stem for p in _CROSSWALK_DIR.glob("*.json"))
        raise CrosswalkError(
            f"未找到 doc_type={doc_type!r} 的 crosswalk（{_CROSSWALK_DIR}）；已有：{available}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    _validate_crosswalk(data, contract=contract, path=path)
    return data


def _resolve_crosswalk_path(doc_type: str) -> Path | None:
    direct = _CROSSWALK_DIR / f"{doc_type}.json"
    if direct.is_file():
        return direct
    # 允许用 outline 的英文 doc_type 匹配（读每个文件的 doc_type 字段）
    for candidate in sorted(_CROSSWALK_DIR.glob("*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("doc_type") == doc_type or data.get("doc_type_zh") == doc_type:
            return candidate
    return None


def _validate_crosswalk(data: object, *, contract: EvidenceContract, path: Path) -> None:
    if not isinstance(data, Mapping):
        raise CrosswalkError(f"{path.name}: 顶层必须是对象")
    if data.get("contract_id") != contract.contract_id:
        raise CrosswalkError(
            f"{path.name}: contract_id={data.get('contract_id')!r} 与契约 "
            f"{contract.contract_id!r} 不一致"
        )
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        raise CrosswalkError(f"{path.name}: sections 必须是非空列表")
    valid_slot_ids = set(contract.slot_ids)
    seen_sections: set[str] = set()
    for idx, sec in enumerate(sections):
        if not isinstance(sec, Mapping):
            raise CrosswalkError(f"{path.name}: sections[{idx}] 必须是对象")
        sid = sec.get("section_id")
        if not isinstance(sid, str) or not sid:
            raise CrosswalkError(f"{path.name}: sections[{idx}].section_id 缺失")
        if sid in seen_sections:
            raise CrosswalkError(f"{path.name}: 重复 section_id {sid!r}")
        seen_sections.add(sid)
        out_of_scope = bool(sec.get("out_of_evidence_scope"))
        asset_map = sec.get("asset_map", [])
        if not isinstance(asset_map, list):
            raise CrosswalkError(f"{path.name}: sections[{idx}].asset_map 必须是列表")
        if out_of_scope and asset_map:
            raise CrosswalkError(
                f"{path.name}: sections[{idx}] out_of_evidence_scope=true 时 asset_map 必须为空"
            )
        for j, entry in enumerate(asset_map):
            if not isinstance(entry, Mapping):
                raise CrosswalkError(f"{path.name}: sections[{idx}].asset_map[{j}] 必须是对象")
            if not entry.get("asset_type"):
                raise CrosswalkError(f"{path.name}: sections[{idx}].asset_map[{j}].asset_type 缺失")
            slot_id = entry.get("slot_id")
            if slot_id not in valid_slot_ids:
                raise CrosswalkError(
                    f"{path.name}: sections[{idx}].asset_map[{j}].slot_id={slot_id!r} "
                    f"不在契约 slot_ids"
                )


# ── 去标识 source ────────────────────────────────────────────────────────
def _deidentified_source(
    ref: EvidenceReference,
    document,  # NormalizedDocument | None
) -> str:
    """构造去标识来源串：《文档标题》· 第X页 / 标题层级。不含 offset / hash / id。"""

    title = getattr(document, "title", None) or "未知来源"
    pages: list[int] = []
    headings: list[str] = []
    if document is not None:
        try:
            prov = provenance_for_range(document, ref.start_char, ref.end_char)
            pages = list(prov.get("pages") or [])
            headings = list(prov.get("headings") or [])
        except Exception:
            pages, headings = [], []
    loc_parts: list[str] = []
    page = ref.page if ref.page else (pages[0] if pages else None)
    if page:
        loc_parts.append(f"第{page}页")
    if headings:
        loc_parts.append(" > ".join(str(h) for h in headings))
    location = " · ".join(loc_parts) if loc_parts else "正文"
    return f"《{title}》· {location}"


# ── Brief 组装 ────────────────────────────────────────────────────────────
def build_generation_brief(
    *,
    report: GapReport,
    documents: Sequence[object],
    crosswalk: Mapping[str, object],
    contract: EvidenceContract,
    project_id: str | None = None,
) -> tuple[dict, dict]:
    """组装 (brief, audit)。brief 是对外层，audit 是供 Claim Audit 使用的内部溯源。"""

    if not isinstance(report, GapReport):
        raise ValueError("report must be GapReport")
    evidence_by_id: dict[str, EvidenceReference] = {
        ref.evidence_id: ref for ref in report.evidence
    }
    doc_by_id = {getattr(d, "document_id", None): d for d in documents}
    slot_by_id: dict[str, SlotGapResult] = {s.slot_id: s for s in report.slots}

    audit: dict[str, dict] = {}
    ref_counter = {"n": 0}

    def _emit_ref(ev: EvidenceReference) -> str:
        ref_counter["n"] += 1
        handle = f"e{ref_counter['n']}"
        audit[handle] = {
            "evidence_id": ev.evidence_id,
            "document_id": ev.document_id,
            "start_char": ev.start_char,
            "end_char": ev.end_char,
            "start_line": ev.start_line,
            "end_line": ev.end_line,
            "page": ev.page,
            "discovery_sources": list(ev.discovery_sources),
        }
        return handle

    def _slot_single_source(slot: SlotGapResult) -> bool:
        doc_ids = {
            evidence_by_id[eid].document_id
            for fc in slot.field_coverage
            if fc.covered
            for eid in fc.evidence_ids
            if eid in evidence_by_id
        }
        return len(doc_ids) == 1

    all_slot_gates: list[SlotGate] = [
        gate_for_slot_status(s.status) for s in report.slots
    ]
    mapped_slot_ids: set[str] = set()
    brief_sections: list[dict] = []

    for sec in crosswalk["sections"]:  # type: ignore[index]
        sid = sec["section_id"]
        title = sec.get("section_title", "")
        if sec.get("out_of_evidence_scope"):
            brief_sections.append({
                "section_id": sid,
                "section_title": title,
                "out_of_evidence_scope": True,
                "reason": sec.get("reason", ""),
                "gate": None,
                "acquired": [],
                "missing_fields": [],
                "conflicts": [],
            })
            continue

        asset_map = sec.get("asset_map", [])
        section_slot_ids = [e["slot_id"] for e in asset_map]
        mapped_slot_ids.update(section_slot_ids)
        section_gate = aggregate_section_gate(
            [gate_for_slot_status(slot_by_id[sid2].status) for sid2 in section_slot_ids]
        )
        # 多个 drafter asset_need 可能映射到同一 Engine 槽（8 槽比 asset_needs 粗）。drafter S4
        # 把 section 的 acquired[] 当**素材池**消费（不严格按 asset_type 键），故这里**按槽去重**：
        # 每槽只出一次证据，asset_type 取首个映射到该槽的 need、asset_types 列全部——避免同一
        # span 在多个 asset_type 下重复。asset_map 顺序决定 asset_type 首选（确定性）。
        slot_to_asset_types: dict[str, list[str]] = {}
        for entry in asset_map:
            slot_to_asset_types.setdefault(entry["slot_id"], []).append(entry["asset_type"])

        acquired: list[dict] = []
        missing_fields: list[dict] = []
        conflicts: list[dict] = []
        conflict_slots_done: set[str] = set()

        for slot_id, asset_types in slot_to_asset_types.items():
            asset_type = asset_types[0]
            slot = slot_by_id[slot_id]
            slot_gate = gate_for_slot_status(slot.status)
            single_source = _slot_single_source(slot)

            covered = [fc for fc in slot.field_coverage if fc.covered]
            uncovered = [fc for fc in slot.field_coverage if not fc.covered]

            if not covered:
                # 全无覆盖（missing）→ 一条待补充占位（assets.json 兼容），不编造 content。
                acquired.append({
                    "asset_type": asset_type,
                    "asset_types": asset_types,
                    "status": "待补充",
                    "content": None,
                    "source": None,
                    "supports_fields": [],
                    "slot_id": slot_id,
                    "gate": slot_gate.value,
                    "single_source": single_source,
                    "ref": None,
                })
            else:
                # 按 span 去重：一个 span 覆盖多个字段时只出一条，supports_fields 列全部字段。
                span_fields: dict[str, list[str]] = {}
                for fc in covered:
                    for eid in fc.evidence_ids:
                        if eid in evidence_by_id:
                            span_fields.setdefault(eid, []).append(fc.field_id)
                for eid, fields in span_fields.items():
                    ev = evidence_by_id[eid]
                    acquired.append({
                        "asset_type": asset_type,
                        "asset_types": asset_types,
                        "status": "acquired",
                        "content": ev.text,                       # 逐字
                        "source": _deidentified_source(ev, doc_by_id.get(ev.document_id)),
                        "supports_fields": fields,
                        "slot_id": slot_id,
                        "gate": slot_gate.value,
                        "single_source": single_source,
                        "ref": _emit_ref(ev),
                    })

            for fc in uncovered:
                missing_fields.append({
                    "asset_types": asset_types,
                    "slot_id": slot_id,
                    "field_id": fc.field_id,
                })

            # conflicting 槽 → 进 conflicts[]（每槽一次）
            if slot.status is SlotStatus.CONFLICTING and slot_id not in conflict_slots_done:
                conflict_slots_done.add(slot_id)
                for dc in slot.conflicts:
                    conflicts.append({
                        "slot_id": slot_id,
                        "conflict_type": dc.conflict_type,
                        "subject": dc.subject,
                        "predicate": dc.predicate,
                        "reason": dc.reason,
                        "conflicting_values": [
                            {"value": f.value, "normalized_value": f.normalized_value}
                            for f in dc.facts
                        ],
                        "sources": [
                            _deidentified_source(
                                evidence_by_id[f.evidence_id],
                                doc_by_id.get(evidence_by_id[f.evidence_id].document_id),
                            )
                            for f in dc.facts
                            if f.evidence_id in evidence_by_id
                        ],
                    })

        brief_sections.append({
            "section_id": sid,
            "section_title": title,
            "out_of_evidence_scope": False,
            "gate": section_gate.value,
            "acquired": acquired,
            "missing_fields": missing_fields,
            "conflicts": conflicts,
        })

    unmapped_slots = [
        {
            "slot_id": s.slot_id,
            "slot_name": s.slot_name,
            "status": s.status.value,
            "gate": gate_for_slot_status(s.status).value,
        }
        for s in report.slots
        if s.slot_id not in mapped_slot_ids
    ]

    top_gate = contract_gate(report.status, all_slot_gates)

    brief = {
        "brief_version": BRIEF_VERSION,
        "meta": {
            "contract_id": report.contract_id,
            "contract_version": report.contract_version,
            "doc_type": crosswalk.get("doc_type"),
            "doc_type_zh": crosswalk.get("doc_type_zh"),
            "project_id": project_id,
        },
        "contract_gate": top_gate.value,
        "overall_status": report.status,
        "sections": brief_sections,
        "unmapped_slots": unmapped_slots,
        "_handoff": (
            "对接 solution-drafter S4：本 brief.sections[].acquired[] 直接对齐 drafter "
            "assets.json.sections.<id>.acquired[]（asset_type/status/content/source 同名）；"
            "gate/missing_fields/conflicts 为额外约束层，drafter 读 4 核心字段即可生成，"
            "读 gate 层可决定 allow/limited/block/needs_resolution。out_of_evidence_scope "
            "的 section 走 drafter L3 领域插件/intake，不经 Engine。内部溯源见 .audit.json。"
        ),
    }
    return brief, audit


def assert_outward_clean(brief: Mapping[str, object]) -> None:
    """自检：对外 brief 递归不含任何手段特有 / 内部字段名（手段无关红线）。违反即抛。

    另拦截明显的内部值形态：``ev-`` 前缀 evidence_id、64 位十六进制 hash。
    """

    violations: list[str] = []

    def _hex64(s: str) -> bool:
        return len(s) == 64 and all(c in "0123456789abcdef" for c in s)

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if isinstance(k, str) and k.lower() in FORBIDDEN_OUTWARD_KEYS:
                    violations.append(f"{path}.{k}（禁用字段名）")
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
        elif isinstance(node, str):
            if node.startswith("ev-") and _hex64(node[3:]):
                violations.append(f"{path}（疑似 evidence_id 值）")
            elif _hex64(node):
                violations.append(f"{path}（疑似 hash 值）")

    walk(brief, "brief")
    if violations:
        raise AssertionError("GenerationBrief 对外层含手段特有/内部字段：" + "；".join(violations))
