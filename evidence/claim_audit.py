"""Claim Audit：AuditBrief、审计单 schema 与确定性复核。

**架构：**
- **Engine 全程 model-free**：本模块**零 LLM**。Engine 只做两件事——产 :func:`build_audit_brief`
  （告诉审计者怎么拆 claim、每级判定标准、每个 section 允许引哪些证据、哪些是禁区），
  以及对交回的审计单做 :func:`recheck_audit_sheet`（**纯规则**复核）。
- **语义判断全在外部审计者**：claim 拆分与 supported/weak/unsupported/out_of_scope 的判级，
  由外部审计者（LLM agent 或人）完成，不在本仓。
- **审计者无关**：AuditBrief 与审计单 schema **不含"审计者是谁"的假设**；同一输入既可交生成者
  自审、也可交独立 agent 复审。``meta.auditor`` 是**自由文本、仅作记录**，不参与任何复核逻辑。

**Engine 复核的边界（关键，防越界）：** 确定性复核判规则可判的硬失败——引用存在性
（FABRICATED_REF）、逐字改写（TAMPERED_CONTENT）、禁区 supported（JURISDICTION_VIOLATION）、
逐 section 引用辖区（CROSS_SECTION_CITATION）、未知 section（UNKNOWN_SECTION）、覆盖完整性
（INCOMPLETE_AUDIT）。**Engine 不复核语义强弱**（supported vs weak 属语义判断，规则不可判）。
因此单 claim 的 ``recheck=PASS`` 只代表"引用真实、未改写、未越禁区、辖区合法"，不代表语义被支持。

本模块**不判语义**，但会**强制执行审计者的 verdict**：``unsupported`` / ``contradicted`` 会**阻断交付版**
（:func:`recheck_audit_sheet` 返回 ``delivery.status=BLOCKED``、``accepted=false``），除非删除/弱化
后重审、或人工签字放行（留签字人+理由）。旧"四类规则无违规"语义保留为 ``rule_check_passed``，
与 ``accepted``（规则过 **且** 交付准入过）不混用。

词表为 ``supported / weak / unsupported / contradicted / out_of_scope``；前两者分别表示充分支持
与相关但不足，后两种事实判定会阻断交付，范围外内容不由证据系统背书。
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

AUDIT_BRIEF_VERSION = "m4-audit-brief-v1"
AUDIT_SHEET_SCHEMA_VERSION = "m4-audit-sheet-v1"
RECHECK_VERSION = "m4-recheck-v2"

#: 判定四级（审计者用；Engine 只读不判）。
VERDICTS = ("supported", "weak", "unsupported", "contradicted", "out_of_scope")
# 这些判定会阻断交付版，除非经处置后重审或人工签字放行。
BLOCKING_VERDICTS = ("unsupported", "contradicted")
#: 禁区 gate：这些 section 的结论性 claim 一律应判 unsupported。
RESTRICTED_GATES = ("block", "needs_resolution")


class RecheckCode(str, Enum):
    """逐 claim 复核态。PASS 只代表规则层通过，不代表语义被支持。"""

    PASS = "PASS"
    FABRICATED_REF = "FABRICATED_REF"                # 引用了 audit.json 里不存在的 span
    TAMPERED_CONTENT = "TAMPERED_CONTENT"            # 被引证据逐字被改写
    JURISDICTION_VIOLATION = "JURISDICTION_VIOLATION"  # 禁区/范围外 section 被判 supported
    CROSS_SECTION_CITATION = "CROSS_SECTION_CITATION"  # 引用了不属于本 section 证据包的 span
    UNKNOWN_SECTION = "UNKNOWN_SECTION"              # claim 的 section_id 不在 AuditBrief 里
    CLAIM_NOT_IN_DRAFT = "CLAIM_NOT_IN_DRAFT"        # claim 未绑定或绑错当前稿的段落


class AuditSheetError(ValueError):
    """审计单格式非法（schema 校验失败）。"""


# ── 1. AuditBrief（Engine → 审计者的指令包） ──────────────────────────────
_CLAIM_EXTRACTION_RULE = (
    "代码已把每个 section 的收货正文机械切成句级 claim_units；这不是语义 atomic-claim 提取。"
    "审计者必须为**每个非占位 unit**提交一条 claim，claim_text 必须复制该 unit 的**完整文本**，"
    "不得只取其中有证据的子串；verdict 也由审计者判定。一句含多个断言时按整个 unit 综合判级。"
)

_VERDICT_RULES = {
    "supported": "该 claim 的事实内容能由本 section 证据包里的逐字证据直接支持；必须在 "
                 "cited_span_refs 列出所依据的 span_ref（至少一条）。",
    "weak": "证据与 claim 相关但不足以定论（口径不一致、部分覆盖、需推断）。可引 span_ref。",
    "unsupported": "证据包中没有能支持该 claim 的内容，或该 claim 属禁区 section 的结论性断言。"
                   "不得为凑 supported 而引不相干 span。**判 unsupported 会阻断交付版**（须处置）。",
    "contradicted": "证据包里的证据与该 claim 相矛盾（口径冲突、数值相反等）。"
                    "**判 contradicted 会阻断交付版**（须处置或修正）。",
    "out_of_scope": "该 claim 属 out_of_evidence_scope section（走 drafter L3/intake），"
                    "**不受本系统证据背书**——不算违规，但 Engine 不对其真伪表态。",
}


def build_audit_brief(
    *,
    brief: Mapping[str, object],
    generated_sections: Mapping[str, str],
    audit_map: Mapping[str, Mapping[str, object]],
    section_paragraphs: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> dict:
    """GenerationBrief + 生成文本 + audit.json → AuditBrief（审计指令 + 逐 section 证据包 + 禁区清单）。

    ``generated_sections``：``{section_id: 生成正文}``（agent 按 GenerationBrief 产出的方案文档）。
    ``audit_map``：``generation-brief.audit.json``（``span_ref -> 内部溯源``），用于给证据包
    标 ``span_ref``，并作为后续复核的存在性基准。

    ``section_paragraphs``：``{section_id: [{paragraph_id, content_hash, text,
    is_placeholder, span_refs}]}``，来自 state 的**已盖章段落**。给了则进入**稿件-审计绑定模式**：
    AuditBrief 逐 section 嵌 ``paragraphs``，审计单的 claim 必须绑定到确切段落（``paragraph_id`` +
    ``paragraph_hash``），且非占位段落必须都被 claim 覆盖（见 :func:`recheck_audit_sheet`）。
    不给则退化为仅供历史报告查看的 section 级行为；该模式不能进入交付路径。
    """

    meta = dict(brief.get("meta") or {})
    sections_out: list[dict] = []
    restricted: list[str] = []
    out_of_scope: list[str] = []
    section_paragraphs = section_paragraphs or {}

    for section in brief.get("sections", []):  # type: ignore[union-attr]
        sid = section["section_id"]
        if sid not in generated_sections:
            continue  # 只审实际生成了正文的 section
        gate = section.get("gate")
        oos = bool(section.get("out_of_evidence_scope"))
        is_restricted = gate in RESTRICTED_GATES
        if is_restricted:
            restricted.append(sid)
        if oos:
            out_of_scope.append(sid)

        pack: list[dict] = []
        for item in section.get("acquired", []):
            ref = item.get("ref")
            if item.get("status") != "acquired" or not ref:
                continue
            if ref not in audit_map:
                raise ValueError(
                    f"AuditBrief 构造失败：section {sid} 的 span_ref {ref!r} 不在 audit.json"
                )
            pack.append({
                "span_ref": ref,
                "content": item["content"],          # 逐字
                "source": item["source"],            # 去标识
                "supports_fields": list(item.get("supports_fields") or []),
                "slot_id": item.get("slot_id"),
            })

        section_out = {
            "section_id": sid,
            "section_title": section.get("section_title", ""),
            "gate": gate,
            "out_of_evidence_scope": oos,
            "restricted": is_restricted,
            "restriction_note": (
                "禁区：本 section 的 gate 为 " + str(gate) + "，任何结论性 claim 一律判 unsupported。"
                if is_restricted else None
            ),
            "generated_text": generated_sections[sid],
            "evidence_pack": pack,
        }
        if sid in section_paragraphs:
            from .integrity import claim_units_for, paragraph_content_hash
            paras_out = []
            for p in section_paragraphs[sid]:
                pid = p.get("paragraph_id")
                text = str(p.get("text", ""))
                claim_units = claim_units_for(str(pid), text)
                # content_hash **现算**（不抄存储值）；机械派生句级 claim unit 供审计者完整绑定。
                paras_out.append({
                    "paragraph_id": pid,
                    "content_hash": paragraph_content_hash(text),
                    "text": text,
                    "is_placeholder": bool(claim_units) and all(
                        unit["is_placeholder"] for unit in claim_units),
                    "span_refs": list(p.get("span_refs") or []),
                    "claim_units": [
                        {"unit_id": u["unit_id"], "text": u["text"],
                         "is_placeholder": u["is_placeholder"]}
                        for u in claim_units
                    ],
                })
            section_out["paragraphs"] = paras_out
        sections_out.append(section_out)

    return {
        "audit_brief_version": AUDIT_BRIEF_VERSION,
        "meta": {
            "contract_id": meta.get("contract_id"),
            "doc_type": meta.get("doc_type"),
            "doc_type_zh": meta.get("doc_type_zh"),
            "project_id": meta.get("project_id"),
            "source_brief_version": brief.get("brief_version"),
            "contract_gate": brief.get("contract_gate"),
        },
        "instructions": {
            "claim_extraction": _CLAIM_EXTRACTION_RULE,
            "verdicts": dict(_VERDICT_RULES),
            "citation_rule": (
                "cited_span_refs 只能取自本 section evidence_pack 的 span_ref；"
                "不得引其它 section 的 ref，不得杜撰 ref，不得改写被引证据逐字内容。"
            ),
            "claim_unit_binding": (
                "每条 claim 必须绑定到本 section 某段落 claim_units 里的确切**单元**（句级）：填 "
                "claim_unit_id，且 claim_text 与该单元**完整正文**规范化后相等（原样提交整句，勿只摘局部"
                "子串；空白差异会被规范化）；"
                "建议一并填 paragraph_id + paragraph_hash（现算，防旧稿）。**每个非占位 claim_unit 都要被 "
                "≥1 条 claim 覆盖**——不得跳过任何一句；近似纯占位文案的单元无需 claim，但实质断言与"
                "【待补充】同句仍须审计。绑定缺失/单元不存在/"
                "claim_text 不等于完整单元 → CLAIM_NOT_IN_DRAFT；有非占位单元未覆盖或整节免审 → "
                "INCOMPLETE_AUDIT。代码只做句级切分；claim 的提交与 supported/weak 等判级均由审计者完成。"
            ),
            "auditor_agnostic": (
                "本指令不假设审计者身份：生成者自审或独立 agent 复审均按同一 schema 产出审计单，"
                "Engine 的确定性复核对两者完全一致。meta.auditor 仅作记录、不影响复核。"
            ),
            "engine_boundary": (
                "Engine 只做确定性复核（引用存在性 / 逐字未改写 / 禁区辖区 / 逐 section 引用辖区 / "
                "覆盖完整性）；supported 与 weak 的语义强弱由审计者判，Engine 不复核。但你的 verdict "
                "**有强制力**：判 unsupported/contradicted 会阻断交付版——请据实判级，不要为放行而抬判级。"
            ),
        },
        "restricted_sections": restricted,
        "out_of_scope_sections": out_of_scope,
        "sections": sections_out,
        "audit_sheet_schema_version": AUDIT_SHEET_SCHEMA_VERSION,
    }


# ── 2. 审计单 schema 校验（审计者无关） ────────────────────────────────────
def validate_audit_sheet(sheet: Mapping[str, object]) -> None:
    """审计单格式合法性校验（**只校格式，不校语义**）。非法即抛 :class:`AuditSheetError`。

    schema（对外契约，变更须过 Review）::

        {
          "audit_sheet_schema_version": "m4-audit-sheet-v1",
          "meta": {"auditor": "<自由文本，仅记录>", "audited_at": "<ISO8601>"},
          "no_claim_sections": ["<section_id>", ...],        # 可选：显式声明该 section 无 claim
          "claims": [
            {"claim_text": "...", "section_id": "...", "verdict": "supported|weak|unsupported|out_of_scope",
             "cited_span_refs": ["e1", ...], "auditor_note": "...",
             "cited_content": {"e1": "<逐字引用，可选>"}}      # 可选：填了则参与改写检测
          ]
        }
    """

    if not isinstance(sheet, Mapping):
        raise AuditSheetError("审计单顶层必须是对象")
    meta = sheet.get("meta")
    if not isinstance(meta, Mapping):
        raise AuditSheetError("meta 缺失或非对象")
    if not isinstance(meta.get("auditor"), str) or not meta["auditor"].strip():
        raise AuditSheetError("meta.auditor 必须是非空字符串（自由文本，仅记录）")
    if not isinstance(meta.get("audited_at"), str) or not meta["audited_at"].strip():
        raise AuditSheetError("meta.audited_at 必须是非空字符串")
    claims = sheet.get("claims")
    if not isinstance(claims, list):
        raise AuditSheetError("claims 必须是列表")
    # no_claim_sections 兼容两种形态：``["sid"]`` 或 ``[{"section_id","reason"}]``。
    ncs = sheet.get("no_claim_sections", [])
    if not isinstance(ncs, list):
        raise AuditSheetError("no_claim_sections 必须是列表")
    for i, item in enumerate(ncs):
        if isinstance(item, str):
            continue
        if isinstance(item, Mapping):
            if not isinstance(item.get("section_id"), str) or not item["section_id"].strip():
                raise AuditSheetError(f"no_claim_sections[{i}].section_id 必须是非空字符串")
            continue
        raise AuditSheetError(
            f"no_claim_sections[{i}] 必须是字符串或 {{section_id, reason}} 对象")
    for idx, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise AuditSheetError(f"claims[{idx}] 必须是对象")
        for field in ("claim_text", "section_id", "verdict"):
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise AuditSheetError(f"claims[{idx}].{field} 必须是非空字符串")
        if claim["verdict"] not in VERDICTS:
            raise AuditSheetError(
                f"claims[{idx}].verdict={claim['verdict']!r} 不在 {VERDICTS}"
            )
        refs = claim.get("cited_span_refs", [])
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            raise AuditSheetError(f"claims[{idx}].cited_span_refs 必须是字符串列表")
        if claim["verdict"] == "supported" and not refs:
            raise AuditSheetError(f"claims[{idx}] verdict=supported 必须至少引一条 span_ref")
        cited_content = claim.get("cited_content", {})
        if not isinstance(cited_content, Mapping):
            raise AuditSheetError(f"claims[{idx}].cited_content 必须是对象（可选）")
        # 稿件绑定字段（可选存在；绑定模式下由 recheck 强制）。类型合法性在此校。
        pid = claim.get("paragraph_id")
        if pid is not None and (not isinstance(pid, str) or not pid.strip()):
            raise AuditSheetError(f"claims[{idx}].paragraph_id 若给必须是非空字符串")
        phash = claim.get("paragraph_hash")
        if phash is not None and not isinstance(phash, str):
            raise AuditSheetError(f"claims[{idx}].paragraph_hash 若给必须是字符串")
        cuid = claim.get("claim_unit_id")                # 绑定到确切 claim 单元
        if cuid is not None and (not isinstance(cuid, str) or not cuid.strip()):
            raise AuditSheetError(f"claims[{idx}].claim_unit_id 若给必须是非空字符串")


# ── 3. 确定性复核（Engine 的地板；纯规则、零 LLM） ────────────────────────
def recheck_audit_sheet(
    *,
    audit_brief: Mapping[str, object],
    sheet: Mapping[str, object],
    audit_map: Mapping[str, Mapping[str, object]],
    documents: Sequence[object] | None = None,
    weak_policy: str = "block",
    signoffs: Mapping[int, Mapping[str, object]] | None = None,
) -> dict:
    """对审计单逐 claim 做确定性复核 + **交付准入判定**，返回复核报告。

    **边界不变**：Engine 不判"证据算强还是算弱"（那是审计者的 verdict）。但 Engine **强制执行**
    审计者的 verdict——`unsupported`/`contradicted` 会**阻断交付版**。

    - ``documents``：默认应传（`build_default` CLI 传）→ 用 audit.json 偏移从**原文重切**核对逐字，
      不只信审计单的 cited_content。重切失败按 TAMPERED_CONTENT。
    - ``weak_policy``：``"block"``（默认最严）weak 也须处置才准入；``"allow"`` 允许 weak 带注交付。
    - ``signoffs``：``{claim_index: {"signer": ..., "reason": ...}}`` 人工签字放行某 blocking/weak claim
      （留痕入档）。Engine 不判该不该放行，只强制"放行须留签字人与理由"。
    """

    validate_audit_sheet(sheet)
    if weak_policy not in ("block", "allow"):
        raise AuditSheetError(f"weak_policy 必须是 'block' 或 'allow'；got {weak_policy!r}")
    signoffs = {int(k): dict(v) for k, v in (signoffs or {}).items()}

    by_section = {s["section_id"]: s for s in audit_brief.get("sections", [])}
    issued: dict[str, str] = {}
    section_pack_refs: dict[str, set[str]] = {}          # 逐 section 证据包的 span_ref（辖区用）
    for section in audit_brief.get("sections", []):
        refs_here = set()
        for item in section["evidence_pack"]:
            issued[item["span_ref"]] = item["content"]
            refs_here.add(item["span_ref"])
        section_pack_refs[section["section_id"]] = refs_here
    restricted = set(audit_brief.get("restricted_sections", []))
    oos = set(audit_brief.get("out_of_scope_sections", []))

    # 稿件-审计绑定：句级 claim unit，不声称语义 atomic。
    # 任一 section 带 paragraphs → 绑定模式：claim 必须绑到确切 claim_unit（句级）、claim_text 与该
    # unit 正文规范化后**完整相等**、**每个非占位 unit 都要被覆盖**。单元由正文现场派生（不读自报）。
    from .integrity import _normalize_for_match
    binding_mode = any("paragraphs" in s for s in audit_brief.get("sections", []))
    units_by_section: dict[str, dict[str, dict]] = {}   # sid → {unit_id: {norm, is_placeholder}}
    factual_units: dict[str, set[str]] = {}             # sid → 非占位 unit_id 集（须被覆盖）
    para_hash: dict[str, dict[str, str]] = {}           # sid → {paragraph_id: content_hash}（辅校验）
    for section in audit_brief.get("sections", []):
        sid0 = section["section_id"]
        paras = section.get("paragraphs")
        if paras is None:
            continue
        umap, fset, hmap = {}, set(), {}
        for p in paras:
            pid = p.get("paragraph_id")
            if pid:
                hmap[pid] = p.get("content_hash")
            for u in p.get("claim_units", []):
                uid = u["unit_id"]
                umap[uid] = {"norm": _normalize_for_match(str(u["text"])),
                             "is_placeholder": bool(u["is_placeholder"])}
                if not u["is_placeholder"]:
                    fset.add(uid)
        units_by_section[sid0] = umap
        factual_units[sid0] = fset
        para_hash[sid0] = hmap
    covered_units: dict[str, set[str]] = {sid0: set() for sid0 in factual_units}

    doc_by_id = {getattr(d, "document_id", None): d for d in (documents or [])}

    results: list[dict] = []
    for index, claim in enumerate(sheet["claims"]):
        code = RecheckCode.PASS
        detail = ""
        sid = claim["section_id"]
        refs = list(claim.get("cited_span_refs", []))
        cited_content = dict(claim.get("cited_content", {}) or {})

        pid = claim.get("paragraph_id")
        phash = claim.get("paragraph_hash")
        cuid = claim.get("claim_unit_id")

        # (0) 拒未知 section：claim 的 section_id 必须在 AuditBrief 里。
        if sid not in by_section:
            code = RecheckCode.UNKNOWN_SECTION
            detail = f"section {sid!r} 不在 AuditBrief（只能审已生成正文的 section）"

        # (0b) 稿件绑定：绑定模式下，非 out_of_scope 的 claim 必须绑到确切 claim_unit，
        #      且 claim_text 与该 unit 正文**规范化后完整相等**；paragraph_hash 若给则须与现算一致。
        #      禁止用整句中的局部有证据子串覆盖整个 unit。缺绑定/单元不存在/全文不等 → 拒。
        if code is RecheckCode.PASS and binding_mode and sid in units_by_section \
                and claim["verdict"] != "out_of_scope":
            umap = units_by_section[sid]
            hmap = para_hash.get(sid, {})
            if not cuid:
                code, detail = RecheckCode.CLAIM_NOT_IN_DRAFT, \
                    "缺 claim_unit_id：绑定模式下 claim 必须绑定到确切 claim 单元"
            elif cuid not in umap:
                code, detail = RecheckCode.CLAIM_NOT_IN_DRAFT, \
                    f"claim_unit_id {cuid!r} 不在 section {sid} 的当前稿单元中"
            elif pid and phash is not None and pid in hmap and phash != hmap[pid]:
                code, detail = RecheckCode.CLAIM_NOT_IN_DRAFT, \
                    f"paragraph_hash 与当前稿段落 {pid} 不一致（审的是旧稿/改过的正文）"
            elif _normalize_for_match(str(claim["claim_text"])) != umap[cuid]["norm"]:
                code, detail = RecheckCode.CLAIM_NOT_IN_DRAFT, \
                    f"claim_text 与单元 {cuid} 正文规范化后不完全相等（不得用局部子串覆盖整句）"
            else:
                covered_units.setdefault(sid, set()).add(cuid)

        # (a) 存在性：引用必须真实存在于 audit.json
        if code is RecheckCode.PASS:
            missing = [r for r in refs if r not in audit_map]
            if missing:
                code, detail = RecheckCode.FABRICATED_REF, f"audit.json 中不存在的 span_ref：{missing}"

        # (c) 辖区（verdict 侧，优先于引用辖区）：禁区/范围外 section 不得被判 supported
        if code is RecheckCode.PASS and claim["verdict"] == "supported":
            if sid in restricted:
                code = RecheckCode.JURISDICTION_VIOLATION
                detail = f"section {sid} 为禁区（gate={by_section.get(sid, {}).get('gate')}），结论性 claim 应判 unsupported"
            elif sid in oos:
                code = RecheckCode.JURISDICTION_VIOLATION
                detail = f"section {sid} 为 out_of_evidence_scope，不受本系统证据背书，不得判 supported"

        # (a2) 引用辖区：引用必须属于本 section 的证据包。
        if code is RecheckCode.PASS:
            out_of_section = [r for r in refs if r not in section_pack_refs.get(sid, set())]
            if out_of_section:
                code = RecheckCode.CROSS_SECTION_CITATION
                detail = f"引用了不属于 section {sid} 证据包的 span_ref：{out_of_section}"

        # (b) 一致性：被引证据逐字不得被改写
        if code is RecheckCode.PASS:
            tampered: list[str] = []
            for ref in refs:
                expected = issued.get(ref)
                if expected is None:
                    continue
                if ref in cited_content and cited_content[ref] != expected:
                    tampered.append(f"{ref}(审计单引用与发出内容不一致)")
                    continue
                entry = audit_map.get(ref) or {}
                document = doc_by_id.get(entry.get("document_id"))
                if document is not None:
                    try:
                        rederived = document.slice(entry["start_char"], entry["end_char"])
                    except Exception:
                        # 记录的偏移在该文档上切不出来 → 溯源本身已不自洽，同样按改写处理。
                        tampered.append(f"{ref}(原文重切失败：偏移越界/文档不匹配)")
                        continue
                    if rederived != expected:
                        tampered.append(f"{ref}(原文重切与发出内容不一致)")
            if tampered:
                code, detail = RecheckCode.TAMPERED_CONTENT, "；".join(tampered)

        results.append({
            "index": index,
            "section_id": claim["section_id"],
            "claim_text": claim["claim_text"],
            "paragraph_id": pid,
            "paragraph_hash": phash,
            "claim_unit_id": cuid,
            "verdict": claim["verdict"],
            "cited_span_refs": refs,
            # 缓存复核当时由 AuditBrief 发出的逐字证据。交付门会用当前 AuditBrief/audit map
            # 重跑规则；快照缺失或与当前内容不一致均不能靠缓存 PASS 放行。
            "cited_content": {ref: issued[ref] for ref in refs if ref in issued},
            "recheck": code.value,
            "detail": detail,
        })

    # (d) 完整性：绑定模式下采用 claim-unit 覆盖与免审合法性校验。
    exemptions: dict[str, str | None] = {}
    for item in sheet.get("no_claim_sections", []):
        if isinstance(item, str):
            exemptions[item] = None
        elif isinstance(item, Mapping) and item.get("section_id"):
            exemptions[str(item["section_id"])] = (
                str(item["reason"]) if item.get("reason") else None)
    expected_sections = set(by_section)

    invalid_exemptions: list[dict] = []
    uncovered_units: dict[str, list[str]] = {}
    if binding_mode:
        # 每个非占位 claim 单元必须被至少一个有效绑定 claim 覆盖。
        for sid0 in expected_sections:
            factual = factual_units.get(sid0, set())
            got = covered_units.get(sid0, set())
            missing_u = sorted(factual - got)
            if sid0 in exemptions:
                # 免审只对“无事实单元”的 section 合法，且必须带明确理由。
                if factual:
                    invalid_exemptions.append({
                        "section_id": sid0, "reason": exemptions[sid0],
                        "why": f"section 有 {len(factual)} 个非占位事实单元，不得整节免审"})
                    uncovered_units[sid0] = missing_u or sorted(factual)
                elif not exemptions[sid0]:
                    invalid_exemptions.append({
                        "section_id": sid0, "reason": None,
                        "why": "免审必须带明确理由（reason 字段）"})
                    uncovered_units[sid0] = ["<no-reason>"]
                # 合法免审（无事实单元 + 有理由）→ 不计入 uncovered。
            elif missing_u:
                uncovered_units[sid0] = missing_u
        uncovered = sorted(uncovered_units)
    else:
        # 兼容旧报告（仅供查看）：section 出现在 claims 或声明 no_claim 即算覆盖。
        # section 级旧报告不得进入交付路径。
        covered_sec = {c["section_id"] for c in sheet["claims"]} | set(exemptions)
        uncovered = sorted(expected_sections - covered_sec)

    counts = {code.value: 0 for code in RecheckCode}
    for item in results:
        counts[item["recheck"]] += 1
    hard_rejects = sum(
        counts[c.value] for c in RecheckCode if c is not RecheckCode.PASS
    )
    rule_check_passed = hard_rejects == 0 and not uncovered and not invalid_exemptions

    # 交付准入：审计者的 verdict 必须有强制力。
    verdict_summary = {v: 0 for v in VERDICTS}
    for item in results:
        verdict_summary[item["verdict"]] = verdict_summary.get(item["verdict"], 0) + 1

    def _signoff_for(i):
        so = signoffs.get(i)
        if so and str(so.get("signer", "")).strip() and str(so.get("reason", "")).strip():
            return {"index": i, "verdict": results[i]["verdict"],
                    "signer": so["signer"], "reason": so["reason"]}
        return None

    blocking, weak_unresolved, applied_signoffs = [], [], []
    for i, item in enumerate(results):
        v = item["verdict"]
        is_block = v in BLOCKING_VERDICTS
        is_weak_block = v == "weak" and weak_policy == "block"
        if not (is_block or is_weak_block):
            continue
        so = _signoff_for(i)
        if so is not None:
            applied_signoffs.append(so)
        elif is_block:
            blocking.append({"index": i, "section_id": item["section_id"], "verdict": v,
                             "claim_text": item["claim_text"]})
        else:
            weak_unresolved.append({"index": i, "section_id": item["section_id"],
                                    "claim_text": item["claim_text"]})

    delivery_admitted = rule_check_passed and not blocking and not weak_unresolved
    if not rule_check_passed:
        delivery_reason = "规则层未过（存在硬拒或漏审 section）"
    elif blocking:
        delivery_reason = f"存在 {len(blocking)} 条 unsupported/contradicted 未处置（须删除/弱化后重审，或人工签字放行）"
    elif weak_unresolved:
        delivery_reason = f"weak_policy=block 下存在 {len(weak_unresolved)} 条 weak 未处置"
    else:
        delivery_reason = "交付准入通过：无 unsupported/contradicted 未处置" + (
            f"（{len(applied_signoffs)} 条经人工签字放行）" if applied_signoffs else "")

    return {
        "recheck_version": RECHECK_VERSION,
        "audit_sheet_schema_version": AUDIT_SHEET_SCHEMA_VERSION,
        "binding_mode": binding_mode,                    # 交付路径只接受 True
        "auditor": dict(sheet["meta"]).get("auditor"),   # 仅记录，不影响复核
        "audited_at": dict(sheet["meta"]).get("audited_at"),
        # 审计单本体不作为授权输入；保留合法免审声明，供交付门重建最小审计输入并复算完整性。
        "no_claim_sections": [
            dict(item) if isinstance(item, Mapping) else item
            for item in sheet.get("no_claim_sections", [])
        ],
        "claims": results,
        "summary": {
            "total_claims": len(results),
            **counts,
            "hard_reject_count": hard_rejects,
            "verdict_summary": verdict_summary,
        },
        "completeness": {
            "mode": "unit" if binding_mode else "section",
            "expected_sections": sorted(expected_sections),
            "uncovered_sections": uncovered,
            "uncovered_units": {k: v for k, v in sorted(uncovered_units.items())},
            "invalid_exemptions": invalid_exemptions,
            "status": "INCOMPLETE_AUDIT" if (uncovered or invalid_exemptions) else "COMPLETE",
        },
        "rule_check_passed": rule_check_passed,           # 旧 accepted 语义（四类规则问题无违规）
        "delivery": {
            "weak_policy": weak_policy,
            "admitted": delivery_admitted,
            "status": "ADMITTED" if delivery_admitted else "BLOCKED",
            "reason": delivery_reason,
            "blocking_claims": blocking,
            "weak_unresolved": weak_unresolved,
            "applied_signoffs": applied_signoffs,
        },
        # 新语义：四类规则无违规 **且** 交付准入通过。旧语义保留为 rule_check_passed，不混用。
        "accepted": rule_check_passed and delivery_admitted,
        "engine_boundary_note": (
            "PASS 只代表规则层通过；supported/weak 的语义强弱由审计者判。但审计者的 "
            "unsupported/contradicted **有强制力**：未处置即阻断交付版（accepted=false）。"
            "Engine 不判该不该弱化，只强制'必须处置且留痕'。"
        ),
    }


# ── 4. CLI ───────────────────────────────────────────────────────────────
def _load_state(session_dir):
    import json
    from pathlib import Path
    return json.loads((Path(session_dir) / "state.json").read_text(encoding="utf-8"))


def _load_generated_sections(session_dir):
    """从编排会话 state.json 取 accepted 各 section 的正文。"""
    state = _load_state(session_dir)
    out = {}
    for sid, oc in state["outcomes"].items():
        if oc["status"] == "accepted" and oc.get("draft"):
            out[sid] = "\n".join(p["text"] for p in oc["draft"]["paragraphs"])
    return out


def _load_section_paragraphs(state):
    """从 state（dict）取各 accepted section 的已盖章段落：paragraph_id +
    content_hash + text + is_placeholder + span_refs。供 build_audit_brief 建绑定模式 AuditBrief。

    仅对已盖章（段落带 paragraph_id/content_hash）的 state 返回段落；未带绑定信息的旧 state
    返回空，AuditBrief 退化为 section 级（向后兼容）。
    """
    out = {}
    for sid, oc in state.get("outcomes", {}).items():
        if oc.get("status") != "accepted" or not oc.get("draft"):
            continue
        paras = oc["draft"]["paragraphs"]
        if not all(p.get("paragraph_id") and p.get("content_hash") for p in paras):
            continue  # 未盖章 → 不进绑定模式
        out[sid] = [{
            "paragraph_id": p["paragraph_id"],
            "content_hash": p["content_hash"],
            "text": p["text"],
            "is_placeholder": "【待补充" in p["text"],
            "span_refs": list(p.get("span_refs") or []),
        } for p in paras]
    return out


def main(argv=None):  # pragma: no cover - 薄 CLI 壳
    import argparse
    import json
    import sys
    from pathlib import Path

    invoked_as = Path(sys.argv[0]).stem.lower()
    program = "evidence-audit" if invoked_as == "evidence-audit" else "python -m evidence.claim_audit"
    parser = argparse.ArgumentParser(
        prog=program,
        description="L7 Claim Audit：生成 AuditBrief、复核审计单并判定交付准入",
    )
    parser.add_argument("--brief", required=True, help="GenerationBrief JSON（gap_checker --brief 产）")
    parser.add_argument("--audit-map", required=True, help="generation-brief.audit.json（span_ref→溯源）")
    parser.add_argument("--session-dir", required=True, help="生成编排会话目录（取已生成正文）")
    parser.add_argument("--emit-brief", help="仅产 AuditBrief 到此路径（交审计者阅读、产审计单）")
    parser.add_argument("--sheet", help="审计者交回的审计单 JSON；给了则做复核 + 交付准入")
    parser.add_argument("--out", help="复核报告输出路径（默认 session-dir/recheck.json）")
    parser.add_argument("--weak-policy", choices=["block", "allow"], default="block",
                        help="weak 处置策略：block（默认，须处置）/ allow（可带注交付）")
    parser.add_argument("--signoffs", help="人工签字放行 JSON：{claim_index: {signer, reason}}")
    parser.add_argument("--library-root", help="L1 库根（配 --project，供原文重切核对）")
    parser.add_argument("--project", help="L1 项目 id（配 --library-root）")

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        audit_map = json.loads(Path(args.audit_map).read_text(encoding="utf-8"))
        state_dict = _load_state(args.session_dir)
        generated = _load_generated_sections(args.session_dir)
        section_paragraphs = _load_section_paragraphs(state_dict)
        audit_brief = build_audit_brief(brief=brief, generated_sections=generated,
                                        audit_map=audit_map,
                                        section_paragraphs=section_paragraphs)
        if args.emit_brief:
            Path(args.emit_brief).write_text(
                json.dumps(audit_brief, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"AuditBrief written: {args.emit_brief}（交审计者阅读、产审计单后 --sheet 复核）")
            if not args.sheet:
                return 0
        if not args.sheet:
            parser.error("需 --sheet 做复核，或 --emit-brief 仅产 AuditBrief")

        sheet = json.loads(Path(args.sheet).read_text(encoding="utf-8"))
        documents = None
        if args.library_root and args.project:
            from .library import FilesystemLibrary
            documents = FilesystemLibrary(args.library_root, args.project).documents()
        signoffs = None
        if args.signoffs:
            signoffs = {int(k): v for k, v in
                        json.loads(Path(args.signoffs).read_text(encoding="utf-8")).items()}
        report = recheck_audit_sheet(
            audit_brief=audit_brief, sheet=sheet, audit_map=audit_map,
            documents=documents, weak_policy=args.weak_policy, signoffs=signoffs)
        # 输入摘要链把“审的是哪份稿和哪份 brief”绑定进复核报告，渲染门会重算。
        from .integrity import compute_input_digest
        report["input_digest"] = compute_input_digest(
            state_dict=state_dict,
            brief=brief,
            audit_brief=audit_brief,
            audit_map=audit_map,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    out = args.out or str(Path(args.session_dir) / "recheck.json")
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    d = report["delivery"]
    vs = report["summary"]["verdict_summary"]
    print(f"复核：rule_check_passed={report['rule_check_passed']} · "
          f"verdicts={ {k: v for k, v in vs.items() if v} }")
    print(f"交付准入：{d['status']} —— {d['reason']}")
    print(f"复核报告：{out}")
    if report["accepted"]:
        # 按实际交付态打印：含待补料占位时是 SHAREABLE_DRAFT，不是 DELIVERABLE。
        from .delivery import DeliveryState, delivery_state
        from .generation.orchestrator import OrchestrationState
        dstate = delivery_state(OrchestrationState.from_dict(state_dict), report)
        if dstate is DeliveryState.DELIVERABLE:
            print("→ DELIVERABLE：无待补料占位，可渲染交付版（render --recheck，无需 --draft）。")
        else:
            print("→ SHAREABLE_DRAFT：准入通过但含待补料占位——render 出进度草稿（"
                  "带'含待补料'水印、文件名 -shareable-draft），非最终交付物。")
        return 0
    # 阻断：打印处置指令
    from .delivery import remediation_directives
    for it in remediation_directives(report):
        print(f"  ✗ [{it['section_id']}] {it['verdict']}: {(it['claim_text'] or '')[:40]}"
              f"  → 处置 {it['options']}")
    print("处置（删除/弱化→改稿→重跑生成+审计，或人工签字 --signoffs）后交付准入方可通过。")
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
