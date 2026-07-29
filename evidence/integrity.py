"""稿件、审计与交付产物之间的完整性绑定。

本模块提供三层确定性绑定，全部零 LLM：

1. **段落内容哈希**（:func:`paragraph_content_hash`）：正文段落的内容指纹，收货时由编排器写入。
   claim 据 ``paragraph_id`` + ``content_hash`` 绑定到确切段落，复核时按当前正文重算比对——claim
   不来自当前稿即拒（见 :mod:`evidence.claim_audit` 的 ``CLAIM_NOT_IN_DRAFT``）。
2. **输入摘要链**（:func:`compute_input_digest`）：对"生成内容 + brief 结构 + AuditBrief +
   audit map"算一个 digest；审计复核把它写进 recheck；**render 交付版重算并强制一致**，不一致
   即拒（``STALE_AUDIT`` / ``INPUT_MISMATCH``）。这条封死"旧报告批新稿/证据工件"。
3. **回稿指纹**（:func:`response_fingerprint`）：回稿内容一变，其指纹即变；编排器据此让旧 state/
   旧审计自动失效（见 orchestrator 的 re-run 变更检测）。

安全边界：普通哈希能发现未同步的陈旧/误修改，不能证明写入者身份；有全工作区写权限者可协调重算
所有产物。当前是 integrity 级，不是对该主体的安全授权。
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

#: 交付版渲染的完整性拒收类。
STALE_AUDIT = "STALE_AUDIT"           # recheck 的输入摘要与当前稿不一致（旧报告批新稿）
INPUT_MISMATCH = "INPUT_MISMATCH"     # recheck 缺输入摘要 / 无法据当前稿复核新鲜度
HASH_MISMATCH = "HASH_MISMATCH"       # 存储的段落哈希与正文现算不一致
INCOMPLETE_REPORT = "INCOMPLETE_REPORT"          # recheck 报告缺必填字段或未通过校验
LEGACY_BINDING_NOT_ALLOWED = "LEGACY_BINDING_NOT_ALLOWED"  # section 级旧报告进入交付路径


class DeliveryGateError(RuntimeError):
    """交付版渲染被交付闸拒绝。``code`` 为上述拒收类之一。所有交付闸拒绝的共同基类。"""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class StaleAuditError(DeliveryGateError):
    """审计复核与当前稿件不是同一份（陈旧或输入不匹配）。code = STALE_AUDIT / INPUT_MISMATCH。"""


class HashMismatchError(DeliveryGateError):
    """加载 state 时正文与存储哈希不一致：**存储的哈希不可信，一切按现算**。

    威胁模型：交互执行者通常能写工作区，写在文件里的哈希是"自报"，
    不是许可。故交付路径**一律从当前 text 现算**，存储值仅缓存；现算 ≠ 存储即拒（`HASH_MISMATCH`）。
    """

    def __init__(self, detail: str) -> None:
        super().__init__(HASH_MISMATCH, detail)


# ── claim 单元切分：确定性、可复现、不依赖模型 ─────────────────────────────
#: 句终止符：中文句号/叹号/问号/分号 + 半角对应 + 换行。切分保留终止符。
_SENTENCE_TERMINATORS = "。！？；!?;\n"


def _is_placeholder_only_unit(text: str) -> bool:
    """仅把“近似等于占位文案”的单元判为占位。

    允许占位块周围有空白、项目符号和标点；一旦移除 ``【待补充…】`` 后仍有字母、数字或汉字，
    就是含实质内容的混合单元，必须进入审计覆盖。
    """
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFC", text)
    placeholder = re.compile(r"【\s*待补充(?:\s*[:：][^】]*)?\s*】")
    if not placeholder.search(normalized):
        return False
    remainder = placeholder.sub("", normalized)
    # 只忽略排版符号；不能忽略任何可能构成实质断言的文字或数字。
    remainder = re.sub(r"[\s\-—–·•、，,。.!！?？:：;；（）()\[\]【】]+", "", remainder)
    return remainder == ""


def split_claim_units(text: str) -> list[dict]:
    """把一段正文**确定性**切成 claim unit（句级；不声称自动拆出语义 atomic claim）。

    规则（可复现、零模型）：在 :data:`_SENTENCE_TERMINATORS` 任一字符**之后**切；对每个片段 strip、
    去空；仅“近似等于 ``【待补充…】`` 占位文案”的单元记为占位（不计入覆盖要求）。占位标记与
    实质断言同句时仍是事实单元。返回有序
    ``[{"index", "text", "is_placeholder", "norm"}]``（``norm`` = NFC+去空白，供整单元相等校验）。

    代码只做句级机械切分，不做语义 claim 拆分；一句含多个断言时仍是一个 unit。审计者必须对该
    **完整 unit** 提交 claim_text 并判级。禁止用其中一个有证据的子串覆盖整句，否则同句里的另一个
    无依据断言会被漏审。
    """
    import re

    parts = re.split(r"(?<=[" + re.escape(_SENTENCE_TERMINATORS) + r"])", text)
    units: list[dict] = []
    for part in parts:
        u = part.strip()
        if not u:
            continue
        units.append({
            "index": len(units),
            "text": u,
            "is_placeholder": _is_placeholder_only_unit(u),
            "norm": _normalize_for_match(u),
        })
    return units


def _normalize_for_match(text: str) -> str:
    """规范化用于整单元相等校验：NFC + 去所有空白。抗排版空格/换行差异，不改字。"""
    import re
    import unicodedata

    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def claim_units_for(paragraph_id: str, text: str) -> list[dict]:
    """给一个段落的正文派生带 unit_id 的 claim 单元（``<paragraph_id>#u<NN>``）。"""
    out = []
    for u in split_claim_units(text):
        out.append({**u, "unit_id": f"{paragraph_id}#u{u['index'] + 1:02d}"})
    return out


def paragraph_content_hash(text: str) -> str:
    """段落正文内容指纹（sha256 前 16 hex）。用 NFC + strip 归一，抗无意义空白抖动。"""

    import unicodedata

    norm = unicodedata.normalize("NFC", text).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def stamp_paragraphs(section_id: str, paragraphs: Sequence["object"]) -> tuple:
    """给一个 section 的段落序列**确定性盖章**：``paragraph_id`` = ``<section_id>#p<NN>``（按顺序、
    从 01 起），``content_hash`` = :func:`paragraph_content_hash`。

    返回盖章后的新 :class:`~evidence.generation.task.DraftParagraph` 元组（frozen dataclass，
    replace 生成新实例）。已带 id/hash 的段落也会被**重新盖章**（以编排器为准、防伪造）。
    """

    from dataclasses import replace

    stamped = []
    for i, para in enumerate(paragraphs, start=1):
        pid = f"{section_id}#p{i:02d}"
        stamped.append(replace(para, paragraph_id=pid,
                               content_hash=paragraph_content_hash(para.text)))
    return tuple(stamped)


def _canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generated_content_fingerprint(state_dict: Mapping[str, object]) -> dict:
    """从 state.json（dict）抽出**只与"生成了什么正文"有关**的规范摘要。

    每个 accepted section → 其段落的 ``(paragraph_id, content_hash, span_refs)`` 有序列表 +
    ``limited_note``。占位/未完成 section 记其状态。用于输入摘要链——正文一改，摘要即变。
    """

    outcomes = state_dict.get("outcomes") or {}
    order = state_dict.get("order") or sorted(outcomes)  # type: ignore[arg-type]
    sections: dict[str, object] = {}
    for sid in order:  # type: ignore[union-attr]
        oc = outcomes.get(sid) or {}                     # type: ignore[union-attr]
        status = oc.get("status")
        draft = oc.get("draft") or {}
        paras = []
        for p in draft.get("paragraphs", []):
            # 哈希从当前 text 现算，不读存储的 content_hash（存储值是 agent 可写的自报，
            # 改正文却留旧哈希即可绕过——现算封死之）。
            paras.append({
                "paragraph_id": p.get("paragraph_id"),
                "content_hash": paragraph_content_hash(str(p.get("text", ""))),
                "span_refs": list(p.get("span_refs") or []),
            })
        sections[str(sid)] = {
            "status": status,
            "paragraphs": paras,
            "limited_note": draft.get("limited_note"),
        }
    return sections


def assert_state_hashes_consistent(state_dict: Mapping[str, object]) -> None:
    """校验 state 里每个已盖章段落的存储 content_hash 等于正文现算值。

    不一致 → :class:`HashMismatchError`（正文被改、哈希没改）。未盖章段落（content_hash 为空）跳过。
    由 :meth:`OrchestrationState.from_dict` 加载时调用——**交付/审计路径一切经此加载，篡改即拒**。
    """
    outcomes = state_dict.get("outcomes") or {}
    for sid, oc in outcomes.items():                     # type: ignore[union-attr]
        draft = (oc or {}).get("draft") or {}
        for p in draft.get("paragraphs", []):
            stored = p.get("content_hash")
            if not stored:
                continue
            actual = paragraph_content_hash(str(p.get("text", "")))
            if stored != actual:
                raise HashMismatchError(
                    f"section {sid} 段落 {p.get('paragraph_id')!r} 的存储哈希与正文现算不一致"
                    f"（stored={stored} actual={actual}）——正文被改而哈希未更新，拒绝加载。")


def compute_input_digest(
    *,
    state_dict: Mapping[str, object],
    brief: Mapping[str, object],
    audit_brief: Mapping[str, object],
    audit_map: Mapping[str, object],
) -> str:
    """审计复核所依据的**输入摘要**（sha256）：生成内容指纹 + brief 结构 + 完整 AuditBrief +
    完整 audit map。render 交付版重算此摘要并与 recheck 内的比对，一致才放行。

    AuditBrief 覆盖逐 section 证据包、证据内容快照、稿件 claim units 与辖区；audit map 覆盖
    span_ref 的存在性和溯源偏移。原始审计单、源文档字节和完整 contract 本体不在摘要中。
    """

    meta = brief.get("meta") or {}
    brief_shape = {
        "brief_version": brief.get("brief_version"),
        "contract_gate": brief.get("contract_gate"),
        "contract_id": (meta or {}).get("contract_id"),  # type: ignore[union-attr]
        "sections": [
            {"section_id": s.get("section_id"), "gate": s.get("gate"),
             "out_of_evidence_scope": bool(s.get("out_of_evidence_scope"))}
            for s in brief.get("sections", [])            # type: ignore[union-attr]
        ],
    }
    payload = {
        "generated": generated_content_fingerprint(state_dict),
        "brief": brief_shape,
        "audit_brief": audit_brief,
        "audit_map": audit_map,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def verify_recheck_fresh(
    recheck: Mapping[str, object],
    *,
    state_dict: Mapping[str, object],
    brief: Mapping[str, object],
    audit_brief: Mapping[str, object],
    audit_map: Mapping[str, object],
) -> None:
    """交付版渲染前的新鲜度闸：recheck 必须携带 ``input_digest``，且当前稿重算
    的摘要必须与之一致。缺摘要 → :data:`INPUT_MISMATCH`；不一致 → :data:`STALE_AUDIT`。均抛
    :class:`StaleAuditError`。这条封死"旧准入报告批改过的正文"。
    """

    stored = recheck.get("input_digest")
    if not stored or not isinstance(stored, str):
        raise StaleAuditError(
            INPUT_MISMATCH,
            "复核报告缺 input_digest——无法证明它审的是当前稿；请用当前 state 重跑 claim_audit。",
        )
    current = compute_input_digest(
        state_dict=state_dict,
        brief=brief,
        audit_brief=audit_brief,
        audit_map=audit_map,
    )
    if current != stored:
        raise StaleAuditError(
            STALE_AUDIT,
            "复核报告的 input_digest 与当前稿不一致（正文/brief 已变，或用了旧报告）——"
            "请对当前 state 重跑 claim_audit 后再渲染交付版。",
        )


def _report_error(detail: str) -> DeliveryGateError:
    return DeliveryGateError(INCOMPLETE_REPORT, detail)


def _current_factual_units(
    state_dict: Mapping[str, object],
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """从当前 state 现场派生交付审计应覆盖的 section / claim unit。

    只纳入 ``accepted`` section 的非占位 unit；placeholder section 与近似纯
    ``【待补充…】`` 的 unit 不要求事实审计，混有实质断言的 unit 仍须覆盖。返回
    ``(expected_sections, {sid: {unit_id: normalized_full_text}})``。
    """

    outcomes = state_dict.get("outcomes") or {}
    order = state_dict.get("order") or sorted(outcomes)  # type: ignore[arg-type]
    expected: list[str] = []
    units: dict[str, dict[str, str]] = {}
    for sid_raw in order:  # type: ignore[union-attr]
        sid = str(sid_raw)
        outcome = outcomes.get(sid) or {}  # type: ignore[union-attr]
        if outcome.get("status") != "accepted":
            continue
        expected.append(sid)
        units[sid] = {}
        draft = outcome.get("draft") or {}
        for paragraph in draft.get("paragraphs", []):
            pid = paragraph.get("paragraph_id")
            if not isinstance(pid, str) or not pid:
                raise _report_error(f"当前 state 的 accepted section {sid} 含未盖章段落，无法派生审计单元")
            text = str(paragraph.get("text", ""))
            for unit in claim_units_for(pid, text):
                if not unit["is_placeholder"]:
                    units[sid][unit["unit_id"]] = unit["norm"]
    return expected, units


def derive_recheck_authorization(
    recheck: Mapping[str, object],
    *,
    state_dict: Mapping[str, object],
    audit_brief: Mapping[str, object],
    audit_map: Mapping[str, Mapping[str, object]],
    deployment_weak_policy: str = "block",
) -> dict:
    """用当前 state、AuditBrief 与 audit map **重跑复核**并派生交付授权。

    ``claims[].recheck``、``summary``、``completeness``、``rule_check_passed``、``accepted`` 和
    ``delivery`` 全是缓存，任何一项与当前输入上的规则重算不一致都拒绝。该模式防陈旧/误修改/
    不完整工件，不构成对拥有整个工作区写权限者的强身份授权。
    """

    from .claim_audit import (
        AUDIT_SHEET_SCHEMA_VERSION,
        RECHECK_VERSION,
        AuditSheetError,
        recheck_audit_sheet,
    )

    if deployment_weak_policy not in ("block", "allow"):
        raise _report_error(
            "deployment_weak_policy 必须是 'block' 或 'allow'")
    if not isinstance(recheck, Mapping):
        raise _report_error("recheck 不是对象")
    if recheck.get("recheck_version") != RECHECK_VERSION:
        raise _report_error(
            f"recheck_version 缺失或不符（需 {RECHECK_VERSION!r}，得 {recheck.get('recheck_version')!r}）")
    if recheck.get("binding_mode") is not True:
        raise DeliveryGateError(
            LEGACY_BINDING_NOT_ALLOWED,
            "section 级 legacy 复核（无段落/单元绑定）不得进入交付路径——请对已盖章 state 重跑 claim_audit。")
    if recheck.get("audit_sheet_schema_version") != AUDIT_SHEET_SCHEMA_VERSION:
        raise _report_error("recheck.audit_sheet_schema_version 缺失或不符")
    for name in ("auditor", "audited_at", "engine_boundary_note", "input_digest"):
        if not isinstance(recheck.get(name), str) or not str(recheck[name]).strip():
            raise _report_error(f"recheck 缺完整报告字段 {name!r} 或字段为空")

    claims = recheck.get("claims")
    delivery = recheck.get("delivery")
    no_claim_sections = recheck.get("no_claim_sections")
    if not isinstance(claims, list):
        raise _report_error("recheck.claims 缺失或非列表")
    if not isinstance(delivery, Mapping):
        raise _report_error("recheck.delivery 缺失或非对象")
    if not isinstance(no_claim_sections, list):
        raise _report_error("recheck.no_claim_sections 缺失或非列表")

    # 当前 state 与传入 AuditBrief 必须描述同一组已收货事实单元；不能用另一份 AuditBrief 重跑。
    expected_sections, factual_units = _current_factual_units(state_dict)
    audit_sections = {
        str(section.get("section_id")): section
        for section in audit_brief.get("sections", [])  # type: ignore[union-attr]
        if isinstance(section, Mapping) and section.get("section_id") is not None
    }
    if set(audit_sections) != set(expected_sections):
        raise _report_error("AuditBrief sections 与当前 state 的 accepted sections 不一致")
    for sid in expected_sections:
        audit_units: dict[str, str] = {}
        paragraphs = audit_sections[sid].get("paragraphs")
        if not isinstance(paragraphs, list):
            raise _report_error(f"AuditBrief section {sid} 缺 paragraphs，不能进入 unit 绑定交付门")
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                raise _report_error(f"AuditBrief section {sid} 含非法 paragraph")
            for unit in paragraph.get("claim_units", []):  # type: ignore[union-attr]
                if not isinstance(unit, Mapping):
                    raise _report_error(f"AuditBrief section {sid} 含非法 claim_unit")
                if not bool(unit.get("is_placeholder")):
                    audit_units[str(unit.get("unit_id"))] = _normalize_for_match(str(unit.get("text", "")))
        if audit_units != factual_units[sid]:
            raise _report_error(f"AuditBrief section {sid} 的事实单元与当前 state 不一致")

    sheet_claims: list[dict] = []
    cached_codes: list[str] = []
    cached_details: list[str] = []
    for position, item in enumerate(claims):
        if not isinstance(item, Mapping):
            raise _report_error(f"recheck.claims[{position}] 非对象")
        if item.get("index") != position:
            raise _report_error(f"recheck.claims[{position}].index 必须等于其列表位置 {position}")
        for name in (
            "section_id", "claim_text", "verdict", "claim_unit_id", "recheck", "detail",
        ):
            if not isinstance(item.get(name), str) or (
                    name != "detail" and not str(item[name]).strip()):
                raise _report_error(f"recheck.claims[{position}].{name} 缺失或非法")
        for name in ("paragraph_id", "paragraph_hash"):
            if item.get(name) is not None and not isinstance(item.get(name), str):
                raise _report_error(f"recheck.claims[{position}].{name} 若存在必须是字符串")
        refs = item.get("cited_span_refs")
        cited_content = item.get("cited_content")
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise _report_error(f"recheck.claims[{position}].cited_span_refs 必须是字符串列表")
        if not isinstance(cited_content, Mapping) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in cited_content.items()):
            raise _report_error(f"recheck.claims[{position}].cited_content 必须是字符串映射")
        # 对真实 ref，v2 报告必须带审计时的内容快照；缺快照不能复核 TAMPERED_CONTENT。
        missing_snapshots = [ref for ref in refs if ref in audit_map and ref not in cited_content]
        if missing_snapshots:
            raise _report_error(
                f"recheck.claims[{position}] 缺证据内容快照：{missing_snapshots}")
        sheet_claims.append({
            "claim_text": item["claim_text"],
            "section_id": item["section_id"],
            "verdict": item["verdict"],
            "cited_span_refs": list(refs),
            "cited_content": dict(cited_content),
            "paragraph_id": item["paragraph_id"],
            "paragraph_hash": item["paragraph_hash"],
            "claim_unit_id": item["claim_unit_id"],
        })
        cached_codes.append(str(item["recheck"]))
        cached_details.append(str(item["detail"]))

    applied_signoffs = delivery.get("applied_signoffs")
    if not isinstance(applied_signoffs, list):
        raise _report_error("recheck.delivery.applied_signoffs 缺失或非列表")
    signoffs: dict[int, dict] = {}
    for position, signoff in enumerate(applied_signoffs):
        if not isinstance(signoff, Mapping) or not isinstance(signoff.get("index"), int):
            raise _report_error(f"recheck.delivery.applied_signoffs[{position}] 非法")
        index = int(signoff["index"])
        if index in signoffs:
            raise _report_error(f"claim {index} 存在重复 signoff")
        signoffs[index] = {
            "signer": signoff.get("signer"),
            "reason": signoff.get("reason"),
        }

    sheet = {
        "audit_sheet_schema_version": AUDIT_SHEET_SCHEMA_VERSION,
        "meta": {"auditor": recheck["auditor"], "audited_at": recheck["audited_at"]},
        "no_claim_sections": list(no_claim_sections),
        "claims": sheet_claims,
    }
    report_weak_policy = delivery.get("weak_policy")
    if report_weak_policy not in ("block", "allow"):
        raise _report_error("recheck.delivery.weak_policy 必须是 'block' 或 'allow'")
    # 部署方策略与审计报告策略取更严者；报告中的 allow 不能自行降低渲染端要求。
    effective_weak_policy = (
        "block"
        if "block" in (report_weak_policy, deployment_weak_policy)
        else "allow"
    )
    try:
        recomputed = recheck_audit_sheet(
            audit_brief=audit_brief,
            sheet=sheet,
            audit_map=audit_map,
            weak_policy=effective_weak_policy,
            signoffs=signoffs,
        )
    except (AuditSheetError, KeyError, TypeError, ValueError) as exc:
        raise _report_error(f"无法从报告缓存重建并复核审计单：{exc}") from exc

    for position, item in enumerate(recomputed["claims"]):
        if cached_codes[position] != item["recheck"]:
            raise _report_error(
                f"recheck.claims[{position}].recheck 缓存={cached_codes[position]!r}，"
                f"当前证据复算={item['recheck']!r}")
        if cached_details[position] != item["detail"]:
            raise _report_error(f"recheck.claims[{position}].detail 与当前证据复算不一致")

    for name in ("summary", "completeness", "rule_check_passed", "accepted"):
        if recheck.get(name) != recomputed.get(name):
            raise _report_error(f"recheck.{name} 与当前证据/claims/completeness/signoff 复算不一致")
    # weak_policy 本身允许因部署方要求而收紧；其余 delivery 字段必须与更严策略下的复算一致。
    for name in (
        "admitted", "status", "reason", "blocking_claims",
        "weak_unresolved", "applied_signoffs",
    ):
        if delivery.get(name) != recomputed["delivery"].get(name):
            raise _report_error(
                f"recheck.delivery.{name} 与部署方 weak 策略下的复算不一致")

    return {
        "rule_check_passed": recomputed["rule_check_passed"],
        "accepted": recomputed["accepted"],
        "delivery_admitted": recomputed["delivery"]["admitted"],
        "expected_sections": sorted(expected_sections),
        "covered_unit_count": sum(len(items) for items in factual_units.values()),
        "effective_weak_policy": effective_weak_policy,
    }


def assert_deliverable_recheck(
    recheck: Mapping[str, object],
    *,
    state_dict: Mapping[str, object],
    brief: Mapping[str, object],
    audit_brief: Mapping[str, object],
    audit_map: Mapping[str, Mapping[str, object]],
    deployment_weak_policy: str = "block",
) -> None:
    """交付版渲染的**唯一入口校验**：完整 RecheckReport + 当前稿现场派生授权。

    逐项（缺任一或不过即抛 :class:`DeliveryGateError`）：
    1. 完整 schema、当前版本、unit binding；
    2. 从当前 state 重新派生应覆盖的完整 unit；
    3. 用当前 AuditBrief/audit map 重跑 FABRICATED_REF / TAMPERED_CONTENT /
       JURISDICTION_VIOLATION 等全部规则，并从 claims / completeness / signoff 重算授权；
    4. 报告 weak 策略与部署方 ``deployment_weak_policy`` 取更严者（默认 block）；
    5. ``input_digest`` 与当前稿、brief、AuditBrief、audit map 现算一致。

    integrity 级只防陈旧、误修改、不完整审计与自报字段漂移；不防拥有全部工作区写权限者重写整套
    state/audit/recheck。强对抗授权需要下一阶段独立审计服务或受隔离签名密钥。
    """
    derived = derive_recheck_authorization(
        recheck,
        state_dict=state_dict,
        audit_brief=audit_brief,
        audit_map=audit_map,
        deployment_weak_policy=deployment_weak_policy,
    )
    if not derived["rule_check_passed"]:
        raise _report_error("逐 claim/completeness 现算未通过规则层")
    if not derived["accepted"] or not derived["delivery_admitted"]:
        raise _report_error("逐 claim/completeness/signoff 现算未获交付准入")
    verify_recheck_fresh(
        recheck,
        state_dict=state_dict,
        brief=brief,
        audit_brief=audit_brief,
        audit_map=audit_map,
    )


def response_fingerprint(payload: object) -> str:
    """回稿内容指纹（sha256）：回稿一变即变。编排器 re-run 用它检测"已 accepted 的 section 其
    回稿被改过”，据此让旧 state 和旧审计失效。``payload`` 是回稿 dict 或其 JSON 文本。
    """

    text = payload if isinstance(payload, str) else _canonical(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
