"""交付状态机：把“生成 settled”“审计准入”“是否含待补料占位”
合成一个交付态。

    UNSETTLED → GENERATED → AUDITED ──admitted──┬─无占位─▶ DELIVERABLE
                              │                 └─含占位─▶ SHAREABLE_DRAFT
                              └─blocked─▶ BLOCKED ──处置(删除/弱化→重审 或 人工签字)──▶ AUDITED …

**交付语义拆分**：此前"通过准入"一律叫 DELIVERABLE，但含关键占位（整节
【待补充】、oos 占位、或 accepted 节里的占位段落）的文档**名不副实**。拆两态：

- **SHAREABLE_DRAFT**：通过准入**但含未补料占位**——可内部流转 / 给客户看进度，须标注"草稿·含待补料"；
- **DELIVERABLE**：通过准入**且无任何待补料占位**——才是可对外交付。

render 按态产出不同文件名与水印；verify_docx 校验"声称的态 ↔ 实际内容"一致（声称 DELIVERABLE
却含占位 → 拒）。三个前置（settled、admitted）不变；占位判定不碰判定层，只读 state 结构。
"""
from __future__ import annotations

from enum import Enum
from typing import Mapping

from .claim_audit import BLOCKING_VERDICTS
from .generation.orchestrator import OrchestrationState


class DeliveryState(str, Enum):
    UNSETTLED = "UNSETTLED"            # 生成未就绪（有 awaiting/rejected/failed）
    GENERATED = "GENERATED"            # settled，但尚未审计
    BLOCKED = "BLOCKED"                # 已审计，但交付准入未过（有 unsupported/contradicted 未处置）
    SHAREABLE_DRAFT = "SHAREABLE_DRAFT"  # 准入通过但含待补料占位——可内部/进度流转，非交付物
    DELIVERABLE = "DELIVERABLE"        # 准入通过且无占位 → 可对外交付


def has_unfilled_placeholders(state: OrchestrationState) -> bool:
    """会话是否含**待补料占位**：整节占位（oos/block/needs_resolution）或 accepted 节里的占位段落。

    只读 state 结构（占位段落 = `【待补充` 前缀），不碰判定层。用于 DELIVERABLE vs SHAREABLE_DRAFT。
    """

    for sid in state.order:
        oc = state.outcomes[sid]
        if oc.status == "placeholder":
            return True
        if oc.status == "accepted" and oc.draft is not None:
            if any(p.is_placeholder for p in oc.draft.paragraphs):
                return True
    return False


def delivery_state(
    state: OrchestrationState,
    recheck: Mapping[str, object] | None,
) -> DeliveryState:
    """由生成状态 + 审计复核报告 + 占位情况推导交付态。``recheck=None`` = 尚未审计。"""

    if not state.settled():
        return DeliveryState.UNSETTLED
    if recheck is None:
        return DeliveryState.GENERATED
    delivery = recheck.get("delivery") or {}
    if not delivery.get("admitted"):
        return DeliveryState.BLOCKED
    return (DeliveryState.SHAREABLE_DRAFT if has_unfilled_placeholders(state)
            else DeliveryState.DELIVERABLE)


def is_deliverable(state: OrchestrationState, recheck: Mapping[str, object] | None) -> bool:
    """严格可交付：准入通过**且无占位**。SHAREABLE_DRAFT 不算 deliverable。"""
    return delivery_state(state, recheck) is DeliveryState.DELIVERABLE


def is_admitted(state: OrchestrationState, recheck: Mapping[str, object] | None) -> bool:
    """准入通过（DELIVERABLE 或 SHAREABLE_DRAFT 都算）——render 出件的门槛。"""
    return delivery_state(state, recheck) in (
        DeliveryState.DELIVERABLE, DeliveryState.SHAREABLE_DRAFT)


def remediation_directives(recheck: Mapping[str, object]) -> list[dict]:
    """把复核报告里 BLOCKED 的原因翻成**处置指令**（给写手/审计者）。

    每条 blocking/weak-未处置 claim → 一条指令，列出三条确定性处置选项：
    - ``delete``：删除该 claim 所在段落 → 重新生成该 section → 重跑审计；
    - ``weaken``：弱化为 `【待补充】` pending 占位 → 重新生成 → 重跑审计；
    - ``signoff``：人工签字放行（须记签字人 + 理由，入 recheck 的 signoffs）。
    Engine 不判"该选哪条"（语义），只强制"必须处置且留痕，处置后必须重审"。
    """

    delivery = recheck.get("delivery") or {}
    items = list(delivery.get("blocking_claims") or []) + list(delivery.get("weak_unresolved") or [])
    directives: list[dict] = []
    for it in items:
        directives.append({
            "index": it.get("index"),
            "section_id": it.get("section_id"),
            "verdict": it.get("verdict", "weak"),
            "claim_text": it.get("claim_text"),
            "options": ["delete", "weaken", "signoff"],
            "note": "删除/弱化 → 改稿后必须重跑审计+复核；签字 → 记 signer+reason 后重跑复核。",
        })
    return directives
