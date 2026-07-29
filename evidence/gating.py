"""Evidence Gate：确定性、model-free 的生成闸门。

Gate 只**机械读取** Verifier / Gap Report 的状态并映射为「当前允许生成什么」，不重新检索、
不做语义判断，也不调用任何 LLM。

两级：
- **槽级 SlotGate**：
  filled → allow / weak → limited / missing → block / conflicting → needs_resolution。
- **契约级 ContractGate**（DESIGN §13 词表）：complete / partial / abstained / conflicted，
  由契约状态（complete/incomplete/conflicted）+ 槽级门聚合得出。

section 级门（drafter 一个 section 可映射多个槽）由 :func:`aggregate_section_gate` 聚合。
纯规则、无副作用、可完全单测。
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from .types import SlotStatus


class SlotGate(str, Enum):
    """槽级生成闸门的四态映射。"""

    ALLOW = "allow"                        # filled：证据充分，允许生成
    LIMITED = "limited"                    # weak：可用但不足，需标注/限制
    BLOCK = "block"                        # missing：证据不足，阻断（下游写【待补充】）
    NEEDS_RESOLUTION = "needs_resolution"  # conflicting：存在未解决冲突，需先消解


class ContractGate(str, Enum):
    """契约级顶层闸门（DESIGN §13 词表）。"""

    COMPLETE = "complete"        # 必要证据完整
    PARTIAL = "partial"          # 仅允许生成已被支持的部分
    ABSTAINED = "abstained"      # 必要证据不足，不允许形成结论
    CONFLICTED = "conflicted"    # 证据存在未解决冲突


_SLOT_STATUS_TO_GATE: dict[SlotStatus, SlotGate] = {
    SlotStatus.FILLED: SlotGate.ALLOW,
    SlotStatus.WEAK: SlotGate.LIMITED,
    SlotStatus.MISSING: SlotGate.BLOCK,
    SlotStatus.CONFLICTING: SlotGate.NEEDS_RESOLUTION,
}

# 契约状态字符串（GapReport.status / DESIGN 聚合口径）。
_CONTRACT_STATUSES = ("complete", "incomplete", "conflicted")


def gate_for_slot_status(status: SlotStatus) -> SlotGate:
    """单槽状态 → 槽级闸门（纯映射，四态全覆盖）。"""

    if not isinstance(status, SlotStatus):
        raise TypeError(f"status must be SlotStatus; got {status!r}")
    try:
        return _SLOT_STATUS_TO_GATE[status]
    except KeyError:  # pragma: no cover - 枚举已穷举，防御未来新增态未接线
        raise ValueError(f"no gate mapping for SlotStatus {status!r}")


def aggregate_section_gate(gates: Iterable[SlotGate]) -> SlotGate:
    """把一个 section 映射到的多个槽级闸门聚合成 section 级闸门。

    优先级（确定性）：任一 needs_resolution → needs_resolution（冲突最需先处理）；
    否则全 allow → allow；全 block → block；其余（allow/limited/block 混合）→ limited。
    """

    gate_list = list(gates)
    if not gate_list:
        raise ValueError("aggregate_section_gate requires at least one gate")
    for gate in gate_list:
        if not isinstance(gate, SlotGate):
            raise TypeError(f"gates must contain SlotGate; got {gate!r}")
    if any(g is SlotGate.NEEDS_RESOLUTION for g in gate_list):
        return SlotGate.NEEDS_RESOLUTION
    if all(g is SlotGate.ALLOW for g in gate_list):
        return SlotGate.ALLOW
    if all(g is SlotGate.BLOCK for g in gate_list):
        return SlotGate.BLOCK
    return SlotGate.LIMITED


def contract_gate(contract_status: str, slot_gates: Iterable[SlotGate]) -> ContractGate:
    """契约状态（complete/incomplete/conflicted）+ 槽级门 → 顶层 ContractGate。

    conflicted → CONFLICTED；complete → COMPLETE；incomplete → 若尚有可生成槽
    （allow/limited）则 PARTIAL，否则（全 block）ABSTAINED。DESIGN §13 词表。
    """

    if contract_status not in _CONTRACT_STATUSES:
        raise ValueError(
            f"contract_status must be one of {_CONTRACT_STATUSES!r}; got {contract_status!r}"
        )
    gate_list = list(slot_gates)
    for gate in gate_list:
        if not isinstance(gate, SlotGate):
            raise TypeError(f"slot_gates must contain SlotGate; got {gate!r}")
    if contract_status == "conflicted":
        return ContractGate.CONFLICTED
    if contract_status == "complete":
        return ContractGate.COMPLETE
    # incomplete
    if any(g in (SlotGate.ALLOW, SlotGate.LIMITED) for g in gate_list):
        return ContractGate.PARTIAL
    return ContractGate.ABSTAINED
