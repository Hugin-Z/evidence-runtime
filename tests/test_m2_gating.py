"""M2 Evidence Gate 单测：确定性四态映射 + section 聚合 + 契约级 + 边界。"""

from __future__ import annotations

import pytest

from evidence.gating import (
    ContractGate,
    SlotGate,
    aggregate_section_gate,
    contract_gate,
    gate_for_slot_status,
)
from evidence.types import SlotStatus


def test_slot_status_four_state_mapping():
    assert gate_for_slot_status(SlotStatus.FILLED) is SlotGate.ALLOW
    assert gate_for_slot_status(SlotStatus.WEAK) is SlotGate.LIMITED
    assert gate_for_slot_status(SlotStatus.MISSING) is SlotGate.BLOCK
    assert gate_for_slot_status(SlotStatus.CONFLICTING) is SlotGate.NEEDS_RESOLUTION


def test_slot_gate_covers_every_status_no_gap():
    # 四态全覆盖：任何 SlotStatus 都有映射，不会漏。
    for status in SlotStatus:
        assert isinstance(gate_for_slot_status(status), SlotGate)


def test_gate_for_slot_status_rejects_non_status():
    with pytest.raises(TypeError):
        gate_for_slot_status("filled")  # 字符串不是 SlotStatus


@pytest.mark.parametrize(
    ("gates", "expected"),
    (
        ([SlotGate.ALLOW], SlotGate.ALLOW),
        ([SlotGate.ALLOW, SlotGate.ALLOW], SlotGate.ALLOW),
        ([SlotGate.BLOCK, SlotGate.BLOCK], SlotGate.BLOCK),
        ([SlotGate.ALLOW, SlotGate.BLOCK], SlotGate.LIMITED),
        ([SlotGate.ALLOW, SlotGate.LIMITED], SlotGate.LIMITED),
        ([SlotGate.LIMITED, SlotGate.BLOCK], SlotGate.LIMITED),
        # needs_resolution 优先（冲突最需先处理）
        ([SlotGate.ALLOW, SlotGate.NEEDS_RESOLUTION], SlotGate.NEEDS_RESOLUTION),
        ([SlotGate.BLOCK, SlotGate.NEEDS_RESOLUTION], SlotGate.NEEDS_RESOLUTION),
    ),
)
def test_aggregate_section_gate(gates, expected):
    assert aggregate_section_gate(gates) is expected


def test_aggregate_section_gate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_section_gate([])


def test_aggregate_section_gate_rejects_non_gate():
    with pytest.raises(TypeError):
        aggregate_section_gate(["allow"])


@pytest.mark.parametrize(
    ("status", "slot_gates", "expected"),
    (
        ("complete", [SlotGate.ALLOW, SlotGate.ALLOW], ContractGate.COMPLETE),
        ("conflicted", [SlotGate.NEEDS_RESOLUTION], ContractGate.CONFLICTED),
        # incomplete 且有可生成槽 → partial
        ("incomplete", [SlotGate.ALLOW, SlotGate.BLOCK], ContractGate.PARTIAL),
        ("incomplete", [SlotGate.LIMITED, SlotGate.BLOCK], ContractGate.PARTIAL),
        # incomplete 且全阻断 → abstained
        ("incomplete", [SlotGate.BLOCK, SlotGate.BLOCK], ContractGate.ABSTAINED),
        # conflicted 优先于其它（即便有 allow）
        ("conflicted", [SlotGate.ALLOW, SlotGate.NEEDS_RESOLUTION], ContractGate.CONFLICTED),
    ),
)
def test_contract_gate_levels(status, slot_gates, expected):
    assert contract_gate(status, slot_gates) is expected


def test_contract_gate_rejects_bad_status():
    with pytest.raises(ValueError):
        contract_gate("ready", [SlotGate.ALLOW])


def test_gate_values_are_json_friendly_strings():
    # 落 JSON 时用 .value；确认字符串稳定（对下游是契约）。
    assert SlotGate.ALLOW.value == "allow"
    assert SlotGate.LIMITED.value == "limited"
    assert SlotGate.BLOCK.value == "block"
    assert SlotGate.NEEDS_RESOLUTION.value == "needs_resolution"
    assert {g.value for g in ContractGate} == {
        "complete", "partial", "abstained", "conflicted"
    }
