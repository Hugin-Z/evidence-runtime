"""M1 Foundation Slice：固定 Contract 与不可变 schema。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from evidence import (
    EvidenceContract,
    EvidenceSlot,
    SlotStatus,
    load_evidence_contract,
    solution_material_completeness_v1,
)


EXPECTED_SLOT_IDS = (
    "background_and_basis",
    "current_state_and_needs",
    "scope_and_subjects",
    "objectives",
    "solution_content_and_capabilities",
    "implementation_conditions_and_boundaries",
    "reference_cases_and_proof",
    "outcomes_and_metrics",
)
EXPECTED_SLOT_NAMES = (
    "背景与依据",
    "现状问题与需求",
    "建设对象与范围",
    "建设目标",
    "建设内容与能力",
    "实施条件与边界",
    "项目案例与证明",
    "建设成效与指标",
)


def _slot(slot_id: str = "slot-a", **overrides) -> EvidenceSlot:
    values = {
        "slot_id": slot_id,
        "name": "测试槽",
        "description": "用于验证不可变 Contract schema。",
        "required": True,
        "minimum_evidence": 1,
        "required_fields": ("subject",),
        "weak_conditions": ("只有标题",),
        "material_request": "补充主体和正文证据。",
    }
    values.update(overrides)
    return EvidenceSlot(**values)


def _contract(*slots: EvidenceSlot) -> EvidenceContract:
    return EvidenceContract(
        contract_id="test_contract_v1",
        version="1.0",
        name="测试 Contract",
        description="只用于 schema 单元测试。",
        slots=tuple(slots),
        allow_partial=True,
        citation_required=True,
        risk_level="high",
    )


def test_slot_status_is_exactly_the_frozen_four_values():
    assert tuple(status.value for status in SlotStatus) == (
        "filled",
        "weak",
        "missing",
        "conflicting",
    )


def test_fixed_contract_has_exactly_eight_stable_slots():
    contract = solution_material_completeness_v1
    assert contract.contract_id == "solution_material_completeness_v1"
    assert contract.slot_ids == EXPECTED_SLOT_IDS
    assert tuple(slot.name for slot in contract.slots) == EXPECTED_SLOT_NAMES
    assert len(contract.slots) == 8


def test_fixed_contract_all_slots_are_required_and_actionable():
    for slot in solution_material_completeness_v1.slots:
        assert slot.required is True
        assert slot.minimum_evidence >= 1
        assert slot.description
        assert slot.required_fields
        assert slot.weak_conditions
        assert slot.material_request


def test_contract_rejects_duplicate_slot_ids():
    with pytest.raises(ValueError, match="duplicate slot_id.*slot-a"):
        _contract(_slot(), _slot(name="另一个名称"))


@pytest.mark.parametrize("minimum_evidence", [0, -1, True, 1.5])
def test_slot_rejects_invalid_minimum_evidence(minimum_evidence):
    with pytest.raises(ValueError, match="minimum_evidence.*integer >= 1"):
        _slot(minimum_evidence=minimum_evidence)


@pytest.mark.parametrize("required_fields", [(), [], "subject"])
def test_slot_rejects_empty_or_non_sequence_required_fields(required_fields):
    with pytest.raises(ValueError, match="required_fields"):
        _slot(required_fields=required_fields)


def test_unknown_slot_lookup_fails_clearly():
    with pytest.raises(KeyError, match="unknown slot_id 'not-present'.*test_contract_v1"):
        _contract(_slot()).get_slot("not-present")


def test_slot_lookup_returns_declared_instance():
    slot = _slot()
    assert _contract(slot).get_slot("slot-a") is slot


def test_fixed_contract_slots_cannot_be_appended_or_reassigned():
    contract = solution_material_completeness_v1
    assert isinstance(contract.slots, tuple)
    with pytest.raises(AttributeError):
        contract.slots.append(_slot())  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        contract.slots = contract.slots + (_slot(),)  # type: ignore[misc]


def test_fixed_slot_fields_cannot_be_reassigned():
    slot = solution_material_completeness_v1.slots[0]
    with pytest.raises(FrozenInstanceError):
        slot.name = "被修改"  # type: ignore[misc]


def test_strict_loader_rejects_unknown_contract_fields(tmp_path: Path):
    source = Path(__file__).parent.parent / "evidence" / "contracts" / "solution_material_completeness_v1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["unexpected"] = "not allowed"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys=.*unexpected"):
        load_evidence_contract(invalid)


def test_strict_loader_rejects_duplicate_slot_ids(tmp_path: Path):
    source = Path(__file__).parent.parent / "evidence" / "contracts" / "solution_material_completeness_v1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["slots"][1]["slot_id"] = raw["slots"][0]["slot_id"]
    invalid = tmp_path / "duplicate.json"
    invalid.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate slot_id"):
        load_evidence_contract(invalid)
