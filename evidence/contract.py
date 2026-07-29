"""固定 Evidence Contract 的严格加载。

``load_evidence_contract`` + 模块级不可变对象 ``solution_material_completeness_v1``；
本模块不实现动态 Contract 编译或运行时 Slot 修改。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import (
    EvidenceContract,
    EvidenceSlot,
)


_CONTRACT_KEYS = {
    "contract_id",
    "version",
    "name",
    "description",
    "slots",
    "allow_partial",
    "citation_required",
    "risk_level",
}
_SLOT_KEYS = {
    "slot_id",
    "name",
    "description",
    "required",
    "minimum_evidence",
    "required_fields",
    "weak_conditions",
    "material_request",
}


def _require_object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object; got {value!r}")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    location: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys={missing!r}")
        if unknown:
            details.append(f"unknown keys={unknown!r}")
        raise ValueError(f"{location} schema mismatch: {', '.join(details)}")


def _require_bool(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{location} must be bool; got {value!r}")
    return value


def _parse_evidence_slot(raw: object, index: int) -> EvidenceSlot:
    location = f"EvidenceContract.slots[{index}]"
    data = _require_object(raw, location)
    _require_exact_keys(data, _SLOT_KEYS, location)
    required_fields = data["required_fields"]
    weak_conditions = data["weak_conditions"]
    if not isinstance(required_fields, list):
        raise ValueError(
            f"{location}.required_fields must be a JSON array; got {required_fields!r}"
        )
    if not isinstance(weak_conditions, list):
        raise ValueError(
            f"{location}.weak_conditions must be a JSON array; got {weak_conditions!r}"
        )
    return EvidenceSlot(
        slot_id=data["slot_id"],
        name=data["name"],
        description=data["description"],
        required=_require_bool(data["required"], f"{location}.required"),
        minimum_evidence=data["minimum_evidence"],
        required_fields=tuple(required_fields),
        weak_conditions=tuple(weak_conditions),
        material_request=data["material_request"],
    )


def load_evidence_contract(path: str | Path) -> EvidenceContract:
    """严格读取不可变 Contract；缺字段、未知字段或非法值均显式失败。"""

    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    data = _require_object(raw, "EvidenceContract")
    _require_exact_keys(data, _CONTRACT_KEYS, "EvidenceContract")
    raw_slots = data["slots"]
    if not isinstance(raw_slots, list):
        raise ValueError(
            f"EvidenceContract.slots must be a JSON array; got {raw_slots!r}"
        )
    slots = tuple(_parse_evidence_slot(slot, index) for index, slot in enumerate(raw_slots))
    return EvidenceContract(
        contract_id=data["contract_id"],
        version=data["version"],
        name=data["name"],
        description=data["description"],
        slots=slots,
        allow_partial=_require_bool(data["allow_partial"], "EvidenceContract.allow_partial"),
        citation_required=_require_bool(
            data["citation_required"],
            "EvidenceContract.citation_required",
        ),
        risk_level=data["risk_level"],
    )


from .resources import contract_file

_SOLUTION_MATERIAL_CONTRACT_PATH = contract_file("solution_material_completeness_v1.json")

solution_material_completeness_v1 = load_evidence_contract(
    _SOLUTION_MATERIAL_CONTRACT_PATH
)
