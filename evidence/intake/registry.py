"""L0 契约选择：``doc_type → (contract, crosswalk)`` 确定性查表。

doc_type 到证据契约的映射是代码查表，不是 agent/LLM 猜。
当前一个 doc_type（对应现有契约）；表结构支持多 doc_type —— 将来加一种输出文档 = 注册一行。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..resources import contracts_dir
from .spec import DocType

_CONTRACTS_DIR = contracts_dir()


@dataclass(frozen=True)
class ContractBinding:
    """一个 doc_type 绑定的契约与 crosswalk。"""

    doc_type: str
    contract_path: Path
    crosswalk_doc_type: str   # 传给 evidence.brief.load_crosswalk 的 doc_type 标识


# doc_type → 绑定。**新增 doc_type 在此注册一行**（并建对应 crosswalk）。
_REGISTRY: dict[str, ContractBinding] = {
    DocType.PROPOSAL_DOCUMENT.value: ContractBinding(
        doc_type=DocType.PROPOSAL_DOCUMENT.value,
        contract_path=_CONTRACTS_DIR / "solution_material_completeness_v1.json",
        crosswalk_doc_type="proposal-document",
    ),
}


class ContractSelectionError(ValueError):
    """doc_type 没有对应的契约绑定。"""


def registered_doc_types() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def has_binding(doc_type: str) -> bool:
    return doc_type in _REGISTRY


def select_contract(doc_type: str) -> ContractBinding:
    """确定性查表。未注册 → 抛 :class:`ContractSelectionError`（不猜、不兜底）。"""

    binding = _REGISTRY.get(doc_type)
    if binding is None:
        raise ContractSelectionError(
            f"doc_type {doc_type!r} has no registered contract; "
            f"registered: {registered_doc_types()!r}"
        )
    return binding
