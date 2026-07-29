"""L0 输入校验：answers → 校验 → ``ProjectSpec``。

**A1 三明治的收货侧**：问答的执行由 agent/人完成（语义），交回来的 answers **由代码校验**：
必填齐全、枚举合法、依赖满足、doc_type 有对应契约、指定章节确实存在。不过则**拒绝并给出
完整缺失清单**（收集全部问题，不 fail-fast），绝不静默兜底。
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

from ..brief import CrosswalkError, load_crosswalk
from ..contract import load_evidence_contract
from ..library import SourceType
from .questions import QUESTIONS, Question
from .registry import ContractSelectionError, has_binding, select_contract
from .spec import ProjectSpec, SectionScope


class IntakeValidationError(ValueError):
    """intake answers 未通过收货校验。``problems`` 是完整缺失/非法清单。"""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        joined = "\n".join(f"  - {item}" for item in self.problems)
        super().__init__(f"intake answers rejected ({len(self.problems)} problem(s)):\n{joined}")


def _normalise_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        # 允许交互模式提交逗号分隔字符串；文件模式建议直接给数组。
        text = value.decode() if isinstance(value, bytes) else value
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def validate_answers(answers: Mapping[str, object]) -> ProjectSpec:
    """把 answers 校验并构造成 ``ProjectSpec``。失败抛 :class:`IntakeValidationError`。"""

    problems: list[str] = []

    # 1) 逐题：必填齐全 / 枚举合法 / 依赖满足（依赖不满足的题跳过校验）
    for question in QUESTIONS:
        active = question.is_active(answers)
        raw = answers.get(question.id)
        present = raw is not None and (not isinstance(raw, str) or raw.strip() != "")

        if not active:
            if present:
                problems.append(
                    f"{question.id}: 依赖未满足（{question.depends_on!r}）却给了值"
                )
            continue

        if not present:
            if question.required:
                problems.append(f"{question.id}: 必填项缺失（{question.text}）")
            continue

        if question.type in {"enum"} and question.choices:
            if str(raw) not in question.choices:
                problems.append(
                    f"{question.id}: 非法取值 {raw!r}，须为 {list(question.choices)!r}"
                )
        elif question.type == "string_list" and question.choices:
            for item in _normalise_list(raw) or []:
                if item not in question.choices:
                    problems.append(
                        f"{question.id}: 非法取值 {item!r}，须为 {list(question.choices)!r}"
                    )

    # 未知问题 id（打错字/多塞字段）也报出来，避免静默丢弃
    known = {question.id for question in QUESTIONS}
    for key in answers:
        if key not in known:
            problems.append(f"{key}: 未知字段（不在问题清单中）")

    # 2) doc_type → 有对应契约（A3 硬门槛）
    doc_type = answers.get("doc_type")
    binding = None
    if isinstance(doc_type, str) and doc_type.strip():
        if not has_binding(doc_type):
            try:
                select_contract(doc_type)
            except ContractSelectionError as exc:
                problems.append(f"doc_type: {exc}")
        else:
            binding = select_contract(doc_type)

    # 3) source_types 合法性（若给）
    for item in _normalise_list(answers.get("source_types")) or []:
        try:
            SourceType.parse(item)
        except ValueError:
            problems.append(
                f"source_types: 非法来源 {item!r}，须为 "
                f"{[i.value for i in SourceType]!r}"
            )

    # 4) 指定章节确实存在于该 doc_type 的 crosswalk（需契约绑定成立才能查）
    if (
        binding is not None
        and str(answers.get("section_scope", "")) == SectionScope.CUSTOM.value
    ):
        requested = _normalise_list(answers.get("section_ids")) or []
        if requested:
            try:
                contract = load_evidence_contract(str(binding.contract_path))
                crosswalk = load_crosswalk(binding.crosswalk_doc_type, contract=contract)
                available = {
                    str(sec.get("section_id"))
                    for sec in crosswalk.get("sections", [])  # type: ignore[union-attr]
                }
                for sid in requested:
                    if sid not in available:
                        problems.append(
                            f"section_ids: 章节 {sid!r} 不在 {doc_type!r} 的 crosswalk "
                            f"（可选：{sorted(available)!r}）"
                        )
            except (CrosswalkError, OSError, ValueError) as exc:
                problems.append(f"section_ids: 无法校验章节（{type(exc).__name__}: {exc}）")

    if problems:
        raise IntakeValidationError(problems)

    return ProjectSpec.from_dict({
        "doc_type": str(doc_type),
        "project_name": str(answers["project_name"]).strip(),
        "client_name": str(answers["client_name"]).strip(),
        "industry_domain": _clean_opt(answers.get("industry_domain")),
        "section_scope": str(answers.get("section_scope", SectionScope.ALL.value)),
        "section_ids": _normalise_list(answers.get("section_ids")) or [],
        "project_id": str(answers["project_id"]).strip(),
        "source_types": _normalise_list(answers.get("source_types")),
        "project_number": _clean_opt(answers.get("project_number")),
        "budget": _clean_opt(answers.get("budget")),
        "duration": _clean_opt(answers.get("duration")),
        "project_nature": _clean_opt(answers.get("project_nature")),
    })


def _clean_opt(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_answers_file(path: str) -> dict:
    """读预填 answers 文件（JSON object）。"""

    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise IntakeValidationError(["answers 文件必须是 JSON 对象（问题 id → 值）"])
    return data


def load_project_spec(path: str) -> ProjectSpec:
    """读已落盘的 ProjectSpec JSON（复用/修改后再驱动全链）。"""

    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise IntakeValidationError(["ProjectSpec 文件必须是 JSON 对象"])
    return ProjectSpec.from_dict(data)
