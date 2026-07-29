"""P-B：L0 Intake（ProjectSpec + 收货校验 + 契约映射 + 两种执行形态）单元测试。"""
from __future__ import annotations

import json

import pytest

from evidence.intake import (
    ContractSelectionError,
    DocType,
    IntakeValidationError,
    ProjectSpec,
    QUESTIONS,
    SectionScope,
    collect_interactive,
    load_project_spec,
    registered_doc_types,
    run_from_file,
    save_spec,
    select_contract,
    validate_answers,
)


def _valid_answers(**overrides) -> dict:
    base = {
        "doc_type": "proposal-document",
        "project_name": "某市 CIM 平台建设项目",
        "client_name": "某市住房和城乡建设局",
        "project_id": "acme-2026",
    }
    base.update(overrides)
    return base


# ── ProjectSpec 数据结构 ─────────────────────────────────────────────────
def test_projectspec_roundtrips_through_dict():
    spec = validate_answers(_valid_answers(
        industry_domain="数字住建", section_scope="custom", section_ids=["01", "02"],
        source_types=["client_provided", "public"], budget="800 万", duration="6 个月",
        project_number="ZB-2026-001", project_nature="platform",
    ))
    restored = ProjectSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.section_ids == ("01", "02")
    assert restored.source_types == ("client_provided", "public")


def test_projectspec_json_is_stable(tmp_path):
    spec = validate_answers(_valid_answers())
    path = save_spec(spec, tmp_path / "spec.json")
    assert load_project_spec(str(path)) == spec
    assert json.loads(path.read_text(encoding="utf-8"))["spec_version"] == spec.spec_version


def test_resolved_source_types_none_means_no_filter():
    spec = validate_answers(_valid_answers())
    assert spec.source_types is None
    assert spec.resolved_source_types() is None


# ── 契约映射（确定性查表）────────────────────────────────────────────────
def test_contract_selection_is_a_deterministic_table_lookup():
    binding = select_contract("proposal-document")
    assert binding.doc_type == "proposal-document"
    assert binding.contract_path.name == "solution_material_completeness_v1.json"
    assert binding.crosswalk_doc_type == "proposal-document"
    assert binding.contract_path.is_file()


def test_unknown_doc_type_is_rejected_not_guessed():
    with pytest.raises(ContractSelectionError, match="no registered contract"):
        select_contract("investment-memo")


def test_every_registered_doc_type_resolves():
    for doc_type in registered_doc_types():
        assert select_contract(doc_type).contract_path.is_file()


def test_doctype_enum_matches_registry():
    assert DocType.PROPOSAL_DOCUMENT.value in registered_doc_types()


# ── 收货校验（A3）：各类拒绝 ─────────────────────────────────────────────
def test_missing_required_field_is_rejected_with_list():
    answers = _valid_answers()
    del answers["project_name"]
    del answers["client_name"]
    with pytest.raises(IntakeValidationError) as exc:
        validate_answers(answers)
    problems = exc.value.problems
    assert any("project_name" in p for p in problems)
    assert any("client_name" in p for p in problems)


def test_illegal_enum_is_rejected():
    with pytest.raises(IntakeValidationError, match="section_scope"):
        validate_answers(_valid_answers(section_scope="partial"))


def test_unknown_doc_type_rejected_at_intake():
    with pytest.raises(IntakeValidationError, match="doc_type"):
        validate_answers(_valid_answers(doc_type="investment-memo"))


def test_dependency_violation_custom_scope_requires_section_ids():
    with pytest.raises(IntakeValidationError, match="section_ids"):
        validate_answers(_valid_answers(section_scope="custom"))


def test_section_ids_must_exist_in_crosswalk():
    with pytest.raises(IntakeValidationError, match="99"):
        validate_answers(_valid_answers(section_scope="custom", section_ids=["01", "99"]))


def test_section_ids_given_without_custom_scope_is_flagged():
    # section_scope=all（默认）时给 section_ids → 依赖未满足却给值
    with pytest.raises(IntakeValidationError, match="section_ids"):
        validate_answers(_valid_answers(section_ids=["01"]))


def test_illegal_source_type_rejected():
    with pytest.raises(IntakeValidationError, match="source_types"):
        validate_answers(_valid_answers(source_types=["client_provided", "confidential"]))


def test_unknown_field_is_flagged_not_silently_dropped():
    with pytest.raises(IntakeValidationError, match="typo_field"):
        validate_answers(_valid_answers(typo_field="oops"))


def test_all_problems_collected_not_fail_fast():
    answers = _valid_answers(doc_type="nope", section_scope="bogus")
    del answers["project_name"]
    with pytest.raises(IntakeValidationError) as exc:
        validate_answers(answers)
    assert len(exc.value.problems) >= 3


def test_valid_minimal_answers_accepted():
    spec = validate_answers(_valid_answers())
    assert spec.doc_type == "proposal-document"
    assert spec.section_scope == SectionScope.ALL.value
    assert spec.section_ids == ()
    assert spec.industry_domain is None


# ── 文件模式 intake 端到端 ───────────────────────────────────────────────
def test_file_mode_intake_answers_to_spec_to_disk(tmp_path):
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_valid_answers(
        section_scope="custom", section_ids=["03", "04"], budget="1200 万",
    ), ensure_ascii=False), encoding="utf-8")
    spec = run_from_file(str(answers_path))
    out = save_spec(spec, tmp_path / "spec.json")
    reloaded = load_project_spec(str(out))
    assert reloaded.section_ids == ("03", "04")
    assert reloaded.budget == "1200 万"


def test_file_mode_rejects_non_object(tmp_path):
    answers_path = tmp_path / "answers.json"
    answers_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(IntakeValidationError, match="JSON 对象"):
        run_from_file(str(answers_path))


def test_file_mode_comma_string_for_lists(tmp_path):
    """文件模式允许 list 字段用逗号字符串（交互模式产物的兼容形态）。"""

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(_valid_answers(
        source_types="client_provided,public",
    ), ensure_ascii=False), encoding="utf-8")
    spec = run_from_file(str(answers_path))
    assert spec.source_types == ("client_provided", "public")


# ── 交互模式（注入 reader/writer）─────────────────────────────────────────
def test_interactive_collects_and_skips_inactive_questions():
    scripted = {
        "要生成什么类型的文档？": "proposal-document",
        "项目名称？": "某项目",
        "甲方 / 采购人全称？": "某局",
        "行业域": "",                       # 可留空 → 跳过
        "关联哪个 L1 材料库项目": "acme-2026",
        "章节范围": "all",                  # → section_ids 不该被问
        "是否按材料来源过滤": "",
        "项目编号": "",
        "项目预算": "",
        "项目工期": "",
        "项目性质": "",
    }

    def reader(prompt: str) -> str:
        for key, value in scripted.items():
            if key in prompt:
                return value
        raise AssertionError(f"unexpected prompt: {prompt}")

    answers = collect_interactive(reader=reader, writer=lambda *_: None)
    assert "section_ids" not in answers          # section_scope=all → 依赖未激活，未问
    assert answers["doc_type"] == "proposal-document"
    spec = validate_answers(answers)
    assert spec.project_name == "某项目"


def test_interactive_custom_scope_asks_section_ids():
    answered = {}

    def reader(prompt: str) -> str:
        mapping = {
            "要生成什么类型的文档？": "proposal-document",
            "项目名称？": "某项目",
            "甲方 / 采购人全称？": "某局",
            "行业域": "",
            "关联哪个 L1 材料库项目": "acme-2026",
            "章节范围": "custom",
            "指定哪些章节": "01,02",
            "是否按材料来源过滤": "",
            "项目编号": "",
            "项目预算": "",
            "项目工期": "",
            "项目性质": "",
        }
        for key, value in mapping.items():
            if key in prompt:
                answered[key] = value
                return value
        raise AssertionError(f"unexpected prompt: {prompt}")

    answers = collect_interactive(reader=reader, writer=lambda *_: None)
    assert answers["section_ids"] == ["01", "02"]   # custom → 被问到且解析成 list
    assert any("指定哪些章节" in k for k in answered)


# ── 问题清单结构自洽 ─────────────────────────────────────────────────────
def test_question_dependencies_reference_existing_questions():
    ids = {q.id for q in QUESTIONS}
    for q in QUESTIONS:
        if q.depends_on is not None:
            assert q.depends_on[0] in ids
