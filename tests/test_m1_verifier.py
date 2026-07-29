"""M1 Step 5 deterministic EvidenceSpan-only Verifier tests。"""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from evidence import (
    EvidenceSpan,
    NormalizedDocument,
    SLOT_VERIFICATION_RULES,
    SlotRule,
    SlotStatus,
    DeterministicSlotVerifier,
    VectorCandidate,
    VerificationReasonCode,
    solution_material_completeness_v1,
    validate_rule_registry,
)


FILLED_TEXT = {
    "background_and_basis": (
        "《智慧城市建设规划》文件要求本项目建设统一数据平台，"
        "用于支撑当前项目业务。"
    ),
    "current_state_and_needs": (
        "目前各部门业务数据分散，现状是审批流程难以协同，"
        "产生具体业务需求。"
    ),
    "scope_and_subjects": (
        "建设对象为市级交通管理部门，业务范围为全市交通监管业务，"
        "范围包括市级与区县系统。"
    ),
    "objectives": (
        "建设目标是实现审批时长降低，针对当前业务问题形成可检查的目标状态。"
    ),
    "solution_content_and_capabilities": (
        "建设统一数据治理模块和告警功能，用于解决监管业务需求并支撑建设目标。"
    ),
    "implementation_conditions_and_boundaries": (
        "部署于政务云环境，必须通过现有系统接口对接，网络条件为专网。"
    ),
    "reference_cases_and_proof": (
        "客户为海城市交通局，项目实施了统一数据平台，平台已上线并通过验收。"
    ),
    "outcomes_and_metrics": (
        "实际结果：交通事件平均处置时长由30分钟缩短至20分钟，"
        "数据来源为验收报告。"
    ),
}

WEAK_TEXT = {
    "background_and_basis": "响应国家战略。",
    "current_state_and_needs": "存在短板，亟须提升。",
    "scope_and_subjects": "面向政府客户，覆盖相关业务，服务各类用户。",
    "objectives": "全面提升，打造一流平台。",
    "solution_content_and_capabilities": "人工智能、大数据、云计算。",
    "implementation_conditions_and_boundaries": (
        "可灵活部署，支持多种环境，具备良好扩展性。"
    ),
    "reference_cases_and_proof": "客户为海城市交通局。",
    "outcomes_and_metrics": "系统可以降本增效。",
}


def _span(tmp_path, text, *, document_id):
    path = tmp_path / f"{document_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    document = NormalizedDocument.from_path(
        path,
        document_id=document_id,
        source_path=f"materials/{document_id}.txt",
    )
    return EvidenceSpan.from_document(
        document,
        0,
        len(document.text),
        discovery_sources=("ripgrep",),
    )


def _filled_span(tmp_path, slot_id, *, suffix="filled"):
    return _span(
        tmp_path,
        FILLED_TEXT[slot_id],
        document_id=f"{slot_id}-{suffix}",
    )


def _verify(slot_id, spans):
    return DeterministicSlotVerifier().verify_slot(
        contract=solution_material_completeness_v1,
        slot_id=slot_id,
        evidence_spans=spans,
    )


def test_no_evidence_is_missing_with_stable_reasons_and_trace():
    result = _verify("background_and_basis", [])
    assert result.status is SlotStatus.MISSING
    assert result.reason_codes == (
        VerificationReasonCode.NO_EVIDENCE,
        VerificationReasonCode.INSUFFICIENT_EVIDENCE_COUNT,
        VerificationReasonCode.MISSING_REQUIRED_FIELD,
    )
    assert result.evidence_ids == ()
    assert result.trace["input_evidence_count"] == 0
    assert result.trace["unique_evidence_count"] == 0


@pytest.mark.parametrize("slot_id", solution_material_completeness_v1.slot_ids)
def test_each_fixed_slot_has_a_conservative_filled_example(tmp_path, slot_id):
    result = _verify(slot_id, [_filled_span(tmp_path, slot_id)])
    assert result.status is SlotStatus.FILLED
    assert result.reason_codes == ()
    assert tuple(item.field_id for item in result.field_coverage) == (
        solution_material_completeness_v1.get_slot(slot_id).required_fields
    )
    assert all(item.covered for item in result.field_coverage)


@pytest.mark.parametrize("slot_id", solution_material_completeness_v1.slot_ids)
def test_each_fixed_slot_has_a_related_but_insufficient_example(tmp_path, slot_id):
    span = _span(tmp_path, WEAK_TEXT[slot_id], document_id=f"{slot_id}-weak")
    result = _verify(slot_id, [span])
    assert result.status is SlotStatus.WEAK
    assert result.status is not SlotStatus.FILLED


def test_case_customer_name_only_is_weak_but_full_case_is_filled(tmp_path):
    weak = _verify(
        "reference_cases_and_proof",
        [_span(tmp_path, "客户为海城市交通局。", document_id="case-customer")],
    )
    filled = _verify(
        "reference_cases_and_proof",
        [_filled_span(tmp_path, "reference_cases_and_proof")],
    )
    assert weak.status is SlotStatus.WEAK
    assert [item.covered for item in weak.field_coverage] == [True, False, False]
    assert filled.status is SlotStatus.FILLED


def test_outcome_slogan_is_weak_and_specific_actual_metric_is_filled(tmp_path):
    weak = _verify(
        "outcomes_and_metrics",
        [_span(tmp_path, "系统可以降本增效。", document_id="outcome-slogan")],
    )
    filled = _verify(
        "outcomes_and_metrics",
        [_filled_span(tmp_path, "outcomes_and_metrics")],
    )
    assert weak.status is SlotStatus.WEAK
    assert filled.status is SlotStatus.FILLED


def test_unrelated_evidence_with_no_field_coverage_is_missing(tmp_path):
    result = _verify(
        "implementation_conditions_and_boundaries",
        [_span(tmp_path, "这是企业历史介绍。", document_id="irrelevant")],
    )
    assert result.status is SlotStatus.MISSING
    assert VerificationReasonCode.MISSING_REQUIRED_FIELD in result.reason_codes


def test_only_existing_four_slot_status_values_are_emitted(tmp_path):
    results = [
        _verify("background_and_basis", []),
        _verify(
            "background_and_basis",
            [_span(tmp_path, "响应国家战略。", document_id="status-weak")],
        ),
        _verify(
            "background_and_basis",
            [_filled_span(tmp_path, "background_and_basis", suffix="status-filled")],
        ),
        _verify(
            "implementation_conditions_and_boundaries",
            [
                _span(tmp_path, "平台A支持离线部署。", document_id="status-pos"),
                _span(tmp_path, "平台A不支持离线部署。", document_id="status-neg"),
            ],
        ),
    ]
    assert {item.status for item in results} == set(SlotStatus)


def test_same_input_is_stable_and_does_not_mutate_input(tmp_path):
    span = _filled_span(tmp_path, "objectives")
    evidence = [span]
    before = tuple(evidence)
    first = _verify("objectives", evidence)
    second = _verify("objectives", evidence)
    assert first == second
    assert tuple(evidence) == before
    assert evidence[0] is span


def test_required_field_coverage_exactly_matches_contract_order(tmp_path):
    for slot in solution_material_completeness_v1.slots:
        result = _verify(slot.slot_id, [_filled_span(tmp_path, slot.slot_id)])
        assert tuple(item.field_id for item in result.field_coverage) == slot.required_fields


def test_registry_matches_frozen_contract_at_import():
    validate_rule_registry(
        solution_material_completeness_v1,
        SLOT_VERIFICATION_RULES,
    )
    assert tuple(SLOT_VERIFICATION_RULES) == solution_material_completeness_v1.slot_ids


def test_unknown_field_rule_and_contract_registry_mismatch_fail():
    registry = dict(SLOT_VERIFICATION_RULES)
    original = registry["background_and_basis"]
    registry["background_and_basis"] = SlotRule(
        original.slot_id,
        (replace(original.field_rules[0], field_id="unknown_field"),)
        + original.field_rules[1:],
    )
    with pytest.raises(ValueError, match="field rule mismatch"):
        validate_rule_registry(solution_material_completeness_v1, registry)

    missing_slot = dict(SLOT_VERIFICATION_RULES)
    missing_slot.pop("objectives")
    with pytest.raises(ValueError, match="slot mismatch"):
        DeterministicSlotVerifier(missing_slot).verify_slot(
            contract=solution_material_completeness_v1,
            slot_id="background_and_basis",
            evidence_spans=(),
        )


def test_evidence_id_is_stably_deduplicated_before_minimum_count(tmp_path):
    original_slot = solution_material_completeness_v1.get_slot("objectives")
    stricter_slot = replace(original_slot, minimum_evidence=2)
    stricter_contract = replace(
        solution_material_completeness_v1,
        slots=tuple(
            stricter_slot if slot.slot_id == "objectives" else slot
            for slot in solution_material_completeness_v1.slots
        ),
    )
    span = _filled_span(tmp_path, "objectives")
    result = DeterministicSlotVerifier().verify_slot(
        contract=stricter_contract,
        slot_id="objectives",
        evidence_spans=[span, span],
    )
    assert result.status is SlotStatus.WEAK
    assert result.evidence_ids == (span.evidence_id,)
    assert result.trace["input_evidence_count"] == 2
    assert result.trace["unique_evidence_count"] == 1
    assert VerificationReasonCode.INSUFFICIENT_EVIDENCE_COUNT in result.reason_codes


def test_vector_candidate_cannot_substitute_for_evidence_span():
    candidate = VectorCandidate(
        slot_id="background_and_basis",
        query="q",
        document_id="doc",
        chunk_id="chunk",
        text="候选",
        start_char=0,
        end_char=2,
        start_line=1,
        end_line=1,
        rank=1,
        distance=0.1,
        discovery_source="vector",
    )
    with pytest.raises(ValueError, match="must be EvidenceSpan"):
        _verify("background_and_basis", [candidate])
    with pytest.raises(ValueError, match="must be EvidenceSpan"):
        _verify("background_and_basis", ["未经定位的普通字符串"])


def test_pure_promotional_language_is_weak_with_reason_code(tmp_path):
    span = _span(
        tmp_path,
        "行业领先，全面赋能，高效协同。",
        document_id="promotion-only",
    )
    result = _verify("objectives", [span])
    assert result.status is SlotStatus.WEAK
    assert VerificationReasonCode.PROMOTIONAL_ONLY in result.reason_codes
    assert result.trace["promotional_phrases"] == (
        "行业领先",
        "全面赋能",
        "高效协同",
    )


def test_promotional_phrase_does_not_cancel_complete_facts(tmp_path):
    span = _span(
        tmp_path,
        "行业领先。建设目标是实现审批时长降低，针对当前业务问题形成可检查目标。",
        document_id="promotion-with-facts",
    )
    result = _verify("objectives", [span])
    assert result.status is SlotStatus.FILLED
    assert VerificationReasonCode.PROMOTIONAL_ONLY not in result.reason_codes


def test_unsupported_percentage_has_reason_code(tmp_path):
    span = _span(tmp_path, "效率提升30%。", document_id="unsupported-percent")
    result = _verify("outcomes_and_metrics", [span])
    assert result.status is SlotStatus.WEAK
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        in result.reason_codes
    )
    assert result.trace["quantitative_claims"][0]["verified"] is False


def test_expected_number_is_not_misrepresented_as_actual_result(tmp_path):
    span = _span(
        tmp_path,
        "预计交通事件处置时长缩短30分钟，属于预期目标。",
        document_id="expected-metric",
    )
    result = _verify("outcomes_and_metrics", [span])
    assert result.status is SlotStatus.FILLED
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )
    assert "预计" in result.trace["quantitative_claims"][0]["support_markers"]


def test_source_and_caliber_make_quantitative_claim_acceptable(tmp_path):
    span = _span(
        tmp_path,
        "实际业务办理时长降低30%，统计口径为全年平均，来源为验收报告。",
        document_id="sourced-metric",
    )
    result = _verify("outcomes_and_metrics", [span])
    assert result.status is SlotStatus.FILLED
    assert all(item["verified"] for item in result.trace["quantitative_claims"])


@pytest.mark.parametrize("year", (2024, 2025))
def test_ordinary_calendar_year_is_not_a_quantitative_outcome(tmp_path, year):
    span = _span(
        tmp_path,
        f"{year}年《智慧城市规划》文件要求本项目建设数据平台。",
        document_id=f"calendar-year-{year}",
    )
    result = _verify("background_and_basis", [span])
    assert result.status is SlotStatus.FILLED
    assert result.trace["quantitative_claims"] == ()


@pytest.mark.parametrize(
    ("positive", "negative"),
    (
        ("平台A支持离线部署。", "平台A不支持离线部署。"),
        ("海城项目已上线。", "海城项目未上线。"),
    ),
)
def test_opposite_assertions_on_same_topic_are_conflicting(
    tmp_path,
    positive,
    negative,
):
    first = _span(tmp_path, positive, document_id="conflict-positive")
    second = _span(tmp_path, negative, document_id="conflict-negative")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is SlotStatus.CONFLICTING
    assert VerificationReasonCode.CONFLICTING_FACTS in result.reason_codes
    assert result.conflicts[0].conflict_type == "opposite_assertion"
    assert set(result.conflicts[0].evidence_ids) == {
        first.evidence_id,
        second.evidence_id,
    }


def test_same_key_different_scalar_values_are_conflicting(tmp_path):
    first = _span(tmp_path, "项目周期：12个月。", document_id="period-12")
    second = _span(tmp_path, "项目周期：18个月。", document_id="period-18")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is SlotStatus.CONFLICTING
    conflict = result.conflicts[0]
    assert conflict.conflict_type == "scalar_mismatch"
    assert {fact.value for fact in conflict.facts} == {"12个月", "18个月"}
    assert set(conflict.evidence_ids) == {first.evidence_id, second.evidence_id}


@pytest.mark.parametrize(
    ("first_text", "second_text", "expected_values"),
    (
        ("并发用户数为 500。", "并发用户数为 1000。", {"500", "1000"}),
        ("预算为 300 万元。", "预算为 500万元。", {"300 万元", "500万元"}),
    ),
)
def test_limited_concurrency_and_budget_scalar_formats_are_supported(
    tmp_path,
    first_text,
    second_text,
    expected_values,
):
    first = _span(tmp_path, first_text, document_id="scalar-first")
    second = _span(tmp_path, second_text, document_id="scalar-second")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is SlotStatus.CONFLICTING
    conflict = result.conflicts[0]
    assert conflict.conflict_type == "scalar_mismatch"
    assert {fact.value for fact in conflict.facts} == expected_values


def test_scalar_spacing_and_decimal_format_do_not_create_false_conflict(tmp_path):
    first = _span(tmp_path, "项目周期：12个月。", document_id="period-compact")
    second = _span(tmp_path, "项目周期是12.0 个月。", document_id="period-spaced")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is not SlotStatus.CONFLICTING
    assert result.conflicts == ()


def test_same_scalar_value_with_only_unit_spacing_is_not_conflicting(tmp_path):
    first = _span(tmp_path, "项目周期：12个月。", document_id="period-no-space")
    second = _span(tmp_path, "项目周期：12 个月。", document_id="period-one-space")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is not SlotStatus.CONFLICTING
    assert result.conflicts == ()


def test_different_topics_do_not_create_opposite_assertion_conflict(tmp_path):
    first = _span(tmp_path, "平台A支持离线部署。", document_id="topic-a")
    second = _span(tmp_path, "平台B不支持离线部署。", document_id="topic-b")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is not SlotStatus.CONFLICTING
    assert result.conflicts == ()


def test_positive_and_negative_phrases_for_different_deployments_do_not_conflict(tmp_path):
    span = _span(
        tmp_path,
        "系统不支持离线部署，但支持云端部署。",
        document_id="different-deployments",
    )
    result = _verify("implementation_conditions_and_boundaries", [span])
    assert result.status is not SlotStatus.CONFLICTING
    assert result.conflicts == ()


def test_online_and_not_online_for_different_systems_do_not_conflict(tmp_path):
    span = _span(
        tmp_path,
        "A 系统已上线，B 系统未上线。",
        document_id="different-systems",
    )
    result = _verify("reference_cases_and_proof", [span])
    assert result.status is not SlotStatus.CONFLICTING
    assert result.conflicts == ()


def test_negative_phrase_is_not_split_into_positive_and_negative_facts(tmp_path):
    span = _span(tmp_path, "平台A不支持离线部署。", document_id="negative-only")
    result = _verify("implementation_conditions_and_boundaries", [span])
    assert result.status is not SlotStatus.CONFLICTING
    facts = result.trace["detected_facts"]
    support = [item for item in facts if item["predicate"] == "support"]
    assert len(support) == 1
    assert support[0]["normalized_value"] == "negative"


def test_opposite_words_for_different_objects_do_not_force_conflict(tmp_path):
    first = _span(tmp_path, "平台支持模块A。", document_id="module-a")
    second = _span(tmp_path, "平台不支持模块B。", document_id="module-b")
    result = _verify("implementation_conditions_and_boundaries", [first, second])
    assert result.status is not SlotStatus.CONFLICTING


def test_conflicting_status_precedes_missing_required_fields(tmp_path):
    first = _span(tmp_path, "平台A支持离线部署。", document_id="missing-fields-pos")
    second = _span(tmp_path, "平台A不支持离线部署。", document_id="missing-fields-neg")
    result = _verify("objectives", [first, second])
    assert result.status is SlotStatus.CONFLICTING
    assert all(not item.covered for item in result.field_coverage)
    assert VerificationReasonCode.MISSING_REQUIRED_FIELD in result.reason_codes
    assert VerificationReasonCode.CONFLICTING_FACTS in result.reason_codes


def _all_filled_evidence(tmp_path):
    return {
        slot_id: [_filled_span(tmp_path, slot_id, suffix="contract")]
        for slot_id in solution_material_completeness_v1.slot_ids
    }


def test_contract_all_required_slots_filled_is_complete(tmp_path):
    result = DeterministicSlotVerifier().verify_contract(
        contract=solution_material_completeness_v1,
        evidence_by_slot=_all_filled_evidence(tmp_path),
    )
    assert result.status == "complete"
    assert tuple(item.slot_id for item in result.slots) == (
        solution_material_completeness_v1.slot_ids
    )


def test_contract_weak_or_missing_required_slot_is_incomplete(tmp_path):
    weak_evidence = _all_filled_evidence(tmp_path / "weak")
    weak_evidence["objectives"] = [
        _span(tmp_path / "weak", "全面提升。", document_id="weak-objective")
    ]
    weak = DeterministicSlotVerifier().verify_contract(
        contract=solution_material_completeness_v1,
        evidence_by_slot=weak_evidence,
    )
    missing_evidence = _all_filled_evidence(tmp_path / "missing")
    missing_evidence.pop("objectives")
    missing = DeterministicSlotVerifier().verify_contract(
        contract=solution_material_completeness_v1,
        evidence_by_slot=missing_evidence,
    )
    assert weak.status == "incomplete"
    assert missing.status == "incomplete"
    missing_slot = next(item for item in missing.slots if item.slot_id == "objectives")
    assert missing_slot.status is SlotStatus.MISSING


def test_contract_required_conflict_has_highest_priority(tmp_path):
    evidence = _all_filled_evidence(tmp_path)
    evidence["implementation_conditions_and_boundaries"] = [
        _span(tmp_path, "平台A支持离线部署。", document_id="contract-pos"),
        _span(tmp_path, "平台A不支持离线部署。", document_id="contract-neg"),
    ]
    result = DeterministicSlotVerifier().verify_contract(
        contract=solution_material_completeness_v1,
        evidence_by_slot=evidence,
    )
    assert result.status == "conflicted"


def test_contract_unknown_extra_slot_fails(tmp_path):
    evidence = _all_filled_evidence(tmp_path)
    evidence["unknown_slot"] = ()
    with pytest.raises(ValueError, match="unknown slot keys"):
        DeterministicSlotVerifier().verify_contract(
            contract=solution_material_completeness_v1,
            evidence_by_slot=evidence,
        )


def test_unknown_slot_fails_clearly():
    with pytest.raises(KeyError, match="unknown slot_id"):
        _verify("not-a-slot", [])


def test_trace_is_json_serializable_and_immutable(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [_filled_span(tmp_path, "outcomes_and_metrics")],
    )
    encoded = json.dumps(result.trace, ensure_ascii=False)
    assert "detected_facts" in encoded
    with pytest.raises(TypeError, match="immutable"):
        result.trace["status"] = "changed"


def test_objectives_gate_rejects_generic_product_promotion(tmp_path):
    spans = [
        _span(
            tmp_path,
            "本产品全面赋能城市数字化转型，打造行业领先的一流平台。",
            document_id="objective-product-promotion",
        ),
        _span(
            tmp_path,
            "实现能力提升，面向业务系统提供服务。",
            document_id="objective-generic-coverage",
        ),
    ]
    result = _verify("objectives", spans)
    assert result.status is SlotStatus.WEAK
    assert VerificationReasonCode.MISSING_REQUIRED_FIELD in result.reason_codes
    assert result.trace["semantic_gate"]["evaluated"] is True
    assert result.trace["semantic_gate"]["passed"] is False
    assert result.trace["semantic_gate"]["failed_aspects"] == (
        "project_or_construction_context",
    )


def test_objectives_gate_rejects_capability_description_as_target(tmp_path):
    spans = [
        _span(
            tmp_path,
            "平台具备强大的数据治理和三维可视化能力。",
            document_id="objective-capability",
        ),
        _span(
            tmp_path,
            "实现能力提升，面向业务系统提供服务。",
            document_id="objective-capability-coverage",
        ),
    ]
    result = _verify("objectives", spans)
    assert result.status is SlotStatus.WEAK
    assert result.trace["semantic_gate"]["passed"] is False


def test_objectives_gate_keeps_specific_project_target_filled(tmp_path):
    result = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "本项目面向省级和市级住建业务，建设统一数据平台，"
                "实现跨层级数据汇聚、服务共享与应用协同。",
                document_id="objective-specific-project",
            )
        ],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


def test_objectives_gate_combines_signals_across_spans(tmp_path):
    spans = [
        _span(
            tmp_path,
            "本项目面向省级和市级住建部门。",
            document_id="objective-context",
        ),
        _span(
            tmp_path,
            "建设目标是形成统一的数据底板。",
            document_id="objective-expression",
        ),
        _span(
            tmp_path,
            "面向住建业务实现跨层级数据共享和业务协同。",
            document_id="objective-specific-content",
        ),
    ]
    result = _verify("objectives", spans)
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


def test_objectives_gate_does_not_reject_promotion_with_complete_project_facts(tmp_path):
    result = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "本项目将打造行业领先的平台，面向住建部门解决数据分散问题，"
                "实现省级和市级数据统一汇聚。",
                document_id="objective-promotion-and-facts",
            )
        ],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


@pytest.mark.parametrize(
    "text",
    (
        "平台支持多源数据接入，可应用于城市治理业务、资源管理和应急管理。",
        "系统提供数据治理、模型管理、服务发布和可视化能力。",
        "项目平台建设三大中心，配置五台服务器和80TB存储。",
    ),
)
def test_outcomes_gate_rejects_capability_scenario_and_configuration_text(
    tmp_path,
    text,
):
    result = _verify(
        "outcomes_and_metrics",
        [_span(tmp_path, text, document_id="outcome-wrong-role")],
    )
    assert result.status is SlotStatus.WEAK
    assert result.status is not SlotStatus.FILLED


def test_outcomes_gate_rejects_cross_span_field_hits_without_outcome_role(tmp_path):
    spans = [
        _span(
            tmp_path,
            "平台支持多源数据接入，可应用于城市治理、自然资源和应急管理。",
            document_id="outcome-capability-scenario",
        ),
        _span(
            tmp_path,
            "平台建设目标和业务规划计划指标。",
            document_id="outcome-unrelated-nature",
        ),
        _span(
            tmp_path,
            "实现系统配置。",
            document_id="outcome-unrelated-field-hit",
        ),
    ]
    result = _verify("outcomes_and_metrics", spans)
    assert result.trace["semantic_gate"]["preliminary_status"] == "filled"
    assert result.trace["semantic_gate"]["evaluated"] is True
    assert result.trace["semantic_gate"]["passed"] is False
    assert result.trace["semantic_gate"]["failed_aspects"] == (
        "explicit_outcome_or_metric_role",
    )
    assert result.status is SlotStatus.WEAK


def test_outcomes_gate_keeps_expected_evaluation_result_filled(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "项目建成后，预计实现住建数据统一汇聚，"
                "以接入系统数量和共享服务调用成功率作为评价指标。",
                document_id="outcome-expected-evaluation",
            )
        ],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


def test_outcomes_gate_keeps_sourced_actual_change_filled(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "平台上线运行后，业务材料平均查询时间由10分钟缩短至2分钟，"
                "统计依据为试运行阶段100次业务查询记录。",
                document_id="outcome-actual-evaluation",
            )
        ],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


def test_outcomes_gate_combines_role_nature_and_subject_across_spans(tmp_path):
    spans = [
        _span(
            tmp_path,
            "项目建成后预期提升跨层级协同效率。",
            document_id="outcome-role-nature",
        ),
        _span(
            tmp_path,
            "评价对象为省级与市级住建业务流程。",
            document_id="outcome-subject",
        ),
        _span(
            tmp_path,
            "以业务协同事项覆盖率作为考核指标。",
            document_id="outcome-evaluation",
        ),
    ]
    result = _verify("outcomes_and_metrics", spans)
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"]["passed"] is True


def test_semantic_gate_does_not_override_conflict_or_missing_priority(tmp_path):
    conflicting = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "平台A支持离线部署。",
                document_id="semantic-priority-positive",
            ),
            _span(
                tmp_path,
                "平台A不支持离线部署。",
                document_id="semantic-priority-negative",
            ),
        ],
    )
    missing = _verify("outcomes_and_metrics", [])
    assert conflicting.status is SlotStatus.CONFLICTING
    assert conflicting.trace["semantic_gate"]["evaluated"] is False
    assert missing.status is SlotStatus.MISSING
    assert missing.trace["semantic_gate"]["evaluated"] is False


def test_semantic_gate_only_intercepts_preliminary_filled_result(tmp_path):
    partial = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "建设目标是形成统一平台。",
                document_id="semantic-partial",
            )
        ],
    )
    assert partial.status is SlotStatus.WEAK
    assert partial.trace["semantic_gate"]["evaluated"] is False
    assert partial.trace["semantic_gate"]["preliminary_status"] == "weak"


def test_semantic_gate_result_is_order_independent_and_json_serializable(tmp_path):
    spans = [
        _span(
            tmp_path,
            "本项目面向省级和市级住建部门。",
            document_id="semantic-order-context",
        ),
        _span(
            tmp_path,
            "建设目标是形成统一平台，面向业务系统实现跨层级协同。",
            document_id="semantic-order-target",
        ),
    ]
    first = _verify("objectives", spans)
    second = _verify("objectives", list(reversed(spans)))
    assert first.status is second.status is SlotStatus.FILLED
    assert first.trace["semantic_gate"] == second.trace["semantic_gate"]
    json.dumps(first.trace["semantic_gate"], ensure_ascii=False)


@pytest.mark.parametrize(
    "slot_id",
    (
        "background_and_basis",
        "current_state_and_needs",
        "scope_and_subjects",
        "solution_content_and_capabilities",
        "implementation_conditions_and_boundaries",
        "reference_cases_and_proof",
    ),
)
def test_semantic_gate_does_not_apply_to_other_slots(tmp_path, slot_id):
    result = _verify(slot_id, [_filled_span(tmp_path, slot_id, suffix="p0-scope")])
    assert result.status is SlotStatus.FILLED
    assert result.trace["semantic_gate"] == {
        "applicable": False,
        "evaluated": False,
        "passed": None,
        "failed_aspects": (),
        "reason": "Slot-specific semantic gate does not apply.",
        "preliminary_status": "filled",
    }


def test_p2_outcome_percentage_without_measurement_support_stays_weak(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "平台上线后，业务效率提升30%。",
                document_id="p2-unsupported-outcome",
            )
        ],
    )
    claim = result.trace["quantitative_claims"][0]
    assert claim["role"] == "outcome_claim"
    assert claim["requires_outcome_support"] is True
    assert claim["verified"] is False
    assert result.status is SlotStatus.WEAK
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        in result.reason_codes
    )


def test_p2_sourced_actual_outcome_is_verified(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "平台上线运行后，业务材料查询时间由10分钟缩短至2分钟，"
                "统计依据为试运行阶段100次业务查询记录。",
                document_id="p2-supported-outcome",
            )
        ],
    )
    claims = result.trace["quantitative_claims"]
    assert claims
    assert all(claim["role"] == "outcome_claim" for claim in claims)
    assert all(claim["verified"] for claim in claims)
    assert result.status is SlotStatus.FILLED
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_expected_percentage_remains_an_outcome_claim(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "项目建成后，预计业务办理效率提升30%。",
                document_id="p2-expected-outcome",
            )
        ],
    )
    claim = result.trace["quantitative_claims"][0]
    assert claim["role"] == "outcome_claim"
    assert claim["verified"] is False
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        in result.reason_codes
    )


@pytest.mark.parametrize(
    "text",
    (
        "配置5台服务器。",
        "存储容量为80TB。",
        "部署3个服务节点。",
        "配置32核CPU和256GB内存。",
        "系统设计支持100个并发用户。",
        "建设3个数据中心。",
        "三维模型容量为1.5GB。",
        "数据分类共19个大类。",
    ),
)
def test_p2_configuration_values_do_not_require_outcome_support(tmp_path, text):
    result = _verify(
        "implementation_conditions_and_boundaries",
        [_span(tmp_path, text, document_id="p2-configuration")],
    )
    claims = result.trace["quantitative_claims"]
    assert claims
    assert all(claim["role"] == "configuration" for claim in claims)
    assert all(claim["requires_outcome_support"] is False for claim in claims)
    assert all(claim["verified"] for claim in claims)
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


@pytest.mark.parametrize(
    "text",
    (
        "建设周期为12个月。",
        "项目预算为500万元。",
        "计划接入40个业务系统。",
        "项目分3个阶段实施。",
        "平台每月推送1次更新数据。",
        "标准编制工作量为4人月。",
        "规划采用5年预测值。",
    ),
)
def test_p2_project_plan_values_do_not_require_outcome_support(tmp_path, text):
    result = _verify(
        "objectives",
        [_span(tmp_path, text, document_id="p2-project-plan")],
    )
    claims = result.trace["quantitative_claims"]
    assert claims
    assert all(claim["role"] == "project_plan" for claim in claims)
    assert all(claim["verified"] for claim in claims)
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_planned_completion_year_is_not_an_outcome_claim(tmp_path):
    result = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "本项目拟于2027年完成建设。",
                document_id="p2-planned-year",
            )
        ],
    )
    assert result.trace["quantitative_claims"] == ()
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


@pytest.mark.parametrize(
    "text",
    (
        "项目于2026年启动。",
        "版本为V2.1。",
        "当前进入第3阶段。",
        "文号为2026-01。",
    ),
)
def test_p2_temporal_or_identifier_text_does_not_fill_outcomes(tmp_path, text):
    result = _verify(
        "outcomes_and_metrics",
        [_span(tmp_path, text, document_id="p2-temporal-identifier")],
    )
    assert result.status is not SlotStatus.FILLED
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_ocr_spaced_date_is_temporal_context(tmp_path):
    result = _verify(
        "solution_content_and_capabilities",
        [
            _span(
                tmp_path,
                "2 0 2 6 年 7 月发布产品说明。",
                document_id="p2-spaced-date",
            )
        ],
    )
    claims = result.trace["quantitative_claims"]
    assert claims
    assert all(claim["role"] == "temporal_or_identifier" for claim in claims)
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


@pytest.mark.parametrize(
    ("slot_id", "text", "expected_status", "expected_role"),
    (
        (
            "solution_content_and_capabilities",
            "建设数据治理模块并配置5台服务器，用于支撑监管业务需求。",
            SlotStatus.FILLED,
            "configuration",
        ),
        (
            "implementation_conditions_and_boundaries",
            "部署环境配置5台服务器，必须接入现有业务系统。",
            SlotStatus.FILLED,
            "configuration",
        ),
        (
            "scope_and_subjects",
            "建设对象为市级管理部门，业务范围包括3个业务系统，"
            "范围覆盖市级和区县系统。",
            SlotStatus.FILLED,
            "configuration",
        ),
        (
            "objectives",
            "本项目面向市级管理部门，计划建设3个数据中心，"
            "形成统一数据汇聚能力，面向业务系统解决数据分散问题。",
            SlotStatus.FILLED,
            "project_plan",
        ),
        (
            "outcomes_and_metrics",
            "项目平台配置5台服务器和80TB存储。",
            SlotStatus.WEAK,
            "configuration",
        ),
    ),
)
def test_p2_roles_respect_slot_required_fields_and_semantic_gates(
    tmp_path,
    slot_id,
    text,
    expected_status,
    expected_role,
):
    result = _verify(slot_id, [_span(tmp_path, text, document_id=f"p2-{slot_id}")])
    assert result.status is expected_status
    assert {claim["role"] for claim in result.trace["quantitative_claims"]} == {
        expected_role
    }
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_mixed_configuration_and_unsupported_outcome_keeps_strict_reason(tmp_path):
    result = _verify(
        "solution_content_and_capabilities",
        [
            _span(
                tmp_path,
                "本项目建设数据处理模块，用于支撑业务需求；"
                "项目配置5台服务器，建成后预计业务查询效率提升30%。",
                document_id="p2-mixed-roles",
            )
        ],
    )
    roles = {claim["value"]: claim["role"] for claim in result.trace["quantitative_claims"]}
    assert roles == {"5台": "configuration", "30%": "outcome_claim"}
    assert result.status is SlotStatus.WEAK
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        in result.reason_codes
    )


def test_p2_plan_count_is_not_reclassified_by_adjacent_evaluation_clause(tmp_path):
    result = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "本项目计划接入40个业务系统，并以系统接入覆盖率作为验收指标。",
                document_id="p2-plan-and-indicator",
            )
        ],
    )
    assert result.trace["quantitative_claims"][0]["role"] == "project_plan"
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_multi_span_roles_and_result_are_order_independent(tmp_path):
    spans = [
        _span(
            tmp_path,
            "本项目面向市级管理部门。",
            document_id="p2-multi-project",
        ),
        _span(
            tmp_path,
            "配置5台服务器。",
            document_id="p2-multi-configuration",
        ),
        _span(
            tmp_path,
            "项目周期为12个月。",
            document_id="p2-multi-plan",
        ),
        _span(
            tmp_path,
            "建设目标是形成统一数据平台，面向业务系统解决数据分散问题。",
            document_id="p2-multi-target",
        ),
    ]
    first = _verify("objectives", spans)
    second = _verify("objectives", list(reversed(spans)))
    assert first.status is second.status is SlotStatus.FILLED
    assert first.trace["quantitative_claims"] == second.trace["quantitative_claims"]
    assert {claim["role"] for claim in first.trace["quantitative_claims"]} == {
        "configuration",
        "project_plan",
    }
    json.dumps(first.trace["quantitative_claims"], ensure_ascii=False)


def test_p2_unrelated_configuration_does_not_globally_downgrade_filled_slot(tmp_path):
    result = _verify(
        "background_and_basis",
        [
            _filled_span(tmp_path, "background_and_basis", suffix="p2-unrelated"),
            _span(
                tmp_path,
                "配置5台服务器。",
                document_id="p2-unrelated-configuration",
            ),
        ],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["quantitative_claims"][0]["role"] == "configuration"
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_bounded_configuration_count_is_role_based_outside_primary_slots(tmp_path):
    result = _verify(
        "reference_cases_and_proof",
        [
            _span(
                tmp_path,
                "数据分类共19个大类。",
                document_id="p2-bounded-count",
            )
        ],
    )
    assert result.trace["quantitative_claims"][0]["role"] == "configuration"
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


def test_p2_unverified_coverage_claim_remains_strict_outside_outcome_slot(tmp_path):
    result = _verify(
        "implementation_conditions_and_boundaries",
        [
            _filled_span(
                tmp_path,
                "implementation_conditions_and_boundaries",
                suffix="p2-strict-outcome",
            ),
            _span(
                tmp_path,
                "产品已广泛应用于29个地区。",
                document_id="p2-unverified-coverage",
            ),
        ],
    )
    claim = result.trace["quantitative_claims"][0]
    assert claim["role"] == "outcome_claim"
    assert claim["verified"] is False
    assert result.status is SlotStatus.WEAK
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        in result.reason_codes
    )


@pytest.mark.parametrize(
    "text",
    (
        "传统GIS以图为中心，业务流程难以协同，产生具体问题。",
        "现阶段审批业务主要依赖人工，流程难以协同。",
        "实际运行中仍然采用线下流程，部门业务难以联动。",
        "既有系统已部署，但业务数据仍然分散，产生具体需求。",
    ),
)
def test_bounded_current_state_predicates_cover_explicit_present_facts(
    tmp_path,
    text,
):
    result = _verify(
        "current_state_and_needs",
        [_span(tmp_path, text, document_id="current-state-present")],
    )
    assert result.status is SlotStatus.FILLED
    assert result.trace["missing_fields"] == ()


@pytest.mark.parametrize(
    "text",
    (
        "产品支持自动化业务处理，能够覆盖多种场景。",
        "平台可实现数据治理，并可提供流程服务。",
        "未来将建设统一系统，业务流程将实现协同。",
        "计划打造先进平台，面向多种业务场景。",
        "适用于政务行业，可覆盖多种业务。",
        "服务广大客户，提供通用产品模块。",
        "未来将不再依赖人工，业务流程将全面升级。",
        "产品不支持离线部署，平台可提供云端服务。",
        "预计系统将采用新流程，业务能力得到提升。",
    ),
)
def test_bounded_current_state_predicates_reject_future_or_product_capability(
    tmp_path,
    text,
):
    result = _verify(
        "current_state_and_needs",
        [_span(tmp_path, text, document_id="current-state-guard")],
    )
    coverage = {item.field_id: item.covered for item in result.field_coverage}
    assert coverage["current_state"] is False
    assert result.status is not SlotStatus.FILLED


def test_illustrative_quantitative_example_does_not_downgrade_objectives(tmp_path):
    spans = [
        _span(
            tmp_path,
            "本项目建设目标是形成统一数据平台，面向业务系统解决数据分散问题。",
            document_id="objective-supported",
        ),
        _span(
            tmp_path,
            "使用后需反馈效果，例如，“业务办理效率提升30%”。",
            document_id="objective-illustrative-example",
        ),
    ]
    result = _verify("objectives", spans)
    example = next(
        item
        for item in result.trace["quantitative_claims"]
        if item["role"] == "illustrative_example"
    )
    assert example["requires_outcome_support"] is False
    assert example["verified"] is True
    assert result.status is SlotStatus.FILLED
    assert (
        VerificationReasonCode.UNVERIFIED_QUANTITATIVE_CLAIM
        not in result.reason_codes
    )


@pytest.mark.parametrize(
    "text",
    (
        "平台上线后，业务效率提升30%。",
        "项目建成后，预计业务办理效率提升30%。",
        "如期实现业务效率提升30%。",
        "如何实现业务效率提升30%。",
    ),
)
def test_illustrative_marker_is_bounded_and_keeps_real_claims_strict(
    tmp_path,
    text,
):
    result = _verify(
        "outcomes_and_metrics",
        [_span(tmp_path, text, document_id="non-illustrative-claim")],
    )
    claim = result.trace["quantitative_claims"][0]
    assert claim["role"] == "outcome_claim"
    assert claim["requires_outcome_support"] is True
    assert claim["verified"] is False
    assert result.status is SlotStatus.WEAK


def test_illustrative_marker_does_not_reclassify_ru_shi_statistics(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "如实统计业务效率提升30%。",
                document_id="truthful-statistics",
            )
        ],
    )
    assert result.trace["quantitative_claims"][0]["role"] == "outcome_claim"


def test_illustrative_marker_does_not_cross_sentence_boundary(tmp_path):
    result = _verify(
        "outcomes_and_metrics",
        [
            _span(
                tmp_path,
                "例如，模块配置为3个。平台上线后，业务效率提升30%。",
                document_id="illustrative-boundary",
            )
        ],
    )
    claim = next(
        item for item in result.trace["quantitative_claims"] if item["value"] == "30%"
    )
    assert claim["role"] == "outcome_claim"
    assert claim["verified"] is False


def test_illustrative_quantitative_example_cannot_fill_slot_by_itself(tmp_path):
    result = _verify(
        "objectives",
        [
            _span(
                tmp_path,
                "例如，“业务覆盖率达到95%”。",
                document_id="illustrative-only",
            )
        ],
    )
    assert result.trace["quantitative_claims"][0]["role"] == "illustrative_example"
    assert result.status is not SlotStatus.FILLED
    assert "verifiability_or_alignment_with_current_problem" in result.trace[
        "missing_fields"
    ]
