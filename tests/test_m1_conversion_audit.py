from __future__ import annotations

import json

from eval.task_acceptance.audit_converted_materials import (
    AutomatedReviewStatus,
    _comparable_pdf_sample_overlap,
    _evaluate_pdf_overlaps,
    audit_normalized_text,
)


def test_empty_document_is_blocked():
    result = audit_normalized_text("")
    assert result.status is AutomatedReviewStatus.BLOCKED
    assert result.human_review_required is True


def test_excessive_garbled_text_is_blocked():
    result = audit_normalized_text("正文" * 100 + "\ufffd" * 20)
    assert result.status is AutomatedReviewStatus.BLOCKED
    assert "excessive_garbled_or_abnormal_characters" in result.detected_issues


def test_normal_chinese_markdown_passes_and_serializes():
    text = "# 建设背景\n\n" + "".join(
        f"第{index}段说明本项目依据、建设对象、范围和目标。\n"
        for index in range(20)
    )
    result = audit_normalized_text(text)
    assert result.status is AutomatedReviewStatus.PASSED
    assert result.heading_count == 1
    json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)


def test_partially_misaligned_markdown_table_is_warning():
    text = "# 清单\n\n| 名称 | 数量 |\n|---|---|\n| 服务器 | 2 | 备注 |\n" + "".join(
        f"说明段落{index}。\n" for index in range(30)
    )
    result = audit_normalized_text(text)
    assert result.status is AutomatedReviewStatus.WARNING
    assert "markdown_table_column_count_inconsistent" in result.detected_issues


def test_repeated_headers_are_warning():
    text = "# 报告\n\n" + "统一页眉\n正文内容与项目范围。\n" * 30
    result = audit_normalized_text(text)
    assert result.status is AutomatedReviewStatus.WARNING
    assert "repeated_headers_or_lines" in result.detected_issues


def test_audit_result_does_not_retain_source_path_or_text():
    result = audit_normalized_text("# 标题\n\n" + "正常正文。" * 100).to_dict()
    rendered = json.dumps(result, ensure_ascii=False)
    assert "source_path" not in rendered
    assert "正常正文" not in rendered


def test_short_non_body_pdf_sample_does_not_define_text_overlap():
    assert _comparable_pdf_sample_overlap("结束语", "完整正文不含结束语") is None


def test_one_layout_heavy_pdf_sample_warns_when_body_sample_is_recoverable():
    issues, blocked = _evaluate_pdf_overlaps([0.0, 1.0])
    assert blocked is False
    assert issues == ["pdf_sample_text_partially_recoverable"]
