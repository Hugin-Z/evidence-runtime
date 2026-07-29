from __future__ import annotations

import json

import pytest

from eval.task_acceptance.ai_label_acceptance import CONTRACT_ID, SLOT_IDS
from eval.task_acceptance.audit_acceptance_discrepancies import (
    DATASET_IDS,
    DiscrepancyAuditError,
    build_mismatch_matrix,
    freeze_inputs,
    redact_sensitive,
    write_matrix,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture_tree(tmp_path):
    acceptance = tmp_path / "acceptance"
    results = tmp_path / "results"
    reviews = tmp_path / "reviews"
    contract = tmp_path / "contract.json"
    _write_json(
        contract,
        {
            "contract_id": CONTRACT_ID,
            "slots": [
                {"slot_id": slot_id, "name": slot_id, "required_fields": ["field"]}
                for slot_id in SLOT_IDS
            ],
        },
    )
    for dataset_id in DATASET_IDS:
        dataset = acceptance / dataset_id
        material = dataset / "materials" / "material.md"
        material.parent.mkdir(parents=True)
        material.write_text("标题\n证据。\n", encoding="utf-8")
        _write_json(
            dataset / "manifest.json",
            {
                "documents": [
                    {
                        "document_id": "doc-a",
                        "title": "材料",
                        "source_path": "materials/material.md",
                        "source_type": "md",
                        "normalized_path": "materials/material.md",
                        "metadata": {},
                    }
                ]
            },
        )
        expected_slots = {
            slot_id: {"expected_status": "weak", "reason": "证据有限", "references": []}
            for slot_id in SLOT_IDS
        }
        _write_json(
            dataset / "expected.json",
            {
                "dataset_id": dataset_id,
                "description": "synthetic",
                "contract_id": CONTRACT_ID,
                "slots": expected_slots,
                "overall_expected_status": "incomplete",
                "notes": "synthetic",
            },
        )
        evidence = {
            "evidence_id": "ev-a",
            "document_id": "doc-a",
            "text": "证据。",
            "start_char": 3,
            "end_char": 6,
            "start_line": 2,
            "end_line": 2,
            "page": None,
            "discovery_sources": ["exact"],
        }
        actual_slots = [
            {
                "slot_id": slot_id,
                "slot_name": slot_id,
                "required": True,
                "status": "weak",
                "evidence_ids": ["ev-a"],
                "field_coverage": 0.5,
                "covered_fields": ["field"],
                "missing_fields": [],
                "reason_codes": [],
                "reason": "synthetic",
                "conflicts": [],
                "material_request": "",
            }
            for slot_id in SLOT_IDS
        ]
        trace = {
            "per_slot": {
                slot_id: {
                    "vector_candidate_count": 1,
                    "anchor_count": 1,
                    "exact_span_count": 1,
                    "fallback_used": False,
                    "verification_status": "weak",
                }
                for slot_id in SLOT_IDS
            }
        }
        _write_json(
            results / dataset_id / "actual-gap-report.json",
            {"slots": actual_slots, "evidence": [evidence], "trace": trace},
        )
        comparison_slots = [
            {
                "slot_id": slot_id,
                "expected_status": "weak",
                "actual_status": "weak",
                "match": True,
                "error_attribution": None,
            }
            for slot_id in SLOT_IDS
        ]
        _write_json(results / dataset_id / "comparison.json", {"slots": comparison_slots})
        judge_evidence = {
            "document_id": "doc-a",
            "filename": "material.md",
            "start_line": 2,
            "end_line": 2,
            "excerpt": "证据。",
        }
        for judge_id in ("judge-a", "judge-b"):
            _write_json(
                reviews / dataset_id / f"{judge_id}.json",
                {
                    "dataset_id": dataset_id,
                    "judge_id": judge_id,
                    "judge_model": "synthetic-model",
                    "judge_independence": "independent",
                    "contract_id": CONTRACT_ID,
                    "slots": {
                        slot_id: {
                            "status": "weak",
                            "reason": "证据有限",
                            "evidence": [judge_evidence],
                            "confidence": 0.9,
                        }
                        for slot_id in SLOT_IDS
                    },
                    "overall_status": "incomplete",
                    "notes": [],
                },
            )
    return acceptance, results, reviews, contract


def test_matrix_covers_all_24_slots_and_validates_readback(tmp_path):
    acceptance, results, reviews, contract = _fixture_tree(tmp_path)
    matrix = build_mismatch_matrix(
        acceptance_root=acceptance,
        results_root=results,
        ai_review_root=reviews,
        contract_path=contract,
    )
    assert matrix["slot_count"] == 24
    assert matrix["mismatch_count"] == 0
    assert all(row["judge_evidence_readback_valid"] for row in matrix["rows"])
    assert all(row["actual_evidence_readback_valid"] for row in matrix["rows"])


def test_freeze_records_hashes_without_modifying_inputs(tmp_path):
    acceptance, results, _, contract = _fixture_tree(tmp_path)
    output = tmp_path / "audit"
    before = (acceptance / "complete" / "expected.json").read_bytes()
    frozen = freeze_inputs(
        acceptance_root=acceptance,
        results_root=results,
        contract_path=contract,
        output_dir=output,
        generated_timestamp="2026-07-17T00:00:00+00:00",
    )
    assert len(frozen["datasets"]) == 3
    assert len(frozen["datasets"][0]["materials"]) == 1
    assert before == (acceptance / "complete" / "expected.json").read_bytes()
    with pytest.raises(DiscrepancyAuditError, match="refusing to overwrite"):
        freeze_inputs(
            acceptance_root=acceptance,
            results_root=results,
            contract_path=contract,
            output_dir=output,
            generated_timestamp="later",
        )


def test_markdown_redaction_masks_sensitive_values():
    raw = "联系人 13800138000 a@example.com，地址 10.1.2.3，预算 123.45 万元"
    redacted = redact_sensitive(raw)
    assert "13800138000" not in redacted
    assert "a@example.com" not in redacted
    assert "10.1.2.3" not in redacted
    assert "123.45" not in redacted
    assert redacted.count("[REDACTED_") == 4


def test_matrix_markdown_never_renders_private_evidence_text(tmp_path):
    acceptance, results, reviews, contract = _fixture_tree(tmp_path)
    matrix = build_mismatch_matrix(
        acceptance_root=acceptance,
        results_root=results,
        ai_review_root=reviews,
        contract_path=contract,
    )
    matrix["rows"][0]["actual_evidence_excerpts"][0]["text"] = "私有精确报价 123.45 万元"
    write_matrix(tmp_path / "audit", matrix)
    markdown = (tmp_path / "audit" / "mismatch-matrix.md").read_text(encoding="utf-8")
    assert "123.45" not in markdown
    assert "PRIVATE_EVIDENCE_REDACTED" in markdown
