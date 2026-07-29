from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from eval.task_acceptance.ai_label_acceptance import (
    AILabelError,
    CONTRACT_ID,
    SLOT_IDS,
    ConsensusSlot,
    aggregate_overall,
    assert_results_postdate_expected,
    compare_judges,
    freeze_expected,
    resolve_consensus,
    validate_blind_input,
    validate_judge_output,
)


AI_LABEL_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "task_acceptance"
    / "ai_label_acceptance.py"
)


def _dataset(tmp_path):
    dataset = tmp_path / "dataset-a"
    materials = dataset / "materials"
    materials.mkdir(parents=True)
    text = "标题\n第一段证据。\n第二段证据。\n"
    (materials / "material.md").write_text(text, encoding="utf-8")
    manifest = {
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
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def _evidence(excerpt="第一段证据。"):
    return {
        "document_id": "doc-a",
        "filename": "material.md",
        "start_line": 2,
        "end_line": 2,
        "excerpt": excerpt,
    }


def _judge(dataset="dataset-a", judge_id="judge-a", status="weak", confidence=0.9):
    return {
        "dataset_id": dataset,
        "judge_id": judge_id,
        "judge_model": "isolated-test-model",
        "judge_independence": "independent_same_model_sessions",
        "contract_id": CONTRACT_ID,
        "slots": {
            slot_id: {
                "status": status,
                "reason": f"独立判断 {slot_id}",
                "evidence": [] if status == "missing" else [_evidence()],
                "confidence": confidence,
            }
            for slot_id in SLOT_IDS
        },
        "overall_status": "incomplete",
        "notes": [],
    }


def test_judge_requires_all_eight_slots(tmp_path):
    dataset = _dataset(tmp_path)
    value = _judge()
    value["slots"].pop(SLOT_IDS[-1])
    with pytest.raises(AILabelError, match="all eight"):
        validate_judge_output(value, dataset_dir=dataset)


def test_judge_rejects_illegal_status_and_confidence(tmp_path):
    dataset = _dataset(tmp_path)
    value = _judge()
    value["slots"][SLOT_IDS[0]]["status"] = "unknown"
    with pytest.raises(AILabelError, match="illegal"):
        validate_judge_output(value, dataset_dir=dataset)
    value = _judge(confidence=1.1)
    with pytest.raises(AILabelError, match="between 0 and 1"):
        validate_judge_output(value, dataset_dir=dataset)


def test_filled_requires_evidence(tmp_path):
    dataset = _dataset(tmp_path)
    value = _judge(status="filled")
    value["slots"][SLOT_IDS[0]]["evidence"] = []
    with pytest.raises(AILabelError, match="filled requires"):
        validate_judge_output(value, dataset_dir=dataset)


def test_excerpt_must_read_back_from_declared_lines(tmp_path):
    dataset = _dataset(tmp_path)
    value = _judge()
    value["slots"][SLOT_IDS[0]]["evidence"] = [_evidence("模型改写的句子")]
    with pytest.raises(AILabelError, match="read back"):
        validate_judge_output(value, dataset_dir=dataset)


def test_overall_aggregation():
    assert aggregate_overall(["filled"] * 8) == "complete"
    assert aggregate_overall(["filled"] * 7 + ["weak"]) == "incomplete"
    assert aggregate_overall(["filled"] * 7 + ["conflicting"]) == "conflicted"


def test_blind_packet_rejects_actual_or_verifier_fields():
    with pytest.raises(AILabelError, match="forbidden"):
        validate_blind_input({"materials": [], "actual_status": "filled"})
    with pytest.raises(AILabelError, match="forbidden"):
        validate_blind_input({"verifier_output": {}})


def test_matching_judges_are_adopted_without_adjudication(tmp_path):
    dataset = _dataset(tmp_path)
    a = validate_judge_output(_judge(), dataset_dir=dataset)
    b = validate_judge_output(_judge(judge_id="judge-b"), dataset_dir=dataset)
    comparison = compare_judges(a, b)
    assert not any(row["needs_adjudication"] for row in comparison["slots"])
    consensus = resolve_consensus(a, b, comparison)
    assert all(item.status == "weak" and not item.unresolved for item in consensus)


def test_status_disagreement_and_low_confidence_enter_adjudication(tmp_path):
    dataset = _dataset(tmp_path)
    a = validate_judge_output(_judge(), dataset_dir=dataset)
    b_value = _judge(judge_id="judge-b")
    b_value["slots"][SLOT_IDS[0]]["status"] = "missing"
    b_value["slots"][SLOT_IDS[0]]["evidence"] = []
    b_value["slots"][SLOT_IDS[1]]["confidence"] = 0.4
    b = validate_judge_output(b_value, dataset_dir=dataset)
    comparison = compare_judges(a, b)
    rows = {row["slot_id"]: row for row in comparison["slots"]}
    assert rows[SLOT_IDS[0]]["classification"] == "STATUS_DISAGREEMENT"
    assert rows[SLOT_IDS[1]]["classification"] == "LOW_CONFIDENCE"


def test_adjudicated_disagreement_generates_resolved_consensus(tmp_path):
    dataset = _dataset(tmp_path)
    a = validate_judge_output(_judge(), dataset_dir=dataset)
    b_value = _judge(judge_id="judge-b")
    b_value["slots"][SLOT_IDS[0]]["status"] = "missing"
    b_value["slots"][SLOT_IDS[0]]["evidence"] = []
    b = validate_judge_output(b_value, dataset_dir=dataset)
    comparison = compare_judges(a, b)
    adjudication = {
        "slots": {
            SLOT_IDS[0]: {
                "final_status": "weak",
                "reason": "裁决采用相关但不充分",
                "supporting_evidence": [_evidence()],
                "confidence": 0.8,
                "requires_human_review": False,
                "human_review_question": "",
            }
        }
    }
    consensus = resolve_consensus(a, b, comparison, adjudication)
    assert consensus[0].status == "weak"
    assert consensus[0].source == "adjudicator"
    assert consensus[0].unresolved is False


def test_unresolved_consensus_writes_draft_not_expected(tmp_path):
    dataset = _dataset(tmp_path)
    a = _judge()
    b = _judge(judge_id="judge-b")
    consensus = tuple(
        ConsensusSlot(slot_id, None, "确有歧义", (), 0.5, "unresolved", True, "请确认")
        if index == 0
        else ConsensusSlot(slot_id, "weak", "一致", tuple([_evidence()]), 0.9, "judge_consensus", False, "")
        for index, slot_id in enumerate(SLOT_IDS)
    )
    result = freeze_expected(
        dataset_dir=dataset,
        consensus=consensus,
        judge_a=a,
        judge_b=b,
        adjudication=None,
        generated_timestamp="2026-07-17T00:00:00+00:00",
    )
    assert result["resolved"] is False
    assert (dataset / "expected.ai-draft.json").is_file()
    assert not (dataset / "expected.json").exists()


def test_resolved_expected_hash_is_stable_and_pre_actual(tmp_path):
    dataset = _dataset(tmp_path)
    a = _judge()
    b = _judge(judge_id="judge-b")
    consensus = tuple(
        ConsensusSlot(slot_id, "weak", "一致", tuple([_evidence()]), 0.9, "judge_consensus", False, "")
        for slot_id in SLOT_IDS
    )
    kwargs = dict(
        dataset_dir=dataset,
        consensus=consensus,
        judge_a=a,
        judge_b=b,
        adjudication=None,
        generated_timestamp="2026-07-17T00:00:00+00:00",
    )
    first = freeze_expected(**kwargs)
    first_bytes = (dataset / "expected.json").read_bytes()
    second = freeze_expected(**kwargs)
    assert first["expected_file_sha256"] == second["expected_file_sha256"]
    assert first_bytes == (dataset / "expected.json").read_bytes()
    assert first["expected_file_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    notes = json.loads(json.loads(first_bytes)["notes"])["ai_label_provenance"]
    assert notes["generated_before_actual_run"] is True


def test_blocked_conflicting_directory_can_remain_without_expected(tmp_path):
    blocked = tmp_path / "conflicting"
    blocked.mkdir()
    (blocked / "blocked.json").write_text('{"status":"blocked"}', encoding="utf-8")
    assert not (blocked / "expected.json").exists()


def test_actual_results_must_postdate_expected_freeze(tmp_path):
    dataset = _dataset(tmp_path)
    consensus = tuple(
        ConsensusSlot(slot_id, "weak", "一致", tuple([_evidence()]), 0.9, "judge_consensus", False, "")
        for slot_id in SLOT_IDS
    )
    freeze_expected(
        dataset_dir=dataset,
        consensus=consensus,
        judge_a=_judge(),
        judge_b=_judge(judge_id="judge-b"),
        adjudication=None,
        generated_timestamp="2026-07-17T00:00:00+00:00",
    )
    result = tmp_path / "actual.json"
    result.write_text("{}", encoding="utf-8")
    assert_results_postdate_expected(dataset / "expected.provenance.json", [result])


def test_ai_label_cli_can_run_directly_outside_repo_cwd(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(AI_LABEL_SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-dir" in completed.stdout
