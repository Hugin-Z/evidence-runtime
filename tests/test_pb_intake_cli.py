"""P-B：`python -m evidence.intake` CLI（文件模式）端到端。"""
from __future__ import annotations

import json

from evidence.intake.__main__ import main as intake_main
from evidence.intake import load_project_spec


def _answers(tmp_path, **overrides):
    data = {
        "doc_type": "proposal-document",
        "project_name": "某市 CIM 平台",
        "client_name": "某市住建局",
        "project_id": "cim-2026",
    }
    data.update(overrides)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_intake_cli_file_mode_writes_spec(tmp_path, capsys):
    answers = _answers(tmp_path, section_scope="custom", section_ids=["01", "05"])
    out = tmp_path / "spec.json"
    code = intake_main(["--answers", answers, "--out", str(out)])
    assert code == 0
    stdout = capsys.readouterr().out
    assert "ProjectSpec written" in stdout
    assert "gap_checker --spec" in stdout
    spec = load_project_spec(str(out))
    assert spec.section_ids == ("01", "05")


def test_intake_cli_rejects_invalid_answers(tmp_path, capsys):
    answers = _answers(tmp_path, doc_type="unknown-doc")
    code = intake_main(["--answers", answers, "--out", str(tmp_path / "spec.json")])
    assert code == 2
    assert "doc_type" in capsys.readouterr().err
    assert not (tmp_path / "spec.json").exists()   # 校验失败不落盘
