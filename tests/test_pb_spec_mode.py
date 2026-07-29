"""P-B：``gap_checker --spec`` 接线测试（零 embedding 调用）。

只验证**接线**：spec 解析 → 确定性选契约 → 定位项目库 → source_types 透传，
且落到与 --project 完全相同的 run() 参数（spec 只是入口翻译，判断链不变）。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evidence.gap_checker import build_argument_parser, main
from evidence.intake import save_spec, validate_answers
from evidence.library import FilesystemLibrary, SourceType


class _RecordingChecker:
    def __init__(self, report):
        self.report = report
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(report=self.report)


@pytest.fixture()
def report():
    from tests.test_m1_gap_checker import _minimal_report

    return _minimal_report("incomplete")


@pytest.fixture()
def project(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "招标文件.md").write_text("本项目建设依据为上级批复。\n", encoding="utf-8")
    (src / "我方方案.md").write_text("我方已在多地部署应用。\n", encoding="utf-8")
    root = tmp_path / "lib"
    library = FilesystemLibrary(root, "p1")
    library.init_project()
    client_id, _ = library.add_material(src / "招标文件.md", source_type=SourceType.CLIENT_PROVIDED)
    self_id, _ = library.add_material(src / "我方方案.md", source_type=SourceType.SELF_PRODUCED)
    return SimpleNamespace(root=str(root), library=library, client_id=client_id, self_id=self_id)


def _spec_file(tmp_path, project_id, **overrides):
    answers = {
        "doc_type": "proposal-document",
        "project_name": "某项目",
        "client_name": "某局",
        "project_id": project_id,
    }
    answers.update(overrides)
    spec = validate_answers(answers)
    return str(save_spec(spec, tmp_path / "spec.json"))


def test_parser_rejects_spec_together_with_project():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--spec", "s.json", "--project", "p1", "--output-dir", "o"])


def test_spec_mode_resolves_contract_and_project(tmp_path, project, report):
    spec_path = _spec_file(tmp_path, "p1")
    checker = _RecordingChecker(report)
    code = main(
        ["--spec", spec_path, "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: checker,
    )
    assert code == 1
    kwargs = checker.calls[0]
    assert kwargs["contract"].contract_id == "solution_material_completeness_v1"
    assert sorted(d.document_id for d in kwargs["documents"]) == sorted(
        [project.client_id, project.self_id]
    )
    assert kwargs["allowed_document_ids"] is None


def test_spec_mode_isolates_index_like_project(tmp_path, project, report):
    spec_path = _spec_file(tmp_path, "p1")
    captured = {}

    def factory(settings):
        captured["vector_db_path"] = settings.vector_db_path
        return _RecordingChecker(report)

    main(
        ["--spec", spec_path, "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=factory,
    )
    assert captured["vector_db_path"] == str(project.library.index_root())


def test_spec_source_types_flow_to_allowed_ids(tmp_path, project, report):
    spec_path = _spec_file(tmp_path, "p1", source_types=["client_provided"])
    checker = _RecordingChecker(report)
    main(
        ["--spec", spec_path, "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: checker,
    )
    assert checker.calls[0]["allowed_document_ids"] == frozenset({project.client_id})


def test_spec_and_project_produce_identical_run_kwargs(tmp_path, project, report):
    """spec 只是入口翻译：--spec 与 --project 落到 run() 的参数必须一致。"""

    spec_path = _spec_file(tmp_path, "p1")
    spec_checker = _RecordingChecker(report)
    proj_checker = _RecordingChecker(report)
    main(["--spec", spec_path, "--library-root", project.root,
          "--output-dir", str(tmp_path / "out1")],
         checker_factory=lambda settings: spec_checker)
    main(["--project", "p1", "--library-root", project.root,
          "--output-dir", str(tmp_path / "out2")],
         checker_factory=lambda settings: proj_checker)

    a, b = spec_checker.calls[0], proj_checker.calls[0]
    assert a["contract"].contract_id == b["contract"].contract_id
    assert sorted(d.document_id for d in a["documents"]) == sorted(
        d.document_id for d in b["documents"]
    )
    assert a["allowed_document_ids"] == b["allowed_document_ids"]
    for key in ("max_chars", "top_k_per_slot", "context_lines",
                "max_span_chars", "max_results_per_slot"):
        assert a[key] == b[key]


def test_spec_with_unknown_project_is_reported(tmp_path, report, capsys):
    spec_path = _spec_file(tmp_path, "ghost")
    code = main(
        ["--spec", spec_path, "--library-root", str(tmp_path / "lib"),
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: _RecordingChecker(report),
    )
    assert code == 2
    assert "not initialised" in capsys.readouterr().err
