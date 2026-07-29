"""P-A：``gap_checker --project`` 项目模式的接线测试（零 embedding 调用）。

只验证**接线**：材料从 L1 取、索引落在项目 ``index/`` 下（项目隔离）、
``--source-types`` 解析成 ``allowed_document_ids`` 传进链路、跑完登记索引指纹。
判定逻辑本身由既有 M1 测试与 e2e 验收覆盖，本文件不重复。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evidence.gap_checker import build_argument_parser, main
from evidence.library import FilesystemLibrary, SourceType


class _RecordingChecker:
    """记录 run() 收到的参数；不做任何检索。"""

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


def test_parser_rejects_project_together_with_input(tmp_path):
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--project", "p1", "--input", str(tmp_path), "--output-dir", "o"])


def test_project_mode_reads_materials_from_library(tmp_path, project, report):
    checker = _RecordingChecker(report)
    code = main(
        ["--project", "p1", "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: checker,
    )
    assert code == 1                                   # incomplete
    kwargs = checker.calls[0]
    assert sorted(d.document_id for d in kwargs["documents"]) == sorted(
        [project.client_id, project.self_id]
    )
    assert kwargs["allowed_document_ids"] is None      # 未传 --source-types = 不过滤


def test_project_mode_isolates_the_vector_index(tmp_path, project, report):
    captured = {}

    def factory(settings):
        captured["vector_db_path"] = settings.vector_db_path
        return _RecordingChecker(report)

    main(
        ["--project", "p1", "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=factory,
    )
    assert captured["vector_db_path"] == str(project.library.index_root())


def test_source_types_resolve_to_allowed_document_ids(tmp_path, project, report):
    checker = _RecordingChecker(report)
    main(
        ["--project", "p1", "--library-root", project.root,
         "--source-types", "client_provided",
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: checker,
    )
    kwargs = checker.calls[0]
    # 索引仍建在全量材料上（一份索引服务任意过滤组合），只有候选层被收窄。
    assert len(kwargs["documents"]) == 2
    assert kwargs["allowed_document_ids"] == frozenset({project.client_id})


def test_successful_run_marks_index_fresh(tmp_path, project, report):
    assert project.library.index_state().is_stale is True
    main(
        ["--project", "p1", "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: _RecordingChecker(report),
    )
    assert project.library.index_state().is_stale is False


def test_failed_run_leaves_index_stale(tmp_path, project, report, capsys):
    class _Boom:
        def run(self, **kwargs):
            raise RuntimeError("embedding backend failed")

    code = main(
        ["--project", "p1", "--library-root", project.root,
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: _Boom(),
    )
    assert code == 2
    assert "embedding backend failed" in capsys.readouterr().err
    assert project.library.index_state().is_stale is True


def test_empty_project_is_reported_not_silently_empty(tmp_path, capsys):
    root = tmp_path / "lib"
    FilesystemLibrary(root, "empty").init_project()
    code = main(
        ["--project", "empty", "--library-root", str(root),
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: _RecordingChecker(None),
    )
    assert code == 2
    assert "no current materials" in capsys.readouterr().err


def test_source_types_matching_nothing_is_an_error(tmp_path, project, report, capsys):
    code = main(
        ["--project", "p1", "--library-root", project.root,
         "--source-types", "public",
         "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: _RecordingChecker(report),
    )
    assert code == 2
    assert "no materials match" in capsys.readouterr().err


def test_input_mode_still_passes_no_filter(tmp_path, report):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "a.md").write_text("材料正文。\n", encoding="utf-8")
    checker = _RecordingChecker(report)
    main(
        ["--input", str(materials), "--output-dir", str(tmp_path / "out")],
        checker_factory=lambda settings: checker,
    )
    kwargs = checker.calls[0]
    assert kwargs["allowed_document_ids"] is None      # 目录模式恒不过滤
    assert [d.source_path for d in kwargs["documents"]] == ["a.md"]
