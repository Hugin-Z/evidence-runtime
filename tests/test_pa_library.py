"""P-A：L1 Library（材料库管理层）单元测试。

覆盖四项职责 + 一条铁律：项目隔离、入库幂等与版本、`source_type` 元数据与过滤、
索引生命周期；铁律 = **material_id 内容派生**（同正文同 id，与文件名/入库顺序无关）。
"""
from __future__ import annotations

import json

import pytest

from evidence.library import (
    FilesystemLibrary,
    LibraryError,
    LibraryProtocol,
    SourceType,
)
from evidence.library.__main__ import main as library_main


def _write(directory, name: str, text: str):
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def sources(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _write(root, "招标文件.md", "# 招标文件\n\n本项目建设依据为上级批复文件。\n")
    _write(root, "我方方案.md", "# 我方方案\n\n我方已在多地部署应用。\n")
    _write(root, "国标.md", "# 国标\n\nGB/T 39786-2021 信息安全技术。\n")
    return root


@pytest.fixture()
def library(tmp_path):
    lib = FilesystemLibrary(tmp_path / "lib", "proj-a")
    lib.init_project()
    return lib


# ── 接口 ────────────────────────────────────────────────────────────────
def test_filesystem_library_satisfies_protocol(library):
    assert isinstance(library, LibraryProtocol)


# ── 项目隔离 ─────────────────────────────────────────────────────────────
def test_projects_are_isolated(tmp_path, sources):
    root = tmp_path / "lib"
    first = FilesystemLibrary(root, "p1")
    second = FilesystemLibrary(root, "p2")
    first.init_project()
    second.init_project()
    first.add_material(sources / "招标文件.md", source_type=SourceType.CLIENT_PROVIDED)

    assert len(first.documents()) == 1
    assert second.documents() == ()
    assert first.index_root() != second.index_root()
    assert first.index_root().is_dir() and second.index_root().is_dir()


def test_init_project_is_idempotent(tmp_path, sources):
    lib = FilesystemLibrary(tmp_path / "lib", "p1")
    lib.init_project()
    lib.add_material(sources / "国标.md", source_type=SourceType.PUBLIC)
    created_at = lib.info().created_at
    again = lib.init_project()                     # 二次 init 不得清空已入库材料
    assert again.material_count == 1
    assert again.created_at == created_at


@pytest.mark.parametrize("bad", ["", "..", "../escape", "a/b", "x" * 65])
def test_invalid_project_id_rejected(tmp_path, bad):
    with pytest.raises(LibraryError):
        FilesystemLibrary(tmp_path / "lib", bad)


def test_operations_require_init(tmp_path, sources):
    lib = FilesystemLibrary(tmp_path / "lib", "never-inited")
    with pytest.raises(LibraryError, match="not initialised"):
        lib.list_materials()


# ── 入库：幂等 + 内容派生 id ──────────────────────────────────────────────
def test_add_is_idempotent_by_content(library, sources):
    first_id, first_added = library.add_material(sources / "招标文件.md")
    second_id, second_added = library.add_material(sources / "招标文件.md")
    assert first_added is True and second_added is False
    assert first_id == second_id
    assert library.info().version_count == 1


def test_material_id_is_content_derived_not_name_derived(tmp_path, library, sources):
    """同正文换个文件名入库 → **同一个 material_id**（承接 M1b 内容确定铁律）。"""

    original = (sources / "国标.md").read_text(encoding="utf-8")
    renamed = _write(tmp_path, "完全不同的名字.md", original)
    first_id, _ = library.add_material(sources / "国标.md")
    second_id, added = library.add_material(renamed)
    assert second_id == first_id
    assert added is False                          # 同内容 = 已在库，与文件名无关


def test_material_id_is_independent_of_ingest_order(tmp_path, sources):
    forward = FilesystemLibrary(tmp_path / "l1", "a")
    backward = FilesystemLibrary(tmp_path / "l2", "b")
    forward.init_project()
    backward.init_project()
    names = ["招标文件.md", "我方方案.md", "国标.md"]
    for name in names:
        forward.add_material(sources / name)
    for name in reversed(names):
        backward.add_material(sources / name)
    assert (
        sorted(d.document_id for d in forward.documents())
        == sorted(d.document_id for d in backward.documents())
    )


def test_add_missing_file_raises(library, tmp_path):
    with pytest.raises(LibraryError, match="material not found"):
        library.add_material(tmp_path / "does-not-exist.md")


# ── 版本 ─────────────────────────────────────────────────────────────────
def test_same_name_new_content_creates_version(library, sources):
    old_id, _ = library.add_material(sources / "我方方案.md", source_type=SourceType.SELF_PRODUCED)
    _write(sources, "我方方案.md", "# 我方方案\n\n我方已在浙江、江苏两地部署应用。\n")
    new_id, added = library.add_material(sources / "我方方案.md", source_type=SourceType.SELF_PRODUCED)

    assert added is True and new_id != old_id
    record = next(r for r in library.list_materials() if r.name == "我方方案.md")
    assert len(record.versions) == 2
    assert record.current is not None and record.current.material_id == new_id
    assert [v.superseded for v in record.versions] == [True, False]

    # 只有当前版本参与检索；旧版本仍可查
    assert [d.document_id for d in library.documents()] == [new_id]
    assert old_id in {v.material_id for v in record.versions}
    assert library.info().material_count == 1
    assert library.info().version_count == 2


# ── source_type 元数据与过滤 ─────────────────────────────────────────────
def test_source_type_recorded_and_filters_documents(library, sources):
    client_id, _ = library.add_material(sources / "招标文件.md", source_type="client_provided")
    self_id, _ = library.add_material(sources / "我方方案.md", source_type="self_produced")
    public_id, _ = library.add_material(sources / "国标.md", source_type=SourceType.PUBLIC)

    assert len(library.documents()) == 3           # 默认不过滤
    filtered = library.documents(source_types=["client_provided", "public"])
    assert sorted(d.document_id for d in filtered) == sorted([client_id, public_id])
    assert self_id not in {d.document_id for d in filtered}


def test_unknown_source_type_rejected(library, sources):
    with pytest.raises(ValueError, match="unknown source_type"):
        library.add_material(sources / "国标.md", source_type="confidential")


def test_documents_are_ordered_deterministically(library, sources):
    for name in ("招标文件.md", "我方方案.md", "国标.md"):
        library.add_material(sources / name)
    ids = [d.document_id for d in library.documents()]
    assert ids == sorted(ids)


def test_documents_carry_original_filename_as_title(library, sources):
    library.add_material(sources / "招标文件.md", source_type="client_provided")
    document = library.documents()[0]
    assert document.title == "招标文件"
    assert document.metadata["runtime_adapter"]["source_type"] == "md"


# ── 索引生命周期 ─────────────────────────────────────────────────────────
def test_index_state_tracks_material_set(library, sources):
    assert library.index_state().is_stale is True          # 从未建索引
    library.add_material(sources / "招标文件.md")
    state = library.mark_indexed()
    assert state.is_stale is False and state.indexed_at is not None

    library.add_material(sources / "国标.md")               # 材料集变了 → 过期
    assert library.index_state().is_stale is True
    assert library.mark_indexed().is_stale is False


def test_index_state_survives_reopen(tmp_path, sources):
    root = tmp_path / "lib"
    FilesystemLibrary(root, "p1").init_project()
    first = FilesystemLibrary(root, "p1")
    first.add_material(sources / "国标.md")
    first.mark_indexed()
    assert FilesystemLibrary(root, "p1").index_state().is_stale is False


def test_corrupt_registry_is_reported(library):
    library.registry_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LibraryError, match="corrupt registry"):
        library.list_materials()


def test_registry_is_json_and_human_readable(library, sources):
    library.add_material(sources / "国标.md", source_type=SourceType.PUBLIC)
    data = json.loads(library.registry_path.read_text(encoding="utf-8"))
    version = data["materials"][0]["versions"][0]
    assert data["project_id"] == "proj-a"
    assert version["source_type"] == "public"
    assert version["original_filename"] == "国标.md"


# ── CLI ──────────────────────────────────────────────────────────────────
def test_library_cli_roundtrip(tmp_path, sources, capsys):
    root = str(tmp_path / "lib")
    assert library_main(["--root", root, "--project", "cli", "init"]) == 0
    assert library_main([
        "--root", root, "--project", "cli", "add", str(sources),
        "--source-type", "client_provided",
    ]) == 0
    assert "3 added, 0 skipped" in capsys.readouterr().out

    assert library_main(["--root", root, "--project", "cli", "add", str(sources)]) == 0
    assert "0 added, 3 skipped" in capsys.readouterr().out

    assert library_main(["--root", root, "--project", "cli", "status"]) == 0
    assert "STALE" in capsys.readouterr().out

    assert library_main(["--root", root, "--project", "cli", "list"]) == 0
    listing = capsys.readouterr().out
    assert "client_provided" in listing and "招标文件.md" in listing

    assert library_main(["--root", root, "--project", "cli", "reindex"]) == 0
    assert "--project cli" in capsys.readouterr().out


def test_library_cli_reports_uninitialised_project(tmp_path, capsys):
    code = library_main(["--root", str(tmp_path / "lib"), "--project", "ghost", "list"])
    assert code == 2
    assert "not initialised" in capsys.readouterr().err
