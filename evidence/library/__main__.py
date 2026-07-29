"""``python -m evidence.library`` —— L1 材料库命令行。

::

    python -m evidence.library init    --project P1
    python -m evidence.library add     --project P1 <文件或目录> [--source-type client_provided]
    python -m evidence.library list    --project P1
    python -m evidence.library status  --project P1
    python -m evidence.library reindex --project P1

**本命令行零 LLM、零向量调用**——``reindex`` 只打印"该跑什么"，真正建索引由
``python -m evidence.gap_checker --project P1`` 在链路中完成（索引与判定同一次遍历，
避免两处各建一份、口径漂移）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .filesystem import DEFAULT_LIBRARY_ROOT, FilesystemLibrary, LibraryError
from .models import SourceType

_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.library",
        description="L1 Library：项目材料库管理",
    )
    parser.add_argument("--root", default=DEFAULT_LIBRARY_ROOT, help="材料库根目录")
    parser.add_argument("--project", required=True, help="项目 id")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="创建（或幂等打开）项目库")

    add = sub.add_parser("add", help="入库材料（文件或目录）")
    add.add_argument("path", help="原始文件或目录（PDF/DOCX/MD/TXT）")
    add.add_argument(
        "--source-type",
        default=SourceType.OTHER.value,
        choices=[item.value for item in SourceType],
        help="材料来源类型（证据效力分档；默认 other）",
    )

    sub.add_parser("list", help="列出材料及版本")
    sub.add_parser("status", help="项目概要 + 索引新旧")
    sub.add_parser("reindex", help="打印重建索引所需的命令（本身不建索引）")
    return parser


def _iter_material_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise LibraryError(f"path not found: {target}")
    return sorted(
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        library = FilesystemLibrary(args.root, args.project)
        if args.command == "init":
            info = library.init_project()
            print(f"project {info.project_id} ready at {library.project_dir}")
            return 0

        if args.command == "add":
            paths = _iter_material_paths(Path(args.path))
            if not paths:
                print(f"error: no supported material under {args.path}", file=sys.stderr)
                return 2
            added = skipped = 0
            for path in paths:
                material_id, is_new = library.add_material(
                    path, source_type=args.source_type
                )
                verb = "added" if is_new else "skipped (same content)"
                print(f"{verb}: {path.name} -> {material_id[:12]}")
                added += int(is_new)
                skipped += int(not is_new)
            print(f"{added} added, {skipped} skipped")
            return 0

        if args.command == "list":
            records = library.list_materials()
            if not records:
                print("(empty)")
                return 0
            for record in records:
                for version in record.versions:
                    mark = "  (superseded)" if version.superseded else ""
                    print(
                        f"{version.material_id[:12]}  {version.source_type:<16}"
                        f"{version.text_length:>8} chars  {record.name}{mark}"
                    )
            return 0

        if args.command == "status":
            info = library.info()
            state = info.index
            print(f"project        : {info.project_id}")
            print(f"created_at     : {info.created_at}")
            print(f"materials      : {info.material_count} current / {info.version_count} versions")
            print(f"index root     : {library.index_root()}")
            print(f"material set   : {state.material_set_fingerprint[:16]}")
            print(f"indexed        : {state.indexed_fingerprint or '(never)'}"
                  f"{'' if state.indexed_at is None else ' @ ' + state.indexed_at}")
            print(f"index status   : {'STALE — reindex needed' if state.is_stale else 'up to date'}")
            return 0

        if args.command == "reindex":
            state = library.index_state()
            if not state.is_stale:
                print("index is already up to date; nothing to do")
                return 0
            print("index is stale. Rebuild it by running the pipeline once:")
            print(
                f"  python -m evidence.gap_checker --project {args.project}"
                f" --library-root {args.root} --output-dir out/gap-report"
            )
            return 0

    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                       # noqa: BLE001 - CLI 边界统一转退出码
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
