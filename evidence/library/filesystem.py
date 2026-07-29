"""L1 Library 默认实现：文件系统材料库。

**目录约定（一项目一库）**::

    <root>/<project_id>/
        library.json          # 注册表：材料 → 版本链 + 索引指纹
        materials/            # 内容寻址存储：<material_id><原扩展名>
        index/                # 该项目的向量索引根（项目隔离的物理落点）

**项目隔离怎么做的**：每个项目有**独立的 ``index/`` 目录**，直接作为
``Settings.vector_db_path`` 传给 L2。因此不同项目的 Chroma collection 物理隔离，
**且 L2 一行不用改**——collection 命名仍用既有指纹机制（已含 chunker 版本、
embedding 后端/模型/维度与文档集），衔接而非替换。

**版本语义**：入库按**内容哈希**识别——同哈希幂等跳过；**同名不同哈希 = 新版本**，
旧版本置 ``superseded`` 保留可查，**只有当前版本参与检索**。

**索引新旧**：``library.json`` 记录上次建索引时的「材料集指纹」
（= 当前版本 material_id 有序集 + chunker 版本的 sha256）。``index_state()`` 比对它与
当前材料集指纹即可回答"索引是否过期"，**无需启动 embedder**。更深一层的 embedding 身份
一致性仍由既有 ``VectorIndexManifest`` 机制在查询时强制。
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..documents import LOSSLESS_CHUNKER_VERSION, NormalizedDocument
from ..runtime_adapter import adapt_file
from .models import IndexState, MaterialRecord, MaterialVersion, ProjectInfo, SourceType

LIBRARY_VERSION = "p-a-library-v1"
REGISTRY_NAME = "library.json"
DEFAULT_LIBRARY_ROOT = "./.library"   # 默认材料库根目录（未跟踪；不入 git）
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class LibraryError(RuntimeError):
    """材料库操作失败（项目不存在、id 非法、注册表损坏等）。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not _PROJECT_ID_RE.match(project_id):
        raise LibraryError(
            f"invalid project_id {project_id!r}: expected [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
        )
    if project_id in {".", ".."}:
        raise LibraryError(f"invalid project_id {project_id!r}")
    return project_id


class FilesystemLibrary:
    """``LibraryProtocol`` 的随仓默认实现（纯文件系统，无新依赖）。"""

    def __init__(self, root: str | Path, project_id: str) -> None:
        self.root = Path(root).resolve()
        self.project_id = _validate_project_id(project_id)
        self.project_dir = self.root / self.project_id

    # ── 路径 ──────────────────────────────────────────────────────────
    @property
    def registry_path(self) -> Path:
        return self.project_dir / REGISTRY_NAME

    @property
    def materials_dir(self) -> Path:
        return self.project_dir / "materials"

    def index_root(self) -> Path:
        return self.project_dir / "index"

    # ── 注册表读写 ────────────────────────────────────────────────────
    def _read_registry(self) -> dict:
        if not self.registry_path.is_file():
            raise LibraryError(
                f"project {self.project_id!r} not initialised under {self.root}"
                " — run `library init` first"
            )
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LibraryError(f"corrupt registry {self.registry_path}: {exc}") from exc
        if not isinstance(data, dict) or "materials" not in data:
            raise LibraryError(f"corrupt registry {self.registry_path}: missing 'materials'")
        return data

    def _write_registry(self, data: dict) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False),
            encoding="utf-8",
        )

    # ── Protocol 实现 ────────────────────────────────────────────────
    def init_project(self, project_id: str | None = None) -> ProjectInfo:
        if project_id is not None:
            _validate_project_id(project_id)
            if project_id != self.project_id:
                raise LibraryError(
                    f"project_id mismatch: library bound to {self.project_id!r}, got {project_id!r}"
                )
        if not self.registry_path.is_file():          # 幂等：已存在则直接打开
            self.materials_dir.mkdir(parents=True, exist_ok=True)
            self.index_root().mkdir(parents=True, exist_ok=True)
            self._write_registry({
                "library_version": LIBRARY_VERSION,
                "project_id": self.project_id,
                "created_at": _utc_now(),
                "indexed_fingerprint": None,
                "indexed_at": None,
                "materials": [],
            })
        return self.info()

    def add_material(
        self,
        source: str | Path,
        *,
        source_type: SourceType | str = SourceType.OTHER,
    ) -> tuple[str, bool]:
        resolved_type = SourceType.parse(source_type)
        path = Path(source)
        if not path.is_file():
            raise LibraryError(f"material not found: {path}")
        data = self._read_registry()

        # 内容派生 id：规范化正文哈希（adapt_file 已做格式探测 + 规范化 + provenance）
        document = adapt_file(path)
        material_id = document.normalized_content_hash
        original = path.name

        records = [MaterialRecord.from_dict(item) for item in data["materials"]]
        for record in records:                        # 幂等：同内容已在库（任何名下）
            for version in record.versions:
                if version.material_id == material_id:
                    return material_id, False

        stored_name = f"{material_id}{path.suffix.lower()}"
        self.materials_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self.materials_dir / stored_name)

        new_version = MaterialVersion(
            material_id=material_id,
            original_filename=original,
            stored_filename=stored_name,
            source_type=resolved_type.value,
            ingested_at=_utc_now(),
            text_length=len(document.text),
            superseded=False,
        )
        updated: list[MaterialRecord] = []
        matched = False
        for record in records:
            if record.name == original:               # 同名不同哈希 → 新版本
                matched = True
                superseded = tuple(
                    MaterialVersion(**{**v.to_dict(), "superseded": True})
                    for v in record.versions
                )
                updated.append(MaterialRecord(record.name, superseded + (new_version,)))
            else:
                updated.append(record)
        if not matched:
            updated.append(MaterialRecord(original, (new_version,)))

        data["materials"] = [record.to_dict() for record in updated]
        self._write_registry(data)
        return material_id, True

    def list_materials(self) -> tuple[MaterialRecord, ...]:
        data = self._read_registry()
        return tuple(MaterialRecord.from_dict(item) for item in data["materials"])

    def documents(
        self,
        *,
        source_types: Sequence[SourceType | str] | None = None,
    ) -> tuple[NormalizedDocument, ...]:
        """当前版本材料 → NormalizedDocument。``source_types=None`` 表示不过滤。"""

        allowed: set[str] | None = None
        if source_types is not None:
            allowed = {SourceType.parse(item).value for item in source_types}
        out: list[NormalizedDocument] = []
        for record in self.list_materials():
            current = record.current
            if current is None:
                continue
            if allowed is not None and current.source_type not in allowed:
                continue
            stored = self.materials_dir / current.stored_filename
            if not stored.is_file():
                raise LibraryError(f"stored material missing: {stored}")
            # document_id 用 material_id（内容派生），title 用原始文件名（去扩展名）
            out.append(adapt_file(
                stored,
                document_id=current.material_id,
                title=Path(current.original_filename).stem,
            ))
        # 稳定顺序：按内容派生 id 排序（不依赖入库顺序）
        return tuple(sorted(out, key=lambda d: d.document_id))

    # ── 索引生命周期 ──────────────────────────────────────────────────
    def material_set_fingerprint(self) -> str:
        ids = sorted(
            record.current.material_id
            for record in self.list_materials()
            if record.current is not None
        )
        payload = json.dumps(
            {"materials": ids, "chunker_version": LOSSLESS_CHUNKER_VERSION},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def index_state(self) -> IndexState:
        data = self._read_registry()
        return IndexState(
            material_set_fingerprint=self.material_set_fingerprint(),
            indexed_fingerprint=data.get("indexed_fingerprint"),
            indexed_at=data.get("indexed_at"),
        )

    def mark_indexed(self) -> IndexState:
        data = self._read_registry()
        data["indexed_fingerprint"] = self.material_set_fingerprint()
        data["indexed_at"] = _utc_now()
        self._write_registry(data)
        return self.index_state()

    def info(self) -> ProjectInfo:
        data = self._read_registry()
        records = self.list_materials()
        return ProjectInfo(
            project_id=self.project_id,
            created_at=str(data.get("created_at", "")),
            material_count=sum(1 for r in records if r.current is not None),
            version_count=sum(len(r.versions) for r in records),
            index=self.index_state(),
        )
