"""L1 Library 数据模型：材料版本、材料记录、项目信息与索引状态。

**确定性要求**：材料 id 由规范化正文的 sha256 派生，
不用随机 id、不用入库顺序。同一份材料无论何时、以何文件名入库，material_id 恒定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class SourceType(str, Enum):
    """材料来源类型（政企场景最小枚举）。

    分档依据是**证据效力与自证风险**，不是文件格式或来源目录：
    - ``client_provided``：客户提供（招标文件、现状说明、需求清单）——效力最高，
      「客户自己说的」本身即可引用的事实来源。
    - ``self_produced``：我方产出（方案、报价、产品介绍）——**自证风险区**：
      用我方材料证明我方能力，审计口径上弱于客户材料。
    - ``public``：公开来源（国标行标、政策文件、公开研究）——可外部核验，效力独立于双方。
    - ``other``：未分类兜底，避免入库被枚举卡住。
    """

    CLIENT_PROVIDED = "client_provided"
    SELF_PRODUCED = "self_produced"
    PUBLIC = "public"
    OTHER = "other"

    @classmethod
    def parse(cls, value: object) -> "SourceType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for item in cls:
                if item.value == value:
                    return item
        raise ValueError(
            f"unknown source_type {value!r}; expected one of {[i.value for i in cls]!r}"
        )


@dataclass(frozen=True)
class MaterialVersion:
    """一条材料的一个版本。``material_id`` 即规范化正文哈希（内容派生）。"""

    material_id: str
    original_filename: str
    stored_filename: str
    source_type: str
    ingested_at: str
    text_length: int
    superseded: bool = False

    def to_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "source_type": self.source_type,
            "ingested_at": self.ingested_at,
            "text_length": self.text_length,
            "superseded": self.superseded,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MaterialVersion":
        return cls(
            material_id=str(data["material_id"]),
            original_filename=str(data["original_filename"]),
            stored_filename=str(data["stored_filename"]),
            source_type=str(data["source_type"]),
            ingested_at=str(data["ingested_at"]),
            text_length=int(data["text_length"]),          # type: ignore[arg-type]
            superseded=bool(data.get("superseded", False)),
        )


@dataclass(frozen=True)
class MaterialRecord:
    """一个**逻辑材料**（按原始文件名归组）及其全部版本。

    同名不同哈希 = 新版本：旧版本保留可查（``superseded=True``），
    **只有当前版本参与检索**。
    """

    name: str
    versions: tuple[MaterialVersion, ...] = field(default_factory=tuple)

    @property
    def current(self) -> MaterialVersion | None:
        for version in self.versions:
            if not version.superseded:
                return version
        return None

    def to_dict(self) -> dict:
        return {"name": self.name, "versions": [v.to_dict() for v in self.versions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MaterialRecord":
        return cls(
            name=str(data["name"]),
            versions=tuple(
                MaterialVersion.from_dict(v) for v in data.get("versions", [])  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True)
class IndexState:
    """索引新旧：比对「上次建索引时的材料集指纹」与「当前材料集指纹」。"""

    material_set_fingerprint: str
    indexed_fingerprint: str | None
    indexed_at: str | None

    @property
    def is_stale(self) -> bool:
        return self.indexed_fingerprint != self.material_set_fingerprint

    def to_dict(self) -> dict:
        return {
            "material_set_fingerprint": self.material_set_fingerprint,
            "indexed_fingerprint": self.indexed_fingerprint,
            "indexed_at": self.indexed_at,
            "stale": self.is_stale,
        }


@dataclass(frozen=True)
class ProjectInfo:
    """项目基本信息（一项目一库）。"""

    project_id: str
    created_at: str
    material_count: int
    version_count: int
    index: IndexState

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at,
            "material_count": self.material_count,
            "version_count": self.version_count,
            "index": self.index.to_dict(),
        }
