"""L0 Intake 数据结构：``ProjectSpec`` 是用户任务到系统契约的翻译产物。

**它在架构里的位置**：L0 是**产品入口**（用户带着一个任务来），此前的
`--contract + 材料` 是**工程师入口**。`ProjectSpec` 是这两者之间的翻译层的**产物**：
一份结构化、可校验、可落盘、可复用的任务说明，由它确定性地驱动契约选择与后续全链。

**三个面**：

- **任务面**：`doc_type`（枚举 → 确定性映射到 contract/crosswalk）、`project_name`、
  `client_name`（甲方/采购人）、`industry_domain`（行业域）、`section_scope` + `section_ids`
  （全量 / 指定章节）。
- **材料面**：`project_id`（关联 L1 材料库）、`source_types`（来源类型过滤偏好，可选）。
- **参数面**：只保留生成时需要的 `project_number`、`budget` 和 `duration`。

**确定性红线**：`doc_type → (contract, crosswalk)` 是**代码查表**（见
:mod:`evidence.intake.registry`），不是 agent/LLM 猜。参数面字段本引擎**不消费**
（Engine 全程 model-free），仅随 spec 落盘并交给下游生成层使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from ..library import SourceType

SPEC_VERSION = "p-b-projectspec-v1"


class DocType(str, Enum):
    """要生成的**输出文档类型**（映射到已有 contract/crosswalk）。

    当前仅一个成员（对应现有契约）。注册表结构支持多 doc_type：将来加一种输出文档 =
    在 :mod:`evidence.intake.registry` 注册一行 + 建一份 crosswalk，本枚举加一个成员。

    **与 drafter S1 的 ``project_type`` 不是一回事**：S1 的 project_type
    （engineering/platform/research/planning/other）分的是**项目性质**、选的是 drafter 内部
    outline 骨架；本 doc_type 分的是**产出文档类型**、选的是本引擎的证据契约。两者是不同的轴，
    project_type 作为可选描述字段另存（见 ``project_nature``），**不参与契约选择**。
    """

    PROPOSAL_DOCUMENT = "proposal-document"

    @classmethod
    def parse(cls, value: object) -> "DocType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for item in cls:
                if item.value == value:
                    return item
        raise ValueError(
            f"unknown doc_type {value!r}; expected one of {[i.value for i in cls]!r}"
        )


class SectionScope(str, Enum):
    """章节范围偏好：全量 crosswalk 章节，或用户指定子集。"""

    ALL = "all"
    CUSTOM = "custom"

    @classmethod
    def parse(cls, value: object) -> "SectionScope":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for item in cls:
                if item.value == value:
                    return item
        raise ValueError(
            f"unknown section_scope {value!r}; expected one of {[i.value for i in cls]!r}"
        )


@dataclass(frozen=True)
class ProjectSpec:
    """一次任务入口的结构化说明。字段全部可校验；可序列化落盘、复用、修改。"""

    # —— 任务面 ——
    doc_type: str
    project_name: str
    client_name: str
    project_id: str
    section_scope: str = SectionScope.ALL.value
    section_ids: tuple[str, ...] = field(default_factory=tuple)
    industry_domain: str | None = None
    # —— 材料面 ——
    source_types: tuple[str, ...] | None = None
    # —— 参数面（S1 精选；Engine 不消费，随 spec 落盘交下游生成层）——
    project_number: str | None = None
    budget: str | None = None
    duration: str | None = None
    project_nature: str | None = None   # drafter S1 project_type，描述用，不参与契约选择
    spec_version: str = SPEC_VERSION

    def to_dict(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "doc_type": self.doc_type,
            "project_name": self.project_name,
            "client_name": self.client_name,
            "industry_domain": self.industry_domain,
            "section_scope": self.section_scope,
            "section_ids": list(self.section_ids),
            "project_id": self.project_id,
            "source_types": None if self.source_types is None else list(self.source_types),
            "project_number": self.project_number,
            "budget": self.budget,
            "duration": self.duration,
            "project_nature": self.project_nature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectSpec":
        def _opt_str(key: str) -> str | None:
            value = data.get(key)
            return None if value is None else str(value)

        def _seq(key: str) -> tuple[str, ...]:
            value = data.get(key) or ()
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{key} must be a list, not a string")
            return tuple(str(item) for item in value)  # type: ignore[union-attr]

        raw_source_types = data.get("source_types")
        source_types: tuple[str, ...] | None
        if raw_source_types is None:
            source_types = None
        elif isinstance(raw_source_types, (str, bytes)):
            raise ValueError("source_types must be a list, not a string")
        else:
            source_types = tuple(str(item) for item in raw_source_types)

        return cls(
            doc_type=str(data["doc_type"]),
            project_name=str(data["project_name"]),
            client_name=str(data["client_name"]),
            project_id=str(data["project_id"]),
            section_scope=str(data.get("section_scope", SectionScope.ALL.value)),
            section_ids=_seq("section_ids"),
            industry_domain=_opt_str("industry_domain"),
            source_types=source_types,
            project_number=_opt_str("project_number"),
            budget=_opt_str("budget"),
            duration=_opt_str("duration"),
            project_nature=_opt_str("project_nature"),
            spec_version=str(data.get("spec_version", SPEC_VERSION)),
        )

    def resolved_source_types(self) -> tuple[SourceType, ...] | None:
        """把 source_types 偏好解析成 SourceType（None = 不过滤）。"""

        if self.source_types is None:
            return None
        return tuple(SourceType.parse(item) for item in self.source_types)
