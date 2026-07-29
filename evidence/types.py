"""证据层数据契约。

``EvidenceSlot``、``EvidenceContract`` 与 ``SlotStatus`` 定义固定契约，
``VectorCandidate`` 表示尚未验证的候选，``EvidenceSpan`` 表示可从规范化材料精确回读的证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional

from evidence.core.types import RetrievalPath

if TYPE_CHECKING:
    from .documents import NormalizedDocument


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
    return value


def _string_tuple(
    values: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings; got {values!r}")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of strings; got {values!r}") from exc
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must contain at least one item")
    for index, item in enumerate(result):
        _non_empty_string(item, f"{field_name}[{index}]")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate items; got {result!r}")
    return result


class SlotStatus(str, Enum):
    """Evidence Slot 的完整且封闭的状态集合。"""

    FILLED = "filled"
    WEAK = "weak"
    MISSING = "missing"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class EvidenceSlot:
    """固定 Contract 中一个不可变的证据要求。"""

    slot_id: str
    name: str
    description: str
    required: bool
    minimum_evidence: int
    required_fields: tuple[str, ...]
    weak_conditions: tuple[str, ...]
    material_request: str

    def __post_init__(self) -> None:
        _non_empty_string(self.slot_id, "EvidenceSlot.slot_id")
        _non_empty_string(self.name, f"EvidenceSlot[{self.slot_id}].name")
        _non_empty_string(self.description, f"EvidenceSlot[{self.slot_id}].description")
        if type(self.required) is not bool:
            raise ValueError(
                f"EvidenceSlot[{self.slot_id}].required must be bool; got {self.required!r}"
            )
        if type(self.minimum_evidence) is not int or self.minimum_evidence < 1:
            raise ValueError(
                f"EvidenceSlot[{self.slot_id}].minimum_evidence must be an integer >= 1; "
                f"got {self.minimum_evidence!r}"
            )
        object.__setattr__(
            self,
            "required_fields",
            _string_tuple(
                self.required_fields,
                f"EvidenceSlot[{self.slot_id}].required_fields",
            ),
        )
        object.__setattr__(
            self,
            "weak_conditions",
            _string_tuple(
                self.weak_conditions,
                f"EvidenceSlot[{self.slot_id}].weak_conditions",
            ),
        )
        _non_empty_string(
            self.material_request,
            f"EvidenceSlot[{self.slot_id}].material_request",
        )


@dataclass(frozen=True)
class EvidenceContract:
    """不可变的显式 Evidence Contract；不支持运行时增删 Slot。"""

    contract_id: str
    version: str
    name: str
    description: str
    slots: tuple[EvidenceSlot, ...]
    allow_partial: bool
    citation_required: bool
    risk_level: str
    _slots_by_id: Mapping[str, EvidenceSlot] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _non_empty_string(self.contract_id, "EvidenceContract.contract_id")
        _non_empty_string(self.version, "EvidenceContract.version")
        _non_empty_string(self.name, "EvidenceContract.name")
        _non_empty_string(self.description, "EvidenceContract.description")
        _non_empty_string(self.risk_level, "EvidenceContract.risk_level")
        if type(self.allow_partial) is not bool:
            raise ValueError(
                f"EvidenceContract.allow_partial must be bool; got {self.allow_partial!r}"
            )
        if type(self.citation_required) is not bool:
            raise ValueError(
                "EvidenceContract.citation_required must be bool; "
                f"got {self.citation_required!r}"
            )
        if isinstance(self.slots, (str, bytes)):
            raise ValueError("EvidenceContract.slots must be a sequence of EvidenceSlot objects")
        try:
            slots = tuple(self.slots)
        except TypeError as exc:
            raise ValueError(
                "EvidenceContract.slots must be a sequence of EvidenceSlot objects"
            ) from exc
        if not slots:
            raise ValueError("EvidenceContract.slots must contain at least one slot")
        for index, slot in enumerate(slots):
            if not isinstance(slot, EvidenceSlot):
                raise ValueError(
                    f"EvidenceContract.slots[{index}] must be EvidenceSlot; got {slot!r}"
                )
        slot_ids = [slot.slot_id for slot in slots]
        duplicates = sorted({slot_id for slot_id in slot_ids if slot_ids.count(slot_id) > 1})
        if duplicates:
            raise ValueError(
                f"EvidenceContract.slots contains duplicate slot_id values: {duplicates!r}"
            )
        object.__setattr__(self, "slots", slots)
        object.__setattr__(
            self,
            "_slots_by_id",
            MappingProxyType({slot.slot_id: slot for slot in slots}),
        )

    @property
    def slot_ids(self) -> tuple[str, ...]:
        """按 Contract 声明顺序返回稳定 Slot ID。"""

        return tuple(slot.slot_id for slot in self.slots)

    def get_slot(self, slot_id: str) -> EvidenceSlot:
        """按 ID 返回 Slot；未知 ID 明确抛 ``KeyError``。"""

        try:
            return self._slots_by_id[slot_id]
        except KeyError:
            raise KeyError(
                f"unknown slot_id {slot_id!r} for contract {self.contract_id!r}"
            ) from None


@dataclass(frozen=True)
class VectorCandidate:
    """Per-slot 语义发现候选；它不是已经定位或验证的证据。

    ``distance`` 保留向量后端的原始距离方向，不重命名为 similarity score。
    字符和行区间来自 canonical ``LosslessChunk``，用于下一步 Exact Locator
    限定候选范围；调用方不得把本对象直接冒充 :class:`EvidenceSpan`。
    """

    slot_id: str
    query: str
    document_id: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    rank: int
    distance: float | None
    discovery_source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "slot_id",
            "query",
            "document_id",
            "chunk_id",
            "text",
            "discovery_source",
        ):
            _non_empty_string(
                getattr(self, field_name),
                f"VectorCandidate.{field_name}",
            )
        for field_name in ("start_char", "end_char", "start_line", "end_line", "rank"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise ValueError(
                    f"VectorCandidate.{field_name} must be an integer; got {value!r}"
                )
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError(
                "VectorCandidate character range must satisfy "
                f"0 <= start_char < end_char; got {self.start_char}, {self.end_char}"
            )
        if self.end_char - self.start_char != len(self.text):
            raise ValueError(
                "VectorCandidate text length must equal end_char - start_char; "
                f"got len(text)={len(self.text)}, range={self.end_char - self.start_char}"
            )
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError(
                "VectorCandidate line range must satisfy 1 <= start_line <= end_line; "
                f"got {self.start_line}, {self.end_line}"
            )
        if self.rank < 1:
            raise ValueError(f"VectorCandidate.rank must be >= 1; got {self.rank}")
        if self.distance is not None:
            if isinstance(self.distance, bool) or not isinstance(self.distance, Real):
                raise ValueError(
                    "VectorCandidate.distance must be None or a finite number; "
                    f"got {self.distance!r}"
                )
            distance = float(self.distance)
            if not math.isfinite(distance):
                raise ValueError(
                    "VectorCandidate.distance must be finite; "
                    f"got {self.distance!r}"
                )
            object.__setattr__(self, "distance", distance)
        if not isinstance(self.metadata, Mapping):
            raise ValueError(
                f"VectorCandidate.metadata must be a mapping; got {self.metadata!r}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, init=False)
class EvidenceSpan:
    """只能通过 :meth:`from_document` 创建的规范原文证据区间。

    字符偏移为 0-based 左闭右开，行号为 1-based 且首尾均包含。
    provenance 指向 ``NormalizedDocument.text``；PDF/DOCX 原始页码映射由
    runtime adapter 提供。禁用公开构造器，避免调用方传入与原文不一致的 ``text``。
    """

    evidence_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    page: int | None
    discovery_sources: tuple[str, ...]
    metadata: Mapping[str, object]

    def __init__(self) -> None:
        raise TypeError("EvidenceSpan must be created with EvidenceSpan.from_document()")

    @classmethod
    def from_document(
        cls,
        document: "NormalizedDocument",
        start_char: int,
        end_char: int,
        *,
        discovery_sources: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
        page: int | None = None,
    ) -> "EvidenceSpan":
        """校验范围并从规范化文档精确回读文本、行号与稳定 ID。"""

        from .documents import NormalizedDocument

        if not isinstance(document, NormalizedDocument):
            raise ValueError(
                f"document must be NormalizedDocument; got {document!r}"
            )
        if type(start_char) is not int or type(end_char) is not int:
            raise ValueError(
                "EvidenceSpan character offsets must be integers; "
                f"got start_char={start_char!r}, end_char={end_char!r}"
            )
        if start_char >= end_char:
            raise ValueError(
                "EvidenceSpan requires start_char < end_char; "
                f"got start_char={start_char}, end_char={end_char}"
            )
        text = document.slice(start_char, end_char)
        start_line, end_line = document.line_range(start_char, end_char)
        sources = _string_tuple(
            discovery_sources,
            "EvidenceSpan.discovery_sources",
            allow_empty=True,
        )
        if page is not None and (type(page) is not int or page < 1):
            raise ValueError(f"EvidenceSpan.page must be None or an integer >= 1; got {page!r}")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(
                f"EvidenceSpan.metadata must be a mapping; got {metadata!r}"
            )
        digest_input = (
            f"{document.document_id}\0{document.normalized_content_hash}\0"
            f"{start_char}\0{end_char}"
        ).encode("utf-8")
        evidence_id = f"ev-{hashlib.sha256(digest_input).hexdigest()}"
        span_metadata = dict(metadata or {})
        span_metadata.update(
            {
                "source_path": document.source_path,
                "normalized_path": document.normalized_path,
                "source_content_hash": document.source_content_hash,
                "normalized_content_hash": document.normalized_content_hash,
                "span_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

        instance = object.__new__(cls)
        object.__setattr__(instance, "evidence_id", evidence_id)
        object.__setattr__(instance, "document_id", document.document_id)
        object.__setattr__(instance, "text", text)
        object.__setattr__(instance, "start_char", start_char)
        object.__setattr__(instance, "end_char", end_char)
        object.__setattr__(instance, "start_line", start_line)
        object.__setattr__(instance, "end_line", end_line)
        object.__setattr__(instance, "page", page)
        object.__setattr__(instance, "discovery_sources", sources)
        object.__setattr__(instance, "metadata", MappingProxyType(span_metadata))
        return instance
