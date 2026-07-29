"""L6 生成编排数据结构：任务包 SectionTask 与回稿 SectionDraft。

**手段无关红线**：任务包里的证据只带逐字 `content` + 去标识 `source`（源自 GenerationBrief，
已过 `assert_outward_clean`），不带 distance/chunk_id/embedding/query/hash/char-offset。
`span_ref` 是 brief 发出的引用句柄（如 ``e1``），**不是** 内部 evidence_id。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TaskEvidence:
    """任务包里的一条证据：逐字内容 + 去标识来源 + 引用句柄。"""

    span_ref: str            # brief 发出的引用句柄（e1/e2…），回稿用它绑定段落
    content: str             # 逐字原文
    source: str              # 去标识来源串（《标题》· 第X页 / 标题层级）
    supports_fields: tuple[str, ...] = ()
    slot_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "span_ref": self.span_ref,
            "content": self.content,
            "source": self.source,
            "supports_fields": list(self.supports_fields),
            "slot_id": self.slot_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TaskEvidence":
        return cls(
            span_ref=str(data["span_ref"]),
            content=str(data["content"]),
            source=str(data["source"]),
            supports_fields=tuple(str(x) for x in data.get("supports_fields", [])),  # type: ignore[union-attr]
            slot_id=None if data.get("slot_id") is None else str(data["slot_id"]),
        )


@dataclass(frozen=True)
class SectionTask:
    """一个 allowed/limited section 的任务包。"""

    section_id: str
    section_title: str
    gate: str                                   # allow / limited
    gate_meaning: str                           # 人读说明
    writing: Mapping[str, object]               # S3 迁入：structure / tone / length_basis
    project_params: Mapping[str, object]        # ProjectSpec 参数
    evidence: tuple[TaskEvidence, ...]
    missing_fields: tuple[str, ...]             # 无证据字段 → 回稿须占位、不得编造
    constraints: tuple[str, ...]                # S4 迁入固定约束段
    response_format: Mapping[str, object]       # 回稿格式要求
    limited_note_required: bool = False
    # 证据里的 prompt-injection 告警（确定性扫描，只标注不改材料）。
    injection_warnings: tuple[Mapping[str, object], ...] = ()

    def evidence_refs(self) -> frozenset[str]:
        return frozenset(e.span_ref for e in self.evidence)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "gate": self.gate,
            "gate_meaning": self.gate_meaning,
            "writing": dict(self.writing),
            "project_params": dict(self.project_params),
            "evidence": [e.to_dict() for e in self.evidence],
            "missing_fields": list(self.missing_fields),
            "constraints": list(self.constraints),
            "response_format": dict(self.response_format),
            "limited_note_required": self.limited_note_required,
            "injection_warnings": [dict(w) for w in self.injection_warnings],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SectionTask":
        return cls(
            section_id=str(data["section_id"]),
            section_title=str(data["section_title"]),
            gate=str(data["gate"]),
            gate_meaning=str(data.get("gate_meaning", "")),
            writing=dict(data.get("writing", {})),  # type: ignore[arg-type]
            project_params=dict(data.get("project_params", {})),  # type: ignore[arg-type]
            evidence=tuple(TaskEvidence.from_dict(e) for e in data.get("evidence", [])),  # type: ignore[union-attr]
            missing_fields=tuple(str(x) for x in data.get("missing_fields", [])),  # type: ignore[union-attr]
            constraints=tuple(str(x) for x in data.get("constraints", [])),  # type: ignore[union-attr]
            response_format=dict(data.get("response_format", {})),  # type: ignore[arg-type]
            limited_note_required=bool(data.get("limited_note_required", False)),
            injection_warnings=tuple(dict(w) for w in data.get("injection_warnings", [])),  # type: ignore[union-attr]
        )


@dataclass(frozen=True)
class DraftParagraph:
    """回稿的一个段落：正文 + 绑定的 span_ref。占位段落 span_refs 可为空。

    **稿件-审计绑定**：``paragraph_id``（稳定 id，如 ``02#p01``）与 ``content_hash``
    （正文内容哈希）由**编排器在收货接受时确定性写入**（非 response 字段；完整性指纹而非身份凭证），
    随 state 落盘。审计单的 claim 据此绑定到确切段落、复核时按当前正文重算比对（见
    :mod:`evidence.integrity`）。agent 回稿可不带这两字段（默认 ``None``，收货后被盖章）。
    """

    text: str
    span_refs: tuple[str, ...] = ()
    paragraph_id: str | None = None
    content_hash: str | None = None

    @property
    def is_placeholder(self) -> bool:
        return "【待补充" in self.text

    def to_dict(self) -> dict:
        out: dict = {"text": self.text, "span_refs": list(self.span_refs)}
        if self.paragraph_id is not None:
            out["paragraph_id"] = self.paragraph_id
        if self.content_hash is not None:
            out["content_hash"] = self.content_hash
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DraftParagraph":
        return cls(
            text=str(data["text"]),
            span_refs=tuple(str(x) for x in data.get("span_refs", [])),  # type: ignore[union-attr]
            paragraph_id=None if data.get("paragraph_id") is None else str(data["paragraph_id"]),
            content_hash=None if data.get("content_hash") is None else str(data["content_hash"]),
        )


@dataclass(frozen=True)
class SectionDraft:
    """执行 agent 交回的一份 section 回稿。"""

    section_id: str
    paragraphs: tuple[DraftParagraph, ...]
    limited_note: str | None = None

    def cited_refs(self) -> frozenset[str]:
        return frozenset(r for p in self.paragraphs for r in p.span_refs)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "limited_note": self.limited_note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SectionDraft":
        raw = data.get("paragraphs", [])
        if isinstance(raw, (str, bytes)):
            raise ValueError("paragraphs must be a list")
        return cls(
            section_id=str(data["section_id"]),
            paragraphs=tuple(DraftParagraph.from_dict(p) for p in raw),  # type: ignore[union-attr]
            limited_note=None if data.get("limited_note") is None else str(data["limited_note"]),
        )
