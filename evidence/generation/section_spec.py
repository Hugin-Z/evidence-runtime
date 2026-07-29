"""每个 section 的写作要求（结构、基调与篇幅）。

drafter S3 生成 outline 的机制（读 scoring_matrix + section_anchors、覆盖度自检表、
撰写指引行号）**舍弃**——section 列表由 crosswalk 给定，证据由 L1–L4 给定。真正迁入的是
**每 section 的写作要求本身**。存为 `evidence/contracts/section_specs/<doc_type>.json`，可审可改。
没有 spec 的 section 落到通用默认（不硬失败）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..resources import section_specs_dir as _section_specs_dir

_SPEC_DIR = _section_specs_dir()

_GENERIC = {
    "structure": [],
    "tone": "以材料为准撰写，证据不足处留白，不编造。",
    "length_basis": "按证据量适度。",
}


@dataclass(frozen=True)
class SectionWritingSpec:
    """一个 section 的写作要求。"""

    section_id: str
    structure: tuple[str, ...] = ()
    tone: str = _GENERIC["tone"]
    length_basis: str = _GENERIC["length_basis"]

    def to_dict(self) -> dict:
        return {
            "structure": list(self.structure),
            "tone": self.tone,
            "length_basis": self.length_basis,
        }


class SectionSpecError(ValueError):
    """section_specs 配置无效。"""


def load_section_specs(doc_type: str) -> dict[str, SectionWritingSpec]:
    """按 doc_type 加载 section 写作要求。缺文件 → 空表（全部落通用默认）。"""

    path = _SPEC_DIR / f"{doc_type}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SectionSpecError(f"corrupt section spec {path}: {exc}") from exc
    sections = data.get("sections")
    if not isinstance(sections, Mapping):
        raise SectionSpecError(f"{path.name}: sections 必须是对象")
    out: dict[str, SectionWritingSpec] = {}
    for sid, spec in sections.items():
        if not isinstance(spec, Mapping):
            raise SectionSpecError(f"{path.name}: sections.{sid} 必须是对象")
        out[str(sid)] = SectionWritingSpec(
            section_id=str(sid),
            structure=tuple(str(x) for x in spec.get("structure", [])),
            tone=str(spec.get("tone", _GENERIC["tone"])),
            length_basis=str(spec.get("length_basis", _GENERIC["length_basis"])),
        )
    return out


def spec_for(specs: Mapping[str, SectionWritingSpec], section_id: str) -> SectionWritingSpec:
    """取某 section 的写作要求；无则通用默认。"""

    return specs.get(section_id) or SectionWritingSpec(section_id=section_id)
