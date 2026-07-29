"""宣传语分析：有限宣传表达识别。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import unicodedata

from ..contract import solution_material_completeness_v1
from ..types import EvidenceContract, EvidenceSpan, SlotStatus
from .results import _freeze_json, _non_empty, _stable_strings

from .predicates import _contains_any, _normalized, _regex


PROMOTIONAL_PHRASES = (
    "行业领先",
    "国内领先",
    "全面赋能",
    "显著提升",
    "大幅降低",
    "一流平台",
    "丰富案例",
    "广泛应用",
    "全面支持",
    "高效协同",
    "全面提升",
    "领先能力",
    "先进平台",
    "良好扩展性",
)

_WEAK_RELEVANCE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "background_and_basis": ("响应国家战略", "顺应行业趋势", "政策大力支持"),
        "current_state_and_needs": ("存在短板", "亟须提升", "需要加强"),
        "scope_and_subjects": ("面向政府客户", "覆盖相关业务", "服务各类用户"),
        "objectives": ("全面提升", "打造一流", "形成领先能力", "建设先进平台"),
        "solution_content_and_capabilities": (
            "人工智能",
            "大数据",
            "云计算",
            "技术名词",
            "产品领先",
        ),
        "implementation_conditions_and_boundaries": (
            "灵活部署",
            "支持多种环境",
            "良好扩展性",
        ),
        "reference_cases_and_proof": (
            "客户名称",
            "logo",
            "案例标题",
            "丰富案例",
            "广泛应用",
        ),
        "outcomes_and_metrics": ("降本增效", "提升治理能力"),
    }
)

_CONCRETE_MARKERS = (
    "项目",
    "客户",
    "系统",
    "平台",
    "模块",
    "功能",
    "业务",
    "流程",
    "部门",
    "用户",
    "数据",
    "接口",
    "上线",
    "验收",
    "部署",
    "范围",
    "来源",
    "口径",
    "预期",
    "实际",
)


def _promotional_analysis(text: str) -> tuple[tuple[str, ...], bool]:
    phrases = tuple(phrase for phrase in PROMOTIONAL_PHRASES if phrase in text)
    if not phrases:
        return (), False
    residue = _normalized(text)
    for phrase in phrases:
        residue = residue.replace(_normalized(phrase), "")
    residue = re.sub(r"[\W_]+", "", residue, flags=re.UNICODE)
    concrete = _contains_any(text, _CONCRETE_MARKERS) or bool(
        re.search(r"\d+(?:\.\d+)?(?:%|倍|万元|元|个月|天|小时|分钟|秒)", _normalized(text))
    )
    return phrases, not concrete and len(residue) <= 12
