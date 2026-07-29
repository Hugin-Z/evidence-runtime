"""量化角色分类：数值表达的角色判定与量化声明抽取。"""

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

from .results import DetectedFact
from .predicates import _contains_any, _normalized, _regex
from .promotional import _CONCRETE_MARKERS, PROMOTIONAL_PHRASES


_QUANTITATIVE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|倍|亿元|万元|元|年|个月|月|天|小时|分钟|秒|人|个|台|户|次|gb|tb|mbps)"
    r"(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)

_QUANT_SUPPORT_MARKERS = (
    "来源",
    "根据",
    "报告",
    "统计",
    "监测",
    "测算",
    "数据表明",
    "口径",
    "按",
    "统计周期",
    "预期",
    "预计",
    "目标",
    "计划",
    "实际",
    "实测",
    "已实现",
    "截至",
    "验收",
)

_QUANT_OBJECT_MARKERS = (
    "业务",
    "流程",
    "事件",
    "告警",
    "订单",
    "系统",
    "项目",
    "能耗",
    "成本",
    "办理",
    "处置",
    "响应",
    "识别",
    "准确率",
    "满意度",
    "吞吐量",
    "销售额",
)


class _QuantitativeRole(str, Enum):
    """Private roles for numeric expressions already found by the fixed regex."""

    OUTCOME_CLAIM = "outcome_claim"
    PROJECT_PLAN = "project_plan"
    CONFIGURATION = "configuration"
    TEMPORAL_OR_IDENTIFIER = "temporal_or_identifier"
    ILLUSTRATIVE_EXAMPLE = "illustrative_example"


_QUANT_OUTCOME_CHANGE_RE = re.compile(
    r"(?:提升|降低|减少|缩短|增长|增加|提高|改善|达到).{0,24}"
    r"\d+(?:\.\d+)?\s*(?:%|倍|亿元|万元|元|年|个月|月|天|小时|分钟|秒|人|个|台|户|次|gb|tb|mbps)|"
    r"\d+(?:\.\d+)?\s*(?:%|倍|亿元|万元|元|年|个月|月|天|小时|分钟|秒|人|个|台|户|次|gb|tb|mbps)"
    r".{0,24}(?:提升|降低|减少|缩短|增长|增加|提高|改善|达到)",
    re.IGNORECASE,
)

_QUANT_OUTCOME_ROLE_MARKERS = (
    "覆盖率",
    "成功率",
    "准确率",
    "满意度",
    "成效",
    "效果",
    "运行结果",
    "验收结果",
    "广泛应用",
)

_QUANT_PLAN_MARKERS = (
    "建设周期",
    "项目周期",
    "实施周期",
    "开发周期",
    "工期",
    "预算",
    "投资",
    "计划",
    "拟建",
    "拟于",
    "规划建设",
    "分期",
    "分期建设",
    "工作量",
    "人月",
    "预测值",
    "预测期",
    "定期推送",
    "每月",
    "每季度",
    "每年",
)

_QUANT_CONFIGURATION_ACTIONS = (
    "配置",
    "部署",
    "建设",
    "提供",
    "支持",
    "接入",
    "设置",
    "配备",
    "容量",
    "规格",
    "沉淀",
)

_QUANT_CONFIGURATION_OBJECTS = (
    "服务器",
    "服务节点",
    "节点",
    "数据中心",
    "中心",
    "接口",
    "并发用户",
    "cpu",
    "内存",
    "存储",
    "带宽",
    "设备",
    "实例",
    "模块",
    "系统",
    "场景",
    "模型",
    "数据分类",
    "类别",
    "大类",
)

_QUANT_CONFIGURATION_COUNT_OBJECTS = (
    "服务节点",
    "节点",
    "数据中心",
    "中心",
    "接口",
    "并发用户",
    "设备",
    "实例",
    "模块",
    "系统",
    "场景",
    "数据分类",
    "类别",
    "大类",
)

_QUANT_ALLOCATION_MARKERS = (
    "权重",
    "评分",
    "评分项",
    "评估项",
    "占比",
    "比例",
)

_QUANT_SOURCE_OR_MEASUREMENT_MARKERS = (
    "来源",
    "根据",
    "报告",
    "统计",
    "监测",
    "测算",
    "数据表明",
    "口径",
    "统计周期",
    "实测",
    "验收",
)

_QUANT_EXPECTED_MARKERS = ("预期", "预计", "目标", "计划")
_QUANT_ACTUAL_MARKERS = ("实际", "实测", "已实现", "截至", "验收结果", "统计结果")
_QUANT_ILLUSTRATIVE_INTRO_RE = (
    r"(?:(?:例如|比如|举例|示例)(?:\s|[:：，,])|如\s*[“\"「‘])"
)
_QUANT_CONCRETE_OUTCOME_OBJECT_MARKERS = (
    "事件",
    "告警",
    "订单",
    "能耗",
    "成本",
    "处置",
    "响应",
    "识别",
    "准确率",
    "满意度",
    "吞吐量",
    "销售额",
    "审批",
    "查询时间",
)
_QUANT_CLAUSE_BOUNDARIES = "。！？；;，,\n"


def _quantitative_clause(text: str, start: int, end: int) -> str:
    left = max(text.rfind(boundary, 0, start) for boundary in _QUANT_CLAUSE_BOUNDARIES)
    right_candidates = tuple(
        position
        for boundary in _QUANT_CLAUSE_BOUNDARIES
        if (position := text.find(boundary, end)) >= 0
    )
    right = min(right_candidates, default=len(text))
    return text[left + 1:right]


def _quantitative_role(
    *,
    slot_id: str,
    number: str,
    unit: str,
    clause: str,
    context: str,
) -> tuple[_QuantitativeRole, tuple[str, ...]]:
    """Classify one already-detected expression with conservative P2 priority."""

    quantitative_value = rf"{re.escape(number)}\s*{re.escape(unit)}"
    illustrative = re.compile(
        _QUANT_ILLUSTRATIVE_INTRO_RE
        + r"[^。！？；;\n]{0,60}"
        + quantitative_value
    )
    if illustrative.search(clause) or illustrative.search(context):
        return (
            _QuantitativeRole.ILLUSTRATIVE_EXAMPLE,
            ("explicit_illustrative_example",),
        )

    outcome_signals: list[str] = []
    if _QUANT_OUTCOME_CHANGE_RE.search(clause):
        outcome_signals.append("outcome_change_predicate")
    if _contains_any(clause, _QUANT_OUTCOME_ROLE_MARKERS):
        outcome_signals.append("explicit_outcome_or_rate_role")
    if outcome_signals:
        return _QuantitativeRole.OUTCOME_CLAIM, tuple(outcome_signals)

    plan_signals: list[str] = []
    if _contains_any(clause, _QUANT_PLAN_MARKERS):
        plan_signals.append("explicit_project_plan_context")
    if _regex(clause, r"(?:分|共).{0,8}\d+(?:\.\d+)?\s*个?阶段"):
        plan_signals.append("planned_stage_count")
    if _contains_any(clause, ("预计", "预期", "目标")) and _contains_any(
        clause,
        ("建设", "完成", "接入", "部署", "配置", "实施"),
    ):
        plan_signals.append("planned_construction_value")
    if unit in {"亿元", "万元", "元"} and _contains_any(
        clause,
        ("项目", "建设", "预算", "投资"),
    ):
        plan_signals.append("project_financial_value")
    if unit in {"年", "个月", "月", "天", "小时"} and _contains_any(
        clause,
        ("周期", "工期", "建设", "实施", "完成"),
    ):
        plan_signals.append("project_duration")
    if plan_signals:
        return _QuantitativeRole.PROJECT_PLAN, tuple(dict.fromkeys(plan_signals))

    configuration_signals: list[str] = []
    if _contains_any(context, _QUANT_ALLOCATION_MARKERS) or _regex(
        context,
        r"\|\s*\d+(?:\.\d+)?\s*%\s*\|",
    ):
        configuration_signals.append("allocation_or_tabular_weight")
    configuration_object = _contains_any(clause, _QUANT_CONFIGURATION_OBJECTS)
    configuration_action = _contains_any(clause, _QUANT_CONFIGURATION_ACTIONS)
    technical_unit = unit in {"gb", "tb", "mbps", "台"}
    bounded_object_count = unit == "个" and _contains_any(
        clause,
        _QUANT_CONFIGURATION_COUNT_OBJECTS,
    )
    configuration_slots = {
        "solution_content_and_capabilities",
        "implementation_conditions_and_boundaries",
        "scope_and_subjects",
        "objectives",
    }
    if configuration_object and (
        configuration_action
        or technical_unit
        or bounded_object_count
        or (slot_id in configuration_slots and unit in {"个", "人", "户"})
    ):
        configuration_signals.append("configured_object_or_capacity")
    if configuration_signals:
        return _QuantitativeRole.CONFIGURATION, tuple(configuration_signals)

    if _regex(context, r"(?:\d\s*){4}年(?:\s*\d+\s*月)?") or _contains_any(
        clause,
        ("版本", "编号", "文号", "日期"),
    ):
        return (
            _QuantitativeRole.TEMPORAL_OR_IDENTIFIER,
            ("calendar_or_identifier_context",),
        )

    # Unknown numeric semantics remain on the strict path.  This avoids
    # relaxing a real outcome merely because the bounded classifier has no
    # positive configuration, plan, or identifier signal.
    return _QuantitativeRole.OUTCOME_CLAIM, ("conservative_unknown_numeric_role",)


def _quantitative_claims(
    span: EvidenceSpan,
    slot_id: str,
) -> tuple[Mapping[str, object], ...]:
    claims: list[Mapping[str, object]] = []
    normalized = _normalized(span.text)
    for match in _QUANTITATIVE_RE.finditer(normalized):
        number = match.group("number")
        unit = match.group("unit").casefold()
        if unit == "年" and number.isdigit() and 1900 <= int(number) <= 2099:
            nearby = normalized[max(0, match.start() - 12): match.end() + 12]
            if not _contains_any(nearby, ("周期", "工期", "时长", "历时")):
                continue
        context = normalized[max(0, match.start() - 40): match.end() + 40]
        clause = _quantitative_clause(normalized, match.start(), match.end())
        role, role_signals = _quantitative_role(
            slot_id=slot_id,
            number=number,
            unit=unit,
            clause=clause,
            context=context,
        )
        markers = tuple(
            marker
            for marker in _QUANT_SUPPORT_MARKERS
            if _normalized(marker) in context
        )
        object_markers = tuple(
            marker
            for marker in _QUANT_OBJECT_MARKERS
            if _normalized(marker) in context
        )
        accepted_markers = markers + object_markers
        source_or_measurement_markers = tuple(
            marker
            for marker in _QUANT_SOURCE_OR_MEASUREMENT_MARKERS
            if _normalized(marker) in context
        )
        expected_markers = tuple(
            marker
            for marker in _QUANT_EXPECTED_MARKERS
            if _normalized(marker) in context
        )
        actual_markers = tuple(
            marker
            for marker in _QUANT_ACTUAL_MARKERS
            if _normalized(marker) in context
        )
        post_state = _regex(
            context,
            r"(?:建成|上线|投产|投入使用|试运行|运行).{0,8}后",
        )
        requires_outcome_support = role is _QuantitativeRole.OUTCOME_CLAIM
        if requires_outcome_support:
            expected_target_is_self_supporting = bool(expected_markers) and (
                unit not in {"%", "倍"}
                or _contains_any(
                    clause,
                    _QUANT_CONCRETE_OUTCOME_OBJECT_MARKERS,
                )
            )
            verified = bool(object_markers) and (
                expected_target_is_self_supporting
                or (
                    bool(source_or_measurement_markers)
                    and (bool(actual_markers) or post_state or bool(markers))
                )
            )
        else:
            verified = True
        claims.append(
            _FrozenJsonDict(
                {
                    "evidence_id": span.evidence_id,
                    "value": f"{number}{unit}",
                    "context": context,
                    "role": role.value,
                    "role_signals": role_signals,
                    "requires_outcome_support": requires_outcome_support,
                    "support_markers": accepted_markers,
                    "source_or_measurement_markers": source_or_measurement_markers,
                    "expected_markers": expected_markers,
                    "actual_markers": actual_markers,
                    "verified": verified,
                }
            )
        )
    return tuple(claims)


_CLAUSE_SPLIT_RE = re.compile(r"[。！？；;\n，,]+")

_RELATION_PATTERNS = (
    ("support", "negative", "不支持", re.compile(r"不支持")),
    ("support", "positive", "支持", re.compile(r"(?<!不)支持")),
    ("online", "negative", "未上线", re.compile(r"(?:未上线|尚未上线)")),
    ("online", "positive", "已上线", re.compile(r"(?:已上线|已经上线)")),
    ("complete", "negative", "未完成", re.compile(r"(?:未完成|尚未完成)")),
    ("complete", "positive", "已完成", re.compile(r"(?:已完成|已经完成)")),
    ("need", "negative", "无需", re.compile(r"(?:无需|不需要)")),
    ("need", "positive", "需要", re.compile(r"(?<!不)需要")),
    ("can", "negative", "不可以", re.compile(r"(?:不可以|不可)")),
    ("can", "positive", "可以", re.compile(r"(?<!不)可以")),
    ("applicable", "negative", "不适用", re.compile(r"不适用")),
    ("applicable", "positive", "适用", re.compile(r"(?<!不)适用")),
    (
        "include",
        "negative",
        "不包含",
        re.compile(r"(?:不包含|未包含|不包括|未包括)"),
    ),
    (
        "include",
        "positive",
        "包含",
        re.compile(r"(?<!不)(?<!未)(?:包含|包括)"),
    ),
)

_SCALAR_RE = re.compile(
    r"(?P<key>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9 _/\-]{1,30}?)\s*"
    r"(?:：|:|为|是)\s*"
    r"(?P<value>\d+(?:\.\d+)?\s*(?:%|倍|亿元|万元|元|年|个月|月|天|小时|分钟|秒|人|个|台|户|次|gb|tb|mbps)?)"
    r"(?=$|[\s，,。；;])",
    re.IGNORECASE,
)
from .results import _FrozenJsonDict
