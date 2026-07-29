"""逐字段文本谓词：每个 required_field 的确定性判定函数。"""

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


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    value = _normalized(text)
    return any(_normalized(term) in value for term in terms)


def _regex(text: str, pattern: str) -> bool:
    return re.search(pattern, _normalized(text)) is not None


def _basis_reference(text: str) -> bool:
    return _contains_any(
        text,
        ("政策", "规划", "通知", "意见", "标准", "规范", "要求", "文件", "指南"),
    ) or _regex(text, r"(?:项目|平台|系统|业务).{0,8}建设背景|建设背景.{0,8}(?:项目|平台|系统|业务)")


def _project_relationship(text: str) -> bool:
    return _contains_any(
        text,
        ("本项目", "当前项目", "本次项目", "本次建设", "当前建设", "项目建设"),
    ) or _regex(
        text,
        r"(?:用于|适用于|支撑|指导|落实|依据|要求).{0,18}(?:项目|建设|平台|系统|业务)",
    )


_CURRENT_STATE_FUTURE_MARKERS = (
    "未来",
    "将建设",
    "将实现",
    "计划",
    "拟",
    "预计",
    "目标是",
)

_CURRENT_STATE_PROMOTIONAL_CAPABILITY_RE = re.compile(
    r"(?:产品|平台|系统).{0,18}(?:支持|可实现|能够|可提供|适用于|面向.{0,8}场景)"
)

_CURRENT_STATE_PREDICATE_RE = re.compile(
    r"(?:传统|原有|现行|日常|线下|人工|现实).{0,28}"
    r"(?:依赖|采用|需要|仍需|存在|很难|难以|无法|未能|受限|周期长|成本高|"
    r"以.{1,16}为中心)|"
    r"(?:现阶段|实际运行中|已经|已具备|已建成|已部署|尚未|仍然).{0,28}"
    r"(?:采用|依赖|运行|具备|建成|部署|实现|存在|缺少|受限)"
)


def _current_state(text: str) -> bool:
    if _contains_any(
        text,
        (
            "现状",
            "当前",
            "目前",
            "现有",
            "既有",
            "当前状态",
            "现阶段",
            "实际运行中",
        ),
    ):
        return True
    clauses = tuple(
        item.strip()
        for item in re.split(r"[。！？；;\n]+", _normalized(text))
        if item.strip()
    )
    for clause in clauses:
        if _contains_any(clause, _CURRENT_STATE_FUTURE_MARKERS):
            continue
        if _CURRENT_STATE_PROMOTIONAL_CAPABILITY_RE.search(clause):
            continue
        if _CURRENT_STATE_PREDICATE_RE.search(clause):
            return True
    return False


def _specific_problem(text: str) -> bool:
    return _contains_any(
        text,
        ("具体问题", "问题", "痛点", "缺口", "不足", "分散", "无法", "难以", "需求", "亟需"),
    )


def _affected_subject(text: str) -> bool:
    return _contains_any(
        text,
        ("业务", "系统", "流程", "环节", "部门", "用户", "客户", "人员", "数据", "平台", "对象", "单位"),
    )


def _construction_subject(text: str) -> bool:
    return _contains_any(text, ("建设对象", "服务对象")) or _regex(
        text,
        r"面向.{0,20}(?:部门|单位|用户|客户|政府|企业|人员|区域|组织)",
    )


def _explicit_scope(text: str) -> bool:
    if _contains_any(
        text,
        ("业务范围", "系统范围", "数据范围", "组织范围", "地域范围", "全市", "市级", "区县", "省级"),
    ):
        return True
    return _regex(
        text,
        r"覆盖(?!相关|各类).{1,24}(?:业务|系统|数据|区域|部门|单位)|"
        r"(?:交通|政务|应急|审批|监管|医疗|教育|能源|园区).{0,8}(?:业务|系统|数据)",
    )


def _basic_boundary(text: str) -> bool:
    return _regex(
        text,
        r"(?:范围|边界)(?:为|包括|包含|覆盖|限于|：|:).{1,30}|"
        r"(?:仅限|不含|除外|从.{1,15}到)|"
        r"覆盖(?!相关|各类).{2,30}",
    )


def _target_state(text: str) -> bool:
    return _regex(text, r"(?:实现|达到|形成|建成|完成|提升|降低|缩短|减少|确保|打造).{2,40}")


def _objective_alignment(text: str) -> bool:
    return _regex(
        text,
        r"(?:解决|针对|满足|对应|支撑|面向).{0,24}(?:问题|需求|业务|对象|目标|能力|流程|系统)",
    )


def _concrete_capability(text: str) -> bool:
    return _contains_any(text, ("建设内容", "功能模块", "能力模块")) or _regex(
        text,
        r"(?:建设|部署|开发|配置|提供|包含|包括).{0,24}(?:模块|功能|平台|系统|能力|服务)",
    )


def _capability_alignment(text: str) -> bool:
    return _regex(
        text,
        r"(?:用于|支撑|解决|满足|对应|服务于|面向).{0,28}(?:需求|目标|业务|对象|流程|问题|管理|用户)",
    )


def _implementation_condition(text: str) -> bool:
    category = _contains_any(
        text,
        ("数据", "接口", "系统", "部署", "网络", "算力", "组织", "人员", "环境", "能力", "适用", "边界"),
    )
    condition = _contains_any(
        text,
        ("需", "必须", "应当", "前提", "条件", "限制", "边界", "仅", "不支持", "依赖", "对接", "接入", "部署于", "运行在", "至少", "不得", "不含", "适用于"),
    )
    return category and condition


_CASE_SUBJECT_RE = re.compile(
    r"(?:项目|客户|实施对象)(?:为|是|：|:)\s*[^，。；;\n]{2,30}|"
    r"(?:在|面向)[^，。；;\n]{2,30}(?:项目|客户|局|院|公司|集团|中心|单位)(?:中|实施|部署)|"
    r"[^，。；;\n]{2,24}(?:局|院|公司|集团|中心|单位|项目)"
)


def _case_subject(text: str) -> bool:
    matches = _CASE_SUBJECT_RE.findall(_normalized(text))
    return any(
        not _contains_any(match, ("丰富项目", "案例项目", "各类项目", "众多客户"))
        for match in matches
    )


def _implementation_content(text: str) -> bool:
    return _contains_any(text, ("实施内容", "建设内容")) or _regex(
        text,
        r"(?:实施了|建设了|部署了|接入了|交付了|完成了).{1,30}(?:平台|系统|模块|功能|服务|数据|能力)",
    )


def _implementation_status(text: str) -> bool:
    return _contains_any(
        text,
        ("已上线", "已经上线", "通过验收", "已验收", "投入使用", "已投产", "已交付", "实际运行", "完成实施", "已应用", "广泛应用"),
    )


def _outcome(text: str) -> bool:
    return _contains_any(text, ("降本增效",)) or _regex(
        text,
        r"(?:提升|降低|缩短|减少|节省|增加|改善|实现).{1,32}",
    )


def _outcome_subject_or_basis(text: str) -> bool:
    return _contains_any(
        text,
        ("来源", "口径", "根据", "报告", "统计", "监测", "测算", "验收", "业务", "流程", "事件", "告警", "用户", "系统", "项目", "能耗", "成本", "办理", "处置", "响应", "准确率", "满意度", "吞吐量"),
    )


def _outcome_nature(text: str) -> bool:
    return _contains_any(
        text,
        ("预期", "预计", "目标", "计划", "实际", "实测", "已实现", "截至", "验收结果", "统计结果"),
    ) or _regex(text, r"(?:建成|上线|投产|投入使用|试运行|运行).{0,8}后")
