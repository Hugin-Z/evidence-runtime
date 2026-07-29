"""生成任务的固定撰写约束与收货边界。

任务模板负责把证据优先、缺料占位、禁止包外材料等要求交给 writer；收货校验只执行
引用存在性、逐字一致、辖区、占位和段落绑定等代码可判的硬约束。排版由渲染层处理，
写作质量与语义支持度由 Claim Audit 处理。
"""
from __future__ import annotations

from .task import SectionTask

# —— 任务模板固定段（写给执行 agent 的约束文本；每个任务包原样带上）——
SECTION_CONSTRAINTS: tuple[str, ...] = (
    "证据优先：每个陈述事实的段落必须绑定至少一个 span_ref，且只能引用本任务包 evidence[] 里的条目。",
    "不得引入任务包外的任何材料、事实、数字、案例或引用——包外即视为编造。",
    "缺证据的字段（见 missing_fields）必须留 `【待补充：字段说明】` 占位，严禁用训练数据或常识补全。",
    "不脑补、不扩展用户字面输入、不为“完整性”自行加配套内容；宁缺毋滥。",
    "不虚构业绩/客户/金额/证书等硬数据；材料没有就留占位。",
    "履约前置条件（配套/环境/第三方协作）不写免责句式，写成投标方主动核查+承担。",
    "正文用自然语言，不出现工具链内部字段名或标识符。",
)

# limited section 额外约束
LIMITED_CONSTRAINT = (
    "本 section 的 gate=limited：证据可用但不足。必须在 limited_note 里**明确标注哪里不足**"
    "（哪些方面证据偏弱/单一来源），不得当作证据充分来写。"
)


def task_constraints(task: SectionTask) -> tuple[str, ...]:
    """一个任务包对应的固定约束段（含 limited 追加）。"""

    items = list(SECTION_CONSTRAINTS)
    if task.gate == "limited":
        items.append(LIMITED_CONSTRAINT)
    return tuple(items)


def receiving_discipline_note() -> str:
    """收货校验边界说明：只验硬约束，不判语义强弱。"""

    return (
        "收货校验只验代码可判的硬约束（引用存在性/逐字一致/辖区/占位/不足标注/段落绑定）；"
        "写得好不好、论证是否有力属语义判断，归审计层 L7（Claim Audit），收货不碰。"
    )
