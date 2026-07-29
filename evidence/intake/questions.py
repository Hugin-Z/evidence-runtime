"""L0 Intake 问题清单：代码定义的结构化 schema。

问题清单是确定性的代码资产（问题 id / 文案 / 类型 / 必填 / 依赖关系）；**问答的执行**
（把这些问题问出去、拿到语义答案）是 agent/人的职责（A1 语义侧）；**收货校验**是代码
（A3，见 :mod:`evidence.intake.validate`）。

问题集合只保留当前 Runtime 与生成任务实际消费的字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..library import SourceType
from .registry import registered_doc_types
from .spec import SectionScope


@dataclass(frozen=True)
class Question:
    """一道 intake 问题。``depends_on`` 满足时才问 / 才必填。"""

    id: str
    text: str
    type: str                                  # enum / string / string_list / project_ref
    required: bool
    choices: tuple[str, ...] = ()              # enum / string_list 的合法取值（空 = 自由）
    depends_on: tuple[str, str] | None = None  # (其它问题 id, 需等于的值)
    help: str = ""

    def is_active(self, answers: Mapping[str, object]) -> bool:
        """依赖是否满足 —— 不满足则本题不问、不校验。"""

        if self.depends_on is None:
            return True
        other_id, required_value = self.depends_on
        return str(answers.get(other_id, "")) == required_value


# 问题顺序 = 交互模式逐问顺序。doc_type 先问（后续依赖它）。
QUESTIONS: tuple[Question, ...] = (
    Question(
        id="doc_type",
        text="要生成什么类型的文档？",
        type="enum",
        required=True,
        choices=registered_doc_types(),
        help="决定选用哪份证据契约（确定性查表，不是猜）。",
    ),
    Question(
        id="project_name",
        text="项目名称？",
        type="string",
        required=True,
    ),
    Question(
        id="client_name",
        text="甲方 / 采购人全称？",
        type="string",
        required=True,
    ),
    Question(
        id="industry_domain",
        text="行业域？（如 数字政务 / 智慧城市 / 住建，可留空）",
        type="string",
        required=False,
    ),
    Question(
        id="project_id",
        text="关联哪个 L1 材料库项目（project_id）？",
        type="project_ref",
        required=True,
        help="材料来自该项目库；须已 `library init` 并入库材料。",
    ),
    Question(
        id="section_scope",
        text="章节范围：全量还是指定章节？",
        type="enum",
        required=False,                            # 留空 = 默认 all（有合理缺省）
        choices=tuple(item.value for item in SectionScope),
        help="all（默认）= crosswalk 全部章节；custom = 只写指定章节。",
    ),
    Question(
        id="section_ids",
        text="指定哪些章节 id？（逗号分隔，须为该 doc_type crosswalk 已有章节）",
        type="string_list",
        required=True,
        depends_on=("section_scope", SectionScope.CUSTOM.value),
    ),
    Question(
        id="source_types",
        text="是否按材料来源过滤候选？（逗号分隔；留空 = 不过滤）",
        type="string_list",
        required=False,
        choices=tuple(item.value for item in SourceType),
        help="按证据效力收窄，如只看 client_provided,public。",
    ),
    Question(
        id="project_number",
        text="项目编号？（可留空）",
        type="string",
        required=False,
    ),
    Question(
        id="budget",
        text="项目预算？（可留空；生成时用）",
        type="string",
        required=False,
    ),
    Question(
        id="duration",
        text="项目工期？（可留空；生成时用）",
        type="string",
        required=False,
    ),
    Question(
        id="project_nature",
        text="项目性质？（engineering/platform/research/planning/other，可留空）",
        type="string",
        required=False,
        help="drafter S1 project_type，描述用；不参与契约选择。",
    ),
)

QUESTIONS_BY_ID: Mapping[str, Question] = {q.id: q for q in QUESTIONS}
