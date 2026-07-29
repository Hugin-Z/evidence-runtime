"""L0 Intake：结构化任务入口。

产品第一次有"从用户任务开始"的入口：用户带着一个任务来（要写什么方案、什么项目），
经结构化 intake 产出 :class:`ProjectSpec`，由它**确定性地**驱动契约选择与后续全链。
直接指定 ``--contract`` 和材料路径的方式保留为工程接口。

- :class:`ProjectSpec` —— 任务说明数据结构（可校验/可落盘/可复用）。
- :data:`QUESTIONS` —— 代码定义的问题清单。
- :func:`validate_answers` —— 输入校验。
- :func:`select_contract` —— doc_type → 契约的确定性查表。
- :func:`run_interactive` / :func:`run_from_file` —— 两种执行形态。
"""
from .interview import (
    IntakeAborted,
    collect_interactive,
    render_answers_summary,
    review_answers,
    run_from_file,
    run_interactive,
    save_spec,
)
from .questions import QUESTIONS, QUESTIONS_BY_ID, Question
from .registry import (
    ContractBinding,
    ContractSelectionError,
    has_binding,
    registered_doc_types,
    select_contract,
)
from .spec import SPEC_VERSION, DocType, ProjectSpec, SectionScope
from .validate import (
    IntakeValidationError,
    load_answers_file,
    load_project_spec,
    validate_answers,
)

__all__ = [
    "ContractBinding",
    "ContractSelectionError",
    "DocType",
    "IntakeValidationError",
    "ProjectSpec",
    "QUESTIONS",
    "QUESTIONS_BY_ID",
    "Question",
    "SPEC_VERSION",
    "IntakeAborted",
    "SectionScope",
    "collect_interactive",
    "has_binding",
    "load_answers_file",
    "load_project_spec",
    "registered_doc_types",
    "render_answers_summary",
    "review_answers",
    "run_from_file",
    "run_interactive",
    "save_spec",
    "select_contract",
    "validate_answers",
]
