"""L6 生成编排层：任务组装、派发、收货校验与状态管理。

在 Gate（L5，`allow/limited/block/needs_resolution`）**之上**，把每个可写 section
组装成**任务包** :class:`SectionTask` 派给执行 agent，回收后经**确定性收货校验**
（A3）接受或退回。

**架构铁律（A1/A3）**：
- **编排器是代码**：任务组装/派发/回收/校验/重试全部确定性。
- **agent 只执行撰写**：拿任务包写内容、交回。它不选契约、不判证据、不定 section 去留
  （gate 已定）、不越权访问任务包之外的材料。
- **收货是代码**：每份回稿必经确定性校验（:mod:`evidence.generation.receiving`）。
- **Engine 仍零 LLM**：默认执行方是交互 agent，
  不是 Engine 内嵌 API 调用。执行方接口（A2）留好：同一任务包将来可派 API 模型或人。
"""
from .discipline import (
    SECTION_CONSTRAINTS,
    receiving_discipline_note,
    task_constraints,
)
from .executor import (
    CallableExecutor,
    DraftNotReady,
    InteractiveFileExecutor,
    TaskExecutorProtocol,
)
from .orchestrator import (
    Orchestrator,
    OrchestrationState,
    SectionOutcome,
    assert_task_clean,
)
from .receiving import ReceivingCheck, ReceivingResult, check_draft
from .section_spec import SectionWritingSpec, load_section_specs
from .task import DraftParagraph, SectionDraft, SectionTask, TaskEvidence

__all__ = [
    "CallableExecutor",
    "DraftNotReady",
    "DraftParagraph",
    "InteractiveFileExecutor",
    "Orchestrator",
    "OrchestrationState",
    "ReceivingCheck",
    "ReceivingResult",
    "SECTION_CONSTRAINTS",
    "SectionDraft",
    "SectionOutcome",
    "SectionTask",
    "SectionWritingSpec",
    "TaskEvidence",
    "TaskExecutorProtocol",
    "assert_task_clean",
    "check_draft",
    "load_section_specs",
    "receiving_discipline_note",
    "task_constraints",
]
