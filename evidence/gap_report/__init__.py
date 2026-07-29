"""Gap Report：报告数据类型、构建器、序列化与渲染。

模块按职责拆成
`models` / `builder` / `serialization` / `markdown` 四个子模块。**纯移动**：函数体逐字符未改，
对外符号与 import 路径（`from evidence.gap_report import X`）完全不变。
"""
from .models import (
    EvidenceReference,
    GapReport,
    PendingActionType,
    PendingMaterialRequest,
    SlotGapResult,
    _json_clone,
)
from .builder import GapReportBuilder, _pending_for
from .serialization import (
    _conflict_to_dict,
    _fact_to_dict,
    gap_report_to_dict,
    gap_report_to_json,
    write_gap_report_json,
)
from .markdown import (
    _one_line,
    render_gap_report_markdown,
    write_gap_report_markdown,
)

__all__ = [
    "PendingActionType", "PendingMaterialRequest", "EvidenceReference",
    "SlotGapResult", "GapReport", "GapReportBuilder",
    "gap_report_to_dict", "gap_report_to_json", "render_gap_report_markdown",
    "write_gap_report_json", "write_gap_report_markdown",
]
