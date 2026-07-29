"""L6′ 渲染层：结构化方案稿 → DOCX + 确定性回验。

- :func:`render_proposal` / :func:`render_to_file`：``OrchestrationState`` + ProjectSpec → DOCX；
  渲染层零内容改写，唯一例外是确定性的中文空格规范化。
- :func:`verify_docx`：检查内容一致、完整性、占位保真与参数正确。
- :func:`normalize_cjk_spacing`：确定性文本规范化。
"""
from .docx_render import render_proposal, render_to_file
from .text_norm import normalize_cjk_spacing
from .verify_docx import VerifyCheck, VerifyResult, load_docx, verify_docx

__all__ = [
    "VerifyCheck",
    "VerifyResult",
    "load_docx",
    "normalize_cjk_spacing",
    "render_proposal",
    "render_to_file",
    "verify_docx",
]
