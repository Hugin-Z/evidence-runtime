"""中文排版空格规范化：渲染层唯一允许的文本变换。

这是一个确定性、纯函数、可单测的步骤：渲染时对每段正文施加，``verify_docx`` 对方案稿施加
同一函数后比对，因此 DOCX 与规范化后的方案稿仍可逐字确定比较。

**只动空格，不动任何其它字符**：仅删除「CJK 与 ASCII 字母/数字相邻」处的半角空格/制表符
（两向）。纯 ASCII 内部空格（英文单词间）、换行、CJK 之间的空格一律不动（后者是 R3 特例：
中文对齐空格允许保留）。
"""
from __future__ import annotations

import re

# CJK：基本区 + 扩展 A（够覆盖政企材料用字）
_CJK = r"㐀-䶿一-鿿"
_ALNUM = r"0-9A-Za-z"

# CJK 后接空格再接字母/数字 → 删空格；反向亦然。只吃半角空格/制表符，不吃换行。
_CJK_THEN_ALNUM = re.compile(rf"([{_CJK}])[ \t]+([{_ALNUM}])")
_ALNUM_THEN_CJK = re.compile(rf"([{_ALNUM}])[ \t]+([{_CJK}])")


def normalize_cjk_spacing(text: str) -> str:
    """删除中文与 ASCII 字母/数字相邻处的半角空格。幂等、确定、纯函数。

    例：``保证金 3000 元`` → ``保证金3000元``；``采用 TLS 加密`` → ``采用TLS加密``。
    不动：``hello world``（纯 ASCII 内部）、``第一行\\n第二行``（换行）、``中文 中文``（CJK 间）。
    """

    if not isinstance(text, str):
        raise ValueError(f"text must be str; got {text!r}")
    # 反复施加直到稳定（形如 ``中 A 中`` 需两个方向各扫一遍；多空格一次清）
    prev = None
    out = text
    while out != prev:
        prev = out
        out = _CJK_THEN_ALNUM.sub(r"\1\2", out)
        out = _ALNUM_THEN_CJK.sub(r"\1\2", out)
    return out
