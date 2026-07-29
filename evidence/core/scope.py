"""低层检索器可选的文件范围。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class Scope:
    """精搜的范围。

    ``files is None`` 表示不限制；文件清单表示只检索指定路径。
    """

    files: Optional[list[str]] = None
