"""L1 Library：材料库管理层。

在判断核心（L2–L7）**下方**回答四个 v0.3 回避的问题：材料从哪来、不同项目怎么隔离、
材料更新了怎么办、检索时怎么按材料属性收窄范围。**L1 不参与任何判定。**

- :class:`LibraryProtocol` —— 抽象契约（接口先行）；
- :class:`FilesystemLibrary` —— 随仓默认实现（纯文件系统，无新依赖）。
"""
from .filesystem import (
    DEFAULT_LIBRARY_ROOT,
    LIBRARY_VERSION,
    FilesystemLibrary,
    LibraryError,
)
from .models import IndexState, MaterialRecord, MaterialVersion, ProjectInfo, SourceType
from .protocol import LibraryProtocol

__all__ = [
    "DEFAULT_LIBRARY_ROOT",
    "FilesystemLibrary",
    "IndexState",
    "LIBRARY_VERSION",
    "LibraryError",
    "LibraryProtocol",
    "MaterialRecord",
    "MaterialVersion",
    "ProjectInfo",
    "SourceType",
]
