"""L1 Library 抽象接口。

``LibraryProtocol`` 定义「材料库」的抽象契约。随仓默认实现是
:class:`evidence.library.filesystem.FilesystemLibrary`；将来接入**外部文档系统**
（SharePoint / 自建 DMS / ``agentic-kb-lite`` 的 PARA 语料库）只需另实现本 Protocol，
**调用方一行不改**。

**层次边界**：L1 只回答「哪些材料、哪个版本、属于哪个项目」，
把结果交给 L2 发现层。L1 **不做**语义路由、不做全文检索、**不参与任何判定**——
Verifier / Gate 的判断逻辑与本层无关。

**接入点（实现外部库时需要提供的语义）**：

======================  ====================================================
方法                    外部系统需要映射到什么
======================  ====================================================
``init_project``        创建/登记一个隔离的材料集合（租户、空间、文件夹）
``add_material``        写入一份材料并返回**内容派生**的 material_id；
                        同内容需幂等，同名不同内容需产生新版本
``list_materials``      列出逻辑材料及其版本（当前版本可辨识）
``documents``           把**当前版本**解析为 ``NormalizedDocument``（含 provenance）；
                        ``source_types`` 用于按来源收窄
``index_state``         回答"索引相对当前材料集是否过期"
``mark_indexed``        记录一次成功建索引后的材料集指纹
``index_root``          给出该项目向量索引的存放位置（实现项目隔离）
======================  ====================================================

**确定性要求（对任何实现都成立）**：``add_material`` 返回的 material_id 必须是
**内容派生**的（同正文 → 同 id），不得使用随机 id 或入库顺序。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from ..documents import NormalizedDocument
from .models import IndexState, MaterialRecord, ProjectInfo, SourceType


@runtime_checkable
class LibraryProtocol(Protocol):
    """材料库抽象接口。所有方法都必须是确定性的（同输入同输出）。"""

    def init_project(self, project_id: str) -> ProjectInfo:
        """创建（或幂等打开）一个项目库。"""
        ...

    def add_material(
        self,
        source: str | Path,
        *,
        source_type: SourceType | str = SourceType.OTHER,
    ) -> tuple[str, bool]:
        """把一份原始材料入库。

        返回 ``(material_id, added)``：``material_id`` **内容派生**；
        ``added=False`` 表示同内容已在库（幂等跳过）。
        同名不同内容 → 追加为**新版本**，旧版本置 ``superseded`` 但保留可查。
        """
        ...

    def list_materials(self) -> tuple[MaterialRecord, ...]:
        """列出全部逻辑材料及其版本（含已被取代的历史版本）。"""
        ...

    def documents(
        self,
        *,
        source_types: Sequence[SourceType | str] | None = None,
    ) -> tuple[NormalizedDocument, ...]:
        """把**当前版本**材料解析为 NormalizedDocument。

        ``source_types=None`` 表示不过滤（默认，行为与目录模式一致）。
        """
        ...

    def index_state(self) -> IndexState:
        """索引相对当前材料集是否过期。"""
        ...

    def mark_indexed(self) -> IndexState:
        """记录「已按当前材料集建好索引」。"""
        ...

    def index_root(self) -> Path:
        """该项目向量索引的存放根目录（项目隔离的物理落点）。"""
        ...

    def info(self) -> ProjectInfo:
        """项目概要（材料数 / 版本数 / 索引状态）。"""
        ...
