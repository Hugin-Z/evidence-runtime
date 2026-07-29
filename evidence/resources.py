"""包内资源定位：契约、crosswalk 与 section specs 随 wheel 分发。

使用 :func:`importlib.resources` 定位 ``evidence.contracts`` 包，并保留源码树回退，
使源码运行与 wheel 安装两种形态都可用。

pip 安装的 wheel 在 site-packages 下是**解压的真实目录**，故这里把 Traversable 当真实路径用
（全仓取档本就是文件系统路径语义）；万一是 zip 安装取不到目录，回退到源码树路径。
"""
from __future__ import annotations

from pathlib import Path

_REPO_CONTRACTS = Path(__file__).resolve().parent / "contracts"


def _resolve(pkg: str, *, repo_fallback: Path) -> Path:
    try:
        from importlib.resources import files

        p = Path(str(files(pkg)))
        if p.is_dir():
            return p
    except (ModuleNotFoundError, TypeError, ValueError, NotADirectoryError, OSError):
        pass
    return repo_fallback


def contracts_dir() -> Path:
    """契约根目录（含 ``solution_material_completeness_v1.json``）。"""
    return _resolve("evidence.contracts", repo_fallback=_REPO_CONTRACTS)


def crosswalk_dir() -> Path:
    """crosswalk 配置目录（``<doc_type_zh>.json``）。"""
    return _resolve("evidence.contracts.crosswalk", repo_fallback=contracts_dir() / "crosswalk")


def section_specs_dir() -> Path:
    """section 写作规格目录（``<doc_type>.json``）。"""
    return _resolve("evidence.contracts.section_specs", repo_fallback=contracts_dir() / "section_specs")


def contract_file(name: str) -> Path:
    """契约根下某个文件的路径。"""
    return contracts_dir() / name
