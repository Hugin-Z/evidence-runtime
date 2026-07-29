"""词法路：ripgrep 精确命中。

把自然语言 query 拆成关键词（去最基本停用词），用 subprocess 调
`rg --json -i -F`（按需 `-C` 给上下文行）在 ``settings.corpus_dir`` 下字面搜，
解析命中的 (文件, 行号, 文本)，把同文件相邻命中合并成行区间，包成
``Snippet``（带 ``start_line``/``end_line``）返回。

特点：零索引延迟、原文保真、按文件返回行区间而非整文件（上下文干净）。
**这条路是"哑"的**：只做字面匹配，不做同义词扩展——换说法的语义靠向量路
（``vector.py``）+ 运行时 AI 操作者循环补，不在这里做。所以一篇"语义相关但
字面不含查询词"的文档，词法路必然漏掉（这是预期行为，也是双路对照的依据）。

约定：
  * 关键词以**字面串**（``-F``）匹配，不当正则解释——避免查询里的元字符注入，
    也契合"精确命中"。子进程一律列表传参、``shell=False``。
  * ``rg`` 默认跳过二进制文件与 ``.gitignore`` / ``.ignore`` 规则命中的文件；
    本实现不传任何关闭 ignore 的开关，故保留该默认行为。

降级（均非静默）：
  * ``rg`` 未安装         → 返回 ``[]`` 并发 ``RuntimeWarning``（明确告知，不静默崩）
  * ``corpus_dir`` 不存在 → 返回 ``[]``
  * 空目录 / 无命中        → 返回 ``[]``
  * ``rg`` 真错误（退出码 ≥2）→ 抛 ``RuntimeError``（硬失败，不吞）

不覆盖：同义词扩展、CJK 分词（关键词按 ``\\w+`` 切，CJK 无空白则整段作一个字面词）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import warnings
from typing import TYPE_CHECKING

from ..config import Settings, DEFAULT_SETTINGS
from ..types import Snippet, RetrievalPath

if TYPE_CHECKING:  # 仅类型,不引入运行时耦合（避免 import 环）
    from ..scope import Scope

# 最基本停用词（功能词）。刻意保持小：词法路只需把"配置/重试/错误码"这类
# 实词留下来字面搜，不追求语言学完备。
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "was", "were", "how", "what", "when", "where", "why", "do", "does",
    "did", "i", "you", "it", "this", "that", "with", "at", "by", "as", "from",
    "into", "about",
})

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _keywords(query: str) -> list[str]:
    """把 query 切成去重、去停用词的小写关键词（保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(query.lower()):
        if len(tok) < 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


class RipgrepRetriever:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS, context_lines: int = 2):
        self.settings = settings
        self.context_lines = max(0, context_lines)

    def search(self, query: str, k: int | None = None,
               scope: "Scope | None" = None) -> list[Snippet]:
        """字面检索，返回按初排分降序、截到 k 的 Snippet 列表。

        ``scope`` 决定在哪搜（精搜永远只认 scope 入参，切片层将来才能随便升级）：
          * ``None`` 或 ``scope.files is None`` → 全量扫 ``settings.corpus_dir``（默认/向后兼容）
          * ``scope.files`` 是清单 → 只在这些文件路径里搜（限定到切片）；空清单 → 空结果

        ``score`` 为 *命中的不同关键词数 / 查询关键词总数*（0~1）。
        **此分仅供初排（pre-rerank）**：用于喂给精排前的粗排，最终相关度由
        ``rerank`` 决定，不应被当作语义相关度读。
        """
        k = k or self.settings.lexical_top_k

        keywords = _keywords(query)
        if not keywords:
            return []

        if shutil.which("rg") is None:
            warnings.warn(
                "ripgrep (rg) 未安装，词法路降级为空结果。"
                "安装：macOS `brew install ripgrep` / Ubuntu `apt install ripgrep`。",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        targets = self._resolve_targets(scope)
        if not targets:  # None=corpus_dir 不存在；[]=scope 限定到空集 —— 两者皆空结果
            return []

        proc = self._run_rg(keywords, targets)
        if proc.returncode >= 2:  # 0=有命中, 1=无命中, ≥2=真错误
            raise RuntimeError(
                f"ripgrep failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )

        snippets = self._parse(proc.stdout, total_keywords=len(keywords))
        snippets.sort(key=lambda s: (-s.score, s.source_path, s.start_line or 0))
        return snippets[:k]

    # —— 内部 ——

    def _resolve_targets(self, scope: "Scope | None") -> list[str] | None:
        """把 scope 解析成 rg 的搜索目标清单。

        scope 限定文件 → 直接返回这些文件路径（精搜只在切片里搜，不扫整个 corpus_dir）。
        无限制 → 返回 ``[corpus_dir]``；corpus_dir 不存在 → 返回 ``None``（上层转空结果）。
        """
        if scope is not None and scope.files is not None:
            return list(scope.files)  # 可能为 [] → 上层 `not targets` 转空结果
        corpus_dir = self.settings.corpus_dir
        if not os.path.isdir(corpus_dir):
            return None
        return [corpus_dir]

    def _run_rg(self, keywords: list[str], targets: list[str]) -> subprocess.CompletedProcess:
        cmd = ["rg", "--json", "-i", "-F"]
        if self.context_lines:
            cmd += ["-C", str(self.context_lines)]
        for kw in keywords:
            cmd += ["-e", kw]
        cmd.extend(targets)
        # encoding 显式 utf-8：rg --json 输出 UTF-8，Windows 默认编码会乱码。
        return subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", shell=False,
        )

    def _parse(self, stdout: str, total_keywords: int) -> list[Snippet]:
        # path -> {"lines": {lineno: text}, "matches": {lineno: set(kw)}}
        files: dict[str, dict] = {}
        for raw in stdout.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            if kind not in ("match", "context"):
                continue
            data = obj["data"]
            path = data["path"]["text"]
            lineno = data["line_number"]
            text = data["lines"]["text"].rstrip("\n")
            entry = files.setdefault(path, {"lines": {}, "matches": {}})
            entry["lines"][lineno] = text
            if kind == "match":
                kws = {sm["match"]["text"].lower() for sm in data.get("submatches", [])}
                if kws:
                    entry["matches"].setdefault(lineno, set()).update(kws)

        snippets: list[Snippet] = []
        ctx = self.context_lines
        for path, entry in files.items():
            if not entry["matches"]:
                continue
            for start, end, mlines in self._intervals(sorted(entry["matches"]), ctx):
                present = [n for n in entry["lines"] if start <= n <= end] or list(mlines)
                real_start, real_end = min(present), max(present)
                text = "\n".join(
                    entry["lines"][n]
                    for n in range(real_start, real_end + 1)
                    if n in entry["lines"]
                )
                matched: set[str] = set()
                for ln in mlines:
                    matched |= entry["matches"][ln]
                snippets.append(Snippet(
                    source_path=path,
                    text=text,
                    score=len(matched) / total_keywords,
                    path=RetrievalPath.LEXICAL,
                    start_line=real_start,
                    end_line=real_end,
                    meta={
                        "match_lines": sorted(mlines),
                        "matched_keywords": sorted(matched),
                        "keyword_hits": len(matched),
                        "keyword_total": total_keywords,
                    },
                ))
        return snippets

    @staticmethod
    def _intervals(match_lines: list[int], ctx: int):
        """把命中行按 ±ctx 展开，相邻/重叠（间隔≤1）的并成一段。

        产出 (start, end, [命中行...])，命中行用于归因关键词与回报 match_lines。
        """
        merged: list[list] = []
        for ln in match_lines:
            start, end = max(1, ln - ctx), ln + ctx
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2].append(ln)
            else:
                merged.append([start, end, [ln]])
        return [(s, e, mls) for s, e, mls in merged]
