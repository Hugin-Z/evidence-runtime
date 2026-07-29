"""Runtime Document Adapter：原始文件（MD/TXT/DOCX/文本型 PDF）→
:class:`evidence.documents.NormalizedDocument`。

**边界**：不做 OCR、不做 Hybrid Retrieval、不碰生成层。扫描件 PDF（无
文本层）不静默产出空文档，而是抛 :class:`NoTextLayerError`。

**与冻结 schema 的关系（扩展而非修改）**：``NormalizedDocument`` 已支持 ``source_type``
"pdf"/"docx" 且带 ``metadata`` 映射；本适配器只**填充** ``text`` 与 ``metadata``，
不改 documents.py，也不被下游（vector_discovery/exact_locator/verifier/gap_checker）读取
——``NormalizedDocument.metadata`` 是 provenance 侧信息，不进入检索/判定，故适配器等价替换
现有 MD/TXT 入口时结果不变。

**命门：字符偏移**。exact_locator 靠 ``NormalizedDocument.text`` 的字符偏移回读原文，故
所有 provenance（页码/标题层级）都以**指向 ``text`` 的字符区间**表达（``blocks``），偏移与
``text`` 严格一致。MD/TXT 复用 ``NormalizedDocument.from_path`` 的规范化，保证与既有链路字节一致。

provenance 结构（存入 ``metadata["runtime_adapter"]``）：
    {"version","source_type","original_filename","page_count","warnings",
     "blocks":[{"start_char","end_char","kind","page","heading_level","heading_path"}]}
``blocks`` 是对 ``text`` 的字符区间标注；``provenance_for_range`` 据此回答「来自第几页/哪个标题下」。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from .documents import NormalizedDocument, normalize_text

ADAPTER_VERSION = "m1b-adapter-v1"
SUPPORTED_INPUT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".docx", ".pdf"})
_SUFFIX_TO_TYPE = {".md": "md", ".markdown": "md", ".txt": "txt", ".docx": "docx", ".pdf": "pdf"}


class RuntimeAdapterError(RuntimeError):
    """适配器基类错误。"""


class UnsupportedInputError(ValueError):
    """扩展名/magic bytes 不在支持范围。"""


class NoTextLayerError(RuntimeAdapterError):
    """PDF 无文本层（扫描件），需要 OCR。"""


@dataclass
class _Block:
    start_char: int
    end_char: int
    kind: str                       # heading | paragraph | table | page
    page: int | None = None
    heading_level: int | None = None
    heading_path: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"start_char": self.start_char, "end_char": self.end_char, "kind": self.kind,
                "page": self.page, "heading_level": self.heading_level,
                "heading_path": list(self.heading_path)}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_source_type(path: str | Path) -> str:
    """按扩展名 + magic bytes 判定来源类型；冲突或不支持则报错。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in _SUFFIX_TO_TYPE:
        raise UnsupportedInputError(
            f"不支持的输入扩展名 {suffix!r}（支持 {sorted(SUPPORTED_INPUT_SUFFIXES)}）：{p.name}")
    stype = _SUFFIX_TO_TYPE[suffix]
    head = p.read_bytes()[:8] if p.is_file() else b""
    if stype == "pdf" and not head.startswith(b"%PDF"):
        raise UnsupportedInputError(f"{p.name} 扩展名为 .pdf 但 magic bytes 非 %PDF")
    if stype == "docx" and not head.startswith(b"PK\x03\x04"):
        raise UnsupportedInputError(f"{p.name} 扩展名为 .docx 但非 zip/OOXML 容器")
    return stype


# ── 各格式 parser：返回 (normalized_text, blocks) ────────────────────────────

def _adapt_text(raw: bytes, source_type: str) -> tuple[str, list[_Block]]:
    """MD/TXT：与 NormalizedDocument.from_path 同口径规范化；blocks 只标注、不重建 text。"""
    text = normalize_text(raw.decode("utf-8-sig"))
    blocks: list[_Block] = []
    if source_type == "txt":
        # 段落 = 空行分隔；只标注，不改 text
        pos = 0
        for para in text.split("\n\n"):
            if para.strip():
                blocks.append(_Block(pos, pos + len(para), "paragraph"))
            pos += len(para) + 2
        if not blocks and text:
            blocks.append(_Block(0, len(text), "paragraph"))
        return text, blocks
    # md：逐行，# 开头为标题，维护 heading 栈
    stack: list[tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        end = pos + len(line)
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            htext = stripped[level:].strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htext))
            blocks.append(_Block(pos, end, "heading", heading_level=level,
                                 heading_path=tuple(h for _, h in stack)))
        elif line.strip():
            blocks.append(_Block(pos, end, "paragraph",
                                 heading_path=tuple(h for _, h in stack)))
        pos = end + 1
    return text, blocks


def _docx_ignored_carriers(doc) -> list[dict]:
    """检测 ``_adapt_docx`` 只读正文段落和表格时会忽略的载体：文本框、图片、图表、
    页眉页脚与批注。检测到存在即告警，但不负责提取。返回 gap 列表。"""
    from docx.oxml.ns import qn

    gaps: list[dict] = []
    body = doc.element.body
    n_txbx = sum(1 for _ in body.iter(qn("w:txbxContent")))
    n_drawing = sum(1 for _ in body.iter(qn("w:drawing")))
    if n_txbx:
        gaps.append({"kind": "docx_textbox", "count": n_txbx,
                     "detail": f"正文含 {n_txbx} 个文本框（w:txbxContent），其中文字未提取——"
                               f"政企材料常把关键信息放文本框，可能影响'材料够不够'的判断。"})
    if n_drawing:
        gaps.append({"kind": "docx_drawing", "count": n_drawing,
                     "detail": f"正文含 {n_drawing} 个图片/图表/绘图对象（w:drawing），其中文字/数据未提取。"})
    # 页眉页脚：_adapt_docx 只读 body，section header/footer 一律不提取
    hf_texts = []
    for section in doc.sections:
        for part in (section.header, section.footer):
            for p in part.paragraphs:
                if p.text.strip():
                    hf_texts.append(p.text.strip())
    if hf_texts:
        gaps.append({"kind": "docx_header_footer", "count": len(hf_texts),
                     "detail": f"含页眉/页脚文字 {len(hf_texts)} 处，未提取。"})
    # 批注：docx 包内若有 comments part 则存在批注
    try:
        rels = doc.part.rels
        if any("comments" in (r.reltype or "") for r in rels.values()):
            gaps.append({"kind": "docx_comments",
                         "detail": "文档含批注（comments），未提取。"})
    except Exception:
        pass
    return gaps


def _adapt_docx(raw: bytes) -> tuple[str, list[_Block], list[dict]]:
    from docx import Document           # python-docx
    from docx.oxml.ns import qn
    import io
    doc = Document(io.BytesIO(raw))
    para_by_p = {p._p: p for p in doc.paragraphs}
    tbl_by_tbl = {t._tbl: t for t in doc.tables}
    pieces: list[str] = []
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    pos = 0

    def emit(seg: str, kind: str, **kw) -> None:
        nonlocal pos
        # 命门：先规范化换行再测量，使 block 偏移与最终 text 字节级一致（seg 内若含 \r
        # 会在最终 normalize_text 处缩短，若在此之前测量偏移就会错位）。normalize_text 幂等。
        seg = normalize_text(seg)
        if not seg:
            return
        if pieces:
            pos += 2                    # 段间的 "\n\n"
        blocks.append(_Block(pos, pos + len(seg), kind,
                             heading_path=tuple(h for _, h in stack), **kw))
        pieces.append(seg)
        pos += len(seg)

    for child in doc.element.body:
        tag = child.tag.split("}")[-1]
        if tag == "p" and child in para_by_p:
            p = para_by_p[child]
            seg = p.text
            if not seg.strip():
                continue
            style = (p.style.name or "") if p.style else ""
            if style.startswith("Heading"):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, seg.strip()))
                emit(seg, "heading", heading_level=level)
            else:
                emit(seg, "paragraph")
        elif tag == "tbl" and child in tbl_by_tbl:
            table = tbl_by_tbl[child]
            rows = ["\t".join(c.text.strip() for c in r.cells) for r in table.rows]
            seg = "\n".join(r for r in rows if r.strip())
            emit(seg, "table")
    text = normalize_text("\n\n".join(pieces))
    return text, blocks, _docx_ignored_carriers(doc)


def _adapt_pdf(raw: bytes) -> tuple[str, list[_Block], list[dict]]:
    import pdfplumber
    import io
    pieces: list[str] = []
    blocks: list[_Block] = []
    gaps: list[dict] = []
    no_text_pages: list[int] = []
    pos = 0
    n_pages = 0
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            # 命门：先规范化换行再 strip 再测量偏移（见 _adapt_docx.emit 注释）。
            seg = normalize_text(page.extract_text() or "").strip()
            if not seg:
                # 无文本层的页被跳过时显式记账，不静默。
                no_text_pages.append(i)
                continue
            if pieces:
                pos += 2
            blocks.append(_Block(pos, pos + len(seg), "page", page=i))
            pieces.append(seg)
            pos += len(seg)
    if not pieces:
        raise NoTextLayerError(
            f"PDF 共 {n_pages} 页但无可抽取文本层（疑扫描件）——需要 OCR，不产出空文档。")
    if no_text_pages:
        gaps.append({
            "kind": "pdf_no_text_page",
            "pages": no_text_pages,
            "detail": (f"混合 PDF：第 {', '.join(map(str, no_text_pages))} 页无文本层（疑扫描/图片页），"
                       f"共 {len(no_text_pages)}/{n_pages} 页未提取——需要 OCR。这些页的内容"
                       f"不在证据判定范围内。"),
        })
    text = normalize_text("\n\n".join(pieces))
    return text, blocks, gaps


def adapt_file(path: str | Path, *, document_id: str | None = None,
               title: str | None = None, warnings: Sequence[str] = ()) -> NormalizedDocument:
    """单个原始文件 → NormalizedDocument（text + provenance metadata + C5 extraction_gaps）。"""
    p = Path(path)
    stype = detect_source_type(p)
    raw = p.read_bytes()
    gaps: list[dict] = []
    if stype in ("md", "txt"):
        text, blocks = _adapt_text(raw, stype)
    elif stype == "docx":
        text, blocks, gaps = _adapt_docx(raw)
    else:  # pdf
        text, blocks, gaps = _adapt_pdf(raw)
    source_hash = _sha256_bytes(raw)
    normalized_hash = _sha256_bytes(text.encode("utf-8"))
    if document_id is None:
        logical = PurePosixPath(p.name).name
        document_id = f"doc-{hashlib.sha256((logical + chr(0) + normalized_hash).encode('utf-8')).hexdigest()}"
    # C5：把每条 extraction gap 也落成人读 warning（"不静默"最低目标）
    gap_warnings = [g["detail"] for g in gaps]
    meta = {"runtime_adapter": {
        "version": ADAPTER_VERSION, "source_type": stype, "original_filename": p.name,
        "page_count": max((b.page for b in blocks if b.page), default=None),
        "warnings": list(warnings) + gap_warnings,
        "extraction_gaps": gaps,
        "blocks": [b.to_dict() for b in blocks]}}
    return NormalizedDocument(
        document_id=document_id, title=title or p.stem, source_path=p.name,
        source_type=stype, normalized_path=None, text=text,
        source_content_hash=source_hash, normalized_content_hash=normalized_hash, metadata=meta)


def adapt_path(path: str | Path) -> tuple[NormalizedDocument, ...]:
    """原始文件或目录 → NormalizedDocument 列表（目录递归、按路径排序、跳过不支持扩展名）。"""
    p = Path(path)
    if p.is_file():
        return (adapt_file(p),)
    if not p.is_dir():
        raise UnsupportedInputError(f"输入路径不存在：{p}")
    files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_INPUT_SUFFIXES)
    if not files:
        raise UnsupportedInputError(f"目录 {p} 下无受支持的输入文件（{sorted(SUPPORTED_INPUT_SUFFIXES)}）")
    return tuple(adapt_file(f) for f in files)


def provenance_for_range(document: NormalizedDocument, start_char: int, end_char: int) -> dict:
    """回答「[start,end) 来自第几页 / 哪个标题下」——据 metadata.blocks 与字符区间重叠判定。"""
    ra = document.metadata.get("runtime_adapter") if isinstance(document.metadata, dict) or hasattr(document.metadata, "get") else None
    blocks = (ra or {}).get("blocks", []) if ra else []
    pages, headings = [], ()
    for b in blocks:
        if b["start_char"] < end_char and start_char < b["end_char"]:   # 重叠
            if b.get("page") and b["page"] not in pages:
                pages.append(b["page"])
            if b.get("heading_path") and not headings:
                headings = tuple(b["heading_path"])
    return {"pages": sorted(pages), "headings": list(headings)}


def _ra_meta(document) -> dict:
    meta = getattr(document, "metadata", None)
    if isinstance(meta, dict) or hasattr(meta, "get"):
        return meta.get("runtime_adapter") or {}
    return {}


def input_completeness_summary(documents: Sequence[NormalizedDocument]) -> dict:
    """跨文档聚合提取缺口，供 Gap Report 的“输入完整性”与 trace 使用。

    返回 ``{"complete": bool, "total_gaps": N, "documents": [{document_id, filename, gaps:[…]}]}``。
    ``complete=True`` = 所有文档都完整提取（无未提取载体）——判定"材料够不够"时可放心；否则该节
    列出所有**可能没看全**的载体（无文本页 / 文本框 / 图片图表 / 页眉页脚 / 批注），让判断带上自知。
    """
    docs_out: list[dict] = []
    total = 0
    for d in documents:
        ra = _ra_meta(d)
        gaps = ra.get("extraction_gaps") or []
        if gaps:
            total += len(gaps)
            docs_out.append({
                "document_id": getattr(d, "document_id", None),
                "filename": ra.get("original_filename") or getattr(d, "source_path", None),
                "gaps": list(gaps),
            })
    return {"complete": total == 0, "total_gaps": total, "documents": docs_out}
