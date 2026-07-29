"""标准 benchmark 适配层（BEIR / C-MTEB 检索格式）。

读三元组 corpus / queries / qrels，并把语料**落地两路**给既有 core（core 不改）：
  * 词法路：每个 passage 写成一个 ``{safe_id}.txt`` 到临时目录，供 ``RipgrepRetriever``（corpus_dir=该目录）
  * 向量路：同一个 txt 目录经 ``core.ingest.build_index`` 进 Chroma；passage id 由**文件名**携带
    （``id_of`` 用文件名反查 passage id）——满足“passage id 进元数据（source_path=文件名）”，不改 ingest。

格式（BEIR 目录布局，其他子集可先导出成同布局）：
  * ``corpus.jsonl``  每行 ``{"_id","title","text"}``
  * ``queries.jsonl`` 每行 ``{"_id","text"}``
  * ``qrels/{split}.tsv`` 制表符分隔、带表头 ``query-id  corpus-id  score``

qrels：**保留分级相关度**（score 原值给 nDCG）；``score >= 1`` 视为相关（二值 gold 给 recall/MRR/precision）。
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

_UNSAFE = re.compile(r"[^0-9A-Za-z_-]")


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(path: str) -> dict[str, dict]:
    """corpus.jsonl → {passage_id: {"title","text"}}。"""
    out = {}
    for r in _read_jsonl(path):
        out[str(r["_id"])] = {"title": r.get("title", "") or "", "text": r.get("text", "") or ""}
    return out


def load_queries(path: str, keep_ids: set[str] | None = None) -> dict[str, str]:
    """queries.jsonl → {query_id: text}。``keep_ids`` 给定时只保留这些（如某 split 的 query）。"""
    out = {}
    for r in _read_jsonl(path):
        qid = str(r["_id"])
        if keep_ids is None or qid in keep_ids:
            out[qid] = r.get("text", "") or ""
    return out


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """qrels/{split}.tsv → {query_id: {passage_id: score}}（保留分级 score，供 nDCG）。跳过表头。"""
    out: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if first and parts[0] == "query-id":   # 表头
                first = False
                continue
            first = False
            qid, pid, score = str(parts[0]), str(parts[1]), int(float(parts[2]))
            out.setdefault(qid, {})[pid] = score
    return out


def load_beir(data_dir: str, split: str = "test"):
    """读一套 BEIR 布局，返回 (corpus, queries, qrels)。queries 只保留该 split 的 query。"""
    qrels = load_qrels(os.path.join(data_dir, "qrels", f"{split}.tsv"))
    corpus = load_corpus(os.path.join(data_dir, "corpus.jsonl"))
    queries = load_queries(os.path.join(data_dir, "queries.jsonl"), keep_ids=set(qrels.keys()))
    return corpus, queries, qrels


def relevant_set(qrels_for_q: dict[str, int]) -> set[str]:
    """某 query 的二值 gold：score >= 1 的 passage id 集合（给 recall/MRR/precision）。"""
    return {pid for pid, s in qrels_for_q.items() if s >= 1}


def _safe_stem(pid: str) -> str:
    return _UNSAFE.sub("_", pid) or "_"


def materialize_corpus(corpus: dict[str, dict], out_dir: str) -> tuple[str, dict[str, str]]:
    """把每个 passage 写成 ``{safe_id}.txt``（title + text）到 out_dir，供 rg 与 ingest 共用。

    返回 (out_dir, fn2id)。``fn2id``：文件名 stem（不含 .txt）→ 原 passage id，供 ``id_of`` 反查
    （文件名做了安全化 + 去重，故必须走这张表、不能从文件名直接推 id）。
    """
    os.makedirs(out_dir, exist_ok=True)
    fn2id: dict[str, str] = {}
    used: set[str] = set()
    for pid, doc in corpus.items():
        stem = _safe_stem(pid)
        cand, n = stem, 1
        while cand in used:
            cand, n = f"{stem}__{n}", n + 1
        used.add(cand)
        fn2id[cand] = pid
        body = (doc.get("title", "") + "\n" + doc.get("text", "")).strip()
        with open(os.path.join(out_dir, cand + ".txt"), "w", encoding="utf-8") as f:
            f.write(body + "\n")
    return out_dir, fn2id


def id_of_factory(fn2id: dict[str, str]) -> Callable[[object], str]:
    """snippet → passage id：取 source_path 的文件名 stem，经 fn2id 反查原 id。"""
    def id_of(snip):
        base = os.path.basename(str(snip.source_path).replace("\\", "/"))
        stem = base[:-4] if base.endswith(".txt") else base
        return fn2id.get(stem, stem)
    return id_of
