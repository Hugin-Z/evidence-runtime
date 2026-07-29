"""下载 C-MTEB CovidRetrieval（parquet，官方 HF resolve URL）→ 转 BEIR 布局。

产出 eval/bench/data/covid/{corpus.jsonl, queries.jsonl, qrels/dev.tsv}（gitignored）。
读 parquet 需可选依赖 pyarrow。下载物不入 Git；使用者自行确认上游许可。
"""
from __future__ import annotations

import json
import os
import urllib.request

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "eval", "bench", "data", "covid")
RAW = os.path.join(OUT, "raw")

URLS = {
    "corpus": "https://huggingface.co/datasets/C-MTEB/CovidRetrieval/resolve/main/data/corpus-00000-of-00001-c6672bba0449b139.parquet",
    "queries": "https://huggingface.co/datasets/C-MTEB/CovidRetrieval/resolve/main/data/queries-00000-of-00001-2b1399da56079d94.parquet",
    "qrels": "https://huggingface.co/datasets/C-MTEB/CovidRetrieval-qrels/resolve/main/data/dev-00000-of-00001-94f5bc4c0377691e.parquet",
}


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    raise KeyError(f"没有匹配列 {cands}（实有 {cols}）")


def main():
    os.makedirs(os.path.join(OUT, "qrels"), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    for name, url in URLS.items():
        dst = os.path.join(RAW, name + ".parquet")
        if not os.path.exists(dst):
            print(f"[fetch] {name} ← {url}", flush=True)
            urllib.request.urlretrieve(url, dst)
        print(f"[fetch] {name}: {os.path.getsize(dst)} bytes", flush=True)

    corpus = pq.read_table(os.path.join(RAW, "corpus.parquet"))
    queries = pq.read_table(os.path.join(RAW, "queries.parquet"))
    qrels = pq.read_table(os.path.join(RAW, "qrels.parquet"))
    print("[schema] corpus", corpus.column_names, "rows", corpus.num_rows)
    print("[schema] queries", queries.column_names, "rows", queries.num_rows)
    print("[schema] qrels", qrels.column_names, "rows", qrels.num_rows)

    c_id = _pick(corpus.column_names, "_id", "id")
    c_text = _pick(corpus.column_names, "text", "content")
    c_title = None
    for t in ("title",):
        if t in [c.lower() for c in corpus.column_names]:
            c_title = _pick(corpus.column_names, t)
    q_id = _pick(queries.column_names, "_id", "id")
    q_text = _pick(queries.column_names, "text", "query")
    r_qid = _pick(qrels.column_names, "qid", "query-id", "query_id")
    r_pid = _pick(qrels.column_names, "pid", "corpus-id", "corpus_id", "docid", "cid")
    r_score = _pick(qrels.column_names, "score", "relevance", "rel")

    cd = corpus.to_pydict()
    with open(os.path.join(OUT, "corpus.jsonl"), "w", encoding="utf-8") as f:
        titles = cd[c_title] if c_title else [""] * corpus.num_rows
        for i in range(corpus.num_rows):
            f.write(json.dumps({"_id": str(cd[c_id][i]), "title": titles[i] or "",
                                "text": cd[c_text][i] or ""}, ensure_ascii=False) + "\n")
    qd = queries.to_pydict()
    with open(os.path.join(OUT, "queries.jsonl"), "w", encoding="utf-8") as f:
        for i in range(queries.num_rows):
            f.write(json.dumps({"_id": str(qd[q_id][i]), "text": qd[q_text][i] or ""}, ensure_ascii=False) + "\n")
    rd = qrels.to_pydict()
    with open(os.path.join(OUT, "qrels", "dev.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for i in range(qrels.num_rows):
            f.write(f"{rd[r_qid][i]}\t{rd[r_pid][i]}\t{int(rd[r_score][i])}\n")

    nq = len(set(str(x) for x in rd[r_qid]))
    print(f"[done] corpus.jsonl={corpus.num_rows}  queries.jsonl={queries.num_rows}  "
          f"qrels rows={qrels.num_rows} (unique qids={nq})")


if __name__ == "__main__":
    main()
