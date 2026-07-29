"""Rerank 服务正确性冒烟与可复现记录。

对 **真数据集**（NFCorpus + CovidRetrieval）取若干 (query, 相关 doc, 无关 doc) 三元组，POST
/v1/rerank，验证相关分 > 无关分且分数有区分度（非退化）。NFCorpus 按 CC BY-SA 留归因后的
样本；CovidRetrieval 的公开记录只写 project-local pair id、原始分数/delta 与聚合，不写
query/qid/document id/qrels/语料 hash。模型只记文件名与 hash，不落本机绝对路径。

用法：
    python eval/bench/rerank_smoke.py --nfcorpus eval/bench/data/nfcorpus --covid eval/bench/data/covid \
        --model-path "<path-to-model.gguf>" \
        --llama-version b10012-win-cuda-12.4 --rerank-ctx 8192 --out <TEMP>/rerank_smoke.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from eval.bench import loader
from eval.bench.reranker import Reranker, DEFAULT_URL

MAX_DOC_CHARS = 6000


def sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha(corpus: dict) -> str:
    h = hashlib.sha256()
    for pid in sorted(corpus):
        h.update(pid.encode("utf-8"))
        h.update((corpus[pid].get("title", "") + "\x00" + corpus[pid].get("text", "")).encode("utf-8"))
    return h.hexdigest()[:12]


def ptext(corpus, pid):
    d = corpus.get(pid, {})
    return (d.get("title", "") + "\n" + d.get("text", "")).strip()[:MAX_DOC_CHARS]


def pick_triples(data_dir, split, lang, n=3):
    """取 n 个 (qid, query, rel_pid, rel_score, irr_pid) —— rel 来自 qrels≥1，irr 取非相关 passage。"""
    corpus, queries, qrels = loader.load_beir(data_dir, split=split)
    all_ids = sorted(corpus)
    triples = []
    for qid in sorted(queries):
        rels = {p: s for p, s in qrels.get(qid, {}).items() if s >= 1 and p in corpus}
        if not rels:
            continue
        rel_pid = sorted(rels, key=lambda p: (-rels[p], p))[0]
        irr_pid = next((p for p in all_ids if p not in qrels.get(qid, {})), None)
        if irr_pid is None:
            continue
        triples.append((qid, queries[qid], rel_pid, rels[rel_pid], irr_pid))
        if len(triples) >= n:
            break
    return corpus, corpus_sha(corpus), triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfcorpus", required=True)
    ap.add_argument("--covid", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--llama-version", required=True)
    ap.add_argument("--rerank-ctx", required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_hash = sha256_file(args.model_path)
    model_size = os.path.getsize(args.model_path)
    rr = Reranker(url=args.url)

    rec = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "service": {"url": args.url, "llama_version": args.llama_version,
                       "rerank_context": args.rerank_ctx,
                       "model_file": os.path.basename(args.model_path),
                       "model_size_bytes": model_size,
                       "model_sha256": model_hash},
           "datasets": {}}
    all_dir = 0
    all_ok = 0
    all_scores = []
    for name, ddir, split, lang in [("NFCorpus", args.nfcorpus, "test", "en"),
                                    ("CovidRetrieval", args.covid, "dev", "zh")]:
        corpus, csha, triples = pick_triples(ddir, split, lang)
        dataset_record = {"split": split, "pairs": []}
        if name == "NFCorpus":
            dataset_record["corpus_sha256_12"] = csha
        rec["datasets"][name] = dataset_record
        for qid, qtext, rel_pid, rel_score, irr_pid in triples:
            scores, dt = rr.scores(qtext, [ptext(corpus, rel_pid), ptext(corpus, irr_pid)])
            s_rel, s_irr = scores[0], scores[1]
            ok = s_rel > s_irr
            all_dir += 1
            all_ok += 1 if ok else 0
            all_scores += [s_rel, s_irr]
            pair = {
                "score_rel": s_rel,
                "score_irr": s_irr,
                "delta": s_rel - s_irr,
                "direction_ok": ok,
                "elapsed_ms": round(dt * 1000, 1),
            }
            if name == "NFCorpus":
                pair.update(
                    {
                        "qid": qid,
                        "query": qtext,
                        "rel_pid": rel_pid,
                        "rel_qrels_score": rel_score,
                        "irr_pid": irr_pid,
                    }
                )
            else:
                pair["pair_id"] = f"covid-pair-{len(dataset_record['pairs']) + 1:02d}"
            dataset_record["pairs"].append(pair)
        if name == "CovidRetrieval":
            deltas = [item["delta"] for item in dataset_record["pairs"]]
            dataset_record["aggregate"] = {
                "pair_count": len(deltas),
                "direction_ok": sum(item["direction_ok"] for item in dataset_record["pairs"]),
                "direction_pass": all(item["direction_ok"] for item in dataset_record["pairs"]),
                "delta_min": min(deltas),
                "delta_max": max(deltas),
                "delta_mean": sum(deltas) / len(deltas),
                "delta_mean_method": "arithmetic mean of pairs[].delta at stored precision",
            }
            dataset_record["upstream_details_retained"] = False

    uniq = len(set(round(s, 6) for s in all_scores))
    degenerate = any(abs(s) < 1e-12 for s in all_scores) and uniq <= 2
    rec["verdict"] = {"direction": f"{all_ok}/{all_dir}", "direction_pass": all_ok == all_dir,
                      "distinct_scores": f"{uniq}/{len(all_scores)}",
                      "score_range": [min(all_scores), max(all_scores)],
                      "degenerate": degenerate,
                      "gate_pass": all_ok == all_dir and not degenerate}

    # —— 写留档（人读摘要 + 完整 JSON）——
    lines = [f"# Rerank 正确性冒烟留档（{rec['generated_at']}）\n",
             f"服务: {args.llama_version} @ {args.url}  |  rerank_context={args.rerank_ctx}",
             f"模型: {os.path.basename(args.model_path)}  size={model_size}  sha256={model_hash}",
             f"判定: 方向 {rec['verdict']['direction']}（pass={rec['verdict']['direction_pass']}）  "
             f"区分度 {rec['verdict']['distinct_scores']}  退化={degenerate}  **GATE_PASS={rec['verdict']['gate_pass']}**\n"]
    for name, d in rec["datasets"].items():
        if name == "NFCorpus":
            lines.append(
                f"## {name}（corpus_sha={d['corpus_sha256_12']}, split={d['split']}；"
                "CC BY-SA 4.0，归因见 THIRD_PARTY_NOTICES.md）"
            )
        else:
            lines.append(f"## {name}（split={d['split']}；公开聚合记录）")
        for p in d["pairs"]:
            if name == "NFCorpus":
                lines.append(f"- [{p['qid']}] rel={p['score_rel']:.5f} irr={p['score_irr']:.5f} "
                             f"Δ={p['delta']:+.5f} ok={p['direction_ok']}  "
                             f"(rel_pid={p['rel_pid']} qrels={p['rel_qrels_score']}, irr_pid={p['irr_pid']}, {p['elapsed_ms']}ms)")
                lines.append(f"    query: {p['query']}")
            else:
                lines.append(
                    f"- [{p['pair_id']}] rel={p['score_rel']:.5f} irr={p['score_irr']:.5f} "
                    f"Δ={p['delta']:+.5f} ok={p['direction_ok']} ({p['elapsed_ms']}ms)"
                )
        if name == "CovidRetrieval":
            a = d["aggregate"]
            lines.append(
                f"- aggregate: pairs={a['pair_count']}, direction_ok={a['direction_ok']}/"
                f"{a['pair_count']}, Δ_min={a['delta_min']:+.5f}, "
                f"Δ_max={a['delta_max']:+.5f}, Δ_mean={a['delta_mean']:+.5f}"
            )
            lines.append(
                "- [2026-07-29 公开前许可收口] query/qid/document id/qrels/语料 hash "
                "均不写入公开记录；逐对分数和 delta 保留，聚合可由 pairs[].delta 复算。"
            )
        lines.append("")
    lines.append("## 完整 JSON\n```json")
    lines.append(json.dumps(rec, ensure_ascii=False, indent=2))
    lines.append("```")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[smoke] GATE_PASS={rec['verdict']['gate_pass']}  direction={rec['verdict']['direction']}  "
          f"distinct={rec['verdict']['distinct_scores']}  degenerate={degenerate}")
    print(f"[smoke] wrote {args.out}")
    sys.exit(0 if rec["verdict"]["gate_pass"] else 3)


if __name__ == "__main__":
    main()
