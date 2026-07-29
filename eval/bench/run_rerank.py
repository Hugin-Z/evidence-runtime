"""第四臂 benchmark：cross-encoder 精排。

**不重跑前三臂、不重建索引**：读取 BENCH 的逐 query 私有衔接件（或旧 NFCorpus 公开附录），
对 **RRF top-10** 候选做精排：
  1. 从 corpus.jsonl 取候选 passage 全文（title+text，与检索时口径一致）、queries.jsonl 取 query 原文；
  2. POST /v1/rerank（独立 llama-server, bge-reranker-v2-m3），按 relevance_score 降序重排；
  3. 在重排序列上算 5 指标（recall@5 / recall@10 / MRR@10 / precision@5 / nDCG@10）。

同时从逐 query 记录**重算前三臂**指标并与主表核对（校验 harness 一致性；偏差应≈0，>2% 视为失败停批）。
精排臂 query 用**原文**（cross-encoder 原生处理中英，不需 jieba）。

用法：
    python eval/bench/run_rerank.py \
        --details-file .local/ZH-Covid-v2.private.json \
        --data-dir eval/bench/data/covid --split dev --lang zh \
        --src-commit a96c03d --out-tag ZH-Covid-v2

CovidRetrieval 的公开输出只含聚合指标，不写 query/qid/qrels/逐 query 排名。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from eval.bench import loader
from eval.bench.reranker import Reranker
from eval.metrics import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k

MAX_DOC_CHARS = 6000   # 头截断上限：中文 ~1 token/char，6000 char + query 远低于模型 8192 max
METRICS = ["recall@5", "recall@10", "MRR@10", "precision@5", "nDCG@10"]
ARMS4 = ["lexical", "vector", "rrf", "rerank"]
LABEL = {"lexical": "LEXICAL", "vector": "VECTOR", "rrf": "RRF（双路）", "rerank": "RERANK（RRF→精排）"}

_ROW = re.compile(r"^\|\s*(\S.*?)\s*\|\s*(\[.*?\])\s*\|\s*(\[.*?\])\s*\|\s*(\[.*?\])\s*\|\s*(\[.*?\])\s*\|\s*$")
_MAIN = re.compile(r"^\|\s*(LEXICAL|VECTOR|RRF（双路）)\s*\|(.+)\|\s*$")


def parse_result_file(path: str):
    """返回 (appendix, main_table)。appendix: qid -> {gold,lexical,vector,rrf}（list[str]）；
    main_table: {arm -> {metric -> float}}（已提交主表，作核对基准）。"""
    appendix, main = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _MAIN.match(line)
            if m:
                arm = {"LEXICAL": "lexical", "VECTOR": "vector", "RRF（双路）": "rrf"}[m.group(1)]
                nums = [float(x) for x in re.findall(r"[0-9]*\.[0-9]+", m.group(2))]
                if len(nums) == len(METRICS):
                    main[arm] = dict(zip(METRICS, nums))
                continue
            r = _ROW.match(line)
            if not r:
                continue
            qid = r.group(1)
            if qid == "qid":   # 表头
                continue
            try:
                gold = ast.literal_eval(r.group(2))
                lex = ast.literal_eval(r.group(3))
                vec = ast.literal_eval(r.group(4))
                rrf = ast.literal_eval(r.group(5))
            except (ValueError, SyntaxError):
                continue
            appendix[qid] = {"gold": gold, "lexical": lex, "vector": vec, "rrf": rrf}
    return appendix, main


def parse_details_file(path: str):
    """Read the private ``bench-private-details-v1`` handoff artifact."""

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema_version") != "bench-private-details-v1":
        raise ValueError("details-file 不是 bench-private-details-v1")
    queries = doc.get("queries")
    metrics = doc.get("metrics")
    if not isinstance(queries, dict) or not isinstance(metrics, dict):
        raise ValueError("details-file 缺 queries/metrics")
    return queries, metrics


def passage_text(corpus, pid):
    doc = corpus.get(pid, {})
    return (doc.get("title", "") + "\n" + doc.get("text", "")).strip()


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--details-file",
        help="run_bench --details-out 生成的未跟踪私有 JSON（Covid 必须用此入口）",
    )
    source.add_argument(
        "--result-file",
        help="旧 BENCH Markdown 附录入口；只用于许可允许公开明细的历史轨道（如 NFCorpus）",
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--src-commit", required=True, help="前三臂来源 commit（附录冻结自此）")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--out-dir", default=None, help="结果落盘目录（默认 eval/bench/results；跑批时先落仓库外临时目录）")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    args = ap.parse_args()

    corpus, queries, qrels = loader.load_beir(args.data_dir, split=args.split)
    if args.details_file:
        appendix, main = parse_details_file(args.details_file)
        source_name = os.path.basename(args.details_file)
        source_description = "未跟踪逐 query 衔接件"
    else:
        appendix, main = parse_result_file(args.result_file)
        source_name = os.path.basename(args.result_file)
        source_description = "旧公开附录"
    qids = [q for q in appendix if q in qrels]
    if not qids:
        raise SystemExit("[rerank] 逐 query 记录与 qrels 无交集；Covid 请传 --details-file")
    print(f"[rerank] {source_description} {len(appendix)} query；与 qrels 交集 {len(qids)}", flush=True)
    if not main:
        print("[rerank] 警告：未解析到已提交主表，跳过核对", flush=True)

    rr = Reranker()
    agg = {arm: {m: [] for m in METRICS} for arm in ARMS4}
    ndcg_exp = {arm: [] for arm in ARMS4}     # 诊断口径（exponential）
    per = {}
    times = []
    n_truncated = 0            # 头截断的候选**槽位**数（跨 query 含重复）
    truncated_pids = set()     # 头截断的**去重** passage id 数
    t0 = time.time()
    for n, q in enumerate(qids, 1):
        qtext = queries[q]
        rrf_ids = appendix[q]["rrf"]
        gold = loader.relevant_set(qrels[q])
        # 前三臂：从附录直接取排名（复用冻结结果，不重跑）
        for arm in ("lexical", "vector", "rrf"):
            ranked = appendix[q][arm]
            agg[arm]["recall@5"].append(recall_at_k(ranked, gold, 5))
            agg[arm]["recall@10"].append(recall_at_k(ranked, gold, 10))
            agg[arm]["MRR@10"].append(mrr_at_k(ranked, gold, 10))
            agg[arm]["precision@5"].append(precision_at_k(ranked, gold, 5))
            agg[arm]["nDCG@10"].append(ndcg_at_k(ranked, qrels[q], 10, gain="linear"))
            ndcg_exp[arm].append(ndcg_at_k(ranked, qrels[q], 10, gain="exponential"))
        # 第四臂：精排 RRF top-10
        if rrf_ids:
            texts = []
            for pid in rrf_ids:
                t = passage_text(corpus, pid)
                if len(t) > MAX_DOC_CHARS:
                    t = t[:MAX_DOC_CHARS]
                    n_truncated += 1
                    truncated_pids.add(pid)
                texts.append(t)
            reranked, dt = rr.rerank(qtext, rrf_ids, texts)
            times.append(dt)
        else:
            reranked = []
        per[q] = reranked
        agg["rerank"]["recall@5"].append(recall_at_k(reranked, gold, 5))
        agg["rerank"]["recall@10"].append(recall_at_k(reranked, gold, 10))
        agg["rerank"]["MRR@10"].append(mrr_at_k(reranked, gold, 10))
        agg["rerank"]["precision@5"].append(precision_at_k(reranked, gold, 5))
        agg["rerank"]["nDCG@10"].append(ndcg_at_k(reranked, qrels[q], 10, gain="linear"))
        ndcg_exp["rerank"].append(ndcg_at_k(reranked, qrels[q], 10, gain="exponential"))
        if n % 100 == 0:
            print(f"  [rerank] {n}/{len(qids)} ({n/max(time.time()-t0,1e-6):.1f} q/s)", flush=True)

    total = {arm: {m: statistics.mean(agg[arm][m]) for m in METRICS} for arm in ARMS4}
    total_exp = {arm: statistics.mean(ndcg_exp[arm]) if ndcg_exp[arm] else 0.0 for arm in ARMS4}

    # ── 前三臂核对（重算 vs 已提交主表）───────────────────────────
    max_dev = 0.0
    check_rows = []
    for arm in ("lexical", "vector", "rrf"):
        for m in METRICS:
            recomputed = total[arm][m]
            committed = main.get(arm, {}).get(m)
            if committed is None:
                continue
            dev = abs(recomputed - committed)
            max_dev = max(max_dev, dev)
            check_rows.append((arm, m, recomputed, committed, dev))
    print(f"[rerank] 前三臂重算 vs 主表 最大绝对偏差 = {max_dev:.4g}", flush=True)
    if max_dev > 0.02:
        print(f"[rerank] 偏差 > 2% → 停批（harness 与冻结结果不一致）", flush=True)
        sys.exit(2)

    # 计时统计
    tstat = {}
    if times:
        st = sorted(times)
        tstat = {"n": len(times), "mean_ms": statistics.mean(times) * 1000,
                 "median_ms": statistics.median(times) * 1000,
                 "p95_ms": st[min(len(st) - 1, int(0.95 * len(st)))] * 1000,
                 "max_ms": max(times) * 1000, "total_s": sum(times)}

    # ── 写四臂结果文件 ────────────────────────────────────────
    short_head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "N/A"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    R = [f"# RERANK {args.out_tag} — 四臂 × 5 指标（{date.today().isoformat()}）\n"]
    R.append("## 配置快照")
    R.append(f"- 前三臂来源: **{source_name}** @ commit `{args.src_commit}`"
             f"（{source_description}中的 RRF top-10 冻结，本次**不重跑、不重建索引**）")
    R.append(f"- 数据集: {args.data_dir}（{args.split}）；query 数 {len(qids)}；**评测单位 = document**（C-1）")
    R.append("- 第四臂 RERANK: 对 **RRF document top-10** 候选做 cross-encoder 精排（bge-reranker-v2-m3-Q8_0，"
             "llama.cpp b10012 win-cuda-12.4，`/v1/rerank`，port 1235，-ngl 99）")
    R.append("- 精排输入: query 原文 + 候选 document 全文（title+text，与检索口径一致）；分数为原始 logit（只用其序）")
    R.append(f"- 头截断规则（既定）: document 全文 **头截断至 {MAX_DOC_CHARS} 字符**再送 reranker（6000 char + query ≪ 模型 8192-token max）——"
             f"本次 **{len(truncated_pids)}** 个去重 document（占候选槽位 **{n_truncated}** 次，跨 query 含重复）被头截断。"
             "（注：向量臂建库按 800 字符硬切 chunk、服务端 context≥8192 无截断；精排见 document 头 6000 字符，上下文更全。此 6000 字符截断 v1 已披露，非勘误对象。）")
    R.append(f"- nDCG@10 主口径 = **linear-gain**（trec_eval/BEIR 官方）；exponential 为诊断值（见下表，勿与公开数字并置）")
    R.append(f"- 超时/重试: timeout {rr.timeout}s，retries {rr.retries}（总尝试 {rr.retries+1}），线性退避 {rr.backoff}s")
    R.append(f"- git_head: `{short_head}`；工作树: {'**dirty**' if dirty else 'clean（可复现）'}")
    R.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")

    R.append("## 四臂主表（全量，均值；nDCG@10 = linear-gain 主口径）\n")
    R.append("| 臂 | " + " | ".join(METRICS) + " |")
    R.append("|---|" + "|".join(["---:"] * len(METRICS)) + "|")
    for arm in ARMS4:
        R.append(f"| {LABEL[arm]} | " + " | ".join(f"{total[arm][m]:.4f}" for m in METRICS) + " |")
    R.append("")
    R.append("### nDCG@10 诊断：exponential-gain（**诊断值，勿与公开数字并置**）\n")
    R.append("| 臂 | nDCG@10 (linear·主) | nDCG@10 (exponential·诊断) |")
    R.append("|---|---:|---:|")
    for arm in ARMS4:
        R.append(f"| {LABEL[arm]} | {total[arm]['nDCG@10']:.4f} | {total_exp[arm]:.4f} |")
    R.append("")
    # RERANK vs RRF 增量
    R.append("### RERANK − RRF（精排增量）\n")
    R.append("| 指标 | RRF | RERANK | Δ |")
    R.append("|---|---:|---:|---:|")
    for m in METRICS:
        d = total["rerank"][m] - total["rrf"][m]
        R.append(f"| {m} | {total['rrf'][m]:.4f} | {total['rerank'][m]:.4f} | {d:+.4f} |")
    R.append("\n> 注：RERANK 仅对 RRF 的同一 top-10 候选集**重排序**，故 **recall@10 恒等于 RRF**"
             "（集合不变）；变化只体现在 recall@5 / MRR@10 / precision@5 / nDCG@10（名次重排）。\n")

    R.append("## 前三臂核对（逐 query 记录重算 vs 已提交主表）\n")
    R.append(f"- 最大绝对偏差 = **{max_dev:.4g}**（阈值 2%；≈0 表示 harness 与冻结记录一致，前三臂确为复用）\n")
    R.append("| 臂 | 指标 | 重算 | 主表 | |Δ| |")
    R.append("|---|---|---:|---:|---:|")
    for arm, m, rc, cm, dev in check_rows:
        R.append(f"| {LABEL[arm]} | {m} | {rc:.4f} | {cm:.4f} | {dev:.4g} |")
    R.append("")

    if tstat:
        R.append("## 精排耗时（每 query 一次 /v1/rerank，10 候选）\n")
        R.append(f"- 样本 {tstat['n']} query；每 query：mean **{tstat['mean_ms']:.1f} ms** / "
                 f"median {tstat['median_ms']:.1f} ms / p95 {tstat['p95_ms']:.1f} ms / max {tstat['max_ms']:.1f} ms")
        R.append(f"- 全量精排累计 {tstat['total_s']:.1f}s\n")

    if args.lang == "zh":
        R.append("## 公开记录边界\n")
        R.append(
            "CovidRetrieval 上游没有明确再分发许可。本文件只保留项目自产聚合指标和运行配置，"
            "不分发 query 文本、qid、document id、qrels/gold 映射或逐 query 排名。"
            "复算所需逐 query 数据留在 `--details-file` 指向的未跟踪私有衔接件中。\n"
        )
    else:
        R.append("## 附录：逐 query RERANK top-10 passage id\n")
        R.append("<details><summary>展开</summary>\n")
        R.append("| qid | gold(rel≥1) | RRF top-10（输入） | RERANK top-10（精排后） |")
        R.append("|---|---|---|---|")
        for q in qids:
            g = sorted(loader.relevant_set(qrels[q]))
            R.append(f"| {q} | {g} | {appendix[q]['rrf']} | {per[q]} |")
        R.append("\n</details>\n")

    outdir = args.out_dir or os.path.join(ROOT, "eval", "bench", "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"RERANK-{args.out_tag}-{date.today().isoformat()}.md")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    print(f"[rerank] 头截断: {len(truncated_pids)} 去重 passage / {n_truncated} 候选槽位")
    print("[rerank] TOTAL:")
    for arm in ARMS4:
        print(f"   {arm:8s} " + "  ".join(f"{m}={total[arm][m]:.4f}" for m in METRICS))
    if tstat:
        print(f"[rerank] per-query: mean {tstat['mean_ms']:.1f}ms  median {tstat['median_ms']:.1f}ms  "
              f"p95 {tstat['p95_ms']:.1f}ms  max {tstat['max_ms']:.1f}ms")
    print(f"[rerank] wrote {os.path.relpath(outpath, ROOT)}")


if __name__ == "__main__":
    main()
