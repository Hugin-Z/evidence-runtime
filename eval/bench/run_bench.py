"""eval/bench/run_bench.py — 标准 benchmark 三臂跑批 v2（BEIR NFCorpus / C-MTEB CovidRetrieval）。

三臂 = LEXICAL / VECTOR / RRF，统一 top_k=10、rrf_k=60。**评测单位 = document**（C-1）：每路按 chunk
过取、保序折叠出 ≥N_DOC 唯一 document，再以 document 排名进 RRF（`core.fusion` 未改，喂 document 哨兵）。
建库用 bench 层硬切切块器（A-1，含指纹 collection）。nDCG 双口径（B-1）：linear 主、exponential 诊断。
中文（--lang zh）：词法臂 query 经 jieba 分词（D13）；向量臂用原始 query。**core 一行不改**。

用法：
    python eval/bench/run_bench.py --dataset nfcorpus --lang en --data-dir eval/bench/data/nfcorpus --out-tag EN-NFCorpus-v2 [--out-dir DIR]
    python eval/bench/run_bench.py --dataset covid --lang zh --data-dir eval/bench/data/covid \
        --out-tag ZH-Covid-v2 --details-out .local/ZH-Covid-v2.private.json [--out-dir DIR]

CovidRetrieval 的公开 Markdown **只写聚合指标**。若第四臂需要逐 query 排名，须显式用
``--details-out`` 把含 qid/qrels/排名的私有衔接件写到 gitignored 或仓库外路径；该文件不得提交。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from eval.bench import loader, index, docrank
from eval.metrics import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k
from evidence.core import Settings, embeddings
from evidence.core.fusion import fuse
from evidence.core.types import RetrievalPath
from evidence.core.retrievers.lexical import RipgrepRetriever

TOP_K = 10
REQUIRED_CTX = 8192          # 前置：embedding 服务 context 须 ≥ 此值（覆盖硬切上限，杜绝服务端截断）
TRUNC_COS_THRESHOLD = 0.98   # 截断经验探测阈值：cos(tail@3000tok) ≥ 此值视为“尾部未参与嵌入”=截断
METRICS = ["recall@5", "recall@10", "MRR@10", "precision@5", "nDCG@10"]
ARMS = ["lexical", "vector", "rrf"]
LABEL = {"lexical": "LEXICAL", "vector": "VECTOR", "rrf": "RRF（双路）"}

PUBLIC = {
    "nfcorpus": [("BM25（Anserini）", "0.325", "BEIR 论文 Table 2（**linear-gain** trec_eval 口径）；词法**排序模型**（BM25 加权），本项目词法臂是 rg 命中率初排、非 BM25"),
                 ("colbertv2 / GTE-ModernColBERT（neural）", "0.338 / 0.379", "晚期交互/大模型 neural；本项目向量臂为 0.6b 小模型、无精排")],
    "covid": [("bge-large-zh", "0.7406", "large 级中文 embedding；本项目为 qwen3-0.6b 小模型、无精排"),
              ("mE5-large", "0.7551", "large 级多语；本项目词法臂用 jieba 通用分词（非训练态）+ rg 命中率初排、非 BM25")],
}


def git_provenance():
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "--short", "HEAD") or "N/A", bool(g("status", "--porcelain"))


def corpus_sha(corpus: dict) -> str:
    h = hashlib.sha256()
    for pid in sorted(corpus):
        h.update(pid.encode("utf-8"))
        h.update((corpus[pid].get("title", "") + "\x00" + corpus[pid].get("text", "")).encode("utf-8"))
    return h.hexdigest()[:12]


def public_dataset_config_line(
    *,
    dataset: str,
    split: str,
    corpus_count: int,
    query_count: int,
    corpus_digest: str,
) -> str:
    """Render a license-aware public dataset snapshot."""

    line = (
        f"- 数据集: **{dataset}**（{split}），corpus **{corpus_count}** passage / "
        f"**{query_count}** query"
    )
    # NFCorpus is attributed under CC BY-SA. CovidRetrieval has no explicit
    # redistribution license, so its public record omits the derived fingerprint.
    if dataset != "covid":
        line += f"；corpus_content_sha256=`{corpus_digest}`"
    return line


def embed_truncation_cos(embedder, *, filler_tokens: int = 3000, tail_reps: int = 300) -> float:
    """经验探测服务端截断：两输入仅在 ~filler_tokens 之后的尾部不同。

    若服务在更早处截断输入，两者尾部被切掉 → 嵌入相同（cos≈1）；cos < 阈值 ⇒ 尾部内容参与
    了嵌入 ⇒ 该 token 深度无截断。比 `/api/v0/models` 的 `loaded_context_length`（对 embedding
    模型实测不可靠，reload 后仍报旧值）稳健，也比 `usage.prompt_tokens`（LM Studio 返 0）可用。
    返回 cos(A,B)，并把探测方式写入结果快照。
    """
    import math
    prefix = "alpha " * filler_tokens
    a = embedder.encode([prefix + "apple " * tail_reps])[0]
    b = embedder.encode([prefix + "zebra " * tail_reps])[0]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 1.0


def probe_max_context(model_id: str, api_base: str) -> int | None:
    """只取 `/api/v0/models` 的 `max_context_length`（此字段可靠，仅作记录，不作闸门）。"""
    host = api_base.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(host + "/api/v0/models", timeout=6) as r:
            data = json.loads(r.read().decode("utf-8"))
        for m in data.get("data", []):
            if m.get("id") == model_id:
                return m.get("max_context_length")
    except Exception:
        pass
    return None


def fmt_table(rows, arms=ARMS):
    out = ["| 臂 | " + " | ".join(METRICS) + " |", "|---|" + "|".join(["---:"] * len(METRICS)) + "|"]
    for arm in arms:
        out.append(f"| {LABEL[arm]} | " + " | ".join(f"{rows[arm][m]:.4f}" for m in METRICS) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--out-dir", default=None, help="结果落盘目录（默认 eval/bench/results；跑批时先落仓库外临时目录）")
    ap.add_argument(
        "--details-out",
        default=None,
        help="可选私有逐 query JSON（含 qid/qrels/排名，供 rerank；必须留在 gitignored 或仓库外）",
    )
    ap.add_argument("--work-dir", default=None, help="稳定工作目录（docs+chroma）；给定则可跨进程 resume 续建（默认每次 mkdtemp）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    jieba_ver = None
    tokenize = (lambda s: s)
    if args.lang == "zh":
        import jieba
        jieba_ver = jieba.__version__
        tokenize = (lambda s: " ".join(w for w in jieba.cut(s) if w.strip()))

    corpus, queries, qrels = loader.load_beir(args.data_dir, split=args.split)
    qids = sorted(queries)
    print(f"[bench] {args.dataset}: corpus {len(corpus)} / queries {len(qids)} (split {args.split})", flush=True)

    if args.work_dir:
        tmp = args.work_dir
        os.makedirs(tmp, exist_ok=True)
        print(f"[bench] work-dir（稳定，可 resume）: {tmp}", flush=True)
    else:
        tmp = tempfile.mkdtemp(prefix=f"bench_{args.dataset}_")
    cdir, fn2id = loader.materialize_corpus(corpus, os.path.join(tmp, "docs"))
    id_of = loader.id_of_factory(fn2id)
    print(f"[bench] materialized {len(fn2id)} passages → rg dir", flush=True)

    # —— 前置闸门：embedding 服务端**无截断**（A-1）——经验探测，稳健于不可靠的 loaded_ctx 字段 ——
    st = Settings()
    embedder = embeddings.load_embedder(st)
    ctx_max = probe_max_context(embedder.model_id, st.embedding_api_base)
    trunc_cos = embed_truncation_cos(embedder)     # tail past ~3000 token 的差异
    print(f"[bench] 截断经验探测: cos(tail@3000tok)={trunc_cos:.4f} (阈值<{TRUNC_COS_THRESHOLD}；max_ctx={ctx_max})", flush=True)
    if trunc_cos >= TRUNC_COS_THRESHOLD:
        raise SystemExit(f"[bench] cos={trunc_cos:.4f} ≥ {TRUNC_COS_THRESHOLD}——3000-token 之后的内容未参与嵌入 = 服务端截断；"
                         f"请把 embedding 模型 context 设 ≥ {REQUIRED_CTX} 并装载后重试；停。")

    dim = len(embedder.encode(["dim-probe"])[0])
    settings = Settings(corpus_dir=cdir, vector_db_path=os.path.join(tmp, "chroma"))
    print(f"[bench] building vector index (batched, real embedder {embedder.model_id})…", flush=True)
    istats = index.build_index_batched(settings, embedder, batch_size=args.batch_size)
    print(f"[bench] index: {istats['chunks']} chunks in {istats['elapsed_s']}s "
          f"(collection={istats['collection']}, over_limit={istats['over_limit']}, max_len={istats['max_len']})", flush=True)
    if istats["over_limit"] != 0:      # 冗余守卫（build 已 raise），并作 ZH grep 断言基础
        raise SystemExit(f"[bench] over_limit={istats['over_limit']} ≠ 0——截断风险，停。")

    coll = istats["collection"]
    lex = RipgrepRetriever(settings)
    short_head, dirty = git_provenance()

    per = {}
    agg = {arm: {m: [] for m in METRICS} for arm in ARMS}
    ndcg_exp = {arm: [] for arm in ARMS}     # 诊断口径（exponential）
    t0 = time.time()
    for n, q in enumerate(qids, 1):
        raw = queries[q]
        lex_snips = lex.search(tokenize(raw), k=docrank.CHUNK_CAP)
        vec_snips = docrank.bench_vector_snippets(settings, embedder, coll, raw, k=docrank.CHUNK_CAP)
        lex_in = docrank.docs_to_fusion_input(
            docrank.fold_to_docs(lex_snips, id_of), RetrievalPath.LEXICAL)
        vec_in = docrank.docs_to_fusion_input(
            docrank.fold_to_docs(vec_snips, id_of), RetrievalPath.VECTOR)
        ranked = {
            "lexical": docrank.fused_doc_ids(fuse([lex_in], top_k=TOP_K)),
            "vector": docrank.fused_doc_ids(fuse([vec_in], top_k=TOP_K)),
            "rrf": docrank.fused_doc_ids(fuse([vec_in, lex_in], top_k=TOP_K)),
        }
        gold = loader.relevant_set(qrels[q])
        per[q] = {}
        for arm in ARMS:
            rk = ranked[arm]
            per[q][arm] = rk
            agg[arm]["recall@5"].append(recall_at_k(rk, gold, 5))
            agg[arm]["recall@10"].append(recall_at_k(rk, gold, 10))
            agg[arm]["MRR@10"].append(mrr_at_k(rk, gold, 10))
            agg[arm]["precision@5"].append(precision_at_k(rk, gold, 5))
            agg[arm]["nDCG@10"].append(ndcg_at_k(rk, qrels[q], 10, gain="linear"))
            ndcg_exp[arm].append(ndcg_at_k(rk, qrels[q], 10, gain="exponential"))
        if n % 100 == 0:
            print(f"  [bench] {n}/{len(qids)} queries ({n/max(time.time()-t0,1e-6):.1f} q/s)", flush=True)
    total = {arm: {m: statistics.mean(agg[arm][m]) for m in METRICS} for arm in ARMS}
    total_exp = {arm: statistics.mean(ndcg_exp[arm]) for arm in ARMS}

    if args.details_out:
        details_path = os.path.abspath(args.details_out)
        os.makedirs(os.path.dirname(details_path), exist_ok=True)
        details = {
            "schema_version": "bench-private-details-v1",
            "dataset": args.dataset,
            "split": args.split,
            "metrics": total,
            "queries": {
                q: {
                    "gold": sorted(loader.relevant_set(qrels[q])),
                    **per[q],
                }
                for q in qids
            },
        }
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(
            "[bench] PRIVATE details written (contains upstream qid/qrels; do not commit): "
            f"{details_path}",
            flush=True,
        )

    zh_nojieba_recall10 = None
    if args.lang == "zh":
        sample = qids[:20]
        nojieba_rec = []
        for q in sample:
            snips = lex.search(queries[q], k=docrank.CHUNK_CAP)   # 原始整句，无分词
            r = docrank.fused_doc_ids(fuse([docrank.docs_to_fusion_input(
                docrank.fold_to_docs(snips, id_of), RetrievalPath.LEXICAL)], top_k=TOP_K))
            nojieba_rec.append(recall_at_k(r, loader.relevant_set(qrels[q]), 10))
        zh_nojieba_recall10 = statistics.mean(nojieba_rec)

    # ── 写结果文件 ─────────────────────────────────────────────
    R = [f"# BENCH {args.out_tag} — Evidence Runtime 三臂 × 5 指标（{date.today().isoformat()}）\n"]
    R.append("## 配置快照")
    R.append(
        public_dataset_config_line(
            dataset=args.dataset,
            split=args.split,
            corpus_count=len(corpus),
            query_count=len(qids),
            corpus_digest=corpus_sha(corpus),
        )
    )
    R.append(f"- embedding_model: `{embedder.model_id}`（真 APIEmbedder / LM Studio），dim {dim}；max_context={ctx_max}")
    R.append(f"- **服务端无截断（经验探测）**: cos(tail@3000tok)=**{trunc_cos:.4f}** < {TRUNC_COS_THRESHOLD} ⇒ 3000-token 后内容仍参与嵌入"
             f"（loaded_ctx 字段对 embedding 不可靠不采信；须 ≥ {REQUIRED_CTX}，`lms ps` 实测 context=8192）")
    R.append(f"- 切块器: **{istats['chunker']}**（bench 层硬切，A-1）；max_chunk_chars={istats['max_chunk_chars']}；"
             f"实测最长 chunk={istats['max_len']} 字符；**over_limit={istats['over_limit']}**（建库断言无 chunk 超上限）")
    R.append(f"- 评测单位: **document**（C-1）；候选深度 N_DOC={docrank.N_DOC}、chunk_cap={docrank.CHUNK_CAP}（每路折叠 ≥N_DOC 唯一 doc 进 RRF）；rerank 候选 = RRF doc top-10")
    R.append(f"- top_k（三臂统一）: {TOP_K}　rrf_k: {Settings().rrf_k}　collection: `{coll}`　建库: 分批 batch={args.batch_size}，{istats['chunks']} chunk / {istats['elapsed_s']}s")
    R.append(f"- git_head: `{short_head}`；工作树: {'**dirty**' if dirty else 'clean（可复现）'}")
    if args.lang == "zh":
        R.append(f"- 中文词法臂: **jieba {jieba_ver} 分词**（D13，仅 query 侧、仅词法臂；向量臂用原始 query）")
    R.append(f"- 指标: {' / '.join(METRICS)}；**nDCG@10 主口径 = linear-gain**（trec_eval / BEIR 官方口径，公开对照唯一口径）")
    R.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")

    R.append("## 三臂主表（全量，均值；nDCG@10 = linear-gain 主口径）\n")
    R.append(fmt_table(total) + "\n")

    R.append("### nDCG@10 诊断：exponential-gain（**诊断值，勿与公开数字并置**）\n")
    R.append("| 臂 | nDCG@10 (linear·主) | nDCG@10 (exponential·诊断) |")
    R.append("|---|---:|---:|")
    for arm in ARMS:
        R.append(f"| {LABEL[arm]} | {total[arm]['nDCG@10']:.4f} | {total_exp[arm]:.4f} |")
    R.append("\n> exponential-gain `2^rel−1` 放大高相关档差异，仅内部诊断；公开榜（BM25 0.325 等）为 linear-gain，勿并置。\n")

    R.append("## 公开对照 + 差距解释（同表并列；对照不是目标）\n")
    R.append("| 对照 | 公开 nDCG@10 | 本项目对应臂 nDCG@10（linear 主口径） | 差距归因（如实） |")
    R.append("|---|---|---:|---|")
    for name, num, why in PUBLIC[args.dataset]:
        R.append(f"| {name} | {num} | VECTOR {total['vector']['nDCG@10']:.4f} / RRF {total['rrf']['nDCG@10']:.4f} / LEXICAL {total['lexical']['nDCG@10']:.4f} | {why} |")
    R.append("\n> 归因总述：本项目向量臂 = **qwen3-embedding-0.6b（小模型）零样本、无精排（rerank 直通）**；"
             "词法臂 = **rg 命中率初排（非 BM25 加权排序）**"
             + ("，中文经 jieba 通用分词（非训练态）。" if args.lang == "zh" else "。")
             + " 与公开榜（多为 large 级 + 训练态）的差距，**候选解释包括模型规模、排序机制差异；未做匹配控制实验，不作排他性归因**；数字原样交，不调参贴榜。\n")

    if args.lang == "zh":
        R.append("## 中文词法臂：jieba 分词抽样（20 条）+ 无分词退化对照\n")
        R.append(f"- **无分词退化小样**（前 20 条 query 原始整句喂词法臂）：recall@10 = **{zh_nojieba_recall10:.4f}**"
                 f"（预期近 0——合成轨道 L2a 问题重演）；jieba 后词法臂全量 recall@10 = {total['lexical']['recall@10']:.4f}。\n")
        R.append(
            "> 公开记录只保留上述聚合结果；抽样 query、qid、jieba token、qrels 与逐 query "
            "排名不写入 Markdown。需要第四臂时通过 `--details-out` 生成未跟踪的私有衔接件。\n"
        )

    if args.dataset == "covid":
        R.append("## 公开记录边界\n")
        R.append(
            "CovidRetrieval 上游没有明确再分发许可。本文件只保留项目自产聚合指标和运行配置，"
            "不分发 query 文本、qid、document id、qrels/gold 映射或逐 query 排名。"
            "逐 query 衔接件仅可通过 `--details-out` 写入 gitignored 或仓库外位置。\n"
        )
    else:
        R.append("## 附录：逐 query 三臂 top-10 document id\n")
        R.append("<details><summary>展开（每行一 query）</summary>\n")
        R.append("| qid | gold(rel≥1) | LEXICAL top-10 | VECTOR top-10 | RRF top-10 |")
        R.append("|---|---|---|---|---|")
        for q in qids:
            g = sorted(loader.relevant_set(qrels[q]))
            R.append(f"| {q} | {g} | {per[q]['lexical']} | {per[q]['vector']} | {per[q]['rrf']} |")
        R.append("\n</details>\n")

    outdir = args.out_dir or os.path.join(ROOT, "eval", "bench", "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"BENCH-{args.out_tag}-{date.today().isoformat()}.md")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    print("[bench] TOTAL (nDCG@10=linear 主口径):")
    for arm in ARMS:
        print(f"   {arm:8s} " + "  ".join(f"{m}={total[arm][m]:.4f}" for m in METRICS)
              + f"  [nDCG@10-exp={total_exp[arm]:.4f}]")
    print(f"[bench] wrote {outpath}")


if __name__ == "__main__":
    main()
