# RERANK ZH-Covid-v2 — 四臂 × 5 指标（2026-07-16）

## 配置快照
- 前三臂来源: **BENCH-ZH-Covid-v2-2026-07-16.md** @ commit `f07a5fe`（当时私有逐 query 记录中的 RRF top-10 冻结，本次**不重跑、不重建索引**）
- 数据集: eval/bench/data/covid（dev）；query 数 949；**评测单位 = document**
- 建库/检索快照（同步 BENCH v2，自包含）: 切块器 **hs800v1**（benchmark 层硬切，over_limit=0）；服务端无截断（经验探测 cos(tail@3000tok)=**0.6818** < 0.98）；embedding context ≥8192；候选深度 **N_DOC=20 / chunk_cap=200**（每路折叠 ≥20 唯一 doc 进 RRF）；中文词法臂 jieba 0.42.1（仅 query 侧）
- 第四臂 RERANK: 对 **RRF document top-10** 候选做 cross-encoder 精排（bge-reranker-v2-m3-Q8_0，llama.cpp b10012 win-cuda-12.4，`/v1/rerank`，port 1235，-ngl 99）
- 精排输入: query 原文 + 候选 document 全文（title+text，与检索口径一致）；分数为原始 logit（只用其序）
- 头截断规则（既定）: document 全文 **头截断至 6000 字符**再送 reranker（6000 char + query ≪ 模型 8192-token max）——本次 **47** 个去重 document（占候选槽位 **1314** 次，跨 query 含重复）被头截断。（注：向量臂建库按 800 字符硬切 chunk、服务端 context≥8192 无截断；精排见 document 头 6000 字符，上下文更全。此 6000 字符截断 v1 已披露，非勘误对象。）
- nDCG@10 主口径 = **linear-gain**（trec_eval/BEIR 官方）；exponential 为诊断值（见下表，勿与公开数字并置）
- 超时/重试: timeout 60s，retries 2（总尝试 3），线性退避 1.0s
- git_head: `f07a5fe`；工作树: clean（可复现）
- 生成时间: 2026-07-16T08:39:25

## 四臂主表（全量，均值；nDCG@10 = linear-gain 主口径）

| 臂 | recall@5 | recall@10 | MRR@10 | precision@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| LEXICAL | 0.7648 | 0.8472 | 0.6176 | 0.1538 | 0.6729 |
| VECTOR | 0.8477 | 0.9004 | 0.7510 | 0.1709 | 0.7873 |
| RRF（双路） | 0.8933 | 0.9457 | 0.7675 | 0.1800 | 0.8110 |
| RERANK（RRF→精排） | 0.9357 | 0.9457 | 0.8710 | 0.1888 | 0.8894 |

### nDCG@10 诊断：exponential-gain（**诊断值，勿与公开数字并置**）

| 臂 | nDCG@10 (linear·主) | nDCG@10 (exponential·诊断) |
|---|---:|---:|
| LEXICAL | 0.6729 | 0.6729 |
| VECTOR | 0.7873 | 0.7873 |
| RRF（双路） | 0.8110 | 0.8110 |
| RERANK（RRF→精排） | 0.8894 | 0.8894 |

### RERANK − RRF（精排增量）

| 指标 | RRF | RERANK | Δ |
|---|---:|---:|---:|
| recall@5 | 0.8933 | 0.9357 | +0.0424 |
| recall@10 | 0.9457 | 0.9457 | +0.0000 |
| MRR@10 | 0.7675 | 0.8710 | +0.1035 |
| precision@5 | 0.1800 | 0.1888 | +0.0089 |
| nDCG@10 | 0.8110 | 0.8894 | +0.0785 |

> 注：RERANK 仅对 RRF 的同一 top-10 候选集**重排序**，故 **recall@10 恒等于 RRF**（集合不变）；变化只体现在 recall@5 / MRR@10 / precision@5 / nDCG@10（名次重排）。

## 前三臂核对（当时逐 query 记录重算 vs 已提交主表）

- 最大绝对偏差 = **4.985e-05**（阈值 2%；≈0 表示 harness 与冻结运行记录一致，前三臂确为复用）

| 臂 | 指标 | 重算 | 主表 | |Δ| |
|---|---|---:|---:|---:|
| LEXICAL | recall@5 | 0.7648 | 0.7648 | 4.763e-05 |
| LEXICAL | recall@10 | 0.8472 | 0.8472 | 7.587e-06 |
| LEXICAL | MRR@10 | 0.6176 | 0.6176 | 3.981e-07 |
| LEXICAL | precision@5 | 0.1538 | 0.1538 | 4.615e-05 |
| LEXICAL | nDCG@10 | 0.6729 | 0.6729 | 1.249e-05 |
| VECTOR | recall@5 | 0.8477 | 0.8477 | 3.446e-05 |
| VECTOR | recall@10 | 0.9004 | 0.9004 | 2.15e-05 |
| VECTOR | MRR@10 | 0.7510 | 0.7510 | 3.563e-06 |
| VECTOR | precision@5 | 0.1709 | 0.1709 | 1.675e-05 |
| VECTOR | nDCG@10 | 0.7873 | 0.7873 | 3.205e-05 |
| RRF（双路） | recall@5 | 0.8933 | 0.8933 | 8.746e-06 |
| RRF（双路） | recall@10 | 0.9457 | 0.9457 | 3.235e-05 |
| RRF（双路） | MRR@10 | 0.7675 | 0.7675 | 3.057e-05 |
| RRF（双路） | precision@5 | 0.1800 | 0.1800 | 2.107e-05 |
| RRF（双路） | nDCG@10 | 0.8110 | 0.8110 | 4.985e-05 |

## 精排耗时（每 query 一次 /v1/rerank，10 候选）

- 样本 949 query；每 query：mean **568.4 ms** / median 548.5 ms / p95 886.9 ms / max 3174.4 ms
- 全量精排累计 539.4s

## Public record boundary

> **2026-07-29 pre-public license hardening:** CovidRetrieval has no explicit upstream redistribution license. This public record retains only project-authored aggregate metrics, run configuration, and method notes. It does not distribute query text, qid, document id, qrels/gold mappings, or per-query rankings.

The benchmark scripts enforce the same default. For a private fourth-arm replay, `run_bench.py --details-out` writes the per-query handoff to an explicitly chosen gitignored or out-of-repository path and `run_rerank.py --details-file` consumes it. That private handoff must not be committed.
