# BENCH ZH-Covid-v2 — Evidence Runtime 检索三臂 × 5 指标（2026-07-16）

## 配置快照
- 数据集: **covid**（dev），corpus **100001** passage / **949** query
- embedding_model: `text-embedding-qwen3-embedding-0.6b`（真 APIEmbedder / LM Studio），dim 1024；max_context=32768
- **服务端无截断（经验探测）**: cos(tail@3000tok)=**0.6818** < 0.98 ⇒ 3000-token 后内容仍参与嵌入（loaded_ctx 字段对 embedding 不可靠不采信；须 ≥ 8192，`lms ps` 实测 context=8192）
- 切块器: **hs800v1**（benchmark 层硬切）；max_chunk_chars=800；实测最长 chunk=800 字符；**over_limit=0**（建库断言无 chunk 超上限）
- 评测单位: **document**；候选深度 N_DOC=20、chunk_cap=200（每路折叠 ≥N_DOC 唯一 doc 进 RRF）；rerank 候选 = RRF doc top-10
- top_k（三臂统一）: 10　rrf_k: 60　collection: `corpus_vectors_hs800v1`　建库: 分批 batch=32，107954 chunk / 1269.0s
- source_snapshot: 公开前冻结运行，原提交由维护者私存；本文件保留配置与聚合指标
- 中文词法臂: **jieba 0.42.1 分词**（仅 query 侧、仅词法臂；向量臂用原始 query）
- 指标: recall@5 / recall@10 / MRR@10 / precision@5 / nDCG@10；**nDCG@10 主口径 = linear-gain**（trec_eval / BEIR 官方口径，公开对照唯一口径）
- 生成时间: 2026-07-16T04:15:46

## 三臂主表（全量，均值；nDCG@10 = linear-gain 主口径）

| 臂 | recall@5 | recall@10 | MRR@10 | precision@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| LEXICAL | 0.7648 | 0.8472 | 0.6176 | 0.1538 | 0.6729 |
| VECTOR | 0.8477 | 0.9004 | 0.7510 | 0.1709 | 0.7873 |
| RRF（双路） | 0.8933 | 0.9457 | 0.7675 | 0.1800 | 0.8110 |

### nDCG@10 诊断：exponential-gain（**诊断值，勿与公开数字并置**）

| 臂 | nDCG@10 (linear·主) | nDCG@10 (exponential·诊断) |
|---|---:|---:|
| LEXICAL | 0.6729 | 0.6729 |
| VECTOR | 0.7873 | 0.7873 |
| RRF（双路） | 0.8110 | 0.8110 |

> exponential-gain `2^rel−1` 放大高相关档差异，仅内部诊断；公开榜（BM25 0.325 等）为 linear-gain，勿并置。

## 公开对照 + 差距解释（同表并列；对照不是目标）

| 对照 | 公开 nDCG@10 | 本项目对应臂 nDCG@10（linear 主口径） | 差距归因（如实） |
|---|---|---:|---|
| bge-large-zh | 0.7406 | VECTOR 0.7873 / RRF 0.8110 / LEXICAL 0.6729 | large 级中文 embedding；本项目为 qwen3-0.6b 小模型、无精排 |
| mE5-large | 0.7551 | VECTOR 0.7873 / RRF 0.8110 / LEXICAL 0.6729 | large 级多语；本项目词法臂用 jieba 通用分词（非训练态）+ rg 命中率初排、非 BM25 |

> 归因总述：本项目向量臂 = **qwen3-embedding-0.6b（小模型）零样本、无精排（rerank 直通）**；词法臂 = **rg 命中率初排（非 BM25 加权排序）**，中文经 jieba 通用分词（非训练态）。 与公开榜（多为 large 级 + 训练态）的差距，**候选解释包括模型规模、排序机制差异；未做匹配控制实验，不作排他性归因**；数字原样交，不调参贴榜。

## 中文词法臂：jieba 分词抽样（20 条）+ 无分词退化对照

- **无分词退化小样**（前 20 条 query 原始整句喂词法臂）：recall@10 = **0.2000**（预期近 0——合成轨道 L2a 问题重演）；jieba 后词法臂全量 recall@10 = 0.8472。

> **2026-07-29 许可处置：**只保留项目自产聚合指标；不再公开 CovidRetrieval 的 query 原文、
> jieba token、qid/document id、qrels/gold 映射或逐 query 排名（上游未给出明确再分发许可）。

## Public record boundary

> **2026-07-29 pre-public license hardening:** CovidRetrieval has no explicit upstream redistribution license. This public record retains only project-authored aggregate metrics, run configuration, and method notes. It does not distribute query text, qid, document id, qrels/gold mappings, or per-query rankings.

The benchmark scripts enforce the same default. For a private fourth-arm replay, `run_bench.py --details-out` writes the per-query handoff to an explicitly chosen gitignored or out-of-repository path and `run_rerank.py --details-file` consumes it. That private handoff must not be committed.
