# 检索 benchmark

本目录评估候选发现与 rerank 子系统，不评估整套方案生成质量。

## 目录

- `run_bench.py`：运行 lexical、vector、RRF 三臂检索；
- `run_rerank.py`：在冻结候选上增加 rerank；
- `rerank_smoke.py`：小规模服务连通与方向检查；
- `results/`：已记录的配置快照和结果；
- `data/`：本地数据位置，第三方数据不随仓库分发。

接口以脚本 `--help` 为准：

```bash
python eval/bench/run_bench.py --help
python eval/bench/run_rerank.py --help
python eval/bench/rerank_smoke.py --help
```

## 阅读结果

公开树只保留修正切块、评测单位和 nDCG 口径后的 `v2` 当前记录。已被替代的 v1 记录保存在维护者
私有档案中，不再让公开读者在两套口径之间选择。结果标题中的 `retrieval-router` 是测量时使用的
历史检索子系统名称，不代表当前产品名。

NFCorpus 记录按其 CC BY-SA 条款保留查询、标识和逐项结果。CovidRetrieval 的公开记录只保留项目自产
测量、聚合指标和不透明 pair ID；上游查询、qid、qrels 和原始文档标识不公开。完整许可与署名见
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。

需要逐 query 私有 handoff 时，必须显式写到 Git 忽略或仓库外的位置，不得提交。

benchmark 配置依赖具体模型和服务，结果只能在记录的配置范围内解释。详见
[评估方法](../../docs/EVALUATION.md)。
