# Demo：材料齐备时生成 DELIVERABLE

本目录与 [`demo-proposal`](../demo-proposal/) 互为对照，使用完全虚构但槽位齐备的材料，预期生成
无待补料占位、无草稿水印的 `DELIVERABLE` DOCX。

| Demo | 证据状态 | 预期输出 |
|---|---|---|
| [`demo-proposal`](../demo-proposal/) | 故意缺料 | `SHAREABLE_DRAFT` |
| 本目录 | 内置契约的 8 个证据槽齐备 | `DELIVERABLE` |

两套 demo 都使用 `--fake-embedder`，用于验证工作流行为，不代表真实检索准确率。

## 运行

```bash
bash examples/demo-proposal-deliverable/run_chain.sh
```

脚本执行 intake、材料入库、gap check、两趟 generation、Claim Audit 和 gated render。主要差异是：

- 使用证据支撑子集的 `方案核心内容` crosswalk；
- 材料包含同类案例的上线/验收状态与建设成效；
- 所有目标 section 都有可用证据，不产生 `【待补充】`；
- Claim Audit 准入后，交付状态为 `DELIVERABLE`。

`expected/` 中的材料、writer 回稿、审计单和 DOCX 都是合成工件，可直接检查。完整命令说明见
[使用指南](../../docs/USAGE.md)。
