# Demo：缺料时生成可分享草稿

本目录用完全虚构的材料跑通 Evidence Runtime 全链。它故意缺少部分案例与对齐证据，因此预期结果是
带 `【待补充】` 和水印的 `SHAREABLE_DRAFT`，不是最终交付物。

这套 demo 用于：

- 展示每一步输入输出；
- 验证任务编排、审计绑定和交付状态机；
- 作为 CI 的端到端回归。

它使用确定性 `--fake-embedder`，不证明真实检索准确率。

## 一键运行

需要 Python 3.11+、`rg` 和 Bash：

```bash
bash examples/demo-proposal/run_chain.sh
```

脚本真正执行：

```text
intake → library → gap check → generation pass 1
       → 放入合成 writer 回稿 → generation pass 2
       → Claim Audit → gated DOCX render
```

gap check 和 generation 第一趟返回 `1` 是预期业务状态：前者表示材料不完整，后者表示正在等待 writer
回稿。脚本会逐步检查预期返回码。

## 材料

| 文件 | 用途 |
|---|---|
| `可行性研究报告.md` | 背景、现状、需求与目标 |
| `建设方案.docx` | 建设内容、能力、实施条件与边界 |
| `需求清单.md` | 范围与对象 |
| `报价单.md` | 报价和工期 |

所有名称、事实和数字均为合成内容。故意缺少的证据会让相应 section 进入 `block` 或 `limited`，
而不是由 writer 自行补全。

## 可查看的产物

`expected/` 保存了合成运行的典型结果：

- `gap/`：Gap Report、GenerationBrief 和 audit map；
- `gen/tasks/`：给 writer 的 section task；
- `gen/responses/`：合成 writer response；
- `gen/state.json`：收货后的盖章段落；
- `audit-sheet-v1-blocked.json`：包含不受支持断言的拒绝路径；
- `audit-sheet-v2-admitted.json`：处置后重新审计的准入路径；
- `proposal-shareable-draft.docx`：仍有待补料项的进度草稿。

材料齐备且输出为无水印 `DELIVERABLE` 的对照 demo 见
[`../demo-proposal-deliverable/`](../demo-proposal-deliverable/)。

完整手动命令见 [使用指南](../../docs/USAGE.md)。
