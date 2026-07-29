[English](README.md) · **中文**

# Evidence Runtime

**把项目材料变成可审计、可交付方案的证据约束型 Agent 工作流。**

Evidence Runtime 面向需要“依据已有材料写方案”的场景。它管理项目材料，检查既定文档契约所需的
证据是否齐备，把可写内容编排成任务包，接收 writer 回稿，再经 Claim Audit 和统一交付门生成 DOCX。

它不是个人知识库，也不是一个内置大模型。它本身可以作为独立的 **Evidence Agent** 使用：
仓库提供完整工作流、模板、状态机和确定性检查；宿主 Agent 或人工负责语义撰写与审计。

## 它实际做什么

| 阶段 | Runtime 负责 | Agent / 人负责 |
|---|---|---|
| 任务与材料 | 形成 `ProjectSpec`，按项目入库 PDF / DOCX / MD / TXT | 确认任务参数与材料权限 |
| 证据准备 | 检索、逐字定位、规则核验，判定 `filled / weak / missing` | 处理缺料与业务取舍 |
| 方案生成 | 按内置文档结构生成受约束任务包，校验并接收回稿 | 根据任务包撰写段落 |
| Claim Audit | 句级切分、覆盖检查、引用/内容/辖区复算、交付门控 | 为每个事实单元提交 claim 并判定支持等级 |
| 交付 | 仅经统一入口渲染草稿或 DOCX，并回验文档结构 | 最终业务审阅与批准 |

当前内置的是面向中文政企方案的固定证据契约和方案模板。契约、交叉映射与 section specs 都随包分发，
可直接使用，也可以按同一协议扩展。

## “防胡写”意味着什么

系统不会阻止 writer 产生错误内容；它阻止的是**未经完整审计的内容被当成交付物输出**：

- 非占位事实单元必须被审计 claim 覆盖；
- 引用必须存在，证据内容和辖区必须一致；
- 审计报告必须绑定当前稿件，旧报告不能批准改过的正文；
- `accepted`、`delivery.admitted` 等缓存布尔不能绕过现场复算；
- 缺料会保留为醒目的 `【待补充】`，不会被静默补写。

默认 `integrity` 模式用于发现陈旧报告、误修改和不完整审计。它**不防**一个拥有整个工作区写权限的主体
协调改写所有工件后自我授权；强授权需要独立审计身份和签名基础设施，当前尚未实现。详见
[安全边界](docs/SECURITY.md)。

## 快速体验

要求 Python 3.11 或 3.12 与 [ripgrep](https://github.com/BurntSushi/ripgrep)。

```bash
python -m pip install -e ".[test]"
python -m pytest -q
bash examples/demo-proposal/run_chain.sh
```

这套合成 demo 会真实跑过 intake、材料库、证据缺口、两趟生成编排、Claim Audit 和 DOCX 渲染。
它故意缺少部分材料，因此最终得到带待补充标记的 `SHAREABLE_DRAFT`。另一套
[`demo-proposal-deliverable`](examples/demo-proposal-deliverable/) 展示材料齐备后的
`DELIVERABLE`。两套 demo 证明管线行为，不代表真实检索准确率。

## 真实使用流程

安装后可使用六个 CLI：

```text
evidence-intake
evidence-library
evidence-gap-check
evidence-generate
evidence-audit
evidence-render
```

典型顺序是：

```text
任务 → ProjectSpec → 项目材料库 → Gap Report / GenerationBrief
     → Writer 回稿 → Claim Audit → Draft 或 Deliverable DOCX
```

完整命令、输出目录和两趟编排方式见 [使用指南](docs/USAGE.md)。

## 与 solution-drafter 的关系

[solution-drafter](https://github.com/Hugin-Z/solution-drafter) 可以作为专门的 **Writer Agent** 接入，
消费 Evidence Runtime 生成的 section task，并返回结构化回稿。它不是本仓库的运行时依赖：
使用 Claude Code、Codex、其他 Agent 或人工撰写都可以。

当前选择保留两边各自独立可用的模板和编排。后续若做统一产品，优先通过稳定的 task/response 协议组成
多 Agent 协作，而不是把 Evidence Runtime 缩成一个只能被调用的内部库。

## 当前状态

版本为 `0.1.0`（Alpha）。主链和交付闸已实现并有端到端回归；它适合试用、研究和受控项目验证，
还不是通用企业内容平台。主要限制包括固定文档契约、无 OCR、无内置语义模型、无 Web/API 服务和
尚未实现签名授权。详见 [当前状态](docs/STATUS.md) 与 [限制](docs/LIMITATIONS.md)。

## 文档

- [文档导航](docs/README.md)
- [使用指南](docs/USAGE.md)
- [架构与边界](docs/DESIGN.md)
- [安全与数据边界](docs/SECURITY.md)
- [评估方法](docs/EVALUATION.md)
- [设计决策](docs/DECISIONS.md)
- [更新记录](CHANGELOG.md)
- [第三方材料与许可](THIRD_PARTY_NOTICES.md)

## 数据与公开边界

真实客户材料、材料库、运行产物、内部验收记录和研发流水账不应提交到 Git。默认忽略
`.library/`、`.local/`、`.internal/` 与 `out/`。公开仓库只保留合成 demo、可复现测试、
按许可处理的评测记录和面向使用者的文档。

如果从曾包含私有材料的旧仓库发布，单纯删除当前文件并不够，还需要审查并清理 Git 历史。

## License

代码与项目文档采用 [MIT License](LICENSE)。第三方数据集不因本项目而重新授权，详情见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
