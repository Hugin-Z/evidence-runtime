# 使用指南

## 1. 环境

- Python 3.11 或 3.12
- `ripgrep`（命令 `rg`）
- 支持文本抽取的 PDF、DOCX、Markdown 或纯文本材料

开发安装：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

真实检索可使用 OpenAI-compatible embedding API，或安装本地后端：

```bash
python -m pip install -e ".[local-embedder]"
```

`--fake-embedder` 只用于 demo、CI 和管线调试，不代表真实检索质量。

## 2. 先跑合成 demo

在 Git Bash、WSL、macOS 或 Linux：

```bash
bash examples/demo-proposal/run_chain.sh
```

脚本会创建临时工作目录并跑完完整链路。缺料版预期输出 `SHAREABLE_DRAFT`；材料齐备版位于
`examples/demo-proposal-deliverable/`。

## 3. 创建任务和材料库

交互生成 `ProjectSpec`：

```bash
evidence-intake --out spec.json
```

也可以用 JSON 答案文件做非交互运行：

```bash
evidence-intake --answers answers.json --out spec.json
```

创建项目库并加入材料：

```bash
evidence-library --project acme-2026 init
evidence-library --project acme-2026 add 招标文件.pdf --source-type client_provided
evidence-library --project acme-2026 add 内部能力说明.docx --source-type self_produced
evidence-library --project acme-2026 status
```

默认库根目录是 `.library/`。可用 `--root <path>` 指向隔离磁盘或项目专属目录。来源类型为
`client_provided`、`self_produced`、`public` 或 `other`。

## 4. 检查证据准备度

让 `ProjectSpec` 驱动项目库与契约：

```bash
evidence-gap-check \
  --spec spec.json \
  --output-dir out/gap \
  --brief 解决方案建议书
```

主要产物：

- `gap-report.json`：机器可读的槽状态与证据；
- `gap-report.md`：面向人的材料缺口报告；
- `generation-brief.json`：生成 section task 的输入；
- `generation-brief.audit.json`：稳定引用到来源偏移的映射。

证据不完整时命令返回非零是业务状态，不一定是程序故障。先阅读 gap report，再决定补料还是保留占位。

兼容入口还支持 `--project`、`--input` 或 `--manifest`。新项目优先使用 `--spec`。

## 5. 配置 embedding

API 后端示例：

```powershell
$env:EVIDENCE_EMBEDDING_API_KEY = "<secret>"
evidence-gap-check --spec spec.json --output-dir out/gap --brief 解决方案建议书 `
  --embedding-backend api `
  --embedding-api-base http://127.0.0.1:1234/v1 `
  --embedding-api-model text-embedding-model
```

本地 sentence-transformers 示例：

```bash
evidence-gap-check --spec spec.json --output-dir out/gap --brief 解决方案建议书 \
  --embedding-backend local \
  --embedding-local-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

配置优先级为：默认值 < JSON 配置 < `EVIDENCE_*` 环境变量 < CLI。JSON 只保存密钥的环境变量名，
不接受 API key 本体。远端 endpoint 会收到材料文本，使用前先确认数据权限，详见
[SECURITY.md](SECURITY.md)。

## 6. 两趟生成编排

第一趟生成 task：

```bash
evidence-generate \
  --brief out/gap/generation-brief.json \
  --session-dir out/gen \
  --spec spec.json
```

Runtime 把任务写到 `out/gen/tasks/`，并等待 writer 在 `out/gen/responses/` 返回对应 JSON。
writer 可以是本仓的宿主 Agent、人工流程，或按同一协议接入的 solution-drafter。

回稿到齐后执行第二趟：

```bash
evidence-generate \
  --brief out/gap/generation-brief.json \
  --session-dir out/gen \
  --spec spec.json
```

第二趟校验并接收回稿，生成带段落 ID 和内容指纹的 `state.json`。修改某一节后使用
`--reset-section <section_id>` 显式重收货。

## 7. Claim Audit

先生成给 auditor 的 `AuditBrief`：

```bash
evidence-audit \
  --brief out/gap/generation-brief.json \
  --audit-map out/gap/generation-brief.audit.json \
  --session-dir out/gen \
  --emit-brief out/audit-brief.json
```

auditor 读取 AuditBrief，为每个非占位事实单元提交 claim 和支持等级，再把审计单保存为 JSON。
审计与判级不是 Runtime 自动完成的。

使用审计单复核，并在材料库原文上重切证据：

```bash
evidence-audit \
  --brief out/gap/generation-brief.json \
  --audit-map out/gap/generation-brief.audit.json \
  --session-dir out/gen \
  --sheet out/audit-sheet.json \
  --library-root .library \
  --project acme-2026 \
  --emit-brief out/audit-brief.json \
  --out out/recheck.json
```

默认 weak 策略为 `block`。需要接受带说明的 weak claim 时，审计和部署方都必须明确选择 `allow`。

## 8. 渲染

未通过审计时，可以生成明确带水印的草稿：

```bash
evidence-render \
  --session-dir out/gen \
  --brief out/gap/generation-brief.json \
  --spec spec.json \
  --out out/proposal.docx \
  --draft
```

非草稿输出必须提供当前审计工件：

```bash
evidence-render \
  --session-dir out/gen \
  --brief out/gap/generation-brief.json \
  --spec spec.json \
  --recheck out/recheck.json \
  --audit-brief out/audit-brief.json \
  --audit-map out/gap/generation-brief.audit.json \
  --out out/proposal.docx \
  --state-suffix
```

输出状态：

- `BLOCKED`：审计未通过，不生成非草稿交付件；
- `SHAREABLE_DRAFT`：审计准入，但仍含待补料占位；
- `DELIVERABLE`：审计准入且无占位。

## 9. 常见问题

**为什么 gap check 返回 1？**

通常表示证据不完整。查看 `gap-report.md`，不要只按传统“零才成功”理解业务返回码。

**为什么 generation 要跑两次？**

第一趟派任务，第二趟接收外部 writer 回稿。Runtime 本身不生成语义正文。

**为什么改了一句话后旧 recheck 不能用？**

审计绑定盖章段落和内容指纹。改稿必须重新收货、重新审计。

**是否能完全离线？**

可以，但 embedding 与 writer/auditor 都要使用本机实现。`--fake-embedder` 只能验证流程。

**运行产物可以提交吗？**

默认不要。项目库、任务包、证据片段、审计工件与 DOCX 都可能包含客户原文。
