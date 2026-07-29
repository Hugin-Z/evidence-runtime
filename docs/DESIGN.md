# 架构与边界

## 1. 产品形态

本仓库是一套可独立使用的 **Evidence Agent 工作流**。它覆盖任务澄清、项目材料库、证据准备、
受约束生成、Claim Audit 和 DOCX 交付。

其中 **Evidence Runtime** 是确定性代码核心，不调用生成式 LLM。语义撰写与语义判级由宿主 Agent
或人工完成。这样的边界使工作流既能单独运行，也能把 `solution-drafter` 等专业 Agent 接到明确的
writer 槽位。

## 2. 处理链

```text
L0 Intake       用户任务 → ProjectSpec → 确定性选择契约
    ↓
L1 Library      项目隔离、材料版本、来源类型与索引生命周期
    ↓
L2 Discovery    向量发现 + 词法候选
    ↓
L3 Location     ripgrep 精确定位 → 可逐字回读的 EvidenceSpan
    ↓
L4 Judgement    固定规则核验 → filled / weak / missing / conflicting
    ↓
L5 Gate         槽状态 → allow / limited / block / needs_resolution
    ↓
L6 Generation   生成 section task，接收并校验 writer 回稿
    ↓
L7 Audit        AuditBrief + 审计单 → 确定性复核与交付准入
    ↓
L6′ Rendering   经统一交付门生成 DOCX，并用 verify_docx 回验
```

L6′ 的编号表示它属于呈现层；实际时序固定为 L6 → L7 → L6′。

## 3. 核心对象

| 对象 | 用途 |
|---|---|
| `ProjectSpec` | 项目标识、目标文档、任务参数与材料库定位 |
| Evidence Contract | 文档必须具备的证据槽及确定性规则 |
| `EvidenceSpan` | 带稳定引用和原文偏移的逐字证据 |
| Gap Report | 面向人的材料齐备度和补料清单 |
| `GenerationBrief` | section gate、可用证据和撰写约束 |
| Section task / response | Runtime 与 writer 之间的结构化协议 |
| `AuditBrief` | 当前盖章稿的句级单元、辖区和证据包 |
| Audit sheet | auditor 提交的 claim、判级、免审与签字处置 |
| `RecheckReport` | Runtime 的复核结果及准入状态 |

## 4. 检索与判断为什么分层

向量检索只负责扩大候选召回，不直接决定“证据成立”。候选必须经过精确定位，形成能回到原文偏移的
span，再由固定契约规则判定。

因此更换 embedding 会影响候选质量，但不会让相似度分数自动变成事实证明。词法通道和精确定位依赖
系统 `ripgrep`；中文分词用于候选发现与锚点处理。

## 5. 生成协议

Runtime 不写正文。首次运行 generation 时，它按 section gate：

- `allow`：提供已满足的证据与完整写作要求；
- `limited`：只允许在当前证据边界内写，并保留缺口；
- `block`：不派发写作任务，直接生成待补料占位；
- `needs_resolution`：要求先解决冲突。

writer 把结构化 response 写回会话目录。第二次运行 generation 会校验引用、格式和响应绑定，
通过后把段落及内容哈希写入 `state.json`。修改已接收段落必须显式重跑对应 section。

## 6. Claim Audit 的责任边界

代码把当前稿机械切成句级 `claim_unit`，用于建立稳定覆盖边界。它**不会**自动完成语义上的
atomic-claim extraction；一句含多个断言时，auditor 仍需作出合适的 claim 表达和支持等级判断。

auditor 负责提交 claim，并判为 `supported / weak / unsupported / contradicted`。Runtime 负责复核
可确定验证的部分：完整单元覆盖、段落绑定、引用存在、逐字内容、辖区、签字处置、工件一致性和新鲜度。

## 7. 交付状态

```text
未完成生成 → DRAFT
审计拒绝   → BLOCKED
审计准入 + 仍有占位 → SHAREABLE_DRAFT
审计准入 + 无占位   → DELIVERABLE
```

非草稿 DOCX 只能通过 `evidence.render` 的统一入口产生。入口使用当前 state、GenerationBrief、
AuditBrief、audit map 与 RecheckReport 现场复算，不把报告中的缓存布尔当作许可。

## 8. integrity 保证

当前实现的是 `integrity` 级：

- 绑定当前盖章正文，拒绝旧审计批准新稿；
- 检查所有非占位事实单元是否被完整审计；
- 复跑引用存在、证据内容和辖区检查；
- 将报告声明的 weak 策略与部署方策略取更严者；
- 摘要绑定完整 AuditBrief 和 audit map。

摘要还绑定 state 正文指纹及 GenerationBrief 的版本、契约和 section gate。它不绑定源文档字节、
source manifest、原始 audit sheet 文件或完整 Contract 本体。渲染门也不会重新加载材料库按偏移切原文；
原文重切发生在带 `--library-root/--project` 的 Claim Audit 阶段。

integrity 不是身份授权。拥有整个工作区写权限的主体仍可能协调重写全部工件。下一阶段可增加
`SignedRecheckEnvelope`，但只有私钥位于工作区外的独立审计服务、隔离 CI 或 KMS/HSM 中才有意义；
当前未实现 signed 模式。

## 9. 扩展点

- 新文档类型：增加 Evidence Contract、crosswalk 和 section spec。
- 新 writer：实现 section task/response 协议。
- 新 auditor：读取 AuditBrief，输出约定格式的 audit sheet。
- 新检索后端：保持候选类型、精确定位和索引一致性元数据。
- 更强授权：部署独立审计身份，再实现并验证 signed envelope。

这些扩展不得旁路 L4 判断、L5 gate、L6 收货或 L7 交付准入。
