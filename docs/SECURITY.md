# 安全与数据边界

Evidence Runtime 处理的客户与第三方材料应视为不可信输入，也可能包含敏感信息。本文件说明数据在哪里、
何时可能离开本机，以及当前交付门能保证什么。

## 1. 主要风险

- **材料中的 prompt injection**：正文可能包含试图影响 writer/auditor 的指令。
- **数据外传**：远端 embedding 或云端 Agent 会接收材料内容。
- **工作区越权**：同一主体若能改写全部运行工件，普通摘要不能证明独立批准。
- **公开泄露**：材料、索引、任务包和审计工件都可能包含客户原文。

## 2. 默认落盘内容

| 位置 | 内容 |
|---|---|
| `<library-root>/<project>/materials/` | 入库材料的明文副本 |
| `<library-root>/<project>/index/` | 向量、元数据和逐字 chunk |
| `--output-dir` | Gap Report、GenerationBrief、逐字证据 |
| 生成会话目录 | task、writer response、正文与状态 |
| AuditBrief / audit sheet / RecheckReport | 证据包、claim 与复核记录 |
| 输出 DOCX | 草稿或交付正文 |

项目没有自有遥测或上传服务，但这些文件默认也没有加密。请用操作系统权限、加密磁盘和隔离环境保护它们。
不要把 `.library/`、`.local/`、`.internal/`、`out/` 或真实运行产物提交到公开仓库。

## 3. embedding 与 Agent 的传输边界

- 本机 embedding endpoint 或本地 sentence-transformers 可以让检索阶段不出机。
- 非本机 endpoint 会收到待索引的材料文本；CLI 会提示远端传输风险。
- embedding 配置文件只应保存密钥的环境变量名，不应保存 API key 本体。
- `--fake-embedder` 完全离线，但只适合 demo/CI，不能代表真实检索。
- Claude Code、Codex、solution-drafter 或其他云端 Agent 作为 writer/auditor 时，任务包中的逐字证据会按
  该产品的机制进入模型上下文。Runtime 无法替这些产品作数据合规承诺。

要实现材料完全不出机，embedding、writer 和 auditor 三层都必须使用本地实现。

## 4. prompt injection 处理

Runtime 把证据放入明确的数据区，并声明其中的指令性文字不是给 Agent 的指令；确定性扫描会标记常见的
元指令、角色改写、伪造证据和结构破坏模式。Claim Audit 的确定性复核不会执行材料内容。

这只是降低风险：

- 模式扫描可能漏报或误报；
- Runtime 不会自动删改原文，因为删改会破坏逐字回读；
- 外部 writer 或 auditor 仍可能被材料语义影响。

敏感任务应使用独立 auditor，并把材料指令告警作为人工审阅项。

## 5. integrity 模式

当前非草稿渲染采用 `integrity` 级交付门。它会：

- 绑定当前盖章正文，拒绝旧报告批准新稿；
- 要求所有非占位事实单元被审计覆盖；
- 用当前 AuditBrief 和 audit map 复跑引用存在、证据内容与辖区检查；
- 从 claim verdict、完整性和 signoff 重新派生准入状态；
- 拒绝与复算不一致的缓存字段；
- 将报告 weak 策略与部署方 weak 策略取更严者；
- 让所有非草稿 DOCX 通过同一个公开渲染入口。

摘要覆盖当前正文指纹、GenerationBrief 的关键结构、完整 AuditBrief 和完整 audit map。它不覆盖：

- 源文档字节与 source manifest；
- 原始 audit sheet 文件；
- 完整 Evidence Contract 本体；
- 执行者身份。

Claim Audit 在同时提供 `--library-root` 和 `--project` 时，会按偏移从材料库重新切原文。render 阶段
不加载材料库，只对当前审计工件进行复算。因此审计后仅替换源材料、同时保持 AuditBrief/audit map 不变，
不会被 render 单独发现。

## 6. integrity 不是授权

默认 CLI 没有 writer、auditor、renderer 的强制 ACL。推荐权限边界是：

| 角色 | 应写 | 不应写 |
|---|---|---|
| writer | `responses/*.response.json` | state、brief、audit map、recheck |
| auditor | audit sheet | 正文、state、brief |
| renderer | DOCX | 所有输入工件 |

但这张表是部署建议，不是当前代码强制。如果一个主体拥有整个工作区写权限，它可以协调重写 state、
审计工件和普通 SHA-256 摘要。`RecheckReport` 是可复核的完整性记录，不是外部授权令牌。

## 7. signed 模式方向

更强保证需要用 `SignedRecheckEnvelope` 绑定 state、brief、audit map、audit sheet、完整 recheck、
source manifest 以及 contract/tool 版本，并由 renderer 使用部署方信任库验证。

只有当私钥位于工作区之外，由独立审计服务、隔离 CI、不同 OS 身份或 KMS/HSM 持有时，签名才建立
不同主体之间的授权边界。私钥与 writer 共处可写工作区没有安全增益。signed 模式当前尚未实现，
验证失败也不应静默降级为 integrity。

## 8. 部署建议

- 入库前做合法性确认、脱敏和最小化；
- 涉密材料使用本机 embedding 与本机 Agent；
- 按项目隔离材料库和运行目录；
- 用 OS、容器或 CI 身份分离 writer、auditor 与 renderer；
- 定期清理不再需要的任务包、索引、审计工件和 DOCX；
- 发布前同时扫描当前树和 Git 历史。

## 9. 漏洞报告

若仓库 Security 标签页已启用 **Private vulnerability reporting**，请使用该渠道报告未修复漏洞。
若尚未启用，可先开一个不含利用细节的 issue 请求私下联系；不要在公开 issue 中披露漏洞细节。
