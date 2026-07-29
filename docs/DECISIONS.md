# 稳定设计决策

本文件只保留影响当前产品的稳定决策。逐轮审计、实验对话、被否决方案和内部验收证据不属于公开文档。

## D1：产品是独立 Evidence Agent，核心是零 LLM Runtime

仓库保留从材料到交付的完整模板和编排，不缩成单一检查库。Runtime 负责确定性工作，writer/auditor
由宿主 Agent 或人工承担。

## D2：固定契约优先于自由规划

当前阶段由代码根据 `ProjectSpec` 确定性选择 Evidence Contract、crosswalk 和 section spec。
Agent 不临时发明判断标准。新增文档类型通过版本化契约扩展。

## D3：向量只做发现，逐字定位才形成证据

相似度不直接等于证据。向量和词法通道只产生候选；候选必须经过精确定位并形成可回读的
`EvidenceSpan`，再由规则判定。

## D4：材料状态先于生成

证据槽先被判为 `filled / weak / missing / conflicting`，L5 再导出 section gate。生成不得自行放宽
gate；缺料必须阻断或显式占位。

## D5：生成采用外部 writer 的两趟协议

第一趟派发 section task，第二趟接收 response。Runtime 校验引用和格式并盖章，不在代码中嵌入特定
模型。solution-drafter 是可选 Writer Agent，不是必需依赖。

## D6：Claim Audit 的语义责任属于 auditor

代码只做句级机械切分和可确定复核，不宣称自动抽取 atomic claims 或自动判定支持等级。auditor 提交
claim、verdict 和必要 signoff。

## D7：审计必须覆盖完整事实单元并绑定当前稿

非占位事实单元必须被覆盖；占位标记与实质断言同句时仍需审计。claim 绑定段落 ID 与内容指纹，改稿后
必须重新收货和审计。

## D8：非草稿交付只走一个门

所有非草稿 DOCX 都通过统一渲染入口。入口接收当前审计工件并现场复算，不读取
`accepted`、`rule_check_passed` 或 `delivery.admitted` 作为独立许可。

## D9：当前保证等级为 integrity

integrity 防陈旧、误修改、漏审和缓存不一致，不防全工作区写入者自授权。signed 模式只在审计身份和
私钥真正独立时实现，且失败不得静默降级。

## D10：公开 demo、benchmark 与私有验收分开

合成 demo 用于证明管线行为；公开 benchmark 只证明对应检索测量；私有真实材料用于本地纠错。
三者不能互相替代，也不能由局部指标推出整套产品能力。

## D11：公开仓库不是研发档案

公开 Git 只保留当前用户文档、稳定决策、代码、测试、合成 demo 和许可允许的评测工件。真实材料、
完整历史、内部验收与审计流水账保存在 Git 忽略的私有位置。
