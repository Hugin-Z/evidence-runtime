# 当前状态

## 发布结论

`evidence-runtime 0.1.0` 处于 **Alpha**。从任务和材料到受门控 DOCX 的主链已经实现，适合公开展示、
技术评估和受控试点；还不应被描述为通用生产平台或无需人工监督的自主 Agent。

## 已实现

- L0 结构化 intake 与 `ProjectSpec`
- L1 按项目隔离的材料库、版本与来源类型
- PDF / DOCX / MD / TXT 文本抽取
- 向量与词法候选发现、ripgrep 精确定位
- 固定 Evidence Contract 的槽级规则判断
- Gap Report、补料信息与 `GenerationBrief`
- 内置 proposal crosswalk 和 section specs
- 两趟 section task/response 生成编排与回稿收货
- 句级单元覆盖、Claim Audit、审计工件绑定
- 单一非草稿渲染门、DOCX 生成与结构回验
- `DRAFT / BLOCKED / SHAREABLE_DRAFT / DELIVERABLE` 状态
- 源码安装和 wheel 安装的 CI 冒烟
- 缺料与材料齐备两套可复现合成 demo

## 真实能力边界

Runtime 能确定性验证引用、逐字内容、覆盖、工件一致性和既定规则。它不能独立理解一句话是否在业务
语义上“被充分证明”，也不自动生成高质量正文。atomic claims 和支持等级由 auditor 提交。

因此当前最准确的定位是：

> 一套独立可用的证据约束工作流，以及其中可复核的确定性 Runtime；不是通用知识库，也不是内置模型的
> 自动方案写作器。

## 公开成熟度

当前工作树的公开材料应只包含代码、合成 demo、可复现测试、按许可处理的 benchmark 记录和用户文档。
真实项目材料、内部验收样本、完整实验历史与研发日记不属于公开发行物。

若仓库历史曾包含不应公开的数据，发布前仍需单独清理 Git 历史；当前树删除文件不会删除旧 commit。

## 下一阶段优先级

1. 在更多可公开、许可清晰的真实任务上建立可复核评估。
2. 把 writer/auditor task 协议稳定为版本化接口，便于 solution-drafter 等 Agent 协作。
3. 增加 OCR 与更多文档契约，而不放松现有 gate。
4. 在独立审计身份和外部密钥成立后实现 signed 授权模式。
5. 再考虑服务化 API、Web UI 和统一多 Agent 产品。

已知缺口详见 [LIMITATIONS.md](LIMITATIONS.md)。
