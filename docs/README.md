# 文档导航

公开文档只描述当前产品、使用方式和可验证边界。研发日记、内部验收记录、真实项目材料与完整历史
保存在本地私有目录，不属于公开产品文档。

## 建议阅读顺序

1. [README 中文版](../README.zh.md)：产品是什么、适合什么场景。
2. [使用指南](USAGE.md)：从材料入库到 DOCX 交付的完整命令。
3. [架构与边界](DESIGN.md)：L0–L7、角色分工和信任边界。
4. [安全与数据边界](SECURITY.md)：材料落盘、远端传输和 integrity 模式。
5. [限制](LIMITATIONS.md)：当前明确没有做什么。

需要评估或参与开发时，再读：

- [当前状态](STATUS.md)
- [评估方法](EVALUATION.md)
- [稳定设计决策](DECISIONS.md)
- [贡献指南](../CONTRIBUTING.md)
- [第三方材料与许可](../THIRD_PARTY_NOTICES.md)
- [更新记录](../CHANGELOG.md)

## 文档口径

- **Evidence Agent** 指可独立运行的完整工作流。
- **Evidence Runtime** 指其中不调用生成式 LLM 的确定性代码核心。
- **writer / auditor** 是外部 Agent 或人工角色，不是仓库内置模型。
- **ADMITTED** 表示通过 Claim Audit 准入；只有无待补料占位时才是 **DELIVERABLE**。
- **SHAREABLE_DRAFT** 可以展示进度，但不是最终交付物。

若文档与 CLI 行为不一致，以当前代码、`--help` 与测试为准，并欢迎提交 issue。
