# 贡献指南

欢迎 issue 与 PR。请把改动保持在清晰、可验证且不泄露真实项目数据的范围内。

## 开发环境

要求 Python 3.11/3.12 和系统命令 `rg`：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q evidence
```

CI 还会从 wheel 安装包，检查六个 console scripts，并运行缺料/材料齐备两套端到端 demo。

## PR 要求

- 说明改动目的、影响的 L0–L7 层和用户可见行为。
- 新行为必须有测试；修复绕过时应增加能复现原绕过的对抗 fixture。
- 修改结构化 JSON 时，说明兼容性和迁移方式。
- 修改契约、crosswalk 或 section spec 时，同步测试和 demo。
- 不把合成 demo 的结果描述成准确率。

涉及检索、规则阈值或交付门的风险改动，请在实现前记录：

1. 假设；
2. 成功指标；
3. 不可接受的回归；
4. 能证伪方案的结果。

完成后把仍然影响产品的稳定结论精炼进 [docs/DECISIONS.md](docs/DECISIONS.md)。实验过程、内部对话和
私有验收记录放在被 Git 忽略的 `.internal/`，不要把公开文档变成研发流水账。

## 数据边界

禁止提交：

- 真实客户或第三方受限材料；
- `.library/`、`.local/`、`.internal/`、`out/`；
- 运行任务包、逐字证据、审计工件和交付 DOCX；
- API key、`.env`、本机路径或账号信息。

第三方 benchmark 变更必须同时核对许可、署名和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 契约字段

中文槽名和描述会参与查询组装、分词和定位。翻译或改名可能改变运行行为，不能作为纯文档修改处理。

## 安全问题

未修复漏洞优先通过 GitHub 的私有漏洞报告渠道提交；若该功能尚未启用，只开一个不含利用细节的
联系请求。详见 [docs/SECURITY.md](docs/SECURITY.md)。
