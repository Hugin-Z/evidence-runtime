# 私有数据任务验收工具

本目录提供本地任务级验收脚本和示例标签 schema。它用于在维护者合法持有的材料上检查：

- 固定 Evidence Contract 的槽级判断；
- expected / actual 标签对比；
- 转换材料与原文件的一致性探针；
- AI 辅助标注输出的格式和差异审计。

目录中**不包含**真实材料、真实标签或具体项目验收记录。使用者自行准备私有输入，并确保材料权限。
输出可能包含原文片段，应写入被 Git 忽略的 `.local/`、`.internal/` 或其他受控目录。

这些工具的结果只能说明对应数据和配置下的表现；私有结果不可作为外部可复核的通用准确率声明。
公开评估原则见 [docs/EVALUATION.md](../../docs/EVALUATION.md)。

脚本接口以 `--help` 为准，例如：

```bash
python eval/task_acceptance/run_acceptance.py --help
python eval/task_acceptance/ai_label_acceptance.py --help
python eval/task_acceptance/audit_acceptance_discrepancies.py --help
python eval/task_acceptance/audit_converted_materials.py --help
```
