# 问题二融合方案修订重跑 v1

本目录是独立重跑产物，不覆盖原始附件、旧版 `q2_v8`、原团队 ZIP 或
`results/q2_scaling_v2`。

## 复现

在仓库根目录执行：

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.q2_fusion_rerun --stage all
```

正式 Q1→Q2 接口由同一入口先行导出到
`results/q1_revision_v2_1/q2_interface/`。接口固定使用第一题选定的
LightGBM，按原始 `index` 连接配比和 Loss，并将 17 维配比逐行归一化。

## 结论边界

- `Q_A` 与附件 B 的 `Q_B` 没有真实配对，不能估计唯一转换关系。
- 附件 B 没有 `(N,D,Q,p,L)` 联合观测，`lambda_p` 和 Form A/B 只能是情景。
- B8 的组内 Q–Loss 方向与 B6/B7 相反，不能合并为统一质量效应。
- B10 是既有外推估算，只能做一致性检查，不能作为真实大模型验证。

完整报告位于 `report/question2_experiment_report.md`；复现哈希位于
`report/reproducibility_summary.json`。
