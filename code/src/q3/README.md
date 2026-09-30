# 问题三融合方案实跑

在仓库根目录执行：

```bash
.venv/bin/python -m pytest -q tests/q3/test_experiment.py
.venv/bin/python src/q3/run_all.py
```

程序依次运行交付包中的八个问题三脚本，再核验和生成 `results/q3/metadata.json`、`results/q3/report.md`。每步日志在 `results/q3/logs/`。输入接口快照在 `results/q3/inputs/`，原始附件保持只读。

移植时作了三处与计算有关的修正：

1. Bootstrap 截距改为与主损失函数相同的 `E_tilde - E1*Q`，且只在 SET_A 的 B6 校准集分组重抽；B7 不参与重拟合。
2. 配比复核的 `L1` 约束使用 17 个逐坐标辅助量，并要求辅助量之和不超过半径；一致性检查要求实际找到可行解。
3. 派生量中 `D/N` 随预算的指数改为 `(α-β)/(α+β)`，并在固定工作点报告数值。

方案中的 `SET_A`、M2 接口属于前两问的**候选冻结接口**。仓库问题二来源文件标记为 `PARTIAL_CANDIDATE_INTERFACE`，因此本次结果是该接口下的条件预测和情景分析。质量成本函数、η 与配比转移系数没有由附件识别。`results/q3/report.md` 记录实测结果与边界。
