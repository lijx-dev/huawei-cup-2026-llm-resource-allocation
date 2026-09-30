# 问题二 q2_v3 复现入口

在仓库根目录使用现有环境运行：

```bash
.venv/bin/python -m src.q2_v3.cli
```

该命令按 P0→P9 顺序执行，每阶段产物写入 `results/q2_v3/`，阶段测试失败或 `BLOCKED` 时停止。全量测试：

```bash
.venv/bin/python -m pytest -q tests/q2_v3
```

固定种子为 7。B1、B6 用于拟合；B7 新增 Loss 在 P4 模型冻结后才用于检验；B8、B9、B10 仅按各自来源类型作压力或外推参考。第一问只读取冻结产物，不重新训练。
