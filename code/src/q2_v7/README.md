# 问题二 q2_v7

这是独立的第二问实现。运行时不导入 `src/q2*` 的旧版本，也不读取 `results/q2*` 的旧版产物。配置与计算结果仅写入 `configs/q2_v7/`、`src/q2_v7/`、`tests/q2_v7/`、`results/q2_v7/`。

## 运行

在仓库根目录执行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/q2_v7
PYTHONPATH=src .venv/bin/python -m q2_v7.pipeline --root .
```

入口先验证交接 ZIP 的文件 SHA-256、Q1 M2 接口结构与零点、19 个 B 附件与 `source_manifest.json` 的字节数。数值模型均由本版源代码重新拟合。输入文件哈希、配置与代码哈希、结果哈希汇总在 `results/q2_v7/metadata.json`。

## 输入边界

- B1–B12 从 `data/real_attachments/B_scaling_laws/` 读取；B1 用于经典律拟合，B6 用于质量律拟合。B7 去除与 B6 完全一致的记录后才用于留出。
- Q1 M2 响应接口及系数只读本次 ZIP；系数仅用于已冻结模型的可控份额转移，并先复算响应表，绝不重新用附件 A 调参。
- 建模文档第 4.5 节的跨规模配对诊断独立读取附件 A 的四个 RegMix 小表。它仅提供 A 端内部量级证据，不训练 B 模型，也不估计 A→B 的 `lambda_p`。
- `Q_A` 的部分量尺情景读取现有问题一域级结果；只纳入 `direct` 与 `near_direct`，不补齐 `inferred` 域。没有 A/B 同样本质量锚点，`Q_A→Q_B` 仅为情景。

## 结果读法

`report.md` 记录主要拟合、分组验证、B8 方向压力测试、两档同配方诊断、边际效应及限制。`scenario/` 和 `compute/mixture_scenarios.csv` 中的 `lambda_p`、`theta_Qp`、Form A/B 是预设情景；不能当作联合估计或独立验证。`regmix/`、`quality/`、`baseline/` 与 `transfer/` 分别保留各自的数据和证据等级。
