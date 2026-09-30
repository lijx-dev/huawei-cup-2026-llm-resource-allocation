# 问题三融合方案｜最新版 2026-09-26

**LATEST / 当前最新版**：`q3-latest-20260926-v1`；状态 `PASS_WITH_WARNINGS`。

本版使用问题二 `q2-fusion-rerun-v1` 的 MQ 主模型和问题一正式 LightGBM 接口；不再使用 q2_v8 SET_A 或 mix_final_model_quad.csv。

## 重跑摘要

- 情景网格 135 行：可行 123，不可行 12。
- 主 η=2e-4 网格 45 行。
- Q2 组 bootstrap 传播成功 60/60 次。
- 局部数值/KKT 复核 9 点，验收=True。
- 参数化等价转换最大绝对误差 0.000e+00。
- Q4 机制腿工作点的总 Loss 预算弹性 epsilon_C=-0.04334897。

## 配比结论

正式接口中加入 p0 后候选 h_agg 范围 [0, 0.333013]。当前 LightGBM 接口下 p0 是这些离散候选的最小值；这不是连续单纯形上的全局最优证明。lambda_p 仍只取情景值。

## 证据边界

质量成本函数、eta 与 lambda_p 均未由联合实验识别；高预算解常触及 N/D 支持边界。所有最优值是给定模型与成本情景的条件解。

Q3 LATEST STATUS: PASS_WITH_WARNINGS
