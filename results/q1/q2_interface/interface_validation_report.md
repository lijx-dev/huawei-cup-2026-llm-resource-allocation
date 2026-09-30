# Q1→Q2 正式接口校验

冻结模型：LightGBM；SHA-256 `39c5d5dc13b6f0fbf68699bb45d2d8d18e28f81f0f46f5475311e6aa66bd1c33`。
接口包含 1214 个配方、17 维归一化输入和 13 维观测/预测 Loss；所有配比与 Loss 均按 `index` 一一连接。
Q_A 与 Q_B 不作数值转换；质量域只输出 direct/near_direct 六条可靠映射。
10B/70B 的 Loss 保持 `estimated_reference` 身份，不作为真实观测。

FINAL_STATUS = PASS
