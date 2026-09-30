# P4 第一问冻结配比接口

使用 LightGBM 冻结模型，SHA-256=39c5d5dc13b6f0fbf68699bb45d2d8d18e28f81f0f46f5475311e6aa66bd1c33；17 输入、13 Loss 输出顺序见 metadata。A4 配方由正式接口逐行归一化，参考配比总和为 1.000000000；未改动 A4。

h_v(p0)=0，13 个目标分开输出。h_agg 为等权 13 个无量纲对数比，仅用于明确标记的情景汇总，不将 B 的标量 Loss 与任一 A 目标天然等同。支持阈值=0.260035。正式接口=results/q1_revision_v2_1/q2_interface/manifest.json。

Q2-P4 STATUS: PASS_WITH_WARNINGS
