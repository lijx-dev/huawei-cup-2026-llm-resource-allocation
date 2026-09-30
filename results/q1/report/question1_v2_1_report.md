# 第一问 q1-revision-v2.1：稳健性补充报告

实验版本 `q1-revision-v2.1`，全局随机种子 7。本次以 A1–A16 原件及经过哈希核验的 v2 中间产物为输入；v2 原结果保持只读。

## 1. 字段 invalid 的记录级核查

v2 字段表中的 reasoning=78、professionalism=36 是**非法列表元素次数**，分别发生在 13 和 6 条记录。两组记录均属于 v2 的 `rejected_corrupt`，没有重复非法副本、没有进入完整 22 维评分；v2 的 11 条可疑记录是其他记录。

| field | raw_invalid_rows | raw_invalid_elements | unique_invalid_ids | duplicate_invalid_rows | already_excluded_structural_invalid | already_excluded_suspected | remaining_invalid_after_dedup | used_in_scoring |
|---|---|---|---|---|---|---|---|---|
| modernbert_reasoning | 13 | 78 | 13 | 0 | 13 | 0 | 0 | False |
| modernbert_professionalism | 6 | 36 | 6 | 0 | 6 | 0 | 0 | False |

原始质量行 272505，有效去重 261056，可评分来源行 272475。异常记录保留原始文件、行号、ID 和哈希指针，详见 `audit/field_invalid_rows.csv`。

## 2. 分层平衡主质量评分与方向敏感性

固定的模型/元评分、DSIR、语言结构三个语义组各占 1/3；组内独立按熵信息差异乘以非冗余系数加权。没有结果驱动的权重 cap 或分组变更。本次未触发退化组等权 fallback。22 维矩阵从原件重建，与 v2 存储矩阵最大绝对差 0。

`rps_lines_numerical_chars_fraction` 的全局权重从 v2 的 0.2520 降为 0.1360；当前最大单指标全局权重为 0.1574。

| domain | n | Q_hierarchical_balanced | Q_hierarchical_balanced_sd | Q_hierarchical_balanced_ci025 | Q_hierarchical_balanced_ci975 | Q_hierarchical_drop4 | Q_hierarchical_equal_within | Q_entropy_redundancy |
|---|---|---|---|---|---|---|---|---|
| stackexchange | 9998 | 64.8150 | 5.6386 | 64.7054 | 64.9290 | 72.6057 | 71.7882 | 46.0357 |
| arxiv | 17523 | 64.2046 | 9.6219 | 64.0642 | 64.3491 | 68.1357 | 70.5346 | 61.5011 |
| commoncrawl | 9636 | 63.6085 | 5.7005 | 63.4820 | 63.7233 | 73.3071 | 72.4317 | 46.6178 |
| c4 | 10000 | 62.2814 | 5.8588 | 62.1664 | 62.3819 | 78.0910 | 70.7549 | 44.2553 |
| wikipedia | 9986 | 60.2105 | 5.7381 | 60.1026 | 60.3187 | 68.1621 | 67.4050 | 41.2631 |
| github | 203742 | 54.0570 | 5.5766 | 54.0342 | 54.0783 | 60.2535 | 64.4309 | 32.9816 |
| book | 171 | 45.8456 | 9.3756 | 44.3081 | 47.2482 | 52.1985 | 54.0029 | 51.0270 |

域内 95% 区间来自 seed=7 的 500 次 bootstrap。样本相关、Top 集合重叠和域排序：

| method | spearman_vs_hierarchical | top_1pct_overlap | top_5pct_overlap | domain_ranking |
|---|---|---|---|---|
| Q_hierarchical_balanced | 1.0000 | 1.0000 | 1.0000 | stackexchange>arxiv>commoncrawl>c4>wikipedia>github>book |
| Q_hierarchical_drop4 | 0.8693 | 0.1536 | 0.4544 | c4>commoncrawl>stackexchange>wikipedia>arxiv>github>book |
| Q_hierarchical_equal_within | 0.9130 | 0.4401 | 0.7584 | commoncrawl>stackexchange>c4>arxiv>wikipedia>github>book |
| Q_entropy_redundancy | 0.8653 | 0.4638 | 0.5616 | arxiv>book>commoncrawl>stackexchange>c4>wikipedia>github |
| Q_group_equal | 0.9130 | 0.4401 | 0.7584 | commoncrawl>stackexchange>c4>arxiv>wikipedia>github>book |
| Q_TOPSIS | 0.7681 | 0.4339 | 0.4082 | arxiv>book>stackexchange>commoncrawl>wikipedia>c4>github |

删除四项方向暂定指标后，样本 Spearman=0.8693，Top 1% 重合=0.1536，5/7 个领域的名次变化。域级绝对排序仍受方向假设影响，不能称为完全稳健。

三个语义组的域内均值、标准差和分位数见 `quality/semantic_group_distribution.csv`；逐样本三组分数及所有 Q 见 `quality/sample_quality_scores.csv.gz`。

## 3. 三种样本指标分歧与固定语义组解释

正式报告并列使用分层权重分歧 `C_weighted_hierarchical`、22 项等权分歧 `C_equal_indicator` 和固定语义组间分歧 `C_semantic_group`。三者 85/90/95 分位阈值都只从 A1 确定，再固定应用于 A2/A3。算法聚类保留 2–7 簇、三种 linkage 与树状图，仅作探索。

| first | second | union_unique_spearman | A1_high90_overlap_fraction |
|---|---|---|---|
| C_equal_indicator | C_weighted_hierarchical | 0.7670 | 0.7090 |
| C_equal_indicator | C_semantic_group | 0.5065 | 0.3950 |
| C_semantic_group | C_weighted_hierarchical | 0.8248 | 0.4956 |

A1 与两个扩展集在同一阈值下的高分歧率（主阈值 90%）：

| view | domain | method | n | mean_C | rate_0.9 |
|---|---|---|---|---|---|
| A1 | arxiv | C_weighted_hierarchical | 1419 | 0.3720 | 0.1092 |
| A1 | arxiv | C_equal_indicator | 1419 | 0.3399 | 0.1078 |
| A1 | arxiv | C_semantic_group | 1419 | 0.4086 | 0.1339 |
| A1 | book | C_weighted_hierarchical | 171 | 0.4344 | 0.1345 |
| A1 | book | C_equal_indicator | 171 | 0.4344 | 0.6257 |
| A1 | book | C_semantic_group | 171 | 0.3210 | 0.0760 |
| A1 | c4 | C_weighted_hierarchical | 10000 | 0.4555 | 0.2404 |
| A1 | c4 | C_equal_indicator | 10000 | 0.4014 | 0.2827 |
| A1 | c4 | C_semantic_group | 10000 | 0.4286 | 0.0659 |
| A1 | commoncrawl | C_weighted_hierarchical | 9636 | 0.4101 | 0.0093 |
| A1 | commoncrawl | C_equal_indicator | 9636 | 0.3588 | 0.0134 |
| A1 | commoncrawl | C_semantic_group | 9636 | 0.4014 | 0.0114 |
| A1 | github | C_weighted_hierarchical | 10000 | 0.4418 | 0.1623 |
| A1 | github | C_equal_indicator | 10000 | 0.3795 | 0.1145 |
| A1 | github | C_semantic_group | 10000 | 0.4588 | 0.2381 |
| A1 | stackexchange | C_weighted_hierarchical | 9998 | 0.3954 | 0.0175 |
| A1 | stackexchange | C_equal_indicator | 9998 | 0.3399 | 0.0093 |
| A1 | stackexchange | C_semantic_group | 9998 | 0.4308 | 0.0457 |
| A1 | wikipedia | C_weighted_hierarchical | 9986 | 0.4276 | 0.0652 |
| A1 | wikipedia | C_equal_indicator | 9986 | 0.3785 | 0.0668 |
| A1 | wikipedia | C_semantic_group | 9986 | 0.4365 | 0.1313 |
| A2 | arxiv | C_weighted_hierarchical | 17523 | 0.3753 | 0.1162 |
| A2 | arxiv | C_equal_indicator | 17523 | 0.3425 | 0.1148 |
| A2 | arxiv | C_semantic_group | 17523 | 0.4110 | 0.1415 |
| A3 | github | C_weighted_hierarchical | 203742 | 0.4420 | 0.1639 |
| A3 | github | C_equal_indicator | 203742 | 0.3796 | 0.1159 |
| A3 | github | C_semantic_group | 203742 | 0.4590 | 0.2416 |

在**相同的 A1 加权高分歧样本**上，`numerical_chars_fraction` 相关指标对占加权贡献 30.6%，而其原始等权指标对差异占比 11.1%。这表明 v2 强主导有相当部分来自权重放大；该字段仍可能有实际分歧，但不能据加权贡献称其为最强现实质量矛盾。

模型评分 vs DSIR、模型评分 vs 语言结构、DSIR vs 语言结构的全量／分域／高分歧组间差异见 `conflict/semantic_group_disagreement.csv`。指标分歧不是已确认脏数据。

## 4. 冻结 v2 模型、配方距离与簇留出

v2 模型文件本次 SHA-256=`39c5d5dc13b6f0fbf68699bb45d2d8d18e28f81f0f46f5475311e6aa66bd1c33`；v2 `metadata.json` **没有模型文件哈希锚点**，因此无法声称完成该项历史哈希匹配。冻结参数、17/13 字段顺序、seed=7 均核对；重载模型与 v2 1M 逐目标预测最大差为 8.88e-16。模型溯源状态为 `prediction_verified_hash_unanchored`。

17 维归一化配比使用与 v2 扰动支持规则相同的欧氏距离；训练配方最近邻排除自身，测试只搜索训练集。10B/70B 配方是训练子集，距离为零仍属 estimated_reference。

| split | n | mean | median | q05 | q25 | q75 | q95 | maximum |
|---|---|---|---|---|---|---|---|---|
| est_10b | 63 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| est_70b | 63 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test_1b | 64 | 0.2153 | 0.2171 | 0.1492 | 0.1908 | 0.2414 | 0.2814 | 0.3238 |
| test_1m | 256 | 0.1716 | 0.1767 | 0.0766 | 0.1364 | 0.2096 | 0.2512 | 0.2968 |
| test_60m | 256 | 0.1716 | 0.1767 | 0.0766 | 0.1364 | 0.2096 | 0.2512 | 0.2968 |
| train_1m | 512 | 0.1706 | 0.1779 | 0.0581 | 0.1283 | 0.2165 | 0.2600 | 0.3315 |

1M 留出集按距离四分位的冻结模型表现：

| model | distance_quartile | n | mean_distance | pooled_r2 | macro_nrmse_train_scale | mean_target_pearson | mean_target_spearman |
|---|---|---|---|---|---|---|---|
| Ridge | Q1_nearest | 64 | 0.0958 | 0.5422 | 0.6260 | 0.7955 | 0.7771 |
| Ridge | Q2 | 64 | 0.1569 | 0.4944 | 0.5648 | 0.7795 | 0.8119 |
| Ridge | Q3 | 64 | 0.1945 | 0.5314 | 0.5418 | 0.8059 | 0.8535 |
| Ridge | Q4_farthest | 64 | 0.2394 | 0.5264 | 0.5665 | 0.7875 | 0.8385 |
| LightGBM | Q1_nearest | 64 | 0.0958 | 0.9766 | 0.1612 | 0.9889 | 0.9822 |
| LightGBM | Q2 | 64 | 0.1569 | 0.9848 | 0.1160 | 0.9923 | 0.9906 |
| LightGBM | Q3 | 64 | 0.1945 | 0.9830 | 0.1166 | 0.9918 | 0.9887 |
| LightGBM | Q4_farthest | 64 | 0.2394 | 0.9806 | 0.1212 | 0.9908 | 0.9863 |

最远四分位的标准化误差未高于最近四分位，本次 1M 检验没有显示性能只集中在最近配方。

KMeans 只在 A4 的 17 维配比上拟合（10 簇、n_init=20、seed=7）；GroupKFold 中每个簇只出现在单侧。Ridge 使用 v2 冻结的逐目标 λ，LightGBM 使用 v2 冻结配置，每折重新拟合。

| evaluation | model | macro_nrmse_train_scale | n_folds |
|---|---|---|---|
| random_repeated_cv_v2 | Ridge | 0.6280 | 50 |
| random_repeated_cv_v2 | LightGBM | 0.1662 | 50 |
| cluster_group_cv_v2_1 | LightGBM | 0.2391 | 5 |
| cluster_group_cv_v2_1 | Ridge | 0.8532 | 5 |

LightGBM 在簇留出 CV 仍低于 Ridge 的误差，但其误差高于随机 repeated CV；优势不能仅归因于随机近邻划分，外推到新簇仍较困难。 各折与 13 目标逐项指标见 `mixture/cluster_cv_metrics.csv`。

## 5. 六域 Qmapped 覆盖率、阈值与置换诊断

仅 A16 的 3 个 direct 与 3 个 near_direct 域用于 Qmapped；11 个 inferred 域不作观测质量补全。Qmapped 可计算不等于六域覆盖充分。

| split | n | n_with_q | mean | median | q10 | q25 | q75 | q90 | minimum | maximum |
|---|---|---|---|---|---|---|---|---|---|---|
| train_1m | 512 | 507 | 0.5646 | 0.6004 | 0.1502 | 0.3393 | 0.7993 | 0.9408 | 0.0000 | 1.0000 |
| test_1m | 256 | 256 | 0.5690 | 0.5830 | 0.1694 | 0.3241 | 0.8238 | 0.9460 | 0.0010 | 1.0000 |
| test_60m | 256 | 256 | 0.5690 | 0.5830 | 0.1694 | 0.3241 | 0.8238 | 0.9460 | 0.0010 | 1.0000 |
| test_1b | 64 | 64 | 0.6226 | 0.6476 | 0.4411 | 0.5265 | 0.7238 | 0.7995 | 0.1380 | 0.8889 |
| est_10b | 63 | 63 | 0.5435 | 0.5714 | 0.1255 | 0.3030 | 0.8035 | 0.9315 | 0.0030 | 1.0000 |
| est_70b | 63 | 63 | 0.5435 | 0.5714 | 0.1255 | 0.3030 | 0.8035 | 0.9315 | 0.0030 | 1.0000 |

coverage≥0.20/0.40/0.60 均为预设敏感性阈值。每个阈值使用同一筛选样本、同一五折及相同 λ 候选比较 Ridge(p) 和 Ridge(p,Qmapped)；样本不足则标记跳过。

| method | coverage_threshold | status | n_train | n_test | cv_macro_nrmse_p | cv_macro_nrmse_p_q | cv_pooled_r2_p | cv_pooled_r2_p_q | test_pooled_r2_p | test_pooled_r2_p_q | cv_target_wins | test_target_wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_hierarchical_balanced | 0.0000 | computed | 507 | 256 | 0.6256 | 0.6232 | 0.5181 | 0.5190 | 0.5434 | 0.5424 | 8 | 7 |
| Q_entropy_redundancy | 0.0000 | computed | 507 | 256 | 0.6256 | 0.6223 | 0.5181 | 0.5243 | 0.5434 | 0.5522 | 8 | 9 |
| Q_hierarchical_balanced | 0.2000 | computed | 445 | 220 | 0.6210 | 0.6200 | 0.5140 | 0.5156 | 0.5434 | 0.5441 | 5 | 11 |
| Q_entropy_redundancy | 0.2000 | computed | 445 | 220 | 0.6210 | 0.6182 | 0.5140 | 0.5182 | 0.5434 | 0.5507 | 10 | 9 |
| Q_hierarchical_balanced | 0.4000 | computed | 354 | 180 | 0.6083 | 0.6086 | 0.5284 | 0.5283 | 0.5352 | 0.5351 | 5 | 9 |
| Q_entropy_redundancy | 0.4000 | computed | 354 | 180 | 0.6083 | 0.6074 | 0.5284 | 0.5311 | 0.5352 | 0.5366 | 6 | 4 |
| Q_hierarchical_balanced | 0.6000 | computed | 257 | 123 | 0.5914 | 0.5896 | 0.5487 | 0.5506 | 0.5331 | 0.5306 | 6 | 5 |
| Q_entropy_redundancy | 0.6000 | computed | 257 | 123 | 0.5914 | 0.5889 | 0.5487 | 0.5506 | 0.5331 | 0.5303 | 10 | 6 |

Placebo 仅使用 A4/A5 训练配方和固定五折；1000 次 seed=7 的非恒等随机置换打乱六个领域质量值对应关系，保留配比、覆盖率、折分及 λ 候选。六域只有 719 个非恒等唯一排列，故 1000 次有重复，表中报告唯一排列数。经验 p 是置换诊断，不是因果显著性。

| quality_method | n_train | repetitions | unique_nonidentity_permutations | real_delta_pooled_r2_cv | placebo_delta_r2_mean | placebo_delta_r2_q025 | placebo_delta_r2_q975 | placebo_percentile | one_sided_empirical_p |
|---|---|---|---|---|---|---|---|---|---|
| Q_hierarchical_balanced | 507 | 1000 | 543 | 0.0009 | 0.0051 | -0.0005 | 0.0161 | 0.1460 | 0.8541 |
| Q_entropy_redundancy | 507 | 1000 | 543 | 0.0062 | 0.0051 | -0.0015 | 0.0142 | 0.6470 | 0.3536 |

- `Q_hierarchical_balanced`：真实 CV pooled R² 增量 +0.000921；placebo 百分位 0.146，单侧经验 p=0.854。真实映射增益未明显超出随机领域赋值，无法区分于一般配比派生特征。
- `Q_entropy_redundancy`：真实 CV pooled R² 增量 +0.006184；placebo 百分位 0.647，单侧经验 p=0.354。真实映射增益未明显超出随机领域赋值，无法区分于一般配比派生特征。

## 6. 结论边界和复现

新 Q 是对 22 个信号的评价规则；三种 C 描述指标一致性；冻结 LightGBM 只说明附件配方范围内的 Loss 预测。有限扰动仍为路径条件下模型响应。Qmapped 是六域覆盖下的配比派生质量代理；置换结果不能建立质量因果效应。60M/1B 是跨规模留出，10B/70B 仅为估算参考。

本次没有按留出集或 placebo 结果调整主 Q、分组或 LightGBM 超参数。原始附件、v2 历史结果的 SHA-256 以及 v2.1 代码和配置哈希见 `metadata.json`；每幅图的输入哈希见 `figures/figure_sources.json`。
