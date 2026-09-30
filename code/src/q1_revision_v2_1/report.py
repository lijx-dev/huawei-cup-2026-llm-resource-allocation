"""仅从 v2.1 实测输出和经核验的 v2 历史表生成完整报告。"""
import json

import numpy as np
import pandas as pd

from .common import paths


def table(frame, columns, digits=4):
    f=frame[columns].copy()
    for col in f:
        if pd.api.types.is_float_dtype(f[col]):
            f[col]=f[col].map(lambda v:"—" if pd.isna(v) else f"{v:.{digits}f}")
    f=f.fillna("—").astype(str)
    return "\n".join(["| "+" | ".join(f.columns)+" |",
                      "|"+"|".join("---" for _ in f.columns)+"|",
                      *["| "+" | ".join(row)+" |" for row in f.to_numpy()]])


def run(root, config):
    _,out,old=paths(root)
    target=out / "report/question1_v2_1_report.md"
    target.parent.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(out / "audit/field_invalid_resolution.csv")
    audit_summary=json.loads((out / "audit/audit_summary.json").read_text())
    quality=json.loads((out / "quality/quality_summary.json").read_text())
    weight=pd.read_csv(out / "quality/hierarchical_weights.csv")
    domains=pd.read_csv(out / "quality/domain_quality_summary.csv")
    methods=pd.read_csv(out / "quality/quality_method_comparison.csv")
    sensitivity=pd.read_csv(out / "quality/quality_sensitivity.csv")
    groups=pd.read_csv(out / "quality/semantic_group_distribution.csv")
    conflict=pd.read_csv(out / "conflict/conflict_method_comparison.csv")
    thresholds=json.loads((out / "conflict/disagreement_thresholds.json").read_text())
    rates=pd.read_csv(out / "conflict/domain_disagreement_summary.csv")
    numerical=pd.read_csv(out / "conflict/numerical_chars_dominance.csv")
    gaps=pd.read_csv(out / "conflict/semantic_group_disagreement.csv")
    model=json.loads((out / "mixture/inherited_v2_model_reference.json").read_text())
    distances=pd.read_csv(out / "mixture/nearest_train_distance_summary.csv")
    strata=pd.read_csv(out / "mixture/distance_stratified_summary.csv")
    folds=pd.read_csv(out / "mixture/cluster_cv_fold_summary.csv")
    cv=pd.read_csv(out / "mixture/random_vs_cluster_cv.csv")
    coverage=pd.read_csv(out / "q_mapping/qmapped_coverage_summary.csv")
    coverage_sensitivity=pd.read_csv(out / "q_mapping/qmapped_coverage_sensitivity.csv")
    placebo=pd.read_csv(out / "q_mapping/qmapped_placebo_summary.csv")
    domain_main=domains[domains.view=="union_unique"].sort_values("Q_hierarchical_balanced",ascending=False)
    valid=audit[audit.raw_invalid_rows>0]
    reason=valid[valid.field=="modernbert_reasoning"].iloc[0]
    professional=valid[valid.field=="modernbert_professionalism"].iloc[0]
    weight_numeric=weight[weight.field=="rps_lines_numerical_chars_fraction"].iloc[0]
    main_vs_drop=methods[methods.method=="Q_hierarchical_drop4"].iloc[0]
    affected=int((sensitivity.rank_shift!=0).sum())
    wnum=numerical[(numerical.measure=="weighted")&(numerical.attachment=="A1")&(numerical.high_set_method=="C_weighted_hierarchical")].iloc[0]
    enum=numerical[(numerical.measure=="equal")&(numerical.attachment=="A1")&(numerical.high_set_method=="C_weighted_hierarchical")].iloc[0]
    lgb_strata=strata[strata.model=="LightGBM"].set_index("distance_quartile")
    nearest=lgb_strata.loc["Q1_nearest","macro_nrmse_train_scale"]
    farthest=lgb_strata.loc["Q4_farthest","macro_nrmse_train_scale"]
    cluster_lgb=cv[(cv.evaluation=="cluster_group_cv_v2_1")&(cv.model=="LightGBM")].iloc[0].macro_nrmse_train_scale
    cluster_ridge=cv[(cv.evaluation=="cluster_group_cv_v2_1")&(cv.model=="Ridge")].iloc[0].macro_nrmse_train_scale
    near_conclusion=("最远四分位的标准化误差高于最近四分位，存在距离相关退化。" if farthest>nearest else
                     "最远四分位的标准化误差未高于最近四分位，本次 1M 检验没有显示性能只集中在最近配方。")
    cluster_conclusion=("LightGBM 在簇留出 CV 仍低于 Ridge 的误差，但其误差高于随机 repeated CV；优势不能仅归因于随机近邻划分，外推到新簇仍较困难。"
                        if cluster_lgb<cluster_ridge else
                        "簇留出 CV 下 LightGBM 未保持对 Ridge 的误差优势，v2 随机 CV 优势应限于相近配方插值。")
    psummary=[]
    for row in placebo.itertuples():
        if row.one_sided_empirical_p<.05:
            interpretation="真实映射优于大多数随机重排；只说明固定对应关系包含更多预测信息，不证明因果。"
        else:
            interpretation="真实映射增益未明显超出随机领域赋值，无法区分于一般配比派生特征。"
        psummary.append(f"- `{row.quality_method}`：真实 CV pooled R² 增量 {row.real_delta_pooled_r2_cv:+.6f}；"
                        f"placebo 百分位 {row.placebo_percentile:.3f}，单侧经验 p={row.one_sided_empirical_p:.3f}。{interpretation}")
    lines=["# 第一问 q1-revision-v2.1：稳健性补充报告","",
           f"实验版本 `{config['experiment_version']}`，全局随机种子 {config['seed']}。本次以 A1–A16 原件及经过哈希核验的 v2 中间产物为输入；v2 原结果保持只读。", "",
           "## 1. 字段 invalid 的记录级核查","",
           f"v2 字段表中的 reasoning={reason.raw_invalid_elements}、professionalism={professional.raw_invalid_elements} 是**非法列表元素次数**，分别发生在 {reason.raw_invalid_rows} 和 {professional.raw_invalid_rows} 条记录。两组记录均属于 v2 的 `rejected_corrupt`，没有重复非法副本、没有进入完整 22 维评分；v2 的 {audit_summary['status_counts'].get('suspected_unreliable',0)} 条可疑记录是其他记录。", "",
           table(valid,["field","raw_invalid_rows","raw_invalid_elements","unique_invalid_ids","duplicate_invalid_rows","already_excluded_structural_invalid","already_excluded_suspected","remaining_invalid_after_dedup","used_in_scoring"],0),"",
           f"原始质量行 {audit_summary['raw_quality_rows']}，有效去重 {audit_summary['valid_unique']}，可评分来源行 {audit_summary['accepted_source_rows']}。异常记录保留原始文件、行号、ID 和哈希指针，详见 `audit/field_invalid_rows.csv`。", "",
           "## 2. 分层平衡主质量评分与方向敏感性","",
           "固定的模型/元评分、DSIR、语言结构三个语义组各占 1/3；组内独立按熵信息差异乘以非冗余系数加权。没有结果驱动的权重 cap 或分组变更。本次未触发退化组等权 fallback。22 维矩阵从原件重建，与 v2 存储矩阵最大绝对差 " + f"{quality['new_matrix_vs_v2_max_abs_diff']:.2g}。", "",
           f"`rps_lines_numerical_chars_fraction` 的全局权重从 v2 的 {weight_numeric.old_v2_global_weight:.4f} 降为 {weight_numeric.global_weight:.4f}；当前最大单指标全局权重为 {quality['max_global_weight']:.4f}。", "",
           table(domain_main,["domain","n","Q_hierarchical_balanced","Q_hierarchical_balanced_sd","Q_hierarchical_balanced_ci025","Q_hierarchical_balanced_ci975","Q_hierarchical_drop4","Q_hierarchical_equal_within","Q_entropy_redundancy"]),"",
           "域内 95% 区间来自 seed=7 的 500 次 bootstrap。样本相关、Top 集合重叠和域排序：","",
           table(methods,["method","spearman_vs_hierarchical","top_1pct_overlap","top_5pct_overlap","domain_ranking"]),"",
           f"删除四项方向暂定指标后，样本 Spearman={main_vs_drop.spearman_vs_hierarchical:.4f}，Top 1% 重合={main_vs_drop.top_1pct_overlap:.4f}，{affected}/7 个领域的名次变化。域级绝对排序仍受方向假设影响，不能称为完全稳健。", "",
           "三个语义组的域内均值、标准差和分位数见 `quality/semantic_group_distribution.csv`；逐样本三组分数及所有 Q 见 `quality/sample_quality_scores.csv.gz`。", "",
           "## 3. 三种样本指标分歧与固定语义组解释","",
           "正式报告并列使用分层权重分歧 `C_weighted_hierarchical`、22 项等权分歧 `C_equal_indicator` 和固定语义组间分歧 `C_semantic_group`。三者 85/90/95 分位阈值都只从 A1 确定，再固定应用于 A2/A3。算法聚类保留 2–7 簇、三种 linkage 与树状图，仅作探索。", "",
           table(conflict,["first","second","union_unique_spearman","A1_high90_overlap_fraction"]),"",
           "A1 与两个扩展集在同一阈值下的高分歧率（主阈值 90%）：","",
           table(rates[rates.view.isin(["A1","A2","A3"])],["view","domain","method","n","mean_C","rate_0.9"]),"",
           f"在**相同的 A1 加权高分歧样本**上，`numerical_chars_fraction` 相关指标对占加权贡献 {wnum.incident_pair_share:.1%}，而其原始等权指标对差异占比 {enum.incident_pair_share:.1%}。这表明 v2 强主导有相当部分来自权重放大；该字段仍可能有实际分歧，但不能据加权贡献称其为最强现实质量矛盾。", "",
           "模型评分 vs DSIR、模型评分 vs 语言结构、DSIR vs 语言结构的全量／分域／高分歧组间差异见 `conflict/semantic_group_disagreement.csv`。指标分歧不是已确认脏数据。", "",
           "## 4. 冻结 v2 模型、配方距离与簇留出","",
           f"v2 模型文件本次 SHA-256=`{model['model_file_sha256']}`；v2 `metadata.json` **没有模型文件哈希锚点**，因此无法声称完成该项历史哈希匹配。冻结参数、17/13 字段顺序、seed=7 均核对；重载模型与 v2 1M 逐目标预测最大差为 {max(model['prediction_max_abs_diff'].values()):.2e}。模型溯源状态为 `{model['verification_status']}`。", "",
           "17 维归一化配比使用与 v2 扰动支持规则相同的欧氏距离；训练配方最近邻排除自身，测试只搜索训练集。10B/70B 配方是训练子集，距离为零仍属 estimated_reference。", "",
           table(distances,["split","n","mean","median","q05","q25","q75","q95","maximum"]),"",
           "1M 留出集按距离四分位的冻结模型表现：","",
           table(strata,["model","distance_quartile","n","mean_distance","pooled_r2","macro_nrmse_train_scale","mean_target_pearson","mean_target_spearman"]),"",
           near_conclusion,"",
           "KMeans 只在 A4 的 17 维配比上拟合（10 簇、n_init=20、seed=7）；GroupKFold 中每个簇只出现在单侧。Ridge 使用 v2 冻结的逐目标 λ，LightGBM 使用 v2 冻结配置，每折重新拟合。","",
           table(cv,["evaluation","model","macro_nrmse_train_scale","n_folds"]),"",
           cluster_conclusion+" 各折与 13 目标逐项指标见 `mixture/cluster_cv_metrics.csv`。", "",
           "## 5. 六域 Qmapped 覆盖率、阈值与置换诊断","",
           "仅 A16 的 3 个 direct 与 3 个 near_direct 域用于 Qmapped；11 个 inferred 域不作观测质量补全。Qmapped 可计算不等于六域覆盖充分。","",
           table(coverage,["split","n","n_with_q","mean","median","q10","q25","q75","q90","minimum","maximum"]),"",
           "coverage≥0.20/0.40/0.60 均为预设敏感性阈值。每个阈值使用同一筛选样本、同一五折及相同 λ 候选比较 Ridge(p) 和 Ridge(p,Qmapped)；样本不足则标记跳过。","",
           table(coverage_sensitivity,["method","coverage_threshold","status","n_train","n_test","cv_macro_nrmse_p","cv_macro_nrmse_p_q","cv_pooled_r2_p","cv_pooled_r2_p_q","test_pooled_r2_p","test_pooled_r2_p_q","cv_target_wins","test_target_wins"]),"",
           "Placebo 仅使用 A4/A5 训练配方和固定五折；1000 次 seed=7 的非恒等随机置换打乱六个领域质量值对应关系，保留配比、覆盖率、折分及 λ 候选。六域只有 719 个非恒等唯一排列，故 1000 次有重复，表中报告唯一排列数。经验 p 是置换诊断，不是因果显著性。","",
           table(placebo,["quality_method","n_train","repetitions","unique_nonidentity_permutations","real_delta_pooled_r2_cv","placebo_delta_r2_mean","placebo_delta_r2_q025","placebo_delta_r2_q975","placebo_percentile","one_sided_empirical_p"]),"",
           *psummary,"",
           "## 6. 结论边界和复现","",
           "新 Q 是对 22 个信号的评价规则；三种 C 描述指标一致性；冻结 LightGBM 只说明附件配方范围内的 Loss 预测。有限扰动仍为路径条件下模型响应。Qmapped 是六域覆盖下的配比派生质量代理；置换结果不能建立质量因果效应。60M/1B 是跨规模留出，10B/70B 仅为估算参考。", "",
           "本次没有按留出集或 placebo 结果调整主 Q、分组或 LightGBM 超参数。原始附件、v2 历史结果的 SHA-256 以及 v2.1 代码和配置哈希见 `metadata.json`；每幅图的输入哈希见 `figures/figure_sources.json`。", ""]
    target.write_text("\n".join(lines),encoding="utf-8")
    return target
