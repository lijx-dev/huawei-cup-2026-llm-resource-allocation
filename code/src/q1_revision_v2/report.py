"""只从本次实验目录读取实测表生成第一问报告。"""
import json

import numpy as np
import pandas as pd

from .common import paths


def _table(frame, columns, digits=4):
    f = frame[columns].copy()
    for key in f:
        if pd.api.types.is_float_dtype(f[key]):
            f[key] = f[key].map(lambda x: "—" if pd.isna(x) else f"{x:.{digits}f}")
    f = f.fillna("—").astype(str)
    header = "| " + " | ".join(f.columns) + " |"
    separator = "|" + "|".join("---" for _ in f.columns) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in f.to_numpy()]
    return "\n".join([header, separator, *rows])


def run(root, config):
    _, out = paths(root)
    report = out / "report/question1_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    audit = json.loads((out / "audit/audit_summary.json").read_text())
    quality = pd.read_csv(out / "quality/domain_scores.csv")
    stability = pd.read_csv(out / "quality/score_stability.csv")
    parameters = pd.read_csv(out / "quality/indicator_parameters.csv")
    uncertain = parameters[parameters.field.isin(config["direction_sensitive"])].sort_values("w_main", ascending=False).iloc[0]
    schema = pd.read_csv(out / "quality/indicator_schema_and_conversion.csv")
    clusters = pd.read_csv(out / "conflict/clusters.csv")
    conflict = json.loads((out / "conflict/conflict_summary.json").read_text())
    rates = pd.read_csv(out / "conflict/domain_conflict_rates.csv")
    pairs = pd.read_csv(out / "conflict/indicator_pair_contributions.csv")
    mixture = json.loads((out / "mixture/mixture_summary.json").read_text())
    holdout = pd.read_csv(out / "mixture/holdout_metrics.csv")
    summary = pd.read_csv(out / "mixture/holdout_summary.csv")
    estimates = pd.read_csv(out / "mixture/estimated_rank_metrics.csv")
    singles = pd.read_csv(out / "mixture/single_domain_effects.csv")
    doubles = pd.read_csv(out / "mixture/pair_interactions.csv")
    common_ame = pd.read_csv(out / "mixture/common_subset_ame.csv")
    cv_variability = pd.read_csv(out / "mixture/cv_repeat_variability.csv")
    integration = json.loads((out / "integration/integration_summary.json").read_text())
    comparison = pd.read_csv(out / "integration/ridge_quality_comparison.csv")
    qhold = pd.read_csv(out / "integration/holdout_summary.csv")
    ledger = pd.read_csv(out / "audit/source_ledger.csv.gz", usecols=["attachment", "record_id", "status"], keep_default_na=False)
    accepted = ledger[ledger.status.isin(["valid", "duplicate_exact"])]
    ids = {name: set(accepted.loc[accepted.attachment == name, "record_id"]) for name in ("A1", "A2", "A3")}
    overlap = {(a,b): len(ids[a] & ids[b]) for a,b in (("A1","A2"),("A1","A3"),("A2","A3"))}
    quality_main = quality[quality.view == "union_unique"].sort_values("Q_entropy_redundancy", ascending=False)
    quality_comparison = quality[quality.view.isin(["A1", "A2", "A3"])][["view", "domain", "n", "Q_entropy_redundancy"]]
    metric_target = holdout[holdout.target != "average_loss"]
    observed_summary = summary.sort_values(["split", "model"])
    estimate_summary = estimates[estimates.target != "average_loss"].groupby(["split", "model"], as_index=False)[["spearman", "kendall", "overlap_5", "overlap_10", "overlap_20"]].mean()
    top_pairs = {a: set(zip(frame.field_j, frame.field_k)) for a,frame in
                 ((a,pairs[pairs.attachment == a].nlargest(5, "H_jk")) for a in ("A1","A2","A3"))}
    pair_overlap_a2 = len(top_pairs["A1"] & top_pairs["A2"])
    pair_overlap_a3 = len(top_pairs["A1"] & top_pairs["A3"])
    raw_counts = ", ".join(f"{item['attachment']}={item['rows_read']}" for item in audit["quality"]["files"])
    qtest = qhold.set_index(["quality_method", "model"])
    main_pooled_gain = qtest.loc[("Q_entropy_redundancy", "Ridge_p_Qmapped"), "pooled_r2"] - qtest.loc[("Q_entropy_redundancy", "Ridge_p"), "pooled_r2"]
    group_pooled_gain = qtest.loc[("Q_group_equal", "Ridge_p_Qmapped"), "pooled_r2"] - qtest.loc[("Q_group_equal", "Ridge_p"), "pooled_r2"]
    q_stable = (main_pooled_gain > 0 and group_pooled_gain > 0 and
                integration["Q_entropy_redundancy"]["relative_cv_gain"] > 0 and
                integration["Q_group_equal"]["relative_cv_gain"] > 0)
    q_conclusion = ("两个质量口径在 CV 和 1M 留出均提升，提示当前映射可提供预测增益；仍不能推断因果效应。"
                    if q_stable else
                    "两种质量评分口径的留出增益不一致；当前附件与可靠映射不足以证明 Q 在 p 之外具有稳定独立预测收益。正式配比主模型保持只用 p。这不表示数据质量不重要。")
    lines = ["# 第一问：q1-revision-v2 正式重算报告", "",
             f"随机种子：{config['seed']}。输入为 A1–A16 原件；各阶段只读取本次 `results/q1_revision_v2/` 产物。报告数字由本次运行表格生成。", "",
             "## 1. 完整性、去重与隔离", "",
             f"A1–A3 实际原始行数：{raw_counts}；有效去重并集 {audit['quality']['union_unique_valid']} 条。"
             f"同 ID 同 22 项指标重复 {audit['quality']['status_counts'].get('duplicate_exact',0)} 条；"
             f"结构损坏 {audit['quality']['status_counts'].get('rejected_corrupt',0)} 条；"
             f"待核实可疑 {audit['quality']['status_counts'].get('suspected_unreliable',0)} 条；"
             f"同 ID 指标冲突 {audit['quality']['status_counts'].get('duplicate_conflict',0)} 条。可疑记录未进入拟合。", "",
             f"有效记录 ID 重叠：A1/A2={overlap['A1','A2']}，A1/A3={overlap['A1','A3']}，A2/A3={overlap['A2','A3']}。重复来源保留在独立视图，去重并集仅计一次。", "",
             "A4–A15 六对配比/Loss 表均通过 index 一一对应；A12–A15 标记为 estimated_reference。A16 的 17 个配方域中 3 个 direct、3 个 near_direct、11 个 inferred。", "",
             "## 2. 22 项指标处理、方向与质量评分", "",
             "ModernBERT 四项六等级 logits 经稳定 softmax 取 0–5 等级期望；fluency_en 取流畅类概率，ad_en 取无广告类概率；fineweb_edu 取唯一元素。QuRater 四个不同维度各自按并集 q01/q99 归一化后等权合成，没有执行 softmax。词数和句数先执行 log1p。每项再按有效去重并集的 q01/q99 截尾缩放，A1/A2/A3 共用尺度。列表语义依据[原始数据卡](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/blob/main/README.md)与修订稿。", "",
             "反向信号是无字母词比例、大写字母比例、高频二元和三元字符比例；广告信号已经转为无广告概率。词数、句数、数字比例、平均词长在主情景暂定正向，并提供删除四项及分组等权两种方向敏感性情景。方向是评分假设，不代表领域无关的客观真值。", "",
             "下表原始类型与异常数来自可解析行的逐字段审计，转换后范围仅统计进入评分的有效记录；每字段边界、熵信息差异及权重另见 `quality/indicator_parameters.csv`。", "",
             _table(schema, ["field", "raw_types", "conversion_method", "direction", "invalid", "transformed_min", "transformed_max"], 3), "",
             "熵信息差异与非冗余系数分别是统计分散度和相关冗余，不可解释为对训练效果的因果重要性。主 Q 为 `Q_entropy_redundancy`，TOPSIS 仅用于独立敏感性比较。", "",
             _table(quality_main, ["domain", "n", "Q_entropy_redundancy", "Q_entropy_redundancy_sd", "Q_entropy_redundancy_ci025", "Q_entropy_redundancy_ci975", "Q_group_equal", "Q_direction_drop4"]), "",
             "领域均值的不确定性为 seed=7 的 500 次域内 bootstrap 95% 区间；book 样本较少，解释须保留区间。A1 与扩展集对照：", "",
             _table(quality_comparison, ["view", "domain", "n", "Q_entropy_redundancy"]), "",
             "评分方法的全量样本 Spearman 与 Top 1% 重叠，以及领域排名：", "",
             _table(stability, ["method", "spearman_vs_main", "top_1pct_overlap", "domain_order"]), "",
             f"主评分中方向待定指标的最高权重为 `{uncertain.field}`={uncertain.w_main:.4f}；删除四项待定指标后，领域排序及 Top 1% 集合均会变化。因此当前主 Q 是明确评分规则下的结果，稳健性分析与其并列报告。", "",
             "## 3. 指标聚类与样本指标分歧", "",
             f"聚类使用有符号 Spearman 距离 `sqrt((1-rho)/2)`；平均连接主方案请求 {config['cluster_primary_k']} 簇，同时保存 2–7 簇与三种 linkage 的敏感性表。簇是探索性统计分组，其语义必须结合字段解释。", "",
             _table(clusters, ["field", "cluster", "weight"]), "",
             "按字段含义核对，本次主聚类把长度、熵、教育/流畅度和部分模型评分混在同一簇；DSIR 与若干词汇、标点信号也混在另一簇，另有近似单字段簇。因此簇编号仅用于统计分组，不将其命名为单一潜在质量能力。", "",
             f"样本指标分歧指数 C 使用 {conflict['pair_count']} 对指标。A1 固定阈值："
             + ", ".join(f"tau{int(float(q)*100)}={v:.6f}" for q,v in conflict['thresholds_from_A1'].items())
             + "；A2/A3 未重估阈值。", "",
             _table(rates, ["attachment", "domain", "n", "C_mean", "rate_0.85", "rate_0.9", "rate_0.95"]), "",
             "A1-arxiv 与 A2-arxiv、A1-github 与 A3-github 在上表按同一阈值比较。每对指标的平均贡献 H_jk 和高分歧样本的最大簇对见 `conflict/indicator_pair_contributions.csv` 与 `conflict/high_conflict_cluster_pairs.csv`。高分歧表示信号不一致，不构成损坏或伪造证据。", "",
             "A1/A2/A3 高分歧贡献前五项（各自独立排序）：", "",
             _table(pairs.sort_values(["attachment", "H_jk"], ascending=[True, False]).groupby("attachment").head(5), ["attachment", "field_j", "field_k", "H_jk"]), "",
             f"以上前五指标对中，A2 与 A1 重合 {pair_overlap_a2}/5，A3 与 A1 重合 {pair_overlap_a3}/5；这是固定 A1 阈值下的贡献模式复现度，不是独立质量标签验证。", "",
             "## 4. 17 域配比与 13 个 Loss", "",
             f"A4/A5 的 {mixture['training_rows']} 条配方是唯一训练和模型选择来源。17 个比例输入、13 个独立 Loss 输出严格按 index 连接。Ridge 省略第 17 个参考域且每折内拟合 StandardScaler；LightGBM 使用全部 17 列、固定 seed=7、deterministic=True、force_col_wise=True。两者共享 RepeatedKFold(5×10)。", "",
             f"Ridge macro NRMSE={mixture['selection']['ridge_macro_nrmse']:.4f}，LightGBM={mixture['selection']['lightgbm_macro_nrmse']:.4f}；后者胜出 {mixture['selection']['lightgbm_target_wins']}/13 个目标与 {mixture['selection']['lightgbm_repeat_wins']}/10 次重复。因此预定选择规则给出主模型 **{mixture['selection']['main_model']}**。模型选择在打开 A6/A7 前完成。", "",
             "重复 CV 各目标 RMSE 的折间均值和标准差：", "",
             _table(cv_variability, ["target_index", "ridge_rmse_mean", "ridge_rmse_sd", "lightgbm_rmse_mean", "lightgbm_rmse_sd"]), "",
             "逐折指标、每目标 Ridge λ 和 LightGBM 参数见 `mixture/cv_fold_metrics.csv`、`mixture/model_selection.json`。以下 1M/60M/1B 汇总以训练集均值和标准差构建标准化宏指标；pooled R² 直接汇总 13 目标残差。跨规模绝对误差和 R² 只描述尺度偏移，排序指标更适合迁移判断。", "",
             _table(observed_summary, ["split", "model", "n", "pooled_r2", "macro_nrmse_train_scale", "J_mae", "mean_target_pearson"]), "",
             "各目标 RMSE、MAE、R²、Pearson、Spearman、Kendall、Overlap@5/10/20 已完整保存于 `mixture/holdout_metrics.csv`。60M/1B 排序指标按 13 目标平均如下：", "",
             _table(metric_target[metric_target["split"].isin(["test_60m", "test_1b"])].groupby(["split", "model"], as_index=False)[["spearman", "kendall", "overlap_5", "overlap_10", "overlap_20"]].mean(), ["split", "model", "spearman", "kendall", "overlap_5", "overlap_10", "overlap_20"]), "",
             "10B/70B 仅是原附件估算表的排序一致性，不能称为真实大模型验证：", "",
             _table(estimate_summary, ["split", "model", "spearman", "kendall", "overlap_5", "overlap_10", "overlap_20"]), "",
             "## 5. 有限替代扰动与路径条件组合响应", "",
             "单域 δ={0.01,0.03,0.05} 与双域 a=b=0.03 均为原始配方上的份额增量；其他域按相对比例缩减，验证非负及行和为 1。仅支持区域中的响应可作为主要解释；支持区域用训练配方留一最近邻距离的 95% 分位数固定。", "",
             f"单域结果覆盖 {singles.domain.nunique()} 域 × {singles.target.nunique()} 目标 × {singles.delta.nunique()} 幅度；"
             f"组合结果覆盖 {doubles[['domain_j','domain_k']].drop_duplicates().shape[0]} 域对 × {doubles.target.nunique()} 目标。每项保存可行数、支持数及 AME/组合响应差异，见 CSV。组合响应受所指定再分配路径影响，不等于固有协同或因果效应。", "",
             "跨 17 个目标域的共同可行样本和共同支持样本数量如下；共同支持不足时不报告虚构 AME，详见 `mixture/common_subset_ame.csv`：", "",
             _table(common_ame.groupby(["model", "subset", "delta"], as_index=False).n.min(), ["model", "subset", "delta", "n"], 3), "",
             "图 `figures/pair_response_J.png` 将 136 对的 13 个目标响应按训练 Loss 标准差转换为等权 J 后展示；空白格表示缺乏训练配方支持。", "",
             "## 6. Qmapped 是否改善 1M 预测", "",
             "仅 A16 的 3 direct 与 3 near_direct 配方域参与 Qmapped，11 inferred 域保持未映射。Qmapped 是已映射配比的加权质量均值，分母为零时记缺失；它不能代表完整 17 域质量。共同可映射样本上 Ridge(p) 和 Ridge(p,Qmapped) 使用同一五折、同一候选 λ、每折内标准化。", "",
             _table(pd.DataFrame([{"Q": key, **value} for key,value in integration.items() if isinstance(value, dict)]), ["Q", "train_rows_mapped", "train_rows_unmapped", "holdout_rows_mapped", "cv_macro_nrmse_p", "cv_macro_nrmse_p_q", "relative_cv_gain", "target_wins"]), "",
             "1M 共同留出配方的两个模型表现：", "",
             _table(qhold, ["quality_method", "model", "n", "pooled_r2", "macro_nrmse_train_scale", "mean_target_pearson"]), "",
             f"主 Q 的 1M pooled R² 增量为 {main_pooled_gain:+.4f}，分组等权 Q 的增量为 {group_pooled_gain:+.4f}。{q_conclusion}", "",
             "## 7. 限制与复现", "",
             "来源清单只提供文件大小等信息；本次 SHA-256 固定了所用字节，但未独立证明附件来源真实性。`valid` 仅表示通过当前结构和数值规则。方向不确定的统计指标会改变 Q 排序；样本量不等同 token 权重；质量数据与配方数据不存在逐记录配对。熵权衡量分布信息而非训练因果贡献。配方扰动是模型预测下的局部替代路径；训练支持以外的树模型响应低可信。10B/70B 是估算参考而非独立实验。", "",
             "输入 SHA-256、配置、设计、全部 v2 Python 文件哈希及库版本记录在 `metadata.json`。源文件只读，隔离记录保留在 `audit/`。此报告没有读取 `results/q1/` 作为正式输入。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
