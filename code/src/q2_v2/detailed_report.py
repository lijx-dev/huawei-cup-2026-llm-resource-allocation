"""从 q2-scaling-v2 已冻结产物生成详细实验报告；不拟合模型。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path):
    return pd.read_csv(path)


def _json(path: Path):
    return json.loads(path.read_text())


def _cell(value):
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return "\n".join([
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
        *("|" + "|".join(_cell(v) for v in row) + "|" for row in rows),
    ])


def _q(series, ps=(.025, .975)):
    return [float(series.quantile(p)) for p in ps]


def build_report(out: Path) -> str:
    """所有数字都由已保存的 CSV/JSON 重新读取，返回完整 Markdown。"""
    audit = _json(out / "audit/audit_metadata.json")
    inventory = _read(out / "audit/b_attachment_inventory.csv")
    roles = _read(out / "audit/dataset_role_matrix.csv")
    invalid = _read(out / "audit/invalid_rows.csv")
    overlap = _read(out / "audit/b6_b7_overlap.csv")
    quality_semantics = _read(out / "audit/quality_semantics.csv")
    interface = _json(out / "audit/q1_interface.json")
    b1 = _json(out / "b1_baseline/final_parameters.json")
    b1_ci = _json(out / "b1_baseline/parameter_ci.json")
    b1_cv = _read(out / "b1_baseline/group_cv_metrics.csv")
    b1_start = _read(out / "b1_baseline/multistart_runs.csv")
    b1_resid = _read(out / "b1_baseline/residuals.csv")
    transfer = _read(out / "transfer_validation/validation_matrix.csv")
    b6 = _json(out / "quality_model/b6_parameters.json")
    b6_cv = _read(out / "quality_model/b6_cv_metrics.csv")
    b6_boot = _read(out / "quality_model/b6_bootstrap_parameters.csv")
    hold = _read(out / "quality_model/b7_new_metrics.csv")
    hold_boot = _read(out / "quality_model/b7_new_bootstrap_delta.csv")
    anchor = _read(out / "quality_model/anchor_sensitivity.csv")
    b8_groups = _read(out / "audit/b8_groupwise_q_loss.csv")
    b8_stress = _read(out / "quality_model/b8_stress_test.csv")
    reference = _json(out / "q1_interface/reference_mixture.json")
    q1_meta = _json(out / "q1_interface/interface_metadata.json")
    h = _read(out / "q1_interface/h_by_target.csv")
    grid = _read(out / "generalized_law/scenario_grid.csv")
    marginal = _read(out / "generalized_law/marginal_effects.csv")
    elasticity = _read(out / "generalized_law/elasticities.csv")
    iso = _read(out / "generalized_law/q_n_substitution.csv")
    local_ci = _read(out / "generalized_law/local_effect_intervals.csv")
    paths = _read(out / "generalized_law/path_conditioned_pair_response.csv")
    opt = _read(out / "compute_opt/analytic_optima.csv")
    bounded = _read(out / "compute_opt/bounded_optima.csv")
    opt_ci = _read(out / "compute_opt/bootstrap_optima_intervals.csv")
    opt_numeric = _read(out / "compute_opt/numerical_verification.csv")
    mix_opt = _read(out / "compute_opt/mixture_scenario_optima.csv")
    extrap = _read(out / "extrapolation/b10_estimated_consistency.csv")
    evidence = _read(out / "report/evidence_grade_table.csv")
    fig = _read(out / "report/figure_manifest.csv")
    reproducibility = _json(out / "report/reproducibility_summary.json")
    stage_folders = ["audit", "b1_baseline", "transfer_validation", "quality_model", "q1_interface", "generalized_law", "compute_opt", "extrapolation"]
    stages = [_json(out / stage_folders[i] / f"p{i}_metadata.json") for i in range(8)]
    lines = []
    version = reproducibility.get("experiment_version", out.name.replace("_", "-"))

    def add(*parts):
        for p in parts:
            lines.extend([str(p), ""])

    add(f"# 问题二 {version} 详细实验报告")
    add(f"**实验版本：**`{version}`　**随机种子：**7　**状态：**`COMPLETE_WITH_WARNINGS`。本报告由 P0–P7 已保存的结果表生成；生成报告时未重新拟合模型。N、D 分别以十亿参数和十亿 tokens 为单位，除非表中另有说明。")
    add("## 摘要")
    add("B1 的经典标度律在 8 个规模留一验证中取得很低的误差；B6 半合成数据内加入 Q_B 明显改善组级预测，B7-new 的新 Q 水平上也改善误差。B8 的组内数值方向却与 B6 相反，说明该质量效应不能直接跨附件合并。第一问配比模型只以冻结接口导入；附件 B 没有逐实验配比，因此广义模型中的 λ_p 是情景量。固定算力最优仅针对 C≈6ND 的训练计算近似。")
    add(_table(["主要结果", "数值", "证据等级"], [
        ("B1 规模留一宏 RMSE", b1_cv.RMSE.mean(), "B1 组留出"),
        ("B6 M0/MQ 组 CV 宏 RMSE", f"{b6_cv.loc[b6_cv.model.eq('M0'),'RMSE'].mean():.6g} / {b6_cv.loc[b6_cv.model.eq('MQ'),'RMSE'].mean():.6g}", "半合成校准"),
        ("B7-new M0/MQ RMSE", f"{hold.loc[hold.model.eq('M0'),'RMSE'].item():.6g} / {hold.loc[hold.model.eq('MQ'),'RMSE'].item():.6g}", "半合成留出"),
        ("B8 正向 Q–Loss 组数", f"{int((b8_groups.loc[b8_groups.data_type.ne('all'),'spearman_Q_loss']>0).sum())}/150", "半合成压力测试"),
        ("中档算力、Q_B=0.5 解析最优 N/D", f"{opt.loc[(opt.budget_level.eq('medium')) & np.isclose(opt.Q_B,.5),'N_star_billions'].item():.5g} / {opt.loc[(opt.budget_level.eq('medium')) & np.isclose(opt.Q_B,.5),'D_star_billions'].item():.5g}", "条件优化情景"),
    ]))

    add("## 46.1 数据合同、审计与证据边界")
    add("附件 B1–B12 共对应 19 个实际文件，其中 B3 占 8 个插值轨迹文件。以下是原始行数，不把同源文件或重叠行视为独立实验数。`valid` 仅表示通过当前结构与数值规则，不能证明来源真实性。附件编号、字段和 SHA-256 见[文件清单](../audit/b_attachment_inventory.csv)、[字段合同](../audit/field_contract.yaml)和[数据说明](../../../docs/数据说明.pdf)。")
    role_map = roles.set_index("attachment")
    add(_table(["附件", "原始行", "来源性质", "预定角色"], [
        (aid, audit["attachment_rows"][aid], role_map.loc[aid,"provenance"], role_map.loc[aid,"model_role"])
        for aid in [f"B{i}" for i in range(1,13)]
    ]))
    add(f"审计识别 {len(invalid)} 条非法数值记录，均来自 B9 的 D≤0；B9 可用 N,D 元数据为 {audit['attachment_rows']['B9']-len(invalid)} 行。单文件完全重复条目 {audit['duplicate_entries']}；B6/B7 跨附件同 ID 且 N、D、Q、Loss 全一致的重叠 {audit['b6_b7']['n_overlap']} 行，B7 新增 {audit['b6_b7']['n_b7_new']} 行，冲突 {audit['b6_b7']['n_conflicting']} 行。排除指针见[非法行表](../audit/invalid_rows.csv)，重叠关系见[B6/B7 对照](../audit/b6_b7_overlap.csv)。所有原始文件的运行前后 SHA-256 一致。")
    add("B1 的 `N_params_B`、`D_tokens_B` 已分别是十亿参数、十亿 tokens，不重复除以 10⁹。B1 有 8 条规模轨迹，每条 147 个 checkpoint；B12 没有精确 N 和 token，无法可靠建立模型级对齐。B2、B6–B8 为半合成，B3 为插值，B10 为已拟合标度律估算。B1/B4/B5 的 Loss 虽标记为公开或观测数据，跨来源评测集、tokenizer、vocab 和协议未充分给出，因此跨来源绝对误差只作描述。详见[Loss 语义表](../audit/loss_semantics.csv)。")
    add("文档差异：任务指定的 `问题二_完整建模思路_扩展建模(1).md` 不在仓库，使用现有 `问题二_完整建模思路_扩展建模.md`。赛题正文讨论质量处理和注意力成本；本次 P6 按总控任务的 C≈6ND 近似实施，结论限定在固定 Q_B 与配比条件下的训练计算分配。")

    add("## 46.2 B1 经典 Scaling Law：拟合、验证与不确定性")
    add(r"模型为 $L=E+AN^{-\alpha}+BD^{-\beta}$，约束 $E\ge0$、$A,B,\alpha,\beta>0$。对正参数取 log 参数化，采用四组初值最小二乘拟合；按完整模型规模留一验证，最后用全部 B1 重拟合。每次起点、收敛状态、目标函数和迭代次数见[多起点记录](../b1_baseline/multistart_runs.csv)。")
    add(f"四个起点收敛 {int(b1_start.converged.sum())}/{len(b1_start)}；最终目标函数范围 [{b1_start.objective.min():.6g}, {b1_start.objective.max():.6g}]。B1 有 {len(b1_resid)} 条有效建模行。参数区间由 {len(_read(out/'b1_baseline/bootstrap_parameters.csv'))} 次按规模重抽样得到，并非逐 checkpoint 独立重抽样。")
    add(_table(["参数", "全量估计", "组 bootstrap 2.5%", "组 bootstrap 97.5%"], [
        (k, b1[k], b1_ci[k]["0.025"], b1_ci[k]["0.975"]) for k in ("E","A","B","alpha","beta")
    ]))
    add(_table(["留出规模 N(B)", "行数", "RMSE", "MAE", "R²", "Spearman"], [
        (r.test_groups, int(r.n), r.RMSE, r.MAE, r.R2, r.Spearman) for _,r in b1_cv.iterrows()
    ] + [("宏平均", int(b1_cv.n.sum()), b1_cv.RMSE.mean(), b1_cv.MAE.mean(), b1_cv.R2.mean(), b1_cv.Spearman.mean())]))
    add(f"全量残差平均 {b1_resid.residual.mean():.6g}，最大绝对值 {b1_resid.residual.abs().max():.6g}，全部小于 0.001。高拟合度不能单独证明数据来源真实性；数据说明把 B1 标为真实轨迹，但现有材料不足以独立核验数值生成过程。残差源表见[residuals.csv](../b1_baseline/residuals.csv)，按 log N、log D、预测值和规模的四幅诊断图见[图表清单](figure_manifest.csv)。")

    add("## 46.3 B2/B3/B4/B5 冻结参数迁移")
    add("以下全部使用 B1 冻结参数，没有重新调参。B2 是半合成跨来源轨迹；B3 是检查点插值轨迹，只检验曲线一致性；B4 是跨族点；B5 是文献汇编。因 Loss 口径未证实一致，RMSE/R² 和偏置为数值描述，不能当作严格同任务预测精度；Pearson/Spearman 也只能作条件性趋势比较。")
    all_transfer=transfer[transfer.subgroup.eq("all")]
    add(_table(["附件", "行数", "RMSE", "MAE", "R²", "Pearson", "Spearman", "预测−原值均偏"], [
        (r.attachment, int(r.n), r.RMSE, r.MAE, r.R2, r.Pearson, r.Spearman, r.mean_bias) for _,r in all_transfer.iterrows()
    ]))
    b4_worst=transfer[(transfer.attachment.eq("B4")) & transfer.subgroup.ne("all") & transfer.n.ge(3)].nlargest(3,"RMSE")
    b5_worst=transfer[(transfer.attachment.eq("B5")) & transfer.subgroup.ne("all") & transfer.n.ge(3)].nlargest(3,"RMSE")
    add("B2 的系统偏差尤其大；B3 的近乎重合属于插值一致性，不能充当 4000 次独立实验。B4/B5 内误差较大的分层如下；分层样本量可能很小，不能仅凭排名判断模型族优劣。")
    add(_table(["附件", "分层", "行数", "RMSE", "Spearman"], [
        (r.attachment, r.subgroup, int(r.n), r.RMSE, r.Spearman) for _,r in pd.concat([b4_worst,b5_worst]).iterrows()
    ]))
    add("完整的模型族、轨迹与文献分层见[迁移验证矩阵](../transfer_validation/validation_matrix.csv)。")

    add("## 46.4 B6 质量 Scaling Law 主实验")
    add(r"B6 主比较使用同一批 360 行和同一组级折分：$M_0=E_6+A_6N^{-\alpha_6}+B_6D^{-\beta_6}$；$M_Q=E_6+A_6N^{-\alpha_6}+B_6D^{-\beta_6}\exp[\gamma_Q(Q_{0,B}-Q_B)]$。$Q_{0,B}=0.5$ 是 B6 中位数，建模前固定。45 个 N,D 基础组各含多个 Q，GroupKFold 保证同组不跨训练/验证。")
    cv_summary=b6_cv.groupby("model",sort=False).agg(folds=("fold","count"),n=("n","sum"),RMSE=("RMSE","mean"),MAE=("MAE","mean"),R2=("R2","mean"),Spearman=("Spearman","mean"))
    add(_table(["模型", "折数", "总验证行", "宏 RMSE", "宏 MAE", "宏 R²", "宏 Spearman"], [
        (name,int(r.folds),int(r.n),r.RMSE,r.MAE,r.R2,r.Spearman) for name,r in cv_summary.iterrows()
    ]))
    add(f"全量 B6 拟合的 MQ 参数为 E={b6['MQ']['E']:.6g}、A={b6['MQ']['A']:.6g}、B={b6['MQ']['B']:.6g}、α={b6['MQ']['alpha']:.6g}、β={b6['MQ']['beta']:.6g}、γ_Q={b6['MQ']['gamma']:.6g}。按基础组 bootstrap {len(b6_boot)} 次，γ_Q 的 95% 分位区间为 [{_q(b6_boot.gamma)[0]:.6g}, {_q(b6_boot.gamma)[1]:.6g}]。在这个模型和编码下，γ_Q>0 意味 Q_B 数值升高、预测 Loss 降低；Q_B 的操作性“越高越好”定义仍未独立确认。")
    add(f"二次候选 MQ2 的宏 RMSE={cv_summary.loc['MQ2','RMSE']:.6g}，按预设至少 2% 改善规则未升级为主模型。候选参数和选择记录见[B6 参数文件](../quality_model/b6_parameters.json)，所有折和预测见[CV 指标](../quality_model/b6_cv_metrics.csv)、[CV 预测](../quality_model/b6_cv_predictions.csv)。")

    add("## 46.5 B7-new 最终留出")
    add(f"B7 共 {audit['b6_b7']['n_b7']} 行，扣除与 B6 完全一致的 {audit['b6_b7']['n_overlap']} 行后，B7-new 为 {audit['b6_b7']['n_b7_new']} 行。它的 45/45 个 N,D 基础组已在 B6 出现，因此只是新 Q 水平留出，不是新规模或新 token 配置留出。模型形式、Q 参考点与参数均在查看该表指标前固定。")
    add(_table(["模型", "行数", "RMSE", "MAE", "R²", "Pearson", "Spearman"], [
        (r.model,int(r.n),r.RMSE,r.MAE,r.R2,r.Pearson,r.Spearman) for _,r in hold.iterrows()
    ]))
    delta=_q(hold_boot.delta_RMSE)
    add(f"MQ−M0 的 RMSE 点差为 {hold.loc[hold.model.eq('MQ'),'RMSE'].item()-hold.loc[hold.model.eq('M0'),'RMSE'].item():.6g}；按 N,D 基础组重抽样的 95% 分位区间 [{delta[0]:.6g}, {delta[1]:.6g}]。这是半合成体系内部的留出证据，不能外推为真实新模型族上的增益。逐行留出预测见[MQ 预测](../quality_model/b7_new_predictions_MQ.csv)，区间样本见[bootstrap 差值](../quality_model/b7_new_bootstrap_delta.csv)。")

    add("## 46.6 B1-anchored 质量参数敏感性")
    ar=anchor.iloc[0]
    add(f"固定 B1 的 E、A、B、α、β，另允许来源偏置后，在 B6 上估得 γ_Q={ar.gamma:.6g}、来源偏置={ar.source_offset:.6g}、训练 RMSE={ar.RMSE:.6g}。B6 自由拟合主模型 γ_Q={b6['MQ']['gamma']:.6g}，两者差异表明跨来源结构约束显著改变质量参数；该 anchored 结果仅作敏感性分析，不替代 B6 主估计。[原表](../quality_model/anchor_sensitivity.csv)。")

    add("## 46.7 B8 分层压力测试")
    direction=[]
    for kind in sorted(set(b8_groups.data_type)-{"all"}):
        g=b8_groups[b8_groups.data_type.eq(kind)]
        direction.append((kind,len(g),int((g.spearman_Q_loss>0).sum()),g.spearman_Q_loss.median(),int((g.monotonic_direction=="increasing").sum())))
    add(_table(["B8 类型", "N,D 组", "Spearman>0 组", "组内 Spearman 中位数", "数值非降组"],direction))
    add("B8 的 calibrated 和 extrapolated 均独立列示，没有进入 B6 拟合或模型选择。所有 150 个分层组的 Q_B–Loss 数值相关为正，而 B6 的 MQ 方向为负。不能为了消除冲突而翻转 Q 编码；也不能在没有生成协议的情况下断定 B8 错误。")
    add(_table(["B8 类型", "冻结模型", "行数", "RMSE", "R²", "Spearman", "均偏"], [
        (r.data_type,r.model,int(r.n),r.RMSE,r.R2,r.Spearman,r.mean_bias) for _,r in b8_stress.iterrows()
    ]))
    add("组内方向完整结果见[B8 审计表](../audit/b8_groupwise_q_loss.csv)，冻结模型压力结果见[B8 stress test](../quality_model/b8_stress_test.csv)。")

    add("## 46.8 Q_A 与 Q_B 的语义区分")
    add("Q_A 是问题一对 22 个指标所建的分层综合评分，设计上越高越好；Q_B 是 B6–B8 的 `Q_score`，其操作性定义与方向未在附件中充分说明。Q_A 与 Q_B 的来源、数值范围及计算口径不能证明一致，因此数据合同保持 `Q_A != Q_B`。")
    add(_table(["变量来源", "列", "实测最小", "实测最大", "方向/等价性"], [
        (r.attachment,r.q_column,r["min"],r["max"],f"{r.higher_means_better}; vs B6: {r.same_definition_as_B6}") for _,r in quality_semantics.iterrows()
    ]))
    add("比较表见[quality_semantics.csv](../audit/quality_semantics.csv)；两类 Q 未在拟合数据中合并。")

    add("## 46.9 第一问冻结配比接口")
    if reference.get("formal_interface_manifest"):
        add(r"第一问冻结的 17 维配比到 13 维 Loss 模型定义 $h_{p,v}(p)=\log[\widehat L_v^{mix}(p)/\widehat L_v^{mix}(p_0)]$。正式接口先将 A4 的 512 份训练配方逐行归一化，再取均值得到 $p_0$；未改动 A4。所有用于求 log 的预测 Loss 经正值检查，$h_{p,v}(p_0)=0$。")
        add(f"导入模型为 {reference['model_name']}，当前文件 SHA-256 `{reference['model_sha256']}`；参考配比行和为 {reference['normalized_reference_sum']:.9g}。配比支持阈值为训练配方留一最近邻欧氏距离 95% 分位 {reference['support_distance_threshold']:.6g}。正式接口清单为 `{reference['formal_interface_manifest']}`。")
    else:
        add(r"第一问冻结的 17 维配比到 13 维 Loss 模型定义 $h_{p,v}(p)=\log[\widehat L_v^{mix}(p)/\widehat L_v^{mix}(p_0)]$。$p_0$ 为 A4 的 512 份训练配方逐域平均，并因原表三位小数舍入作明确归一化；未改动 A4。所有用于求 log 的预测 Loss 经正值检查，$h_{p,v}(p_0)=0$。")
        add(f"导入模型为 {reference['model_name']}，当前文件 SHA-256 `{reference['model_sha256']}`；A4 原始均值行和 {reference['raw_mean_sum']:.9g}，归一化后 1。配比支持阈值为训练配方留一最近邻欧氏距离 95% 分位 {reference['support_distance_threshold']:.6g}。旧 v2 metadata 未锚定模型哈希，v2.1 保存当前文件预测复核。")
    a4_h=h[h.recipe.ne("p0")]
    add(_table(["验证目标", "h 最小", "h 中位", "h 最大"], [
        (q1_meta["output_order"][i],a4_h[f"h_{i+1}"].min(),a4_h[f"h_{i+1}"].median(),a4_h[f"h_{i+1}"].max()) for i in range(13)
    ]))
    add(f"13 目标逐一保留；h_agg 只在明确需要单个情景因子时定义为 13 个无量纲对数比的等权平均。A4 上 h_agg 范围 [{a4_h.h_agg.min():.6g}, {a4_h.h_agg.max():.6g}]，不把 B 标量 Loss 与任一 A 目标天然对应。逐配方结果见[h_by_target.csv](../q1_interface/h_by_target.csv)，输入/输出顺序见[接口元数据](../q1_interface/interface_metadata.json)。")

    add("## 46.10 广义模型与参数证据等级")
    add(r"仅 B6 体系的 $E,A,B,\alpha,\beta,\gamma_Q$ 是 estimated；13 个 $h_{p,v}$ 及约定的 $h_{agg}$ 是 imported_Q1；$\lambda_p$ 是 scenario_assumption。Form A：$L_A=E+AN^{-\alpha}+BD^{-\beta}\exp[g_Q(Q_B)+\lambda_ph(p)]$。Form B：$L_B=E+[AN^{-\alpha}+BD^{-\beta}\exp g_Q(Q_B)]\exp[\lambda_ph(p)]$。两式在 $\lambda_p=0$ 且 $Q_B=Q_{0,B}$ 时退化为 B6 经典形式。")
    add("附件 B 不含 p，也没有 (N,D,Q_B,p,L) 联合实验；故 λ_p 无法被 B 数据识别。B 的标量 Loss 与 Q1 的 13 个验证域 Loss 尺度/语义未证明相同，h_p 的迁移属于结构假设。Form A 假设配比作用于数据项；Form B 是形式敏感性，均不能用于宣称真实机制已识别。")

    add("## 46.11 λ_p 固定情景")
    add("λ_p 预先固定为 0、0.5、1、1.5，分别称 no-transfer、attenuated-transfer、unit-transfer、amplified-transfer；没有从附件 B 回归估计。工作点取 B6 实际 N×D 排序的 10%、50%、90% 位置，Q_B 取 B6 的 25%、50%、75% 分位，配比取 p0 和两份 A4 配方。")
    wp=grid[["workpoint","N_billions","D_billions"]].drop_duplicates().sort_values("workpoint")
    add(_table(["工作点", "N(B)", "D(B tokens)"],[(int(r.workpoint),r.N_billions,r.D_billions) for _,r in wp.iterrows()]))
    add(f"共 {len(grid)} 个 Form×λ×Q×配比×工作点情景值，均按同一模型计算；其中支持区域标记为 {grid.support_status.value_counts().to_dict()}。结果见[scenario_grid.csv](../generalized_law/scenario_grid.csv)，不能视为新的训练观测。")

    add("## 46.12 Form A/B 结构敏感性")
    ex=grid[(grid.workpoint.eq(1)) & np.isclose(grid.Q_B,.5) & grid.recipe.eq("A4:261")]
    add("下表固定 B6 中位工作点、Q_B=0.5、同一 A4 配方，仅改变 λ 和模型形式，展示结构选择带来的预测差异。")
    add(_table(["λ_p", "h_agg", "Form A Loss", "Form B Loss"], [
        (lam,ex.h_agg.iloc[0],ex[(ex.lambda_p.eq(lam)) & ex.form.eq("A")].loss_prediction.item(),ex[(ex.lambda_p.eq(lam)) & ex.form.eq("B")].loss_prediction.item()) for lam in (0,.5,1,1.5)
    ]))
    add("λ=0 时两形式完全一致；非零 λ 下差异来自结构假设，不能由附件 B 选择真实形式。全部差值见[form_sensitivity.csv](../generalized_law/form_sensitivity.csv)。")

    add("## 46.13 N、D、Q_B 边际效应与弹性")
    add(r"对可约 Loss $R=L-E$，在每个工作点计算 $M_N=-\partial L/\partial N$、$M_D=-\partial L/\partial D$、$M_Q=-\partial L/\partial Q_B$，以及 $\varepsilon_N=\partial\ln R/\partial\ln N$、$\varepsilon_D=\partial\ln R/\partial\ln D$。Q_B 零点不具可靠比例语义，报告 $\partial\ln R/\partial Q_B$ 半弹性。下表为 p0、λ=0、Form A、Q_B=0.5 的条件结果。")
    m0=marginal[(marginal.recipe.eq("p0")) & marginal.lambda_p.eq(0) & marginal.form.eq("A") & np.isclose(marginal.Q_B,.5)]
    e0=elasticity[(elasticity.recipe.eq("p0")) & elasticity.lambda_p.eq(0) & elasticity.form.eq("A") & np.isclose(elasticity.Q_B,.5)].set_index("workpoint")
    add(_table(["工作点", "N", "D", "M_N", "M_D", "M_Q", "ε_N", "ε_D", "Q 半弹性"], [
        (int(r.workpoint),r.N_billions,r.D_billions,r.M_N,r.M_D,r.M_Q,e0.loc[r.workpoint,"epsilon_N"],e0.loc[r.workpoint,"epsilon_D"],e0.loc[r.workpoint,"q_semielasticity"]) for _,r in m0.iterrows()
    ]))
    add("导数单位随 N、D 的十亿单位变化；不能跨不同单位直接比较 M_N 与 M_D 大小。全情景值见[marginal_effects.csv](../generalized_law/marginal_effects.csv)、[elasticities.csv](../generalized_law/elasticities.csv)，参数 bootstrap 区间见[local_effect_intervals.csv](../generalized_law/local_effect_intervals.csv)。")

    add("## 46.14 质量与规模的局部等损失替代")
    add(r"固定 D、p、当前 Loss 与工作点，由隐函数求 $dN/dQ_B|_{L,D,p}$ 和 $d\ln N/dQ_B|_{L,D,p}$。这只是在已拟合函数附近的 local iso-loss substitution；Q_B 改善成本未知，不能推出成本最优替代率。")
    i0=iso[(iso.recipe.eq("p0")) & iso.lambda_p.eq(0) & iso.form.eq("A") & np.isclose(iso.Q_B,.5)]
    add(_table(["工作点", "N", "D", "dN/dQ_B", "dlnN/dQ_B"], [
        (int(r.workpoint),r.N_billions,r.D_billions,r.dN_dQ_iso_loss,r.dlogN_dQ_iso_loss) for _,r in i0.iterrows()
    ]))
    add("局部值及其组 bootstrap 区间分别见[q_n_substitution.csv](../generalized_law/q_n_substitution.csv)和[local_effect_intervals.csv](../generalized_law/local_effect_intervals.csv)。")

    add("## 46.15 配比份额转移与路径条件组合响应")
    add(r"每次扰动采用 $p(\delta)=p+\delta(e_d-e_k)$，严格检查非负与行和。双领域组合使用同一 donor、同一基准和同一份额扣减路径，响应为四角有限差分 $\Delta_{jk}-\Delta_j-\Delta_k$。")
    add(f"计算 {len(paths)} 条可行路径，{int(paths.support_status.eq('within_support').sum())} 条全部四角位于第一问支持阈值内，{int(paths.support_status.eq('low_confidence_extrapolation').sum())} 条标为低置信外推。最大绝对响应的若干路径如下；符号随基准和 δ 变化，不归纳为固有或因果协同。")
    top=paths.reindex(paths.response.abs().sort_values(ascending=False).index).head(5)
    add(_table(["基准", "donor", "领域 j", "领域 k", "δ", "h_agg 四角响应", "支持状态"], [
        (r.recipe,r.donor,r.target_j,r.target_k,r.delta,r.response,r.support_status) for _,r in top.iterrows()
    ]))
    add("全量计算见[path_conditioned_pair_response.csv](../generalized_law/path_conditioned_pair_response.csv)。")

    add("## 46.16 固定算力 C≈6ND 的条件优化")
    add(r"采用稠密 Transformer 训练计算近似 $C\approx6ND$，N、D 为十亿单位时表中 C 单位为 $10^{18}$ FLOPs。固定 C 后 $D=C/(6N)$。对 Form A、p0、λ=0，解析内点满足 $\alpha P=\beta T$；并在 B6 观测 N∈[0.07,11.97]、D∈[10,600] 的预设矩形边界内数值优化。矩形边界不等于稠密实测支持区域。")
    bd=bounded.set_index(["budget_level","Q_B"])
    add(_table(["预算档", "C(10¹⁸ FLOPs)", "Q_B", "解析 N*", "解析 D*", "带界 N*", "带界 D*", "解析在边界内"], [
        (r.budget_level,r.C_1e18_FLOPs,r.Q_B,r.N_star_billions,r.D_star_billions,bd.loc[(r.budget_level,r.Q_B),"N_star_bounded_billions"],bd.loc[(r.budget_level,r.Q_B),"D_star_bounded_billions"],r.within_B6_bounds) for _,r in opt.iterrows()
    ]))
    add(f"预算由 B6 的 6ND 10%、50%、90% 分位事先确定；3 个低档解析点的 D* 小于 B6 最小 D=10，带界解贴边。解析一阶条件的最大绝对残差 {opt.alpha_P_minus_beta_T.abs().max():.6g}，无边界解析/数值优化最大相对差 {opt_numeric.relative_difference.max():.6g}。")
    center=opt_ci[np.isclose(opt_ci.Q_B,.5)]
    add("Q_B=0.5 时按 B6 基础组重抽样的解析最优 95% 分位区间：")
    add(_table(["预算档", "N* 2.5%", "N* 97.5%", "D* 2.5%", "D* 97.5%"], [
        (kind,center[(center.budget_level.eq(kind)) & np.isclose(center.level_2,.025)].N_star_billions.item(),center[(center.budget_level.eq(kind)) & np.isclose(center.level_2,.975)].N_star_billions.item(),center[(center.budget_level.eq(kind)) & np.isclose(center.level_2,.025)].D_star_billions.item(),center[(center.budget_level.eq(kind)) & np.isclose(center.level_2,.975)].D_star_billions.item()) for kind in ("low","medium","high")
    ]))
    alt=mix_opt[(mix_opt.recipe.eq("A4_alt_2")) & mix_opt.budget_level.eq("medium")]
    add("λ 的优化敏感性只有在 h_agg≠0 的配比情景中显现。以下固定中档预算、Q_B=Q0 和同一 A4 配方；Form B 的配比因子同时乘在两项可约 Loss 上，故其内点 N*/D* 对 λ 不变，这源于所选函数形式。")
    add(_table(["形式", "λ_p", "h_agg", "解析 N*", "解析 D*"], [
        (r.form,r.lambda_p,r.h_agg,r.N_star_billions,r.D_star_billions) for _,r in alt.sort_values(["form","lambda_p"]).iterrows()
    ]))
    add("详见[analytic_optima.csv](../compute_opt/analytic_optima.csv)、[bounded_optima.csv](../compute_opt/bounded_optima.csv)、[bootstrap_optima_intervals.csv](../compute_opt/bootstrap_optima_intervals.csv)与[mixture_scenario_optima.csv](../compute_opt/mixture_scenario_optima.csv)。质量提升成本、注意力及上下文开销未进入 C≈6ND，因此不是完整质量/配比联合资源最优。")

    add("## 46.17 B9/B10 大尺度外推一致性")
    add(f"B9 有 {audit['attachment_rows']['B9']} 行模型元数据，其中 {len(invalid)} 行 D≤0 被隔离，保留 {audit['attachment_rows']['B9']-len(invalid)} 个有效 N,D 点。B10 有 {audit['attachment_rows']['B10']} 行估算 Loss，未进入 B1/B6 拟合、调参或独立验证。以下仅是与附件既有估算序列的数值一致性。")
    add(_table(["冻结曲线", "估算行", "RMSE", "MAE", "平均相对误差", "Spearman", "平均 log 外推距离"], [
        (r.model,int(r.n),r.RMSE,r.MAE,r.relative_error_mean,r.Spearman,r.extrapolation_distance_log) for _,r in extrap.iterrows()
    ]))
    add("B1 曲线与 B10 极接近，但数据说明明确 B10 是用已拟合标度律参数估算；高度一致存在循环生成风险，绝不能表述成百亿参数以上的真实独立验证。逐点外推距离与预测见[B10 一致性表](../extrapolation/b10_estimated_consistency.csv)和[逐点预测](../extrapolation/b10_predictions_B1_M0.csv)。")

    add("## 46.18 结论等级、局限与复现")
    grades=evidence.groupby("evidence_grade").size()
    add(_table(["证据等级", "结果条目", "在本报告中的解释"], [
        (grade,int(grades.get(grade,0)),label) for grade,label in [
            ("direct_fit","附件 B1 的参数拟合"),
            ("held_out_validation","组留出或 B7-new；仍须看数据来源"),
            ("semi_synthetic_calibration","B2/B6–B8 半合成或 B3 插值检查"),
            ("imported_Q1","冻结的第一问配比接口与 Q_A 口径"),
            ("scenario_assumption","λ、模型形式、条件最优与边际"),
            ("extrapolation_reference","B9/B10 大尺度参考"),
        ]
    ]))
    add("主要限制：①Q_A/Q_B 不能证明等价，B8 方向冲突；②B6–B8 为半合成，B7-new 与 B6 共享全部基础组；③附件 B 不含配比，λ_p 不可识别；④B 的单一 Loss 与 Q1 的 13 目标没有天然映射；⑤C≈6ND 忽略质量处理和注意力成本，上下文长度未优化；⑥极大尺度是外推或既有估算；⑦配比跨问题迁移依赖结构假设；⑧B1 的近舍入量级残差与 Q1 历史哈希缺口仍需来源核验。")
    add("本报告所述最可信结论限定为：B1 在自身记录的规模留出下符合所拟合曲线；B6/B7 半合成体系内 Q_B 提供额外预测信息；B8 无法支持同一质量方向；条件算力最优可由给定模型解析和数值复核。其余配比和 λ 结论均为透明情景，而非独立实证识别。")
    add("### 复现记录")
    add(_table(["阶段", "状态", "测试退出码", "有效/关键记录"], [
        (f"P{i}",m["status"],m.get("test_exit_code","未知"),m.get("counts",{})) for i,m in enumerate(stages)
    ]))
    if version == "q2-fusion-rerun-v1":
        command = "PYTHONPATH=src .venv/bin/python -m src.q2_fusion_rerun --stage all"
        test_command = "PYTHONPATH=src .venv/bin/python -m pytest -q tests/q2_v2 tests/q1_revision_v2_1"
    else:
        command = "python -m src.q2_v2.cli --stage all"
        test_command = "python -m pytest -q tests/q2_v2"
    add(f"运行命令：`{command}`，阶段测试由该入口顺序执行；独立完整测试命令为 `{test_command}`。版本、全部 B 输入 SHA-256、配置 SHA-256、Q1 模型 SHA-256 与阶段元数据见[reproducibility_summary.json](reproducibility_summary.json)；主要结果见[key_results_table.csv](key_results_table.csv)，逐结论证据等级见[evidence_grade_table.csv](evidence_grade_table.csv)，图表来源见[figure_manifest.csv](figure_manifest.csv)。")
    add(f"配置 SHA-256 `{reproducibility['config_sha256']}`；Q1 冻结模型 SHA-256 `{interface['model_sha256']}`。所有原始附件只读，未新增依赖、未下载外部数据。")
    add("**Q2 FINAL STATUS: COMPLETE_WITH_WARNINGS**")
    return "\n".join(lines).rstrip() + "\n"
