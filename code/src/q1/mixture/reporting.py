"""P3/M3 附加核验、结果矩阵和中文实验报告。"""
import json
from pathlib import Path
from itertools import combinations
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from q1.mixture.dataset import fields, load_pair
from q1.mixture.models import predict
from q1.mixture.perturbation import single, nearest_distance
from q1.mixture.figures import generate


def markdown_table(frame, columns, precision=4):
    head = "|" + "|".join(columns) + "|"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in frame[columns].itertuples(index=False, name=None):
        body.append("|" + "|".join((f"{v:.{precision}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)) for v in row) + "|")
    return "\n".join([head, sep, *body])


def finalize(root):
    root = Path(root)
    mix, integ = root / "results/q1/mixture", root / "results/q1/integration"
    cfg = json.loads((root / "configs/q1/mixture.json").read_text())
    metadata = json.loads((mix / "mixture_model_metadata.json").read_text())
    stored = joblib.load(mix / "frozen_models.joblib")
    mixfields, targets = fields(root)
    domains = [s.removeprefix("train_the_pile_") for s in mixfields]
    ids, x, y, _ = load_pair(root, "train_1m", mixfields, targets, cfg["sum_tolerance"])
    ref = stored["reference_index"]
    # 冻结模型重新加载后的预测与保存的独立检验预测逐目标逐行核对。
    tid, tx, ty, _ = load_pair(root, "test_1m", mixfields, targets, cfg["sum_tolerance"])
    splits_for_overlap = {name: load_pair(root, name, mixfields, targets, cfg["sum_tolerance"])
                          for name in ("train_1m", "test_1m", "test_60m", "test_1b", "est_10b", "est_70b")}
    overlap_rows = []
    for left, right in combinations(splits_for_overlap, 2):
        lid, lx, _, _ = splits_for_overlap[left]
        rid, rx, _, _ = splits_for_overlap[right]
        overlap_rows.append({"left_split": left, "right_split": right,
                             "shared_index_labels": len(set(lid) & set(rid)),
                             "identical_mixture_vectors": len(set(map(tuple, lx)) & set(map(tuple, rx)))})
    pd.DataFrame(overlap_rows).to_csv(mix / "cross_split_overlap.csv", index=False)
    saved = pd.read_csv(mix / "predictions_test_1m_lightgbm.csv", dtype={"index": str})
    reloaded = predict(stored["models"], "lightgbm", tx, ref)
    if saved["index"].tolist() != tid.tolist() or not np.allclose(saved[[f"predicted:{t}" for t in targets]].to_numpy(), reloaded, atol=1e-9):
        raise RuntimeError("冻结模型重新加载后预测不一致")
    coeff = []
    for t, model in zip(targets, stored["models"]["ridge"]):
        row = {"target": t, "intercept": model.intercept_, "alpha": model.alpha, "reference_domain": domains[ref]}
        row.update({d: c for d, c in zip([d for i, d in enumerate(domains) if i != ref], model.coef_)})
        coeff.append(row)
    pd.DataFrame(coeff).to_csv(mix / "ridge_coefficients.csv", index=False)
    # Ridge 与主模型有限扰动作为模型形式敏感性，使用相同配方和支持规则。
    max_share = x.max(axis=0)
    threshold = metadata["support_threshold"]
    ridge_base = predict(stored["models"], "ridge", x, ref).mean(axis=1)
    robust = []
    for j, domain in enumerate(domains):
        modified, feasible = single(x, j, cfg["interaction_delta"])
        use = feasible.copy()
        if use.any():
            use[feasible] = (nearest_distance(modified[feasible], x) <= threshold) & np.all(modified[feasible] <= max_share + 1e-12, axis=1)
        if use.any():
            change = predict(stored["models"], "ridge", modified[use], ref).mean(axis=1) - ridge_base[use]
            robust.append({"domain": domain, "delta": cfg["interaction_delta"], "ridge_mean_predicted_delta": float(change.mean()), "supported_n": int(use.sum())})
        else:
            robust.append({"domain": domain, "delta": cfg["interaction_delta"], "ridge_mean_predicted_delta": np.nan, "supported_n": 0})
    robust = pd.DataFrame(robust)
    lgb = pd.read_csv(mix / "marginal_effects.csv")
    lgb = lgb[(lgb.target == "average_loss") & np.isclose(lgb.delta, cfg["interaction_delta"])]
    robust = robust.merge(lgb[["domain", "mean_delta_loss_supported"]], on="domain", validate="one_to_one")
    robust.rename(columns={"mean_delta_loss_supported": "lightgbm_mean_predicted_delta"}, inplace=True)
    robust.to_csv(mix / "model_form_sensitivity.csv", index=False)
    ii = pd.read_csv(mix / "interaction_effects.csv").query('target == "average_loss"')
    grid = pd.DataFrame(np.nan, index=domains, columns=domains)
    for row in ii.itertuples():
        grid.loc[row.domain_j, row.domain_k] = row.mean_interaction_supported
        grid.loc[row.domain_k, row.domain_j] = row.mean_interaction_supported
    grid.to_csv(mix / "interaction_matrix_average_loss.csv")
    manifest = generate(root)
    scores = []
    for split in ("test_1m", "test_60m", "test_1b", "est_10b", "est_70b"):
        frame = pd.read_csv(mix / (f"estimated_{split.removeprefix('est_')}_comparison.csv" if split.startswith("est") else f"{split}_metrics.csv"))
        scores.extend(frame[frame.target == "average_loss"][["split", "data_role", "model", "n", "rmse", "r2", "spearman", "kendall", "overlap_10"]].to_dict("records"))
    scores = pd.DataFrame(scores)
    coverage = pd.read_csv(integ / "mapping_coverage.csv")
    ass = pd.read_csv(integ / "quality_loss_association.csv")
    a17 = pd.read_csv(integ / "a17_audit_report.csv")
    support = pd.read_csv(mix / "marginal_effects.csv").query('target == "average_loss"')
    interactions = pd.read_csv(mix / "interaction_effects.csv").query('target == "average_loss"')
    method_corr = robust[["ridge_mean_predicted_delta", "lightgbm_mean_predicted_delta"]].dropna()
    rank_corr = float(spearmanr(method_corr.iloc[:, 0], method_corr.iloc[:, 1]).statistic) if len(method_corr) >= 3 else float("nan")
    inventory = pd.read_csv(mix / "mixture_integrity_report.csv")
    map_main = coverage[coverage.split == "train_1m"]
    q_ass = ass[(ass.split == "train_1m") & (ass.scenario == "a17_assisted_main")]
    text = f"""# P3/M3：领域配比、质量映射与 Loss 预测

## 1. 数据、来源与切分

原始附件 A4–A17 保持只读。配比表与 Loss 表严格按 `index` 一对一连接；17 个配比输入、13 个验证 Loss 均完整且为有限非负值。原始配比行和在 P0 已核验的 `{cfg['sum_tolerance']}` 容限内；建模输入逐行除以实际行和，审计表记录了规范化行数与最大误差。每一对原始文件的 SHA-256、角色及有效行数见 `mixture_integrity_report.csv` 和 `mixture_model_metadata.json`。

{markdown_table(inventory, ['split', 'role', 'rows', 'invalid', 'normalized_rows', 'max_sum_error'])}

A4/A5 为唯一训练与调参数据。A6/A7、A8/A9、A10/A11 是留出检验；A12–A15 的 63 组配方为训练集子集，Loss 是已有外推估算，不是独立真实实验。不同切分可以复用 `index` 标签，故不能仅靠它判断跨切分重叠；`cross_split_overlap.csv` 另列完全相同配比向量的数量。训练配比与三个留出切分均无完全相同向量；A6 与 A8 配方文件 SHA-256 相同，跨尺度比较的是相同 256 个配方在不同规模的 Loss，不是两组独立配方。

## 2. 模型选择和冻结

按训练集内 3 折固定随机种子交叉验证。Ridge 删除 `{domains[ref]}` 参考配比列并保留截距；LightGBM 对 13 个目标逐一拟合，完整保留 17 个配比。各目标的候选参数与 CV RMSE 见 `ridge_cv_results.csv`、`lightgbm_cv_results.csv`；Ridge 系数见 `ridge_coefficients.csv`。均值基线、Ridge、LightGBM 均用相同输出口径。正式扰动使用冻结的 LightGBM；其模型文件已重新加载并逐行核对 1M 预测一致。

## 3. 1M 独立检验

综合 Loss 是 13 个目标的等权平均，只有完整 13 维记录才计算。逐目标完整指标见 `test_1m_metrics.csv`；下表为综合结果，Loss 越低越好。

{markdown_table(scores[scores.split == 'test_1m'], ['model', 'n', 'rmse', 'r2', 'spearman', 'overlap_10'])}

LightGBM 在 1M 留出集上的综合 RMSE 与排序均优于 Ridge 和均值基线。这是当前表格数据的预测性能，不能据此认定领域配比的因果作用。

## 4. 单领域有限扰动与支持区域

对每个训练配方、17 域和 `0.01/0.03/0.05` 增量，目标域增加，其余 16 域按原比例扣减。基线与扰动均在单纯形上。支持区域采用训练配方最近非自身邻居距离的 {cfg['support_quantile']:.0%} 分位数 `{threshold:.6f}`，且每个配比不得超过训练集中该领域观察到的最大值；此第二约束是在首轮诊断发现纯距离规则过宽后加入，详见配置与运行记录。`marginal_effects.csv` 给出每个验证域和综合 Loss 的支持样本均值、可行样本均值和覆盖率。支持外预测仅供探索；即使支持内也仍是模型预测。

各幅度综合 Loss 支持覆盖率：

{markdown_table(support.groupby('delta', as_index=False).agg(min_support_fraction=('support_fraction', 'min'), mean_support_fraction=('support_fraction', 'mean')), ['delta', 'min_support_fraction', 'mean_support_fraction'])}

`model_form_sensitivity.csv` 比较相同增量与支持规则下 Ridge 和 LightGBM 的响应，17 域响应排序 Spearman 为 {rank_corr:.4f}；差异反映模型形式不确定性，不是已经重复训练验证的因果效应。

## 5. 双领域交互

全部 136 个不同域对均按同一 15 个供给域定义三个扰动：仅增 j、仅增 k、同时增加。`I=Δjk−Δj−Δk`；线性模型微型测试验证交互接近零。`interaction_effects.csv` 保存 13 个目标及综合 Loss、可行/支持计数，`interaction_matrix_average_loss.csv` 为对称 17×17 综合矩阵。对角线为不适用。{int((interactions.supported_n > 0).sum())}/136 个域对至少有一个支持样本；支持比例最低 {interactions.support_fraction.min():.3f}。响应面仅保留有支持样本的网格。所有交互是冻结模型的非加性预测。

## 6. A17 审计、A16 映射与质量代理

A17 为 17 行 × 5 列。其 `sample_rows`、`source_path`、`avg_text_chars` 已对压缩原文抽样逐域复核，{int(a17.verified.sum())}/17 行通过。对两个严格正的特征先取自然对数，再按 A17 全部 17 域的均值和总体标准差做 z 分数；17×17 欧氏距离及 `exp(-D)` 相似性见矩阵文件。这仅描述样本数与平均文本长度的统计接近程度，不是文本语义或数据质量相似度。`sample_rows` 受抽样程序影响，分别仅用长度和仅用样本数的敏感性方案已经保存。

A16 有 3 个 direct、3 个 near_direct，另 11 个 inferred 且未给质量域。A16 单独仅有 6/17 域可映射；对缺失域不填均值或零。A17 辅助方案仅在有可靠 A16 代表域的 6 个质量域之间按 `softmax(-D/T)` 分配推断权重；c4 没有代表域，因此其推断权重为零。此 11 域映射是**低可信度统计代理**，并未证明训练语料的真实质量，不能与 6 个参考映射同等看待。主温度 T=1，在关联分析前固定，T=0.5/2.0 及单特征方案仅作敏感性。P1 A1 七域主评分和等权评分均除以 100 后使用，版本为 `{metadata['quality_model_version']}`，未重新拟合。

训练配方映射覆盖：

{markdown_table(map_main, ['scenario', 'mapped_domains', 'complete_recipes', 'total_recipes', 'mean_share_covered'])}

`projected_domain_quality.csv`、`mixture_quality_proxy.csv` 和 `mapping_coverage.csv` 保留映射类型、完整性及已覆盖配比；配方代理包含全部六个切分，并保留 `data_role`，其中估算参考与真实观测严格区分。完整代理只在所有正配比域都有映射时计算；A17 辅助方案虽可计算 512/512 条训练配方完整代理，解释时必须带低可信度标记。计算式 `Q17=M Q7`、`p7=Mᵀp17`、`Qmix=p17ᵀM Q7` 已在人工微型样例中核验。

## 7. 质量代理与 Loss

`quality_loss_association.csv` 分别记录冻结模型预测和训练观测 Loss 的 Pearson/Spearman，且按情景及切分标明角色。训练集主 A17 辅助方案：

{markdown_table(q_ass, ['loss_role', 'n', 'pearson', 'spearman'])}

Qmix 是固定领域代理评分和配比构成的确定性函数，质量数据与配方数据没有逐文本配对；这些相关性不识别独立质量效应，也不能称为质量弹性。A16 单独方案只有少数完整配方，其相关统计极不稳定。等权、温度、仅长度、仅样本数方案的全部结果已保存，不能根据相关方向挑选正式方案。

## 8. 跨规模真实检验

60M/1B 的绝对误差为未经跨规模校准的描述性统计；重点比较同规模内部配方排序。逐验证域、综合 Loss 的 MAE/RMSE/R²/Pearson/Spearman/Kendall/Overlap@5/10/20 分别见 `test_60m_metrics.csv`、`test_1b_metrics.csv`。

{markdown_table(scores[(scores.model == 'lightgbm') & scores.split.isin(['test_60m', 'test_1b'])], ['split', 'n', 'rmse', 'r2', 'spearman', 'kendall', 'overlap_10'])}

## 9. 10B/70B 估算参考

这两张 Loss 表来自既有外推估算，配方均为 1M 训练配方子集。以下只是与既有估算排序的一致性，不是大模型真实验证；负相关意味着 1M 模型排序不能直接外推到这些估算表。

{markdown_table(scores[(scores.model == 'lightgbm') & scores.split.isin(['est_10b', 'est_70b'])], ['split', 'n', 'spearman', 'kendall', 'overlap_10'])}

## 10. 与新版 Question 1.3 草案的必要数学修正

草案 Step7 同时写了 `17×6` 与七个质量域、把统计距离 `D` 与相似度 `S` 混用、并给出维度不合的配比投影式。本实现按实际 P1 七域固定为 `M∈R^(17×7)`，推断权重使用 `exp(-D/T)`，配比投影使用 `p7=Mᵀp17`。草案中 `wikipedia.cn`、`widipedia`、`gutenbrg_pg_19`、`commandcrawl` 等拼写与实际 A16 不一致，均以 A16 实际字段和 P1 七域名为准。双域扰动按最新版任务要求使用 `1−p_j−p_k−δ_j−δ_k` 除以原 15 域总份额，避免把绝对目标比例与增量混用。Ridge 草案中的系数负号不预设，符号由数据决定。以上修正由公式维度、单纯形约束及附件实际字段支持。

## 11. 图表、复现与边界

已生成 {len(manifest)} 张 SVG，图源逐项见 `results/q1/figures/m3_figure_manifest.csv`。全部图以保存的结构化计算结果绘制；估算图标签与真实检验分开。运行入口：`PYTHONPATH=src .venv/bin/python -m q1.mixture.run`，该命令自动完成模型、整合、图表和报告；固定种子、配置、输入、设计文档和代码 SHA-256、依赖版本、冻结模型 SHA-256 均见 `mixture_model_metadata.json`。本阶段不含下游 Benchmark 直接观测，因此未产生相应评分。

尚存关键限制：A17 仅有样本数和平均文本长度，无法证明 11 个 inferred 域的质量对应；c4 无 A17 代表；跨规模绝对 Loss 未校准；10B/70B 是估算且排序与 1M 模型相反；有限扰动与交互均为模型计算，不是实际再训练实验。
"""
    (mix / "mixture_report.md").write_text(text, encoding="utf-8")
    (integ / "quality_mapping_report.md").write_text("# P3/M3 质量映射说明\n\n完整审计、映射、代理及关联分析见 [mixture_report.md](../mixture/mixture_report.md) 第 6–7 节。\n\nA16 参考映射仅覆盖 6/17 域；A17 辅助推断的其余 11 域为低可信度统计代理。`domain_mapping_matrix.csv` 为 A17 辅助方案，`mapping_coverage.csv` 含 A16 单独方案与敏感性。\n", encoding="utf-8")
    return {"figures": len(manifest), "ridge_vs_lightgbm_effect_rank_spearman": rank_corr}


if __name__ == "__main__":
    print(finalize(Path.cwd()))
