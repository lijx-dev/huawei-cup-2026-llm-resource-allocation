"""q2_v3 按证据门控的 P1–P9 阶段。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from .audit import FILES
from .common import BROOT, OUT, ROOT, SEED, csv, digest, dump, finish, metrics, read, sha, valid_ndl
from .model import EXTRA, analytic_opt, bounded_opt, fit, names, predict


def _save_report(path: Path, title: str, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")


def _base_group(d):
    # CSV 的整数/浮点推断不同：10 与 10.0 必须是同一基础实验组。
    n = pd.to_numeric(d["N_params_B"], errors="raise")
    tok = pd.to_numeric(d["D_tokens_B"], errors="raise")
    return pd.Series([f"{a:.12g}|{b:.12g}" for a, b in zip(n, tok)], index=d.index)


def p1() -> str:
    dest = OUT / "b1_baseline"
    raw = read(FILES["B1"])
    d, exclusions = valid_ndl(raw, "P1", FILES["B1"])
    csv(dest / "exclusions.csv", exclusions)
    if len(d) < 20 or d.N_params_B.nunique() < 5: return "BLOCKED"
    p, runs = fit("M0", d, starts=12)
    csv(dest / "multistart_runs.csv", runs)
    dump(dest / "final_parameters.json", {"model": "M0", "units": "billions", "parameters": dict(zip(names("M0"), p)), "n": len(d), "sample_hash": digest(d[["run_id", "N_params_B", "D_tokens_B", "val_loss"]].to_dict("records"))})
    folds, residuals = [], []
    for n in sorted(d.N_params_B.unique()):
        train, test = d[d.N_params_B != n], d[d.N_params_B == n]
        q, _ = fit("M0", train, starts=4)
        pred = predict("M0", q, test.N_params_B, test.D_tokens_B)
        folds.append({"held_out_N": n, **metrics(test.val_loss, pred)})
    pred = predict("M0", p, d.N_params_B, d.D_tokens_B)
    for (_, row), yp in zip(d.iterrows(), pred):
        residuals.append({"run_id": row.run_id, "N_billions": row.N_params_B, "D_billions": row.D_tokens_B, "observed": row.val_loss, "prediction": yp, "residual": row.val_loss - yp})
    csv(dest / "group_cv_metrics.csv", folds)
    csv(dest / "macro_group_metrics.csv", [{"aggregation": "unweighted_mean_of_held_out_scales", **{k: float(np.nanmean([f[k] for f in folds])) for k in ("RMSE", "MAE", "R2", "Pearson", "Spearman")}}])
    csv(dest / "residuals.csv", residuals)
    rng = np.random.default_rng(SEED)
    scales = np.array(sorted(d.N_params_B.unique()))
    boots = []
    for i in range(200):
        pick = rng.choice(scales, len(scales), replace=True)
        sample = pd.concat([d[d.N_params_B == x] for x in pick], ignore_index=True)
        if sample.N_params_B.nunique() < 3: continue
        try:
            bp, _ = fit("M0", sample, seed=SEED + i)
            boots.append({"replicate": i, **dict(zip(names("M0"), bp))})
        except ValueError: continue
    csv(dest / "bootstrap_parameters.csv", boots)
    ci = pd.DataFrame(boots)[list(names("M0"))].quantile([.025, .975]).to_dict() if boots else {}
    med_rmse = float(np.median([x["RMSE"] for x in folds]))
    r = pd.DataFrame(residuals)
    for x, name in ((np.log(r.N_billions), "log_n"), (np.log(r.D_billions), "log_d"), (r.prediction, "prediction")):
        csv(dest / f"residual_vs_{name}.csv", pd.DataFrame({"x": x, "residual": r.residual}))
    csv(dest / "residual_by_scale.csv", r.groupby("N_billions").residual.agg(["mean", "std", "count"]).reset_index())
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for axis, x, label in ((ax[0, 0], np.log(r.N_billions), "log N"),
                           (ax[0, 1], np.log(r.D_billions), "log D"),
                           (ax[1, 0], r.prediction, "predicted loss")):
        axis.scatter(x, r.residual, s=3, alpha=.35)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel=label, ylabel="observed - predicted")
    by_scale = [g.residual.to_numpy() for _, g in r.groupby("N_billions")]
    labels = [f"{x:g}" for x in sorted(r.N_billions.unique())]
    ax[1, 1].boxplot(by_scale, tick_labels=labels, showfliers=False)
    ax[1, 1].set(xlabel="N (billions)", ylabel="residual")
    ax[1, 1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    figure = OUT / "figures/b1_residual_diagnostics.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    warns = ["检查点同轨迹相关；按 N 规模留一验证", "数值生成机制缺独立核验；极高拟合度不能等价于现实世界泛化"]
    _save_report(dest / "p1_report.md", "P1 B1 经典标度律", [f"样本 {len(d)}；参数 {dict(zip(names('M0'), p))}。", f"LOSO 各折 RMSE 中位数 {med_rmse:.6g}；macro 结果见 macro_group_metrics.csv。", f"95% group bootstrap 区间：{ci}。残差四图：figures/b1_residual_diagnostics.png，数据见 residuals.csv。", *warns])
    finish(1, "PASS_WITH_WARNINGS", [BROOT / FILES["B1"]], dest / "p1_report.md", warns)
    return "PASS_WITH_WARNINGS"


def _b1p():
    x = json.loads((OUT / "b1_baseline/final_parameters.json").read_text())
    return np.array([x["parameters"][k] for k in names("M0")])


def p2() -> str:
    dest = OUT / "loss_comparability"
    p = _b1p()
    allrows, offsets, family, literature, exclusions = [], [], [], [], []
    frames = {"B2": read(FILES["B2"]), "B4": read(FILES["B4"]), "B5": read(FILES["B5"])}
    for aid, raw in frames.items():
        d, ex = valid_ndl(raw, "P2", FILES[aid]); exclusions.extend(ex)
        yhat = predict("M0", p, d.N_params_B, d.D_tokens_B)
        m = metrics(d.val_loss, yhat)
        allrows.append({"source": aid, "evidence": "trend_only", **m})
        offset = float(np.mean(d.val_loss.to_numpy() - yhat))
        offsets.append({"source": aid, "offset": offset, **metrics(d.val_loss, yhat + offset)})
        group_col = "run_id" if aid == "B2" else "source" if aid == "B5" else "family"
        for key, g in d.groupby(group_col):
            gpred = predict("M0", p, g.N_params_B, g.D_tokens_B)
            item = {"source": aid, "group": key, **metrics(g.val_loss, gpred)}
            (literature if aid == "B5" else family).append(item)
    b3 = pd.concat([pd.read_csv(f) for f in sorted((BROOT / FILES["B3"]).glob("*.csv"))], ignore_index=True)
    b3, ex = valid_ndl(b3, "P2", FILES["B3"]); exclusions.extend(ex)
    b3pred = predict("M0", p, b3.N_params_B, b3.D_tokens_B)
    allrows.append({"source": "B3", "evidence": "interpolation_consistency", **metrics(b3.val_loss, b3pred)})
    for n, g in b3.groupby("N_params_B"):
        family.append({"source": "B3", "group": n, **metrics(g.val_loss, predict("M0", p, g.N_params_B, g.D_tokens_B))})
    csv(dest / "zero_dof_metrics.csv", allrows); csv(dest / "source_offset_metrics.csv", offsets)
    csv(dest / "per_family_metrics.csv", family); csv(dest / "per_literature_metrics.csv", literature)
    csv(dest / "exclusions.csv", exclusions)
    _save_report(dest / "p2_report.md", "P2 跨来源 Loss 可比性", ["B1 参数保持冻结；零自由度预测与仅诊断用 source offset 分开。", "B2/B4/B5 的 tokenizer、评估语料与 Loss 定义未充分证明相同；绝对数值仅作诊断，趋势可描述。B3 是插值一致性。", pd.DataFrame(allrows).to_string(index=False)])
    finish(2, "PASS_WITH_WARNINGS", [BROOT / FILES[x] for x in ("B2", "B4", "B5")] + list((BROOT / FILES["B3"]).glob("*.csv")), dest / "p2_report.md", ["绝对 Loss 口径未证实一致"])
    return "PASS_WITH_WARNINGS"


def _amplitudes(d):
    rows = []
    # 固定绝对 Q_B 档位；B6 中所有 N/D 组共用同一阈值。
    fixed_low, fixed_high = .3, .8
    for (n, tok), g in d.groupby(["N_params_B", "D_tokens_B"]):
        if g.Q_score.nunique() < 4: continue
        low = g[g.Q_score <= fixed_low]
        high = g[g.Q_score >= fixed_high]
        if low.empty or high.empty: continue
        low_loss, high_loss = float(low.val_loss.mean()), float(high.val_loss.mean())
        slope = float(np.polyfit(g.Q_score, g.val_loss, 1)[0])
        rows.append({"N_billions": n, "D_billions": tok, "group": f"{n}|{tok}", "n": len(g),
                     "A_max": float(g.val_loss.max() - g.val_loss.min()), "A_80_20": float(np.quantile(g.val_loss, .8) - np.quantile(g.val_loss, .2)),
                     "A_fixed": abs(low_loss - high_loss), "fixed_low_Q_cut": fixed_low, "fixed_high_Q_cut": fixed_high,
                     "dL_dQ": slope, "low_Q_mean_loss": low_loss, "high_Q_mean_loss": high_loss})
    return pd.DataFrame(rows)


def _ampl_reg(frame, col):
    a = frame[frame[col] > 0]
    x = np.c_[np.ones(len(a)), np.log(a.N_billions), np.log(a.D_billions)]
    if len(a) < 5 or np.linalg.matrix_rank(x) < 3: return None
    coef = np.linalg.lstsq(x, np.log(a[col]), rcond=None)[0]
    return dict(zip(("intercept", "b_N", "b_D"), map(float, coef)))


def p3() -> str:
    dest = OUT / "quality_position"; d = read(FILES["B6"])
    frame = _amplitudes(d)
    if len(frame) < 8: return "BLOCKED"
    csv(dest / "amplitude_by_group.csv", frame)
    cols = ("A_max", "A_80_20", "A_fixed")
    regs = [{"amplitude": col, **_ampl_reg(frame, col)} for col in cols]
    csv(dest / "amplitude_regressions.csv", regs)
    rng = np.random.default_rng(SEED)
    boots = []
    for col in cols:
        for i in range(1000):
            sample = frame.iloc[rng.choice(len(frame), len(frame), replace=True)]
            z = _ampl_reg(sample, col)
            if z: boots.append({"amplitude": col, "replicate": i, **z})
    csv(dest / "amplitude_bootstrap.csv", boots)
    csv(dest / "local_quality_sensitivity.csv", frame[["group", "N_billions", "D_billions", "dL_dQ"]])
    local_frame = frame.assign(A_local=frame.dL_dQ.abs())
    local_reg = _ampl_reg(local_frame, "A_local")
    csv(dest / "local_quality_sensitivity_regression.csv", [{"outcome": "log_abs_dL_dQ", **(local_reg or {})}])
    stability = []
    for col in cols:
        for key in ("N_billions", "D_billions"):
            for level in frame[key].unique():
                z = _ampl_reg(frame[frame[key] != level], col)
                stability.append({"amplitude": col, "left_out_dimension": key, "left_out_level": level, **(z or {})})
    csv(dest / "stability_checks.csv", stability)
    ci = pd.DataFrame(boots).groupby("amplitude").b_N.quantile([.025, .975]).unstack()
    robust = (all(ci.loc[c, .975] < 0 for c in cols)
              and all(x.get("b_N", 0) < 0 for x in stability)
              and local_reg is not None and local_reg["b_N"] < 0)
    dump(dest / "quality_position_decision.json", {"reject_D_only_within_B6": robust, "criteria": "all three group-bootstrap upper CIs <0; all leave-one-level-out b_N <0; local |dL/dQ| b_N <0", "provenance": "B6 semi_synthetic"})
    _save_report(dest / "p3_report.md", "P3 质量作用位置", [f"基础组 {len(frame)}；A_fixed 对所有组固定 Q_B≤0.3 与 Q_B≥0.8。", pd.DataFrame(regs).to_string(index=False), f"b_N bootstrap 区间：{ci.to_dict()}；局部质量灵敏度回归：{local_reg}。", f"多项一致时仅在 B6 半合成机制内拒绝质量只作用于 D 通道：{robust}；不能当作真实训练机制的因果证明。留一 N/D 层结果见 stability_checks。"])
    finish(3, "PASS_WITH_WARNINGS", [BROOT / FILES["B6"]], dest / "p3_report.md", ["B6 为半合成；诊断不等同机制因果识别"])
    return "PASS_WITH_WARNINGS"


def _quality_data():
    d, exclusions = valid_ndl(read(FILES["B6"]), "P4", FILES["B6"])
    q = pd.to_numeric(d.Q_score, errors="coerce")
    ok = np.isfinite(q) & q.between(0, 1)
    for i in np.where(~ok)[0]:
        exclusions.append({"source_file": FILES["B6"], "row_id_or_experiment_id": str(d.iloc[i].experiment_id), "reason": "invalid_Q_B", "evidence": str(q.iloc[i]), "stage": "P4", "action": "exclude"})
    return d.loc[ok].copy(), exclusions


def _parameters_json(kind, p, d):
    return {"model_name": kind, "parameters": dict(zip(names(kind), map(float, p))), "units": "N/D billions; Q_B as raw Q_score",
            "fit_sample_hash": digest(d[["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_dict("records"))}


def p4() -> str:
    dest = OUT / "quality_models"
    d, exclusions = _quality_data()
    csv(dest / "exclusions.csv", exclusions)
    groups = _base_group(d)
    fold_list = list(GroupKFold(n_splits=5).split(d, groups=groups))
    fold_def = [{"train_ids": sorted(d.iloc[tr].experiment_id.tolist()), "test_ids": sorted(d.iloc[te].experiment_id.tolist())} for tr, te in fold_list]
    fold_hash = digest(fold_def)
    candidates = list(EXTRA)
    fitted, pars, criteria, cv_rows, diffs = {}, [], [], [], []
    predictions = {}
    for kind in candidates:
        p, runs = fit(kind, d, starts=8)
        fitted[kind] = p
        for row in runs: pars.append({"model": kind, **row})
        yhat = predict(kind, p, d.N_params_B, d.D_tokens_B, d.Q_score)
        sse = float(np.sum((d.val_loss.to_numpy() - yhat) ** 2))
        k = len(p); n = len(d)
        criteria.append({"model": kind, "SSE": sse, "AIC": n * np.log(max(sse / n, 1e-20)) + 2 * k, "BIC": n * np.log(max(sse / n, 1e-20)) + k * np.log(n), "n_parameters": k})
        hold = np.full(n, np.nan)
        for fold, (tr, te) in enumerate(fold_list):
            fp, _ = fit(kind, d.iloc[tr], starts=3, seed=SEED + fold)
            hold[te] = predict(kind, fp, d.iloc[te].N_params_B, d.iloc[te].D_tokens_B, d.iloc[te].Q_score)
            cv_rows.append({"model": kind, "fold": fold, **metrics(d.iloc[te].val_loss, hold[te])})
        predictions[kind] = hold
        cv_rows.append({"model": kind, "fold": "pooled", **metrics(d.val_loss, hold)})
    csv(dest / "candidate_parameters.csv", pars)
    csv(dest / "information_criteria.csv", criteria)
    csv(dest / "candidate_model_metrics.csv", cv_rows)
    pooled = {x["model"]: x for x in cv_rows if x["fold"] == "pooled"}
    minimum = min(x["RMSE"] for x in pooled.values())
    # 预定义 1% 容差内选最简参数化；此规则完全由 B6 CV 决定。
    selected = min((k for k in candidates if pooled[k]["RMSE"] <= 1.01 * minimum), key=lambda k: (len(names(k)), pooled[k]["RMSE"]))
    rng = np.random.default_rng(SEED)
    unique_groups = groups.unique()
    residual = {k: (d.val_loss.to_numpy() - predictions[k]) ** 2 for k in candidates}
    for k in candidates:
        if k == "M0": continue
        for i in range(500):
            picked = rng.choice(unique_groups, len(unique_groups), replace=True)
            idx = np.concatenate([np.where(groups.to_numpy() == g)[0] for g in picked])
            diffs.append({"model": k, "replicate": i, "method": "group_bootstrap_cv_sse_difference", "M0_minus_candidate": float(np.mean(residual["M0"][idx] - residual[k][idx]))})
    # 参数 bootstrap 至少 400 次；每次完整基础组重抽样。
    bootstrap = []
    for i in range(400):
        picked = rng.choice(unique_groups, len(unique_groups), replace=True)
        sample = pd.concat([d.loc[groups == g] for g in picked], ignore_index=True)
        try:
            bp, _ = fit(selected, sample, seed=SEED + i)
            bootstrap.append({"replicate": i, **dict(zip(names(selected), bp))})
        except ValueError: continue
    csv(dest / "parameter_bootstrap.csv", bootstrap)
    bootframe = pd.DataFrame(bootstrap)
    csv(dest / "parameter_correlation.csv", bootframe[list(names(selected))].corr().reset_index().rename(columns={"index": "parameter"}))
    # 零模型下参数化高斯生成，标准差按基础组估计；仅作为模型证据。
    m0hat = predict("M0", fitted["M0"], d.N_params_B, d.D_tokens_B)
    err = d.val_loss.to_numpy() - m0hat
    grouped_sd = {g: max(float(np.std(err[np.where(groups.to_numpy() == g)[0]], ddof=1)), 1e-6) for g in unique_groups}
    parametric = []
    for i in range(100):
        simulated = d.copy()
        simerr = np.array([rng.normal(0, grouped_sd[g]) for g in groups])
        simulated["val_loss"] = m0hat + simerr
        try:
            a, _ = fit("M0", simulated)
            b, _ = fit(selected, simulated)
            s0 = np.sum((simulated.val_loss - predict("M0", a, simulated.N_params_B, simulated.D_tokens_B)) ** 2)
            s1 = np.sum((simulated.val_loss - predict(selected, b, simulated.N_params_B, simulated.D_tokens_B, simulated.Q_score)) ** 2)
            parametric.append({"replicate": i, "model": selected, "method": "parametric_gaussian_null_by_group", "M0_minus_candidate": float(s0-s1)})
        except ValueError: continue
    csv(dest / "bootstrap_model_differences.csv", diffs + parametric)
    # 全模型 profile，含最终模型不一定含有的参数，以诊断结构不可识别性。
    full = fitted["MCHANNEL_ADD"]
    for key, name in (("rho_n", "profile_rho_n.csv"), ("rho_d", "profile_rho_d.csv"), ("e1", "profile_e1.csv")):
        idx = names("MCHANNEL_ADD").index(key)
        grid = np.linspace(max(-5 if key != "e1" else -3, full[idx] - 1), min(5 if key != "e1" else 3, full[idx] + 1), 25)
        rows = []
        for value in grid:
            p, _ = fit("MCHANNEL_ADD", d, anchor={key: value})
            rows.append({key: value, "SSE": float(np.sum((d.val_loss - predict("MCHANNEL_ADD", p, d.N_params_B, d.D_tokens_B, d.Q_score)) ** 2))})
        csv(dest / name, rows)
    trade = []
    for rd in np.linspace(max(-5, full[6] - .6), min(5, full[6] + .6), 13):
        for e1 in np.linspace(max(-3, full[7] - .6), min(3, full[7] + .6), 13):
            p, _ = fit("MCHANNEL_ADD", d, anchor={"rho_d": rd, "e1": e1})
            trade.append({"rho_d": rd, "e1": e1, "SSE": float(np.sum((d.val_loss - predict("MCHANNEL_ADD", p, d.N_params_B, d.D_tokens_B, d.Q_score)) ** 2))})
    csv(dest / "rho_d_e1_tradeoff.csv", trade)
    b1 = dict(zip(names("M0"), _b1p()))
    anchored = []
    for kind in candidates:
        p, _ = fit(kind, d, anchor={k: b1[k] for k in ("E", "alpha", "beta")})
        anchored.append({"model": kind, **metrics(d.val_loss, predict(kind, p, d.N_params_B, d.D_tokens_B, d.Q_score)), "parameters": json.dumps(dict(zip(names(kind), map(float, p))))})
    csv(dest / "anchored_sensitivity.csv", anchored)
    chosen = {"model_name": selected, "formula": "L=E+A*N^-alpha*exp(-rho_n*Q_B)+B*D^-beta*exp(-rho_d*Q_B)-e1*Q_B; absent terms set zero",
              "q_transform": "raw Q_B from B6; no flip", "parameter_constraints": "E>=0; A,B,alpha,beta>0; rho_n,rho_d,e1 in bounded signed intervals",
              "parameters": dict(zip(names(selected), map(float, fitted[selected]))), "fit_sample_hash": digest(d[["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_dict("records")),
              "fold_definition_hash": fold_hash, "selection_evidence": {"rule": "lowest group-CV RMSE; within 1% choose fewest parameters", "pooled_CV": pooled, "AIC_BIC": criteria},
              "code_hash": sha(ROOT / "src/q2_v3/model.py")}
    dump(dest / "selected_model.json", chosen)
    (dest / "model_freeze.sha256").write_text(sha(dest / "selected_model.json") + "  selected_model.json\n", encoding="utf-8")
    _save_report(dest / "p4_report.md", "P4 质量结构与可识别性", [f"同一 B6 样本 {len(d)}，基础组 {groups.nunique()}，5 折同组留出；fold hash={fold_hash}。", f"预定选择规则选出 {selected}，不使用 B7 Loss。", pd.DataFrame([pooled[k] for k in candidates]).to_string(index=False), "400 次组 bootstrap 与全模型 profile/trade-off 见结构化产物；若相关性或 profile 宽，内部通道参数弱识别，应优先解释总质量导数。"])
    finish(4, "PASS_WITH_WARNINGS", [BROOT / FILES["B6"], OUT / "b1_baseline/final_parameters.json"], dest / "p4_report.md", ["候选结构基于半合成 B6；参数可识别性需看 profile 与 bootstrap"])
    return "PASS_WITH_WARNINGS"


def _selected():
    x = json.loads((OUT / "quality_models/selected_model.json").read_text())
    if sha(OUT / "quality_models/selected_model.json") != (OUT / "quality_models/model_freeze.sha256").read_text().split()[0]:
        raise ValueError("P4 模型冻结哈希不一致")
    return x["model_name"], np.array([x["parameters"][k] for k in names(x["model_name"])])


def _m0_b6():
    d = pd.read_csv(OUT / "quality_models/candidate_parameters.csv")
    x = d[d.model == "M0"].sort_values("sse").iloc[0]
    return np.array(json.loads(x.parameters))


def p5() -> str:
    dest = OUT / "locked_validation"
    kind, p = _selected()
    b6 = read(FILES["B6"]); b7 = read(FILES["B7"])
    identity = pd.read_csv(OUT / "audit/b6_b7_overlap.csv")
    newids = set(identity.loc[identity.role == "b7_new", "experiment_id"])
    d = b7[b7.experiment_id.isin(newids)].copy()
    if len(d) != len(newids): raise ValueError("B7-new 身份与原件不一致")
    baseline = _m0_b6()
    pred = predict(kind, p, d.N_params_B, d.D_tokens_B, d.Q_score)
    basepred = predict("M0", baseline, d.N_params_B, d.D_tokens_B)
    csv(dest / "b7_new_metrics.csv", [{"model": kind, **metrics(d.val_loss, pred)}, {"model": "M0_B6", **metrics(d.val_loss, basepred)}])
    rng = np.random.default_rng(SEED)
    groups = _base_group(d)
    unique = groups.unique()
    diff = []
    for i in range(1000):
        picked = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.where(groups.to_numpy() == g)[0] for g in picked])
        diff.append({"replicate": i, "RMSE_M0_minus_selected": metrics(d.val_loss.iloc[idx], basepred[idx])["RMSE"] - metrics(d.val_loss.iloc[idx], pred[idx])["RMSE"]})
    csv(dest / "model_error_differences.csv", diff)
    dump(dest / "pre_validation_parameters.json", _parameters_json(kind, p, b6))
    combined = pd.concat([b6, d], ignore_index=True)
    post, _ = fit(kind, combined, starts=12)
    dump(dest / "post_validation_parameters.json", _parameters_json(kind, post, combined))
    post_groups = _base_group(combined)
    unique_post = post_groups.unique()
    post_boot = []
    for i in range(400):
        picked = rng.choice(unique_post, len(unique_post), replace=True)
        sample = pd.concat([combined.loc[post_groups == g] for g in picked], ignore_index=True)
        try:
            bp, _ = fit(kind, sample, seed=SEED + i)
            post_boot.append({"replicate": i, **dict(zip(names(kind), bp))})
        except ValueError: continue
    csv(dest / "post_validation_bootstrap.csv", post_boot)
    seen_groups = set(_base_group(b6))
    validation_type = "new-Q interpolation validation" if set(_base_group(d)).issubset(seen_groups) else "partly_new_N_D_group_validation"
    dump(dest / "validation_summary.json", {"validation_type": validation_type, "b7_new_rows": len(d), "new_N_D_groups": len(set(_base_group(d)) - seen_groups),
                                           "new_Q_levels": sorted(set(map(float, d.Q_score)) - set(map(float, b6.Q_score))),
                                           "frozen_model_sha256": sha(OUT / "quality_models/selected_model.json")})
    _save_report(dest / "p5_report.md", "P5 B7-new 锁定验证", [f"冻结模型 {kind}；首次检验 {len(d)} 条 B7-new；{validation_type}。", pd.read_csv(dest / "b7_new_metrics.csv").to_string(index=False), "先完成锁定验证，再使用 B6+B7-new 重拟合；验证参数与论文参数已分开保存。"])
    finish(5, "PASS_WITH_WARNINGS", [BROOT / FILES["B7"], OUT / "quality_models/selected_model.json"], dest / "p5_report.md", ["B7-new 仅新增质量档且为半合成，不能声称独立真实验证"])
    return "PASS_WITH_WARNINGS"


def p6() -> str:
    dest = OUT / "stress_extrapolation"
    kind, _ = _selected()
    post = json.loads((OUT / "locked_validation/post_validation_parameters.json").read_text())
    p = np.array([post["parameters"][k] for k in names(kind)])
    b8, b9, b10 = (read(FILES[x]) for x in ("B8", "B9", "B10"))
    directions, errors = [], []
    for (typ, n, d), g in b8.groupby(["data_type", "N_params_B", "D_tokens_B"]):
        rho = float(spearmanr(g.Q_score, g.val_loss).statistic) if g.Q_score.nunique() >= 3 else np.nan
        directions.append({"data_type": typ, "N_billions": n, "D_billions": d, "n": len(g), "spearman_Q_loss": rho,
                           "sign": "negative" if rho < 0 else "positive" if rho > 0 else "unresolved"})
    for typ, g in b8.groupby("data_type"):
        errors.append({"data_type": typ, **metrics(g.val_loss, predict(kind, p, g.N_params_B, g.D_tokens_B, g.Q_score))})
    csv(dest / "b8_groupwise_direction.csv", directions); csv(dest / "b8_model_errors.csv", errors)
    b9valid = b9[(b9.N_params_B > 0) & (b9.D_tokens_B > 0)]
    b9excluded = b9[~b9.index.isin(b9valid.index)]
    b1 = read(FILES["B1"])
    csv(dest / "b9_scale_range.csv", [{"source": "B9", "raw_rows": len(b9), "valid_N_D_rows": len(b9valid), "N_min": b9valid.N_params_B.min(), "N_max": b9valid.N_params_B.max(), "D_min": b9valid.D_tokens_B.min(), "D_max": b9valid.D_tokens_B.max(),
                                        "B1_N_max": b1.N_params_B.max(), "B1_D_max": b1.D_tokens_B.max(), "median_log_N_distance_beyond_B1": float(np.median(np.log(b9valid.N_params_B / b1.N_params_B.max()))),
                                        "median_log_D_distance_beyond_B1": float(np.median(np.log(b9valid.D_tokens_B / b1.D_tokens_B.max())))}])
    csv(dest / "exclusions.csv", [{"source_file": FILES["B9"], "row_id_or_experiment_id": x.model_name, "reason": "N_or_D_nonpositive_or_missing", "evidence": f"N={x.N_params_B}; D={x.D_tokens_B}", "stage": "P6", "action": "exclude_from_range"} for _, x in b9excluded.iterrows()])
    b10valid, _ = valid_ndl(b10, "P6", FILES["B10"])
    csv(dest / "b10_consistency.csv", [{"model": "frozen_B1_M0", "role": "estimated_consistency", **metrics(b10valid.val_loss, predict("M0", _b1p(), b10valid.N_params_B, b10valid.D_tokens_B))}])
    (dest / "generation_risk.md").write_text("B10 文档明确为已拟合标度律生成的估算 Loss；与本模型高一致性有循环验证风险，不计为独立实证。B8 为半合成且有外推层。\n", encoding="utf-8")
    neg = sum(x["sign"] == "negative" for x in directions)
    pos = sum(x["sign"] == "positive" for x in directions)
    _save_report(dest / "p6_report.md", "P6 压力与外推", [f"B8 固定 N,D 组 Q-Loss 负向 {neg}、正向 {pos}；原方向保留。", pd.DataFrame(errors).to_string(index=False), f"B9 {len(b9)} 条元数据、其中 N/D 正值 {len(b9valid)} 条，无 Loss。B10 仅作估算一致性，不能当独立验证。"])
    finish(6, "PASS_WITH_WARNINGS", [BROOT / FILES[x] for x in ("B8", "B9", "B10")], dest / "p6_report.md", ["B8 半合成与 B6/B7 生成机制可能不同", "B10 存在循环验证风险"])
    return "PASS_WITH_WARNINGS"


def _q1_predict(obj, matrix):
    models = obj["models"]["LightGBM"]
    out = np.column_stack([m.predict(matrix) for m in models])
    if out.shape[1] != 13 or not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError("Q1 冻结模型预测非正或目标数不对")
    return out


def p7() -> str:
    dest = OUT / "q1_interface"
    audit = json.loads((OUT / "audit/q1_interface.json").read_text())
    model_file = ROOT / audit["model_path"]
    if sha(model_file) != audit["model_sha256"]: raise ValueError("Q1 模型哈希变化")
    obj = joblib.load(model_file)
    if len(obj["mix_fields"]) != 17 or len(obj["loss_fields"]) != 13: raise ValueError("Q1 输入/目标维度错误")
    q1opt = json.loads((ROOT / "results/q1/mixture/optimal_mixture.json").read_text())
    p0 = np.array([q1opt["best_training_recipe_P"][f.replace("train_the_pile_", "")] for f in obj["mix_fields"]])
    candidate = np.array([q1opt["P"][f.replace("train_the_pile_", "")] for f in obj["mix_fields"]])
    if np.any(p0 < 0) or np.any(candidate < 0) or abs(p0.sum()-1) > 1e-8 or abs(candidate.sum()-1) > 1e-8: raise ValueError("p0/candidate 不在单纯形")
    pred = _q1_predict(obj, pd.DataFrame(np.vstack([p0, candidate]), columns=obj["mix_fields"]))
    hp = np.log(pred[1] / pred[0])
    rows = [{"target": key, "p0_loss": float(pred[0, i]), "candidate_loss": float(pred[1, i]), "h_p_v_p0": 0., "h_p_v_candidate": float(hp[i])} for i, key in enumerate(obj["loss_fields"])]
    csv(dest / "hp_by_target.csv", rows)
    hagg = float(hp.mean())
    csv(dest / "hp_aggregate.csv", [{"mixture": "p0", "h_agg": 0., "weights": "equal_1_of_13"}, {"mixture": "q1_imported_candidate", "h_agg": hagg, "weights": "equal_1_of_13"}])
    csv(dest / "support_diagnostics.csv", [{"mixture": "p0", "simplex_sum": p0.sum(), "support_status": "training_recipe", "nearest_train_distance": 0.},
                                           {"mixture": "q1_imported_candidate", "simplex_sum": candidate.sum(), "support_status": "reported_supported_by_Q1_search", "nearest_train_distance": q1opt["nearest_train_distance"], "threshold": audit["support_threshold"]}])
    dump(dest / "reference_mixture.json", {"definition": "Q1 frozen reported best training recipe; chosen before Q2 mixture scenario calculation", "input_columns": obj["mix_fields"], "output_columns": obj["loss_fields"], "p0": p0.tolist(), "candidate": candidate.tolist(), "model_sha256": sha(model_file)})
    qa = pd.read_csv(ROOT / "results/q1_revision_v2_1/quality/domain_quality_summary.csv")
    qa_col = "Q_hierarchical_balanced"
    if qa_col not in qa: raise ValueError("最新 QA 列缺失")
    low, high = float(qa[qa_col].min()), float(qa[qa_col].max())
    csv(dest / "qa_qb_mapping_scenarios.csv", [{"mapping": "linear_rank_preserving", "formula": "QB=0.1+0.9*r; r=(QA-QA_min)/(QA_max-QA_min)", "power": 1, "QA_min": low, "QA_max": high, "QB_low": .1, "QB_high": 1., "role": "scenario_assumption"},
                                                   {"mapping": "power_rank_preserving", "formula": "QB=0.1+0.9*r^2; r=(QA-QA_min)/(QA_max-QA_min)", "power": 2, "QA_min": low, "QA_max": high, "QB_low": .1, "QB_high": 1., "role": "scenario_assumption"}])
    csv(dest / "mixture_structure_scenarios.csv", [{"structure": s, "lambda_p": lam, "h_agg_candidate": hagg, "role": "scenario_assumption"} for s in ("P-D", "P-R") for lam in (0, .5, 1, 1.5)])
    _save_report(dest / "p7_report.md", "P7 第一问冻结接口与情景", [f"模型哈希 {sha(model_file)}；17 输入、13 输出；p0 为 Q1 已报告最佳训练配方。", f"候选配方 h_agg={hagg:.6g}，13 个逐目标值见 CSV。", "Q_A 与 Q_B 定义及尺度不同；线性/幂次映射仅作秩保持情景，不称真实校准。lambda_p 无 B 数据识别。"])
    finish(7, "PASS_WITH_WARNINGS", [model_file, ROOT / "results/q1/mixture/optimal_mixture.json", ROOT / "results/q1_revision_v2_1/quality/domain_quality_summary.csv"], dest / "p7_report.md", ["Q1 模型缺历史哈希锚，但 v2.1 参考哈希一致", "QA→QB 与 p 作用位置仅为情景假设"])
    return "PASS_WITH_WARNINGS"


def _post_model():
    x = json.loads((OUT / "locked_validation/post_validation_parameters.json").read_text())
    return x["model_name"], np.array([x["parameters"][k] for k in names(x["model_name"])])


def _marginals(kind, p, n, d, q, hp=0., lam=0., structure="P-D"):
    z = dict(zip(names(kind), p))
    term_n = z["A"] * n ** (-z["alpha"]) * np.exp(-z.get("rho_n", 0) * q)
    term_d = z["B"] * d ** (-z["beta"]) * np.exp(-z.get("rho_d", 0) * q)
    factor = np.exp(lam * hp)
    if structure == "P-D": total_n, total_d = term_n, term_d * factor
    else: total_n, total_d = term_n * factor, term_d * factor
    loss = z["E"] - z.get("e1", 0) * q + total_n + total_d
    reducible = total_n + total_d
    dq = -(z.get("rho_n", 0) * total_n + z.get("rho_d", 0) * total_d + z.get("e1", 0))
    mn = z["alpha"] * total_n / n
    md = z["beta"] * total_d / d
    return {"loss": loss, "M_N": mn, "M_D": md, "M_Q": -dq,
            "dlogL_dlogN": -z["alpha"] * total_n / loss,
            "dlogL_dlogD": -z["beta"] * total_d / loss,
            "dlogL_dQ": dq / loss, "dlogR_dQ": dq / reducible,
            "dlogN_dQ_at_fixed_L_D_p": dq / (z["alpha"] * total_n)}


def p8() -> str:
    dest = OUT / "compute_opt"
    kind, p = _post_model()
    b1 = read(FILES["B1"]); b6 = read(FILES["B6"]); b7 = read(FILES["B7"])
    valid = pd.concat([b1[["N_params_B", "D_tokens_B"]], b6[["N_params_B", "D_tokens_B"]], b7[["N_params_B", "D_tokens_B"]]])
    limits = (float(valid.N_params_B.min()), float(valid.N_params_B.max()), float(valid.D_tokens_B.min()), float(valid.D_tokens_B.max()))
    qlow, qhigh = float(min(b6.Q_score.min(), b7.Q_score.min())), float(max(b6.Q_score.max(), b7.Q_score.max()))
    qmid = (qlow + qhigh) / 2
    budgets = (1e19, 1e20, 1e21, 1e22)
    analytic, numeric, bounded, qsens = [], [], [], []
    for c in budgets:
        a = analytic_opt(kind, p, c, qmid)
        n = bounded_opt(kind, p, c, qmid, (1e-8, 1e8, 1e-8, 1e8))
        relative = abs(a[0] - n[0]) / a[0]
        if relative > 1e-5: raise RuntimeError(f"解析/数值最优不一致: {relative}")
        analytic.append({"train_compute": c, "Q_B": qmid, "N_billions": a[0], "D_billions": a[1], "pred_loss": a[2]})
        numeric.append({"train_compute": c, "N_analytic": a[0], "N_numerical": n[0], "relative_gap": relative})
        for q in (qlow, qmid, qhigh):
            z = bounded_opt(kind, p, c, q, limits)
            if z:
                item = {"train_compute": c, "Q_B": q, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2], "N_at_boundary": np.isclose(z[0], limits[0]) or np.isclose(z[0], limits[1]), "evidence": "C_semi_synthetic_calibration"}
                qsens.append(item)
                if q == qmid: bounded.append(item)
    csv(dest / "analytic_optima.csv", analytic); csv(dest / "numerical_verification.csv", numeric)
    csv(dest / "bounded_optima.csv", bounded); csv(dest / "quality_sensitivity.csv", qsens)
    hp = float(pd.read_csv(OUT / "q1_interface/hp_aggregate.csv").iloc[1].h_agg)
    scenarios = []
    for c in budgets:
        for q in (qlow, qmid, qhigh):
            for structure in ("P-D", "P-R"):
                for lam in (0, .5, 1, 1.5):
                    for mixture, effect in (("p0", 0.), ("q1_imported_candidate", hp)):
                        z = bounded_opt(kind, p, c, q, limits, effect, lam, structure)
                        if z:
                            scenarios.append({"train_compute": c, "Q_B": q, "structure": structure, "lambda_p": lam, "mixture": mixture, "h_agg": effect, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2], "evidence": "F_scenario_assumption"})
    csv(dest / "mixture_scenario_optima.csv", scenarios)
    boot = []
    for source, file, bk in (("B1_group_bootstrap", OUT / "b1_baseline/bootstrap_parameters.csv", "M0"),
                             ("quality_post_validation_group_bootstrap", OUT / "locked_validation/post_validation_bootstrap.csv", kind)):
        frame = pd.read_csv(file)
        for i, row in frame.iterrows():
            bp = np.array([row[k] for k in names(bk)], float)
            for c in budgets:
                z = bounded_opt(bk, bp, c, qmid, limits)
                if z: boot.append({"source": source, "replicate": i, "train_compute": c, "Q_B": qmid, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2]})
    csv(dest / "bootstrap_optima.csv", boot)
    structural = []
    frame = pd.read_csv(OUT / "quality_models/candidate_parameters.csv")
    for name in EXTRA:
        row = frame[frame.model == name].sort_values("sse").iloc[0]
        bp = np.array(json.loads(row.parameters))
        for c in budgets:
            for q in (qlow, qhigh):
                z = bounded_opt(name, bp, c, q, limits)
                if z: structural.append({"source": "quality_model_structure", "model": name, "train_compute": c, "Q_B": q, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2]})
    mapping = pd.read_csv(OUT / "q1_interface/qa_qb_mapping_scenarios.csv")
    qa_ref = float((mapping.iloc[0].QA_min + mapping.iloc[0].QA_max) / 2)
    rank = (qa_ref - mapping.iloc[0].QA_min) / (mapping.iloc[0].QA_max - mapping.iloc[0].QA_min)
    for method, q in (("linear_rank_preserving", .1 + .9 * rank), ("power_rank_preserving", .1 + .9 * rank ** 2)):
        z = bounded_opt(kind, p, 1e21, q, limits)
        if z: structural.append({"source": "QA_to_QB_mapping", "model": method, "train_compute": 1e21, "Q_B": q, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2]})
    for row in scenarios:
        if row["train_compute"] == 1e21 and row["Q_B"] == qmid and row["mixture"] != "p0":
            structural.append({"source": "p_structure_and_lambda", "model": f"{row['structure']}_lambda_{row['lambda_p']}", "train_compute": 1e21, "Q_B": qmid, "N_billions": row["N_billions"], "D_billions": row["D_billions"], "pred_loss": row["pred_loss"]})
    # 用联合 bootstrap 的参数极端组合表示通道参数权衡，保留其相关性。
    quality_boot = pd.read_csv(OUT / "locked_validation/post_validation_bootstrap.csv")
    if "rho_d" in quality_boot and "e1" in quality_boot:
        for column in ("rho_d", "e1"):
            for which in (quality_boot[column].idxmin(), quality_boot[column].idxmax()):
                bp = np.array([quality_boot.loc[which, key] for key in names(kind)], float)
                z = bounded_opt(kind, bp, 1e21, qmid, limits)
                if z: structural.append({"source": "weak_identification_joint_parameter_draw", "model": f"{column}_extreme_{which}", "train_compute": 1e21, "Q_B": qmid, "N_billions": z[0], "D_billions": z[1], "pred_loss": z[2]})
    csv(dest / "structural_uncertainty.csv", structural)
    uncertainty = []
    for (source, c), group in pd.DataFrame(boot).groupby(["source", "train_compute"]):
        for key in ("N_billions", "D_billions", "pred_loss"):
            uncertainty.append({"source": source, "train_compute": c, "quantity": key, "low": float(group[key].quantile(.025)), "high": float(group[key].quantile(.975)), "summary_type": "bootstrap_95pct"})
    for (source, c), group in pd.DataFrame(structural).groupby(["source", "train_compute"]):
        for key in ("N_billions", "D_billions", "pred_loss"):
            uncertainty.append({"source": source, "train_compute": c, "quantity": key, "low": float(group[key].min()), "high": float(group[key].max()), "summary_type": "scenario_range"})
    csv(dest / "uncertainty_summary.csv", uncertainty)
    local = []
    for row in scenarios:
        if row["train_compute"] == 1e21 and row["Q_B"] == qmid:
            m = _marginals(kind, p, row["N_billions"], row["D_billions"], qmid, row["h_agg"], row["lambda_p"], row["structure"])
            local.append({"train_compute": 1e21, "N_billions": row["N_billions"], "D_billions": row["D_billions"], "Q_B": qmid, "p_scenario": row["mixture"], "structure": row["structure"], "lambda_p": row["lambda_p"], **m})
    csv(dest / "local_effects_and_substitution.csv", local)
    dump(dest / "support_bounds.json", {"N_billions": [limits[0], limits[1]], "D_billions": [limits[2], limits[3]], "Q_B": [qlow, qhigh], "train_compute_formula": "6*N_raw*D_raw"})
    _save_report(dest / "p8_report.md", "P8 固定训练算力", ["算力近似 train_compute≈6ND（典型稠密 Transformer 训练数量级，不是硬件精确成本）；N/D 表内为十亿单位。", f"支持边界 N={limits[:2]}、D={limits[2:]}、Q_B=({qlow},{qhigh})，在优化之前从 B1/B6/B7 固定；矩形边界内仍可能有未观测组合。", pd.DataFrame(bounded).to_string(index=False), "解析式与数值复核相符；资源建议以边界最优为主。B1、质量参数、弱识别联合抽样、结构、QA→QB、配比和 lambda 的区间与情景保存在 CSV。"])
    finish(8, "PASS_WITH_WARNINGS", [OUT / "locked_validation/post_validation_parameters.json", OUT / "q1_interface/hp_aggregate.csv"], dest / "p8_report.md", ["优化是在简化训练算力代理和半合成质量模型下的条件建议"])
    return "PASS_WITH_WARNINGS"


def p9() -> str:
    dest = OUT / "report"
    selected = json.loads((OUT / "quality_models/selected_model.json").read_text())
    b1 = json.loads((OUT / "b1_baseline/final_parameters.json").read_text())
    p1cv = pd.read_csv(OUT / "b1_baseline/group_cv_metrics.csv")
    p2 = pd.read_csv(OUT / "loss_comparability/zero_dof_metrics.csv")
    p3 = pd.read_csv(OUT / "quality_position/amplitude_regressions.csv")
    p4 = pd.read_csv(OUT / "quality_models/candidate_model_metrics.csv")
    p5 = pd.read_csv(OUT / "locked_validation/b7_new_metrics.csv")
    validation = json.loads((OUT / "locked_validation/validation_summary.json").read_text())
    position = json.loads((OUT / "quality_position/quality_position_decision.json").read_text())
    p6 = pd.read_csv(OUT / "stress_extrapolation/b8_groupwise_direction.csv")
    b8errors = pd.read_csv(OUT / "stress_extrapolation/b8_model_errors.csv")
    hp = pd.read_csv(OUT / "q1_interface/hp_by_target.csv")
    bounded = pd.read_csv(OUT / "compute_opt/bounded_optima.csv")
    local = pd.read_csv(OUT / "compute_opt/local_effects_and_substitution.csv")
    inv = pd.read_csv(OUT / "audit/b_attachment_inventory.csv")
    report = dest / "question2_v3_experiment_report.md"
    sections = [
        ("1. 问题目标与数据边界", "B 附件无完整 (N,D,Q_B,p,L) 联合实验；本报告不声称联合机制被识别。"),
        ("2. 数据来源等级", inv[["attachment_id", "raw_rows", "provenance"]].to_string(index=False)),
        ("3. B1 经典 Scaling Law", f"L=E+A N^-alpha+B D^-beta，N/D 单位十亿。参数 {b1['parameters']}。LOSO macro RMSE={p1cv.RMSE.mean():.6g}。组 bootstrap 和残差详见 b1_baseline。"),
        ("4. 跨来源 Loss 可比性", p2.to_string(index=False) + "\nB2/B4/B5 仅趋势迁移诊断；B3 插值轨迹一致性；绝对 Loss 口径未证明一致。"),
        ("5. 质量作用位置诊断", p3.to_string(index=False) + f"\n在 B6 半合成机制内，多项稳定性条件下拒绝仅 D 通道：{position['reject_D_only_within_B6']}。组 bootstrap、局部 dL/dQ 与留一 N/D 层检查见 quality_position。"),
        ("6. 候选质量结构", p4[p4.fold == "pooled"].to_string(index=False) + "\nAIC/BIC 和 bootstrap 差见 quality_models。"),
        ("7. 参数可识别性", "400 次组 bootstrap、参数相关、rho_n/rho_d/E1 profile、rho_d/E1 权衡与 B1 anchored 敏感性已保存。若通道分解弱识别，优先用总 -dL/dQ。"),
        ("8. B7-new 锁定验证", p5.to_string(index=False) + f"\n验证类型：{validation['validation_type']}；新增 N/D 基础组 {validation['new_N_D_groups']}、新增 Q 档 {validation['new_Q_levels']}；冻结选择先于 B7-new Loss 读取。"),
        ("9. post-validation refit", "pre_validation_parameters 与 post_validation_parameters 分别保存；仅后者进入资源情景。"),
        ("10. B8 stress test", f"固定 N,D 组 Q-Loss 负向 {(p6.sign=='negative').sum()}、正向 {(p6.sign=='positive').sum()}，不翻转 Q。\n" + b8errors.to_string(index=False)),
        ("11. B9/B10", "B9 无 Loss，只提供 N/D 外推距离；B10 Loss 为估算且存在循环生成风险，仅作自洽参考。"),
        ("12. QA 与 QB", "Q_A 与 Q_B 非同尺度；线性和幂次秩保持映射仅为 scenario_assumption。"),
        ("13. 第一问配比接口", hp.to_string(index=False) + "\nh_agg 为 13 目标等权平均。"),
        ("14. 广义情景结构", f"selected={selected['model_name']} 为半合成估计；h_p 为第一问导入；P-D/P-R 与 lambda∈{{0,.5,1,1.5}} 为情景假设。"),
        ("15. 边际效应与弹性", "M_N、M_D、M_Q、∂logL/∂logN、∂logL/∂logD、∂logL/∂Q_B 在 compute_opt/local_effects_and_substitution.csv，条件工作点与假设同列保存。"),
        ("16. 局部等损失替代", "dlogN/dQ_B|L,D,p 仅在表列出的 N、D、Q_B、配比情景及 lambda 条件下成立；不是普遍替代。\n" + local.head(8).to_string(index=False)),
        ("17. 固定训练算力资源配置", "train_compute≈6ND 是数量级近似；解析和边界数值最优及不确定性见 compute_opt/uncertainty_summary.csv。\n" + bounded.to_string(index=False)),
        ("18. 局限", "QA/QB 非同尺度；跨来源 Loss 语义差异；B6/B7 半合成；B7-new 只验证新 Q 档，未增加 N/D 基础组；B8 方向冲突；rho_D/E1 可能存在参数权衡；p 未在 B 中观测；lambda 未识别；train_compute≈6ND 是近似；B9/B10 外推非独立实证。")]
    conclusions = ("## 核心结论与证据等级\n\n"
                   f"- [A_direct_real_fit] B1 经典律按规模留一 CV 的 macro RMSE={p1cv.RMSE.mean():.6g}；B1 原始残差接近 CSV 舍入量级，生成机制缺独立核验。\n"
                   "- [B_external_trend_validation] B2/B4/B5 只能比较跨来源趋势；绝对 Loss 同口径未获证明。\n"
                   f"- [C_semi_synthetic_calibration] B6 中质量仅作用 D 通道的简化假设被多种诊断一致拒绝={position['reject_D_only_within_B6']}；冻结模型 {selected['model_name']} 在 B7-new 的 RMSE={p5.iloc[0].RMSE:.6g}。\n"
                   f"- [D_stress_test] B8 固定 N/D 的 Q-Loss 组正向 {(p6.sign=='positive').sum()}，与 B6/B7 的方向相反，原样保留。\n"
                   "- [E_imported_Q1/F_scenario_assumption] 13 目标配比修正来自第一问冻结模型；QA→QB、作用位置及 lambda 未由 B 数据识别。\n")
    _save_report(report, "问题二 q2_v3 总体实验报告", [conclusions] + [f"## {title}\n\n{content}" for title, content in sections])
    key = [{"key": "B1_LOSO_macro_RMSE", "value": p1cv.RMSE.mean(), "grade": "A_direct_real_fit"},
           {"key": "selected_quality_model", "value": selected["model_name"], "grade": "C_semi_synthetic_calibration"},
           {"key": "B7_new_RMSE", "value": p5.iloc[0].RMSE, "grade": "C_semi_synthetic_calibration"},
           {"key": "B8_positive_Q_loss_groups", "value": int((p6.sign == "positive").sum()), "grade": "D_stress_test"}]
    anchor = bounded[(bounded.train_compute == 1e21) & (bounded.Q_B == bounded.Q_B.median())]
    if len(anchor):
        key.extend([{"key": "bounded_N_billions_at_1e21", "value": anchor.iloc[0].N_billions, "grade": "F_scenario_assumption"},
                    {"key": "bounded_D_billions_at_1e21", "value": anchor.iloc[0].D_billions, "grade": "F_scenario_assumption"}])
    csv(dest / "key_results_table.csv", key)
    csv(dest / "evidence_grade_table.csv", [{"grade": a, "meaning": b} for a, b in (
        ("A_direct_real_fit", "B1 公开轨迹主拟合"), ("B_external_trend_validation", "B2/B4/B5 趋势诊断"),
        ("C_semi_synthetic_calibration", "B6/B7 半合成质量关系"), ("D_stress_test", "B8 压力测试"),
        ("E_imported_Q1", "第一问冻结配比模型"), ("F_scenario_assumption", "QA→QB、p 结构、lambda"),
        ("G_extrapolation_reference", "B9/B10 估算参考"))])
    figure = OUT / "figures/b1_residual_diagnostics.png"
    source = OUT / "b1_baseline/residuals.csv"
    csv(dest / "figure_manifest.csv", [{"figure": str(figure.relative_to(OUT)), "figure_sha256": sha(figure),
                                       "source": str(source.relative_to(OUT)), "source_sha256": sha(source),
                                       "generator": "src/q2_v3/stages.py:p1", "evidence": "A_direct_real_fit"}])
    metadata = sorted(OUT.rglob("*metadata.json"))
    dump(dest / "reproducibility_summary.json", {"experiment_version": "q2_v3", "seed": SEED, "stage_metadata": {str(x.relative_to(OUT)): sha(x) for x in metadata}, "model_freeze_sha256": sha(OUT / "quality_models/selected_model.json"), "report_code_sha256": sha(ROOT / "src/q2_v3/stages.py")})
    finish(9, "PASS_WITH_WARNINGS", [OUT / "quality_models/selected_model.json", OUT / "compute_opt/bounded_optima.csv"], report, ["统计与优化结论受半合成质量数据和情景假设限制"])
    return "PASS_WITH_WARNINGS"
