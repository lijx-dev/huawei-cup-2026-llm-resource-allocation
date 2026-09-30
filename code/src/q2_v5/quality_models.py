"""P04：仅以 B6 选择质量作用结构并在 B7 前冻结。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

from .baseline import PARAMS, fit as baseline_fit
from .common import (config, finalize, load_b, metrics, paths_for,
                     require_pass, stage_dir, write_json)


MODELS = {"M0": (), "MD": ("rho_D",), "MN": ("rho_N",),
          "MND": ("rho_N", "rho_D"),
          "MCHANNEL_ADD": ("rho_N", "rho_D", "E1")}
QUALITY_PARAMS = ("rho_N", "rho_D", "E1")


def train_loader(stage: str) -> pd.DataFrame:
    if stage != "P04":
        raise PermissionError("B6 quality selection loader 仅允许 P04")
    return load_b("B6")


def quality_predict(model: str, params: dict[str, float], n: np.ndarray,
                    d: np.ndarray, q: np.ndarray, q0: float) -> np.ndarray:
    n, d, q = np.asarray(n, float), np.asarray(d, float), np.asarray(q, float)
    if np.any(n <= 0) or np.any(d <= 0) or np.any(~np.isfinite(q)):
        raise ValueError("N、D、Q 输入非法")
    if model not in MODELS:
        raise ValueError(f"未知模型: {model}")
    dq = q - q0
    n_term = params["A"] * n ** (-params["alpha"])
    d_term = params["B"] * d ** (-params["beta"])
    if "rho_N" in MODELS[model]:
        n_term = n_term * np.exp(-params["rho_N"] * dq)
    if "rho_D" in MODELS[model]:
        d_term = d_term * np.exp(-params["rho_D"] * dq)
    value = params["E"] + n_term + d_term
    if "E1" in MODELS[model]:
        value = value - params["E1"] * dq
    return value


@dataclass
class QualityFit:
    parameters: dict[str, float]
    rss: float
    runs: pd.DataFrame


def fit_quality(model: str, frame: pd.DataFrame, q0: float, *,
                anchored: dict[str, float] | None = None,
                starts: int = 4) -> QualityFit:
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    q = frame.Q_score.to_numpy(float)
    y = frame.val_loss.to_numpy(float)
    if not (np.isfinite(np.column_stack([n, d, q, y])).all()
            and (n > 0).all() and (d > 0).all() and (y > 0).all()):
        raise ValueError("B6 含非法建模输入")
    active = MODELS[model]
    if anchored is None:
        base = baseline_fit(n, d, y, starts=3).parameters
        free = list(PARAMS) + list(active)
    else:
        base = anchored
        free = list(active)
        if not free:
            residual = quality_predict(model, base, n, d, q, q0) - y
            return QualityFit(dict(base), float(np.sum(residual**2)),
                              pd.DataFrame([{"start": 0, "success": True,
                                             "rss": float(np.sum(residual**2)), "nfev": 0}]))
    lower = {"E": 1e-9, "A": 1e-8, "B": 1e-8, "alpha": 1e-3,
             "beta": 1e-3, "rho_N": -10, "rho_D": -10, "E1": -10}
    upper = {"E": 20, "A": 1e6, "B": 1e6, "alpha": 3,
             "beta": 3, "rho_N": 10, "rho_D": 10, "E1": 10}
    log_names = set(PARAMS)
    lo = [np.log(lower[name]) if name in log_names else lower[name] for name in free]
    hi = [np.log(upper[name]) if name in log_names else upper[name] for name in free]

    def decode(z: np.ndarray) -> dict[str, float]:
        p = dict(base)
        for name, value in zip(free, z):
            p[name] = float(np.exp(value) if name in log_names else value)
        return p

    def encode(p: dict[str, float]) -> np.ndarray:
        return np.array([np.log(p[name]) if name in log_names else p[name] for name in free])

    rows, solutions = [], []
    for index, signed_init in enumerate((0.0, 0.2, -0.2, 0.7)[:starts]):
        initial = dict(base)
        initial.update({name: signed_init for name in active})
        z0 = np.clip(encode(initial), np.array(lo) + 1e-8, np.array(hi) - 1e-8)
        result = least_squares(lambda z: quality_predict(model, decode(z), n, d, q, q0) - y,
                               z0, bounds=(lo, hi), max_nfev=2500,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        rss = float(np.sum(result.fun**2))
        rows.append({"model": model, "start": index, "initial_quality": signed_init,
                     "rss": rss, "success": bool(result.success), "nfev": int(result.nfev)})
        if result.success and np.isfinite(rss):
            solutions.append((rss, decode(result.x)))
    if not solutions:
        raise RuntimeError(f"{model} 多起点未收敛")
    rss, params = min(solutions, key=lambda item: item[0])
    return QualityFit(params, rss, pd.DataFrame(rows))


def group_ids(frame: pd.DataFrame) -> np.ndarray:
    # 同一基础 N,D 的所有 Q 水平使用相同整数 group，不看 Loss。
    return pd.factorize(pd.MultiIndex.from_frame(frame[["N_params_B", "D_tokens_B"]]))[0]


def freeze_digest(out) -> str:
    names = ("model_selection_rule.json", "selection_decision.json",
             "pre_validation_parameters.json", "candidate_model_metrics.csv",
             "pre_validation_m0_parameters.json", "information_criteria.csv")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((out / name).read_bytes())
    return digest.hexdigest()


def run() -> dict:
    require_pass("P03")
    out = stage_dir("P04")
    out.mkdir(parents=True, exist_ok=True)
    cfg = config()
    # 选择规则先写入磁盘；以下代码路径只调用 B6 loader。
    rule = {**cfg["model_selection"], "candidate_order": list(MODELS),
            "folds": 5, "selection_formula":
            "choose fewest parameters among candidates with mean fold RMSE <= best mean fold RMSE + SEM(best folds); tie lexicographic"}
    write_json(out / "model_selection_rule.json", rule)
    frame = train_loader("P04")
    valid = np.isfinite(frame[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_numpy(float)).all(axis=1)
    valid &= (frame[["N_params_B", "D_tokens_B", "val_loss"]].to_numpy(float) > 0).all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    q0 = float(frame.Q_score.median())
    groups = group_ids(frame)
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(frame, groups=groups))
    rows, criteria, oof_rows, fit_rows, full_fits = [], [], [], [], {}
    for model in MODELS:
        oof = np.full(len(frame), np.nan)
        fold_rmse = []
        for fold, (train, test) in enumerate(folds):
            fitted = fit_quality(model, frame.iloc[train], q0, starts=3)
            pred = quality_predict(model, fitted.parameters,
                                   frame.N_params_B.to_numpy(float)[test],
                                   frame.D_tokens_B.to_numpy(float)[test],
                                   frame.Q_score.to_numpy(float)[test], q0)
            oof[test] = pred
            fold_rmse.append(metrics(frame.val_loss.to_numpy(float)[test], pred)["rmse"])
            oof_rows.extend({"model": model, "fold": fold,
                             "experiment_id": frame.at[int(index), "experiment_id"],
                             "group": int(groups[index]), "observed": float(frame.at[int(index), "val_loss"]),
                             "prediction": float(value), "evidence_type": "semi_synthetic"}
                            for index, value in zip(test, pred))
        if not np.isfinite(oof).all():
            raise RuntimeError(f"{model} OOF 预测不完整")
        m = metrics(frame.val_loss, oof)
        rows.append({"model": model, "n_parameters": len(PARAMS) + len(MODELS[model]),
                     "macro_fold_rmse": float(np.mean(fold_rmse)),
                     "fold_rmse_sem": float(np.std(fold_rmse, ddof=1) / np.sqrt(len(fold_rmse))),
                     "fold_rmse_json": json.dumps(fold_rmse), **m})
        full = fit_quality(model, frame, q0)
        full_fits[model] = full
        fit_rows.append(full.runs)
        k = len(PARAMS) + len(MODELS[model])
        criteria.append({"model": model, "rss": full.rss,
                         "aic": float(len(frame) * np.log(max(full.rss / len(frame), 1e-300)) + 2 * k),
                         "bic": float(len(frame) * np.log(max(full.rss / len(frame), 1e-300)) + k * np.log(len(frame))),
                         "parameters": k})
    candidate = pd.DataFrame(rows)
    candidate.to_csv(out / "candidate_model_metrics.csv", index=False)
    pd.DataFrame(criteria).to_csv(out / "information_criteria.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    pd.concat(fit_rows).to_csv(out / "multistart_runs.csv", index=False)
    best = candidate.loc[candidate.macro_fold_rmse.idxmin()]
    threshold = float(best.macro_fold_rmse + best.fold_rmse_sem)
    eligible = candidate.loc[candidate.macro_fold_rmse <= threshold + 1e-12]
    selected = eligible.sort_values(["n_parameters", "model"]).iloc[0]["model"]
    decision = {"selected_model": selected, "best_macro_fold_rmse_model": best.model,
                "threshold": threshold, "eligible_models": eligible.model.tolist(),
                "q0": q0, "rule": rule["selection_formula"],
                "B7_loss_accessed": False}
    write_json(out / "selection_decision.json", decision)
    write_json(out / "pre_validation_parameters.json", {
        "selected_model": selected, "parameters": full_fits[selected].parameters,
        "q0": q0, "fitted_on": "B6_only", "evidence_type": "semi_synthetic"})
    write_json(out / "pre_validation_m0_parameters.json", {
        "model": "M0", "parameters": full_fits["M0"].parameters,
        "q0": q0, "fitted_on": "B6_only", "evidence_type": "semi_synthetic"})
    # OOF paired group bootstrap，比较每个候选对 M0 的误差差。
    oof_frame = pd.DataFrame(oof_rows)
    pivot = oof_frame.pivot(index="experiment_id", columns="model", values="prediction")
    observed = frame.set_index("experiment_id").loc[pivot.index, "val_loss"].to_numpy(float)
    gp = frame.set_index("experiment_id").loc[pivot.index]
    group_pivot = group_ids(gp)
    rng = np.random.default_rng(cfg["seed"])
    unique_groups = np.unique(group_pivot)
    diffs = []
    for iteration in range(cfg["validation_bootstrap_resamples"]):
        sample = rng.choice(unique_groups, len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(group_pivot == g) for g in sample])
        base_rmse = metrics(observed[idx], pivot["M0"].to_numpy()[idx])["rmse"]
        for model in MODELS:
            delta = metrics(observed[idx], pivot[model].to_numpy()[idx])["rmse"] - base_rmse
            diffs.append({"resample": iteration, "model": model,
                          "rmse_minus_M0": delta})
    pd.DataFrame(diffs).to_csv(out / "bootstrap_model_differences.csv", index=False)
    # 仅对冻结入选模型做组 bootstrap 参数区间；失败明示而不补造。
    boot = []
    for iteration in range(cfg["quality_bootstrap_resamples"]):
        sample = rng.choice(np.unique(groups), len(np.unique(groups)), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sample])
        try:
            result = fit_quality(selected, frame.iloc[idx], q0, starts=2)
            boot.append({"resample": iteration, "success": True, **result.parameters})
        except (ValueError, RuntimeError):
            boot.append({"resample": iteration, "success": False})
    boot_frame = pd.DataFrame(boot)
    boot_frame.to_csv(out / "parameter_bootstrap.csv", index=False)
    numeric = boot_frame.select_dtypes(include="number").drop(columns=["resample"], errors="ignore")
    numeric.corr().to_csv(out / "parameter_correlation.csv")
    ci = pd.DataFrame([{"parameter": name, "median": numeric[name].median(),
                        "q025": numeric[name].quantile(.025), "q975": numeric[name].quantile(.975)}
                       for name in full_fits[selected].parameters])
    ci.to_csv(out / "parameter_ci.csv", index=False)
    # 固定其他参数的一维 RSS 剖面：属于诊断，不声称 profile-likelihood CI。
    profiles = []
    p = full_fits[selected].parameters
    for name in MODELS[selected]:
        for value in np.linspace(p[name] - 1, p[name] + 1, 21):
            trial = dict(p)
            trial[name] = float(value)
            resid = quality_predict(selected, trial, frame.N_params_B, frame.D_tokens_B,
                                    frame.Q_score, q0) - frame.val_loss.to_numpy(float)
            profiles.append({"parameter": name, "value": value, "rss_fixed_others": float(np.sum(resid**2))})
    pd.DataFrame(profiles, columns=["parameter", "value", "rss_fixed_others"]).to_csv(out / "parameter_profile.csv", index=False)
    b1 = json.loads((stage_dir("P02") / "final_parameters.json").read_text())["parameters"]
    anchored = fit_quality(selected, frame, q0, anchored=b1)
    write_json(out / "anchored_sensitivity.json", {"model": selected,
                "parameters": anchored.parameters, "rss": anchored.rss,
                "role": "B1_anchored_sensitivity_only"})
    sign = {name: ("positive" if p[name] > 0 else "negative" if p[name] < 0 else "zero")
            for name in MODELS[selected]}
    write_json(out / "sign_diagnostics.json", sign)
    (out / "model_freeze.sha256").write_text(freeze_digest(out) + "\n", encoding="ascii")
    (out / "report.md").write_text(
        "# P04 B6 质量模型选择与冻结\n\n"
        f"Q0={q0:.6g}；按相同 N,D 分组五折交叉验证。预注册 1-SE 简约规则选择 {selected}。"
        f"最优 fold macro RMSE={best.macro_fold_rmse:.6g}，门限={threshold:.6g}。"
        "候选的 AIC/BIC、组 bootstrap、参数相关和固定其余参数的剖面已保存。"
        "本阶段不读取 B7 Loss。符号诊断不等同因果作用。\n", encoding="utf-8")
    metadata = finalize("P04", paths_for("B6") + [stage_dir("P02") / "final_parameters.json",
                        stage_dir("P03") / "p03_metadata.json"],
                        [f"组 bootstrap 成功 {int(boot_frame.success.sum())}/{len(boot_frame)}",
                         "固定其他参数的 RSS 剖面不是正式 profile-likelihood CI"],
                        ["1-SE 规则和候选集在读取 B7 Loss 前固定"], ["semi_synthetic"])
    return {"decision": decision, "metadata": metadata}
