"""P02：B1 经典标度律与按模型规模分组验证。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .common import config, finalize, load_b, metrics, paths_for, stage_dir, write_json


PARAMS = ("E", "A", "B", "alpha", "beta")
LOWER = np.log([1e-9, 1e-8, 1e-8, 1e-3, 1e-3])
UPPER = np.log([20, 1e6, 1e6, 3, 3])


def predict(params: dict[str, float] | np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    p = np.array([params[name] for name in PARAMS]) if isinstance(params, dict) else np.asarray(params)
    n, d = np.asarray(n, float), np.asarray(d, float)
    if np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("N,D 必须为正")
    return p[0] + p[1] * n ** (-p[3]) + p[2] * d ** (-p[4])


@dataclass
class Fit:
    parameters: dict[str, float]
    runs: pd.DataFrame
    converged: bool


def fit(n: np.ndarray, d: np.ndarray, y: np.ndarray, *, starts: int = 6) -> Fit:
    n, d, y = np.asarray(n, float), np.asarray(d, float), np.asarray(y, float)
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Loss 必须为正且有限")
    if len(y) < 6 or len(set(n)) < 2 or len(set(d)) < 2:
        raise ValueError("数据支撑不足")
    floor = max(1e-5, min(float(y.min() * 0.7), 10.0))
    base = [
        [floor, 1, 1, 0.3, 0.3],
        [max(1e-5, y.min() * 0.5), 2, 1, 0.5, 0.2],
        [max(1e-5, y.min() * 0.3), 1, 2, 0.2, 0.5],
        [max(1e-5, y.min() * 0.9), 0.5, 0.5, 0.1, 0.1],
        [max(1e-5, y.min() * 0.6), 5, 5, 0.8, 0.8],
        [max(1e-5, y.min() * 0.8), 1, 3, 0.4, 0.4],
    ]
    rows, solutions = [], []
    for index, values in enumerate(base[:starts]):
        initial = np.clip(np.log(values), LOWER + 1e-9, UPPER - 1e-9)
        result = least_squares(lambda z: predict(np.exp(z), n, d) - y,
                               initial, bounds=(LOWER, UPPER), max_nfev=3000,
                               xtol=1e-10, ftol=1e-10, gtol=1e-10)
        rss = float(np.sum(result.fun**2))
        row = {"start": index, "initial": json.dumps(values), "rss": rss,
               "success": bool(result.success), "nfev": int(result.nfev),
               "message": str(result.message)}
        row.update({name: float(value) for name, value in zip(PARAMS, np.exp(result.x))})
        rows.append(row)
        if result.success and np.isfinite(rss):
            solutions.append((rss, result.x))
    if not solutions:
        raise RuntimeError("B1 多起点均未收敛")
    _, best = min(solutions, key=lambda item: item[0])
    return Fit(dict(zip(PARAMS, map(float, np.exp(best)))), pd.DataFrame(rows), True)


def valid_b1() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_b("B1")
    cols = ["N_params_B", "D_tokens_B", "val_loss"]
    good = np.isfinite(frame[cols].to_numpy(float)).all(axis=1) & (frame[cols].to_numpy(float) > 0).all(axis=1)
    return frame.loc[good].copy(), frame.loc[~good].copy()


def run() -> dict:
    from .common import require_pass

    require_pass("P01")
    out = stage_dir("P02")
    out.mkdir(parents=True, exist_ok=True)
    frame, invalid = valid_b1()
    if len(frame.N_params_B.unique()) < 3:
        raise RuntimeError("B1 模型规模不足，不能执行 LOSO")
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    y = frame.val_loss.to_numpy(float)
    fitted = fit(n, d, y)
    full_pred = predict(fitted.parameters, n, d)
    write_json(out / "final_parameters.json", {"parameters": fitted.parameters,
                                                "units": {"N": "billion parameters", "D": "billion tokens"},
                                                "fit_rows": len(frame), "evidence_type": "direct_observation"})
    fitted.runs.to_csv(out / "multistart_runs.csv", index=False)
    fold_rows, oof = [], frame[["run_id", "N_params_B", "D_tokens_B", "val_loss", "_source_row"]].copy()
    oof["prediction"] = np.nan
    for scale in sorted(frame.N_params_B.unique()):
        test = n == scale
        model = fit(n[~test], d[~test], y[~test])
        pred = predict(model.parameters, n[test], d[test])
        oof.loc[test, "prediction"] = pred
        fold_rows.append({"held_out_N_params_B": float(scale), **metrics(y[test], pred)})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "group_cv_metrics.csv", index=False)
    oof["residual"] = oof.prediction - oof.val_loss
    oof["evidence_type"] = "direct_observation"
    oof.to_csv(out / "oof_predictions.csv", index=False)
    pooled = metrics(y, oof.prediction.to_numpy(float))
    macro = {"rmse_macro": float(folds.rmse.mean()), "fold_count": len(folds),
             "definition": "arithmetic mean of held-out-scale RMSE"}
    write_json(out / "macro_metrics.json", macro)
    write_json(out / "pooled_metrics.json", pooled)
    rng = np.random.default_rng(config()["seed"])
    scales = np.sort(frame.N_params_B.unique())
    boot = []
    for iteration in range(config()["b1_bootstrap_resamples"]):
        sampled = rng.choice(scales, len(scales), replace=True)
        indices = np.concatenate([np.flatnonzero(n == scale) for scale in sampled])
        try:
            result = fit(n[indices], d[indices], y[indices], starts=3)
            boot.append({"resample": iteration, "success": True, **result.parameters})
        except (RuntimeError, ValueError):
            boot.append({"resample": iteration, "success": False,
                         **{name: np.nan for name in PARAMS}})
    boots = pd.DataFrame(boot)
    boots.to_csv(out / "bootstrap_parameters.csv", index=False)
    if boots.success.mean() < 0.7:
        raise RuntimeError("B1 group bootstrap 收敛率低于 70%")
    ci = pd.DataFrame([{"parameter": name, "median": boots[name].median(),
                        "q025": boots[name].quantile(0.025), "q975": boots[name].quantile(0.975)}
                       for name in PARAMS])
    ci.to_csv(out / "parameter_ci.csv", index=False)
    residuals = frame[["run_id", "N_params_B", "D_tokens_B", "val_loss"]].copy()
    residuals["fitted"] = full_pred
    residuals["residual"] = y - full_pred
    residuals["log_N"] = np.log(n)
    residuals["log_D"] = np.log(d)
    residuals["evidence_type"] = "direct_observation"
    residuals.to_csv(out / "residuals.csv", index=False)
    (out / "report.md").write_text(
        "# P02 B1 经典标度律\n\n"
        f"有效记录 {len(frame)}，排除 {len(invalid)}；模型规模 {len(scales)}。\n\n"
        f"Macro LOSO RMSE = {macro['rmse_macro']:.6g}；pooled OOF RMSE = {pooled['rmse']:.6g}。"
        "前者对每个规模等权，后者对每个检查点等权，因此通常不同。\n\n"
        f"Group bootstrap 成功 {int(boots.success.sum())}/{len(boots)}。"
        "参数区间为按模型规模重抽样的经验分位数。\n", encoding="utf-8")
    metadata = finalize("P02", paths_for("B1") + [stage_dir("P01") / "p01_metadata.json"],
                        [f"B1 排除非法行 {len(invalid)}"],
                        ["N/D 输入分别使用十亿参数、十亿 token 单位"], ["direct_observation"])
    return {"macro": macro, "pooled": pooled, "parameters": fitted.parameters,
            "metadata": metadata}
