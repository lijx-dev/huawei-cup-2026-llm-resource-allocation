"""P09/P10：解析边际效应、弹性与局部等 Loss 替代。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import finalize, load_b, require_pass, stage_dir
from .generalized_law import components, generalized_predict


def derivatives(model: str, params: dict[str, float], n: float, d: float,
                q: float, q0: float, h: float, lam: float, form: str) -> dict[str, float]:
    nterm, dterm, _ = components(model, params, n, d, q, q0)
    nterm, dterm = float(nterm), float(dterm)
    factor = float(np.exp(lam * h))
    rho_n, rho_d, e1 = (params.get(name, 0.0) for name in ("rho_N", "rho_D", "E1"))
    if form == "A":
        dn = -params["alpha"] * nterm / n
        dd = -params["beta"] * dterm * factor / d
        dq = -rho_n * nterm - rho_d * dterm * factor - e1
    elif form == "B":
        dn = -params["alpha"] * nterm * factor / n
        dd = -params["beta"] * dterm * factor / d
        dq = -(rho_n * nterm + rho_d * dterm) * factor - e1
    else:
        raise ValueError("未知 Form")
    return {"dL_dN": dn, "dL_dD": dd, "dL_dQ": dq}


def finite_derivative(model: str, params: dict[str, float], n: float, d: float,
                      q: float, q0: float, h: float, lam: float, form: str,
                      variable: str) -> float:
    args = {"n": n, "d": d, "q": q}
    step = 1e-5 * (args[variable] if variable != "q" else 1.0)
    plus, minus = dict(args), dict(args)
    plus[variable] += step
    minus[variable] -= step
    return float((generalized_predict(model, params, **plus, q0=q0, h=h, lam=lam, form=form)
                  - generalized_predict(model, params, **minus, q0=q0, h=h, lam=lam, form=form))
                 / (2 * step))


def representative_points(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = np.column_stack([np.log(frame.N_params_B), np.log(frame.D_tokens_B), frame.Q_score])
    center = np.median(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0), 1e-9)
    rows = []
    for label, quantile in (("low", .1), ("mid", .5), ("high", .9)):
        target = np.quantile(matrix, quantile, axis=0)
        idx = int(np.argmin(np.sum(((matrix - target) / scale)**2, axis=1)))
        sample = frame.iloc[idx]
        rows.append({"workpoint": label, "N_params_B": float(sample.N_params_B),
                     "D_tokens_B": float(sample.D_tokens_B), "Q_B": float(sample.Q_score),
                     "observed_experiment_id": sample.experiment_id,
                     "support_status": "observed_B6_or_B7_new",
                     "evidence_type": "semi_synthetic"})
    return pd.DataFrame(rows)


def run_p09() -> dict:
    require_pass("P08")
    out = stage_dir("P09")
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    model, p, q0, form = (contract[k] for k in ("quality_model", "quality_parameters", "q0", "primary_form"))
    frame = pd.concat([load_b("B6"), pd.read_csv(stage_dir("P01") / "b7_new.csv")], ignore_index=True)
    workpoints = representative_points(frame)
    h_table = pd.read_csv(stage_dir("P07") / "hp_aggregate.csv")
    h_scenario = float(h_table.loc[h_table.scenario != "p0", "h_p_aggregate_equal_weight"].iloc[0])
    rows, checks = [], []
    for _, row in workpoints.iterrows():
        for p_label, h in (("p0", 0.0), ("one_pair_transfer_0.01", h_scenario)):
            for lam in (0., 0.5, 1., 1.5):
                n, d, q = row.N_params_B, row.D_tokens_B, row.Q_B
                loss = float(generalized_predict(model, p, n, d, q, q0, h, lam, form))
                deriv = derivatives(model, p, n, d, q, q0, h, lam, form)
                entry = {"workpoint": row.workpoint, "N_params_B": n, "D_tokens_B": d,
                         "Q_B": q, "p_scenario": p_label, "h_p": h, "lambda_p": lam,
                         "mapping": "direct_QB_no_QA_mapping", "support_status": row.support_status,
                         "evidence_type": "semi_synthetic_plus_scenario", "predicted_loss": loss,
                         "M_N": -deriv["dL_dN"], "M_D": -deriv["dL_dD"],
                         "M_Q": -deriv["dL_dQ"], **deriv,
                         "epsilon_N": deriv["dL_dN"] * n / loss,
                         "epsilon_D": deriv["dL_dD"] * d / loss,
                         "dlogL_dQ": deriv["dL_dQ"] / loss}
                rows.append(entry)
                for var, name in (("n", "dL_dN"), ("d", "dL_dD"), ("q", "dL_dQ")):
                    finite = finite_derivative(model, p, n, d, q, q0, h, lam, form, var)
                    relative = abs(finite - deriv[name]) / max(abs(finite), abs(deriv[name]), 1e-10)
                    checks.append({"workpoint": row.workpoint, "p_scenario": p_label,
                                   "lambda_p": lam, "variable": var,
                                   "analytic": deriv[name], "finite_difference": finite,
                                   "relative_error": relative})
    checked = pd.DataFrame(checks)
    if checked.relative_error.max() > 1e-5:
        raise RuntimeError(f"解析导数与有限差分不一致: {checked.relative_error.max()}")
    result = pd.DataFrame(rows)
    result[["workpoint", "N_params_B", "D_tokens_B", "Q_B", "p_scenario", "lambda_p",
            "mapping", "support_status", "evidence_type"]].to_csv(out / "workpoints.csv", index=False)
    result[["workpoint", "N_params_B", "D_tokens_B", "Q_B", "p_scenario", "lambda_p",
            "M_N", "M_D", "M_Q", "dL_dN", "dL_dD", "dL_dQ", "predicted_loss",
            "mapping", "support_status", "evidence_type"]].to_csv(out / "marginal_effects.csv", index=False)
    result[["workpoint", "p_scenario", "lambda_p", "epsilon_N", "epsilon_D",
            "dlogL_dQ", "evidence_type"]].to_csv(out / "elasticities.csv", index=False)
    checked.to_csv(out / "derivative_checks.csv", index=False)
    (out / "p09_report.md").write_text(
        "# P09 解析边际效应\n\n"
        f"在 B6+B7-new 支撑域自动选 low/mid/high 三个观测工作点。"
        f"解析导数与中心差分最大相对误差 {checked.relative_error.max():.3g}。"
        "N/D 弹性及质量半弹性均按当前工作点计算；配比和 lambda 为情景。\n",
        encoding="utf-8")
    metadata = finalize("P09", [stage_dir("P08") / "model_contract.json",
                        stage_dir("P07") / "hp_aggregate.csv", stage_dir("P01") / "b7_new.csv"],
                        [], ["工作点来自半合成 B6/B7；配比扰动和 lambda 为情景"],
                        ["semi_synthetic", "scenario_assumption"])
    return {"max_derivative_error": float(checked.relative_error.max()), "metadata": metadata}


def run_p10() -> dict:
    require_pass("P09")
    out = stage_dir("P10")
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    model, p, q0, form = (contract[k] for k in ("quality_model", "quality_parameters", "q0", "primary_form"))
    marginal = pd.read_csv(out / "marginal_effects.csv")
    base_rows = marginal.loc[(marginal.p_scenario == "p0") & (marginal.lambda_p == 0)]
    n_rows, d_rows, nonlin = [], [], []
    for _, row in base_rows.iterrows():
        n, d, q = row.N_params_B, row.D_tokens_B, row.Q_B
        ln, ld, lq = row.dL_dN, row.dL_dD, row.dL_dQ
        if abs(ln) < 1e-12 or abs(ld) < 1e-12:
            raise RuntimeError("局部替代分母过小")
        dn_dq, dd_dq = -lq / ln, -lq / ld
        n_rows.append({"workpoint": row.workpoint, "dN_dQ": dn_dq,
                       "dlogN_dQ": dn_dq / n, "delta_Q": .1,
                       "first_order_delta_N": dn_dq * .1,
                       "evidence_type": "local_first_order_scenario"})
        d_rows.append({"workpoint": row.workpoint, "dD_dQ": dd_dq,
                       "dlogD_dQ": dd_dq / d, "delta_Q": .1,
                       "first_order_delta_D": dd_dq * .1,
                       "evidence_type": "local_first_order_scenario"})
        baseline = float(generalized_predict(model, p, n, d, q, q0, 0, 0, form))
        for delta in (.01, .05, .1):
            for variable, rate in (("N", dn_dq), ("D", dd_dq)):
                new_value = (n if variable == "N" else d) + rate * delta
                feasible = new_value > 0
                changed = float(generalized_predict(model, p,
                                new_value if variable == "N" else n,
                                new_value if variable == "D" else d,
                                q + delta, q0, 0, 0, form)) if feasible else np.nan
                nonlin.append({"workpoint": row.workpoint, "variable": variable,
                               "delta_Q": delta, "first_order_new_value": new_value,
                               "feasible": feasible,
                               "local_linearization_error": changed - baseline if feasible else np.nan,
                               "relative_loss_error": (changed - baseline) / baseline if feasible else np.nan,
                               "evidence_type": "local_first_order_scenario"})
    pd.DataFrame(n_rows).to_csv(out / "q_n_substitution.csv", index=False)
    pd.DataFrame(d_rows).to_csv(out / "q_d_substitution.csv", index=False)
    pd.DataFrame(nonlin).to_csv(out / "substitution_nonlinearity.csv", index=False)
    (out / "p10_report.md").write_text(
        "# P10 局部等 Loss 替代\n\n"
        "以 dN/dQ=-L_Q/L_N、dD/dQ=-L_Q/L_D 计算 low/mid/high 工作点，"
        "并将 ΔQ=0.01/0.05/0.1 的一阶变动重新代入完整模型，保留非线性误差。"
        "结果仅是局部等损失近似。\n", encoding="utf-8")
    metadata = finalize("P10", [out / "marginal_effects.csv",
                        stage_dir("P08") / "model_contract.json"],
                        [], ["固定另一规模维度、配比和当前工作点"], ["scenario_assumption"])
    return {"rows": len(nonlin), "metadata": metadata}
