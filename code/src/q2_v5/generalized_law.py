"""P08：固定质量结构并附加 Q1 配比情景。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import config, finalize, load_b, require_pass, stage_dir, write_json
from .quality_models import MODELS


def components(model: str, params: dict[str, float], n: float | np.ndarray,
               d: float | np.ndarray, q: float | np.ndarray, q0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d, q = np.asarray(n, float), np.asarray(d, float), np.asarray(q, float)
    if np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("N,D 必须严格为正")
    dq = q - q0
    nterm = params["A"] * n ** (-params["alpha"])
    dterm = params["B"] * d ** (-params["beta"])
    if "rho_N" in MODELS[model]:
        nterm = nterm * np.exp(-params["rho_N"] * dq)
    if "rho_D" in MODELS[model]:
        dterm = dterm * np.exp(-params["rho_D"] * dq)
    additive = params.get("E1", 0) * dq if "E1" in MODELS[model] else np.zeros_like(dq)
    return nterm, dterm, additive


def generalized_predict(model: str, params: dict[str, float], n: float | np.ndarray,
                        d: float | np.ndarray, q: float | np.ndarray, q0: float,
                        h: float | np.ndarray, lam: float, form: str) -> np.ndarray:
    if form not in ("A", "B"):
        raise ValueError("未知配比附加结构")
    nterm, dterm, additive = components(model, params, n, d, q, q0)
    factor = np.exp(lam * np.asarray(h, float))
    result = params["E"] + nterm + dterm * factor - additive if form == "A" else (
        params["E"] + (nterm + dterm) * factor - additive)
    if np.any(~np.isfinite(result)):
        raise ValueError("广义标度律返回非有限数值")
    return result


def run() -> dict:
    require_pass("P07")
    out = stage_dir("P08")
    out.mkdir(parents=True, exist_ok=True)
    post = json.loads((stage_dir("P05") / "post_validation_parameters.json").read_text())
    structure = json.loads((stage_dir("P07") / "form_structure_diagnostic.json").read_text())
    model, params, q0 = post["model"], post["parameters"], post["q0"]
    primary = structure["primary_structure"]
    contract = {
        "quality_model": model, "quality_parameters": params, "q0": q0,
        "quality_parameter_source": "B6_plus_B7_new_post_validation_fit",
        "mixture_function": "h_p_v=log(f_Q1_v(p)/f_Q1_v(p0)); aggregate equal 13-target weight",
        "mixture_reference": "equal-share p0, scenario assumption because Q1 numeric p0 not saved",
        "primary_form": primary, "alternative_form": "B" if primary == "A" else "A",
        "lambda_p": {"values": config()["lambda_scenarios"], "status": "NOT_IDENTIFIED; scenario_assumption"},
        "QA_to_QB": "NOT_IDENTIFIED; three scenario mappings",
        "parameter_categories": {"B_data_estimated": list(params),
                                 "Q1_imported": ["h_p_v", "Q_A_domain"],
                                 "scenario_unidentified": ["lambda_p", "QA_to_QB_mapping", "form_attachment", "p0"]},
        "units": {"N": "billion parameters", "D": "billion tokens", "Q_B": "B6/B7 score"},
        "evidence_type": ["semi_synthetic", "q1_imported", "scenario_assumption"],
    }
    write_json(out / "model_contract.json", contract)
    b = pd.concat([load_b("B6"), pd.read_csv(stage_dir("P01") / "b7_new.csv")], ignore_index=True)
    maps = pd.read_csv(stage_dir("P07") / "qa_qb_mapping_scenarios.csv")
    hp_values = pd.read_csv(stage_dir("P07") / "hp_aggregate.csv")
    chosen_hp = float(hp_values.loc[hp_values.scenario != "p0", "h_p_aggregate_equal_weight"].iloc[0])
    mid = b.iloc[(b.N_params_B - b.N_params_B.median()).abs().argsort()[:1]].iloc[0]
    n, d = float(mid.N_params_B), float(mid.D_tokens_B)
    records = []
    for _, row in maps.iterrows():
        for lam in config()["lambda_scenarios"]:
            for form in ("A", "B"):
                value = float(generalized_predict(model, params, n, d, row.Q_B_scenario,
                                                  q0, chosen_hp, lam, form))
                records.append({"domain": row.domain, "mapping": row.mapping, "lambda_p": lam,
                                "form": form, "N_params_B": n, "D_tokens_B": d,
                                "Q_B_scenario": row.Q_B_scenario, "h_p_aggregate": chosen_hp,
                                "predicted_loss": value, "evidence_type": "scenario_assumption"})
    pd.DataFrame(records).to_csv(out / "scenario_predictions.csv", index=False)
    pd.DataFrame([{"parameter": key, "value": value, "source": "B6+B7-new post-validation fit",
                   "evidence_type": "semi_synthetic"} for key, value in params.items()]).to_csv(
                       out / "parameter_table.csv", index=False)
    (out / "report.md").write_text(
        "# P08 广义标度律合同\n\n"
        f"实际选择 {model}，主配比结构 Form {primary}。"
        "B 数据估计 E,A,B,alpha,beta 和可用质量系数；Q1 导入 h_p；"
        "lambda、QA→QB 映射、p0 及 Form 附加位置均为未识别情景。"
        "在 Q=Q0、p=p0 时退化为经典 N-D 形式。\n", encoding="utf-8")
    metadata = finalize("P08", [stage_dir("P05") / "post_validation_parameters.json",
                        stage_dir("P07") / "form_structure_diagnostic.json",
                        stage_dir("P07") / "hp_aggregate.csv",
                        stage_dir("P07") / "qa_qb_mapping_scenarios.csv"],
                        ["配比参考点、映射、lambda 与 Form 附加结构均未识别"],
                        ["最终模型是跨数据源结构拼接，不是联合观测直接估计"],
                        ["semi_synthetic", "q1_imported", "scenario_assumption"])
    return {"contract": contract, "metadata": metadata}
