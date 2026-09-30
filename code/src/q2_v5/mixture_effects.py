"""P11：冻结 Q1 模型上的 simplex 转移与局部联合响应。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import finalize, require_pass, stage_dir
from .generalized_law import generalized_predict
from .q1_bridge import frozen_dir, hp, load_frozen_model


def transfer(p0: np.ndarray, receiver: int, donor: int, delta: float) -> np.ndarray:
    p = np.asarray(p0, float).copy()
    if receiver == donor or delta < 0 or p[donor] < delta - 1e-12:
        raise ValueError("不可行的 simplex 转移")
    p[receiver] += delta
    p[donor] -= delta
    if np.any(p < -1e-12) or not np.isclose(p.sum(), 1, atol=1e-12):
        raise ValueError("扰动后离开 simplex")
    return p


def run() -> dict:
    require_pass("P10")
    out = stage_dir("P11")
    out.mkdir(parents=True, exist_ok=True)
    artifact = load_frozen_model()
    reference = json.loads((frozen_dir() / "reference_mixture.json").read_text())
    p0 = np.array(reference["values"], float)
    fields = artifact["mix_fields"]
    targets = artifact["loss_fields"]
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    work = pd.read_csv(stage_dir("P09") / "workpoints.csv")
    mid = work.loc[(work.workpoint == "mid") & (work.p_scenario == "p0")].iloc[0]
    n, d, q = mid.N_params_B, mid.D_tokens_B, mid.Q_B
    model, params, q0, form = (contract[k] for k in
                                ("quality_model", "quality_parameters", "q0", "primary_form"))
    lam = 1.0  # 明确的 unit-transfer 情景，不是拟合值。
    baseline = float(generalized_predict(model, params, n, d, q, q0, 0, lam, form))
    pair_rows, aggregate_rows, delta_rows = [], [], []
    deltas = [0.005, 0.02]
    eps = min(1e-4, p0.min() / 100)
    for j in range(17):
        for k in range(17):
            if j == k:
                continue
            # 中心差分可用反向转移；若树模型局部平台，斜率可能严格为零。
            plus, minus = transfer(p0, j, k, eps), transfer(p0, k, j, eps)
            hplus, hminus = hp(np.vstack([plus, minus]), p0, artifact)
            slopes = []
            for t in range(13):
                lplus = float(generalized_predict(model, params, n, d, q, q0, hplus[t], lam, form))
                lminus = float(generalized_predict(model, params, n, d, q, q0, hminus[t], lam, form))
                slope = (lplus - lminus) / (2 * eps)
                slopes.append(slope)
                pair_rows.append({"receiver": fields[j], "donor": fields[k], "target": targets[t],
                                  "derivative_at_zero_centered": slope, "epsilon": eps,
                                  "lambda_p": lam, "support_status": "unverifiable_from_frozen_Q1",
                                  "evidence_type": "q1_imported_plus_scenario"})
            aggregate_rows.append({"receiver": fields[j], "donor": fields[k],
                                   "mean_target_slope": float(np.mean(slopes)),
                                   "epsilon": eps, "lambda_p": lam,
                                   "evidence_type": "q1_imported_plus_scenario"})
            for delta in deltas:
                point = transfer(p0, j, k, delta)
                values = hp(point, p0, artifact)[0]
                agg_loss = float(generalized_predict(model, params, n, d, q, q0,
                                                     values.mean(), lam, form))
                delta_rows.append({"receiver": fields[j], "donor": fields[k],
                                   "delta": delta, "aggregate_loss_change": agg_loss - baseline,
                                   "mean_h_p_v": float(values.mean()),
                                   "evidence_type": "q1_imported_plus_scenario"})
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_transfer_by_target.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(out / "pairwise_transfer_aggregate.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out / "delta_sensitivity.csv", index=False)
    joint_rows, joint_aggregate = [], []
    delta = .01
    for j in range(17):
        for k in range(j + 1, 17):
            donor = next(x for x in range(17) if x not in (j, k))
            p_j = transfer(p0, j, donor, delta)
            p_k = transfer(p0, k, donor, delta)
            p_both = transfer(p_j, k, donor, delta)
            values = hp(np.vstack([p0, p_j, p_k, p_both]), p0, artifact)
            for t in range(13):
                loss = [float(generalized_predict(model, params, n, d, q, q0, values[i, t], lam, form))
                        for i in range(4)]
                joint_rows.append({"receiver_j": fields[j], "receiver_k": fields[k],
                                   "donor": fields[donor], "target": targets[t],
                                   "delta": delta, "L0": loss[0], "Lj": loss[1],
                                   "Lk": loss[2], "Ljk": loss[3],
                                   "local_joint_response": loss[3] - loss[1] - loss[2] + loss[0],
                                   "evidence_type": "q1_imported_plus_scenario"})
            loss_agg = [float(generalized_predict(model, params, n, d, q, q0,
                                                  values[i].mean(), lam, form)) for i in range(4)]
            joint_aggregate.append({"receiver_j": fields[j], "receiver_k": fields[k],
                                    "donor": fields[donor], "delta": delta,
                                    "local_joint_response": loss_agg[3] - loss_agg[1] - loss_agg[2] + loss_agg[0],
                                    "evidence_type": "q1_imported_plus_scenario"})
    pd.DataFrame(joint_rows).to_csv(out / "joint_response_by_target.csv", index=False)
    pd.DataFrame(joint_aggregate).to_csv(out / "joint_response_aggregate.csv", index=False)
    (out / "report.md").write_text(
        "# P11 领域配比局部响应\n\n"
        "等权 p0 为情景参考点，转移均在 17 维 simplex 内。"
        "保留 13 目标的单向配比转移与四角局部联合响应，lambda=1 是 unit-transfer 情景。"
        "树模型中心差分可能因局部平台得到零斜率；同时提供有限幅度敏感性。"
        "Q1 未保存训练配方，无法独立验证这些情景点的支持距离；不解释为因果互补。\n",
        encoding="utf-8")
    metadata = finalize("P11", [frozen_dir() / "fitted_models.joblib",
                        frozen_dir() / "reference_mixture.json",
                        stage_dir("P08") / "model_contract.json"],
                        ["Q1 训练配方未保存，配比情景支持距离无法核验"],
                        ["unit-transfer lambda=1 与等权 p0 均为情景"],
                        ["q1_imported", "scenario_assumption"])
    return {"pairs": len(aggregate_rows), "joint_pairs": len(joint_aggregate), "metadata": metadata}
