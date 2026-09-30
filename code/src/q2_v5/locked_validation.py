"""P05：核验冻结摘要后读取 B7-new 目标并执行锁定检验。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import config, finalize, load_b, metrics, paths_for, require_pass, stage_dir, write_json
from .quality_models import fit_quality, freeze_digest, group_ids, quality_predict


def verify_freeze() -> dict:
    out = stage_dir("P04")
    for name in ("model_freeze.sha256", "selection_decision.json", "pre_validation_parameters.json",
                 "pre_validation_m0_parameters.json"):
        if not (out / name).is_file():
            raise RuntimeError(f"冻结文件缺失: {name}")
    saved = (out / "model_freeze.sha256").read_text().strip()
    if saved != freeze_digest(out):
        raise RuntimeError("P04 冻结摘要不一致")
    decision = json.loads((out / "selection_decision.json").read_text())
    prior = json.loads((out / "pre_validation_parameters.json").read_text())
    if decision["selected_model"] != prior["selected_model"] or prior["fitted_on"] != "B6_only":
        raise RuntimeError("冻结选择与参数不一致")
    return prior


def run() -> dict:
    require_pass("P04")
    prior = verify_freeze()  # 严格在任何 B7 目标读取之前。
    out = stage_dir("P05")
    out.mkdir(parents=True, exist_ok=True)
    b7new = pd.read_csv(stage_dir("P01") / "b7_new.csv")
    b6 = load_b("B6")
    if len(b7new) != 90:
        raise RuntimeError("B7-new 行数或 P01 审计完整性失效")
    keys = ["experiment_id", "N_params_B", "D_tokens_B", "Q_score"]
    if b7new.merge(b6[keys], on=keys, how="inner").shape[0]:
        raise RuntimeError("B7-new 与 B6 存在识别键重叠")
    quality_model = prior["selected_model"]
    selected = quality_predict(quality_model, prior["parameters"], b7new.N_params_B,
                               b7new.D_tokens_B, b7new.Q_score, prior["q0"])
    m0 = json.loads((stage_dir("P04") / "pre_validation_m0_parameters.json").read_text())
    baseline = quality_predict("M0", m0["parameters"], b7new.N_params_B,
                               b7new.D_tokens_B, b7new.Q_score, prior["q0"])
    pred = b7new[["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss"]].copy()
    pred["selected_prediction"] = selected
    pred["m0_prediction"] = baseline
    pred["evidence_type"] = "semi_synthetic"
    pred["validation_type"] = "new-Q validation"
    pred.to_csv(out / "b7_new_predictions.csv", index=False)
    metric_rows = [{"model": quality_model, **metrics(pred.val_loss, selected)},
                   {"model": "M0", **metrics(pred.val_loss, baseline)}]
    pd.DataFrame(metric_rows).to_csv(out / "b7_new_metrics.csv", index=False)
    diff = pd.DataFrame({"experiment_id": pred.experiment_id,
                         "absolute_error_selected_minus_m0": np.abs(selected - pred.val_loss) - np.abs(baseline - pred.val_loss),
                         "squared_error_selected_minus_m0": (selected - pred.val_loss)**2 - (baseline - pred.val_loss)**2})
    diff.to_csv(out / "model_error_differences.csv", index=False)
    rng = np.random.default_rng(config()["seed"])
    groups = group_ids(pred)
    unique = np.unique(groups)
    boots = []
    y = pred.val_loss.to_numpy(float)
    for iteration in range(config()["validation_bootstrap_resamples"]):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        boots.append({"resample": iteration,
                      "rmse_difference": metrics(y[idx], selected[idx])["rmse"] - metrics(y[idx], baseline[idx])["rmse"],
                      "mae_difference": metrics(y[idx], selected[idx])["mae"] - metrics(y[idx], baseline[idx])["mae"]})
    pd.DataFrame(boots).to_csv(out / "bootstrap_validation.csv", index=False)
    support = {"new_N": sorted(set(map(float, pred.N_params_B)) - set(map(float, b6.N_params_B))),
               "new_D": sorted(set(map(float, pred.D_tokens_B)) - set(map(float, b6.D_tokens_B))),
               "new_Q": sorted(set(map(float, pred.Q_score)) - set(map(float, b6.Q_score)))}
    summary = {"selected_model": quality_model, "validation_type": "new-Q validation",
               "metrics": metric_rows, "support_novelty": support,
               "bootstrap_rmse_difference_ci": pd.DataFrame(boots).rmse_difference.quantile([.025, .975]).tolist(),
               "uses_pre_validation_parameters": True}
    write_json(out / "validation_summary.json", summary)
    # 锁定验证结果落盘后，才允许相同结构的事后参数估计。
    combined = pd.concat([b6, b7new], ignore_index=True)
    post = fit_quality(quality_model, combined, prior["q0"])
    write_json(out / "post_validation_parameters.json", {
        "model": quality_model, "parameters": post.parameters, "q0": prior["q0"],
        "fitted_on": "B6_plus_B7_new", "used_for_holdout_metrics": False,
        "evidence_type": "semi_synthetic"})
    (out / "report.md").write_text(
        "# P05 B7-new 锁定验证\n\n"
        f"先核验 P04 冻结哈希，再读取 B7-new 的 {len(pred)} 行 Loss。"
        f"入选 {quality_model} RMSE={metric_rows[0]['rmse']:.6g}，M0 RMSE={metric_rows[1]['rmse']:.6g}。"
        f"新 N={support['new_N']}，新 D={support['new_D']}，新 Q={support['new_Q']}。"
        "指标来自 pre-validation 参数；post-validation 参数仅用于后续效应估计。\n",
        encoding="utf-8")
    metadata = finalize("P05", [stage_dir("P01") / "b7_new.csv",
                        stage_dir("P04") / "model_freeze.sha256",
                        stage_dir("P04") / "pre_validation_parameters.json",
                        stage_dir("P04") / "pre_validation_m0_parameters.json"] + paths_for("B6"),
                        [], ["B7-new 为半合成新 Q 留出，非独立真实训练实验"], ["semi_synthetic"])
    return {"summary": summary, "metadata": metadata}
