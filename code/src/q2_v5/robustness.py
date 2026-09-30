"""P06：B8 压力测试及 B9/B10 外推证据边界。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .audit import direction_by_group
from .common import finalize, load_b, metrics, paths_for, require_pass, stage_dir, write_json
from .quality_models import quality_predict


def log_support_distance(n: np.ndarray, d: np.ndarray, reference: pd.DataFrame) -> np.ndarray:
    base = np.log(reference[["N_params_B", "D_tokens_B"]].to_numpy(float))
    query = np.log(np.column_stack([n, d]))
    return np.min(np.sqrt(np.sum((query[:, None, :] - base[None, :, :])**2, axis=2)), axis=1)


def run() -> dict:
    require_pass("P05")
    out = stage_dir("P06")
    out.mkdir(parents=True, exist_ok=True)
    post = json.loads((stage_dir("P05") / "post_validation_parameters.json").read_text())
    b8, b9, b10 = (load_b(name) for name in ("B8", "B9", "B10"))
    train = pd.concat([load_b("B6"), pd.read_csv(stage_dir("P01") / "b7_new.csv")], ignore_index=True)
    support = train[["N_params_B", "D_tokens_B", "Q_score"]].agg(["min", "max"])
    valid8 = np.isfinite(b8[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_numpy(float)).all(axis=1)
    b8 = b8.loc[valid8].copy()
    b8["prediction"] = quality_predict(post["model"], post["parameters"],
                                        b8.N_params_B, b8.D_tokens_B, b8.Q_score, post["q0"])
    b8["support_status"] = np.where(
        (b8.N_params_B.between(support.at["min", "N_params_B"], support.at["max", "N_params_B"]))
        & (b8.D_tokens_B.between(support.at["min", "D_tokens_B"], support.at["max", "D_tokens_B"]))
        & (b8.Q_score.between(support.at["min", "Q_score"], support.at["max", "Q_score"])),
        "range_interpolation", "extrapolation")
    b8["evidence_type"] = "stress_test"
    b8.to_csv(out / "b8_predictions.csv", index=False)
    directions = direction_by_group(b8)
    directions.to_csv(out / "b8_direction.csv", index=False)
    b8_metrics = []
    for (dtype, region), part in b8.groupby(["data_type", "support_status"]):
        b8_metrics.append({"data_type": dtype, "support_status": region,
                           **metrics(part.val_loss, part.prediction)})
    pd.DataFrame(b8_metrics).to_csv(out / "b8_metrics.csv", index=False)
    valid9 = np.isfinite(b9[["N_params_B", "D_tokens_B"]].to_numpy(float)).all(axis=1)
    valid9 &= (b9[["N_params_B", "D_tokens_B"]].to_numpy(float) > 0).all(axis=1)
    b9 = b9.copy()
    b9["log_nd_nearest_distance"] = np.nan
    b9.loc[valid9, "log_nd_nearest_distance"] = log_support_distance(
        b9.loc[valid9, "N_params_B"], b9.loc[valid9, "D_tokens_B"], train)
    b9["evidence_type"] = "metadata_only"
    b9[["model_name", "N_params_B", "D_tokens_B", "log_nd_nearest_distance", "evidence_type"]].to_csv(
        out / "b9_support_distance.csv", index=False)
    b10 = b10.copy()
    b10["prediction_at_q0"] = quality_predict(post["model"], post["parameters"],
                                               b10.N_params_B, b10.D_tokens_B,
                                               np.full(len(b10), post["q0"]), post["q0"])
    b10["evidence_type"] = "estimated_reference"
    b10.to_csv(out / "b10_estimated_consistency.csv", index=False)
    summary = {"b8_positive_groups": int((directions.direction == "positive").sum()),
               "b8_negative_groups": int((directions.direction == "negative").sum()),
               "b8_metrics": b8_metrics, "b9_missing_or_invalid_nd": int((~valid9).sum()),
               "b10_metrics": metrics(b10.val_loss, b10.prediction_at_q0),
               "b10_interpretation": "estimated-reference consistency only"}
    write_json(out / "stress_summary.json", summary)
    (out / "report.md").write_text(
        "# P06 压力与外推边界\n\n"
        f"B8 固定 N,D 的 Q–Loss：正向组 {summary['b8_positive_groups']}，"
        f"负向组 {summary['b8_negative_groups']}。"
        "若与 B6/B7 相反，记为跨附件结构不一致，不翻转 Q。"
        f"B9 有 {summary['b9_missing_or_invalid_nd']} 行缺失或非法 N/D，仅用于支持距离。"
        "B10 的 Loss 是估算参考，只报告自洽性。\n", encoding="utf-8")
    metadata = finalize("P06", paths_for("B8") + paths_for("B9") + paths_for("B10")
                        + [stage_dir("P05") / "post_validation_parameters.json"],
                        ["B8 压力测试可能与 B6/B7 质量方向冲突；不调整模型"],
                        ["B10 用 Q0 情景比较，不是已观测 Q 的真实验证"],
                        ["stress_test", "metadata_only", "estimated_reference"])
    return {"summary": summary, "metadata": metadata}
