"""P03：冻结 B1 参数后的跨来源验证。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .baseline import predict
from .common import EVIDENCE, finalize, load_b, metrics, paths_for, require_pass, stage_dir


INTERPRETATION = {"B2": "trend_only", "B3": "interpolation_consistency",
                  "B4": "cross_family", "B5": "literature_cross_source"}


def run() -> dict:
    require_pass("P02")
    out = stage_dir("P03")
    out.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((stage_dir("P02") / "final_parameters.json").read_text())["parameters"]
    matrix, families, sources, predictions = [], [], [], []
    all_paths = [stage_dir("P02") / "final_parameters.json"]
    for attachment in INTERPRETATION:
        all_paths.extend(paths_for(attachment))
        frame = load_b(attachment)
        cols = ["N_params_B", "D_tokens_B", "val_loss"]
        valid = np.isfinite(frame[cols].to_numpy(float)).all(axis=1) & (frame[cols].to_numpy(float) > 0).all(axis=1)
        frame = frame.loc[valid].copy()
        frame["prediction"] = predict(parameters, frame.N_params_B, frame.D_tokens_B)
        frame["attachment"] = attachment
        frame["evidence_type"] = EVIDENCE[attachment]
        frame["interpretation_level"] = INTERPRETATION[attachment]
        matrix.append({"attachment": attachment, "evidence_type": EVIDENCE[attachment],
                       "interpretation_level": INTERPRETATION[attachment],
                       **metrics(frame.val_loss, frame.prediction)})
        cols_out = ["attachment", "_source_file", "_source_row", "N_params_B", "D_tokens_B",
                    "val_loss", "prediction", "evidence_type", "interpretation_level"]
        for optional in ("family", "source", "run_id"):
            if optional in frame:
                cols_out.append(optional)
        predictions.append(frame[cols_out])
        if "family" in frame:
            for group, part in frame.groupby("family"):
                families.append({"attachment": attachment, "family": group,
                                 **metrics(part.val_loss, part.prediction)})
        if "source" in frame:
            for group, part in frame.groupby("source"):
                sources.append({"attachment": attachment, "source": group,
                                **metrics(part.val_loss, part.prediction)})
    pd.DataFrame(matrix).to_csv(out / "validation_matrix.csv", index=False)
    pd.DataFrame(families).to_csv(out / "per_family_metrics.csv", index=False)
    pd.DataFrame(sources).to_csv(out / "per_source_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "transfer_predictions.csv", index=False)
    (out / "report.md").write_text(
        "# P03 冻结 B1 参数迁移\n\n"
        "所有预测使用 P02 最终参数，未对 B2–B5 重拟合。B2 仅用于趋势，B3 是插值一致性，"
        "B4/B5 跨来源 Loss 口径未证明严格相同；绝对误差仅作条件比较。\n\n"
        + "\n".join(
            f"- {row['attachment']}: n={row['n']}, RMSE={row['rmse']:.6g}, "
            f"Spearman={row['spearman']:.6g} ({row['interpretation_level']})"
            for row in matrix
        ) + "\n", encoding="utf-8")
    metadata = finalize("P03", all_paths,
                        ["跨来源验证 Loss 口径未证明严格可比"],
                        ["P02 B1 参数完全冻结，不做偏置校准"],
                        sorted({EVIDENCE[a] for a in INTERPRETATION}))
    return {"metrics": matrix, "metadata": metadata}
