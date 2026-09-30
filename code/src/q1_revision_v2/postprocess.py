"""保存共同可行配方 AME 和重复 CV 的均值、方差。"""
import json

import joblib
import numpy as np
import pandas as pd

from q1.mixture.dataset import load_pair
from q1.mixture.perturbation import single, nearest_distance, support_reference
from .common import paths
from .mixture import predict


def run(root, config):
    root, out = paths(root)
    dest = out / "mixture"
    audit = json.loads((out / "audit/audit_summary.json").read_text())
    fields, targets = audit["tables"]["mixture_fields"], audit["tables"]["loss_fields"]
    _, x, _, _ = load_pair(root, "train_1m", fields, targets, config["mixture_sum_tolerance"])
    models = joblib.load(dest / "fitted_models.joblib")["models"]
    threshold = json.loads((dest / "support_reference.json").read_text())["nearest_neighbor_95pct_distance"]
    recomputed_threshold, train_nearest = support_reference(x, config["support_nn_quantile"])
    if not np.isclose(threshold, recomputed_threshold, atol=1e-12):
        raise RuntimeError("共同子集支持阈值与正式扰动参考不一致")
    rows = []
    for delta in config["perturbation_deltas"]:
        changed = [single(x, j, delta) for j in range(len(fields))]
        common = np.logical_and.reduce([ok for _, ok in changed])
        supported = common & (train_nearest <= threshold)
        for matrix, _ in changed:
            idx = np.flatnonzero(supported)
            if len(idx):
                supported[idx] &= nearest_distance(matrix[idx], x) <= threshold
        for model_name, fitted in models.items():
            for subset_name, mask in (("common_feasible", common), ("common_supported", supported)):
                idx = np.flatnonzero(mask)
                if len(idx):
                    baseline = predict(fitted, x[idx], model_name)
                for j, field in enumerate(fields):
                    ame = ((predict(fitted, changed[j][0][idx], model_name) - baseline).mean(axis=0)
                           if len(idx) else np.full(len(targets), np.nan))
                    for t, target in enumerate(targets):
                        rows.append({"model": model_name, "subset": subset_name, "domain": field,
                                     "delta": delta, "target": target, "n": len(idx), "AME": ame[t]})
    pd.DataFrame(rows).to_csv(dest / "common_subset_ame.csv", index=False)
    folds = pd.read_csv(dest / "cv_fold_metrics.csv")
    folds.groupby("target_index", as_index=False).agg(
        ridge_rmse_mean=("ridge_rmse", "mean"), ridge_rmse_sd=("ridge_rmse", "std"),
        lightgbm_rmse_mean=("lightgbm_rmse", "mean"), lightgbm_rmse_sd=("lightgbm_rmse", "std")
    ).to_csv(dest / "cv_repeat_variability.csv", index=False)
