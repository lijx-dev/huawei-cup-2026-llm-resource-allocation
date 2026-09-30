"""仅直接/近直接 A16 映射的质量摘要，以及同折 Ridge 增益检验。"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from q1.mixture.dataset import load_pair
from q1.mixture.evaluation import metrics
from .common import paths, save_json


def mapped_quality(x, mix_fields, mapping, domain_q):
    allowed = mapping[mapping.mapping_type.isin(["direct", "near_direct"])]
    cols, values = [], []
    for row in allowed.itertuples():
        field = "train_the_pile_" + row.mixture_domain
        if field in mix_fields and row.quality_domain in domain_q:
            cols.append(mix_fields.index(field))
            values.append(domain_q[row.quality_domain])
    mass = x[:, cols].sum(axis=1)
    numerator = x[:, cols] @ np.asarray(values)
    q = np.divide(numerator, mass, out=np.full(len(x), np.nan), where=mass > 0)
    return q, mass, cols


def fit_ridge(x, y, alpha):
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return scaler, model


def predict(model, x):
    scaler, ridge = model
    return ridge.predict(scaler.transform(x))


def run(root, config, audit):
    root, out = paths(root)
    dest = out / "integration"
    dest.mkdir(parents=True, exist_ok=True)
    mix_fields = audit["tables"]["mixture_fields"]
    loss_fields = audit["tables"]["loss_fields"]
    mapping = pd.read_csv(root / "data/real_attachments/A_data_value/domain_mapping_guide.csv", keep_default_na=False)
    mapping.to_csv(dest / "mapping_used.csv", index=False)
    quality = pd.read_csv(out / "quality/domain_scores.csv")
    quality = quality[quality.view == "union_unique"]
    result = {}
    comparison = []
    holdout_summary = []
    for quality_method in ("Q_entropy_redundancy", "Q_group_equal"):
        domain_q = dict(zip(quality.domain, quality[quality_method]))
        ids, x, y, info = load_pair(root, "train_1m", mix_fields, loss_fields, config["mixture_sum_tolerance"])
        q, mass, cols = mapped_quality(x, mix_fields, mapping, domain_q)
        keep = np.isfinite(q)
        if keep.sum() < config["cv_folds"] * 2:
            raise RuntimeError("有质量映射的训练行不足以比较 Ridge")
        train_ids, tx, ty, tq = ids[keep], x[keep], y[keep], q[keep]
        reference = np.column_stack([tx[:, :-1], tq])
        baseline = tx[:, :-1]
        folds = list(KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"]).split(tx))
        alpha_grid = config["q_ridge_alphas"]
        cv_rmse = {}
        for label, features in (("Ridge_p", baseline), ("Ridge_p_Qmapped", reference)):
            fold_error = np.zeros((len(alpha_grid), len(folds), ty.shape[1]))
            for a, alpha in enumerate(alpha_grid):
                for f, (tr, va) in enumerate(folds):
                    model = fit_ridge(features[tr], ty[tr], alpha)
                    fold_error[a, f] = np.mean((ty[va] - predict(model, features[va])) ** 2, axis=0)
            selected = np.argmin(fold_error.mean(axis=1), axis=0)
            cv_rmse[label] = np.array([np.sqrt(fold_error[selected[t], :, t].mean()) for t in range(ty.shape[1])])
            result[quality_method + "/" + label + "/alphas"] = [alpha_grid[a] for a in selected]
            for t, target in enumerate(loss_fields):
                comparison.append({"quality_method": quality_method, "model": label, "target": target,
                                   "cv_rmse": cv_rmse[label][t], "alpha": alpha_grid[selected[t]]})
        std = ty.std(axis=0, ddof=1)
        base_macro = float(np.mean(cv_rmse["Ridge_p"] / std))
        augmented_macro = float(np.mean(cv_rmse["Ridge_p_Qmapped"] / std))
        gain = (base_macro - augmented_macro) / base_macro
        result[quality_method] = {"train_rows_mapped": int(keep.sum()), "train_rows_unmapped": int((~keep).sum()),
                                  "mapped_domain_count": len(cols), "cv_macro_nrmse_p": base_macro,
                                  "cv_macro_nrmse_p_q": augmented_macro, "relative_cv_gain": gain,
                                  "target_wins": int((cv_rmse["Ridge_p_Qmapped"] < cv_rmse["Ridge_p"]).sum())}
        test_ids, vx, vy, _ = load_pair(root, "test_1m", mix_fields, loss_fields, config["mixture_sum_tolerance"])
        vq, vmass, _ = mapped_quality(vx, mix_fields, mapping, domain_q)
        vkeep = np.isfinite(vq)
        result[quality_method]["holdout_rows_mapped"] = int(vkeep.sum())
        pred_frame = pd.DataFrame({"index": test_ids[vkeep], "Qmapped": vq[vkeep], "covered_mass": vmass[vkeep]})
        for label, feature_train, feature_test in (("Ridge_p", baseline, vx[vkeep, :-1]),
                                                   ("Ridge_p_Qmapped", reference, np.column_stack([vx[vkeep, :-1], vq[vkeep]]))):
            alphas = result[quality_method + "/" + label + "/alphas"]
            pred = np.column_stack([predict(fit_ridge(feature_train, ty[:, t], alphas[t]), feature_test)
                                    for t in range(ty.shape[1])])
            pooled = 1 - np.sum((vy[vkeep] - pred)**2) / np.sum((vy[vkeep] - vy[vkeep].mean(axis=0))**2)
            macro_nrmse = float(np.mean(np.sqrt(np.mean((vy[vkeep] - pred)**2, axis=0)) / ty.std(axis=0, ddof=1)))
            for row in metrics(vy[vkeep], pred, loss_fields, config["top_k"]):
                row.update({"quality_method": quality_method, "model": label, "split": "test_1m",
                            "n_excluded_unmapped": int((~vkeep).sum())})
                comparison.append(row)
            relevant = [row for row in comparison if row.get("quality_method") == quality_method and
                        row.get("model") == label and row.get("split") == "test_1m" and row.get("target") != "average_loss"]
            holdout_summary.append({"quality_method": quality_method, "model": label, "n": int(vkeep.sum()),
                                    "pooled_r2": float(pooled), "macro_nrmse_train_scale": macro_nrmse,
                                    "mean_target_pearson": float(np.nanmean([row["pearson"] for row in relevant]))})
            for t, name in enumerate(loss_fields):
                pred_frame[f"actual/{name}"] = vy[vkeep, t]
                pred_frame[f"{label}/{name}"] = pred[:, t]
        pred_frame.to_csv(dest / f"predictions_{quality_method}.csv", index=False)
    pd.DataFrame(comparison).to_csv(dest / "ridge_quality_comparison.csv", index=False)
    pd.DataFrame(holdout_summary).to_csv(dest / "holdout_summary.csv", index=False)
    save_json(dest / "integration_summary.json", result)
    return result
