"""训练内重复 CV、冻结模型、真实留出与估算表评估、有限替代扰动。"""
import json
import warnings
import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from q1.mixture.dataset import load_pair
from q1.mixture.evaluation import metrics
from q1.mixture.perturbation import single, pair, nearest_distance, support_reference
from .common import paths, save_json


SPLITS = ("train_1m", "test_1m", "test_60m", "test_1b", "est_10b", "est_70b")


def model_lgb(params, seed):
    return lgb.LGBMRegressor(**params, random_state=seed, bagging_seed=seed,
                            feature_fraction_seed=seed, data_random_seed=seed,
                            deterministic=True, force_col_wise=True, verbosity=-1,
                            n_jobs=1)


def ridge_fit(x, y, alpha):
    scaler = StandardScaler().fit(x[:, :-1])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[:, :-1]), y)
    return scaler, model


def ridge_predict(fitted, x):
    scaler, model = fitted
    return model.predict(scaler.transform(x[:, :-1]))


def macro_nrmse(actual, pred, train_std):
    return float(np.mean(np.sqrt(np.mean((actual - pred) ** 2, axis=0)) / train_std))


def load_all(root, audit, config):
    mix_fields, loss_fields = audit["tables"]["mixture_fields"], audit["tables"]["loss_fields"]
    data = {}
    records = []
    for name in SPLITS:
        ids, x, y, info = load_pair(root, name, mix_fields, loss_fields, config["mixture_sum_tolerance"])
        data[name] = (ids, x, y)
        records.append(info)
    return data, mix_fields, loss_fields, records


def cv_models(x, y, config, dest):
    seed = config["seed"]
    splitter = RepeatedKFold(n_splits=config["cv_folds"], n_repeats=config["cv_repeats"], random_state=seed)
    folds = list(splitter.split(x))
    pd.DataFrame([{"fold": f, "repeat": f // config["cv_folds"], "row": int(row), "role": role}
                  for f, (tr, va) in enumerate(folds) for role, ids in (("train", tr), ("validation", va))
                  for row in ids]).to_csv(dest / "cv_fold_membership.csv", index=False)
    std = y.std(axis=0, ddof=1)
    if (std <= 0).any():
        raise RuntimeError("训练 Loss 常数列，无法标准化 CV 指标")
    nr, nt = len(folds), y.shape[1]
    alphas = config["ridge_alphas"]
    ridge_sq = np.zeros((len(alphas), nr, nt))
    lgb_sq = np.zeros((len(config["lightgbm_candidates"]), nr, nt))
    lgb_pred_rows = []
    for fold, (tr, va) in enumerate(folds):
        scaler = StandardScaler().fit(x[tr, :-1])
        xt, xv = scaler.transform(x[tr, :-1]), scaler.transform(x[va, :-1])
        for a, alpha in enumerate(alphas):
            pred = Ridge(alpha=alpha).fit(xt, y[tr]).predict(xv)
            ridge_sq[a, fold] = np.mean((y[va] - pred) ** 2, axis=0)
        for c, candidate in enumerate(config["lightgbm_candidates"]):
            for target in range(nt):
                model = model_lgb(candidate, seed)
                model.fit(x[tr], y[tr, target])
                pred = model.predict(x[va])
                lgb_sq[c, fold, target] = np.mean((y[va, target] - pred) ** 2)
        if fold % 5 == 4:
            print(f"重复 CV：{fold+1}/{nr} 折已完成", flush=True)
    ridge_mean = ridge_sq.mean(axis=1)
    selected_alpha = [int(v) for v in np.argmin(ridge_mean, axis=0)]
    ridge_by_fold = np.stack([ridge_sq[selected_alpha[t], :, t] for t in range(nt)], axis=1)
    lgb_candidate_score = np.mean(np.sqrt(lgb_sq.mean(axis=1)) / std, axis=1)
    chosen_lgb = int(np.argmin(lgb_candidate_score))
    ridge_target_rmse = np.sqrt(ridge_by_fold.mean(axis=0))
    lgb_target_rmse = np.sqrt(lgb_sq[chosen_lgb].mean(axis=0))
    ridge_macro = float(np.mean(ridge_target_rmse / std))
    lgb_macro = float(np.mean(lgb_target_rmse / std))
    target_wins = int((lgb_target_rmse < ridge_target_rmse).sum())
    repeat_wins = int(sum(np.mean(np.sqrt(lgb_sq[chosen_lgb, r*5:(r+1)*5].mean(axis=0)) / std) <
                              np.mean(np.sqrt(ridge_by_fold[r*5:(r+1)*5].mean(axis=0)) / std)
                              for r in range(config["cv_repeats"])))
    main_lgb = ((ridge_macro - lgb_macro) / ridge_macro >= config["lightgbm_main_min_macro_nrmse_gain"]
                and target_wins >= config["lightgbm_main_min_target_wins"]
                and repeat_wins >= config["lightgbm_main_min_repeat_wins"])
    fold_rows = []
    for f in range(nr):
        for t in range(nt):
            fold_rows.append({"fold": f, "repeat": f // config["cv_folds"], "target_index": t,
                              "ridge_alpha": alphas[selected_alpha[t]],
                              "ridge_rmse": np.sqrt(ridge_by_fold[f, t]),
                              "lightgbm_candidate": chosen_lgb,
                              "lightgbm_rmse": np.sqrt(lgb_sq[chosen_lgb, f, t]),
                              "train_target_sd": std[t]})
    pd.DataFrame(fold_rows).to_csv(dest / "cv_fold_metrics.csv", index=False)
    for label, sq in (("ridge", ridge_sq), ("lightgbm", lgb_sq)):
        pd.DataFrame([{"model": label, "candidate": a, "target_index": t,
                       "mean_rmse": np.sqrt(sq[a, :, t].mean()), "normalized_rmse": np.sqrt(sq[a, :, t].mean()) / std[t]}
                      for a in range(len(sq)) for t in range(nt)]).to_csv(dest / f"cv_{label}_candidates.csv", index=False)
    result = {"ridge_alphas_by_target": [alphas[a] for a in selected_alpha],
              "lightgbm_candidate": chosen_lgb, "lightgbm_params": config["lightgbm_candidates"][chosen_lgb],
              "lightgbm_random_settings": {"random_state": seed, "bagging_seed": seed,
                                           "feature_fraction_seed": seed, "data_random_seed": seed,
                                           "deterministic": True, "force_col_wise": True, "n_jobs": 1},
              "ridge_macro_nrmse": ridge_macro, "lightgbm_macro_nrmse": lgb_macro,
              "lightgbm_target_wins": target_wins, "lightgbm_repeat_wins": repeat_wins,
              "main_model": "LightGBM" if main_lgb else "Ridge"}
    save_json(dest / "model_selection.json", result)
    return result


def train_models(x, y, selection, config):
    ridge = [ridge_fit(x, y[:, t], selection["ridge_alphas_by_target"][t]) for t in range(y.shape[1])]
    gbm = [model_lgb(selection["lightgbm_params"], config["seed"]).fit(x, y[:, t])
           for t in range(y.shape[1])]
    return ridge, gbm


def predict(models, x, kind):
    if kind == "Ridge":
        return np.column_stack([ridge_predict(model, x) for model in models])
    return np.column_stack([model.predict(x) for model in models])


def evaluate_holdouts(data, models, fields, train_y, dest, config):
    records, estimated_records, summary_rows, preds = [], [], [], {}
    train_mu, train_std = train_y.mean(axis=0), train_y.std(axis=0, ddof=1)
    for split in SPLITS[1:]:
        ids, x, y = data[split]
        for kind, fitted in models.items():
            p = predict(fitted, x, kind)
            preds[split, kind] = p
            for result in metrics(y, p, fields, config["top_k"]):
                result.update({"split": split, "model": kind,
                               "evidence_type": "estimated_reference" if split.startswith("est_") else "observed_holdout"})
                if split.startswith("est_"):
                    estimated_records.append({k: result[k] for k in ("split", "model", "evidence_type", "target", "n", "spearman", "kendall", "overlap_5", "overlap_10", "overlap_20")})
                else:
                    records.append(result)
            if not split.startswith("est_"):
                actual_z, pred_z = (y - train_mu) / train_std, (p - train_mu) / train_std
                target_mean = y.mean(axis=0)
                pooled = 1 - np.sum((y - p)**2) / np.sum((y - target_mean)**2)
                summary_rows.append({"split": split, "model": kind, "n": len(y),
                                     "pooled_r2": float(pooled),
                                     "macro_nrmse_train_scale": macro_nrmse(y, p, train_std),
                                     "J_mae": float(np.mean(np.abs(actual_z.mean(axis=1) - pred_z.mean(axis=1)))),
                                     "mean_target_pearson": float(np.nanmean([r["pearson"] for r in records if r["split"] == split and r["model"] == kind and r["target"] != "average_loss"]))})
        frame = pd.DataFrame({"index": ids, "split": split})
        for target, name in enumerate(fields):
            frame[f"actual/{name}"] = y[:, target]
            for kind in models:
                frame[f"predicted_{kind}/{name}"] = preds[split, kind][:, target]
        frame.to_csv(dest / f"predictions_{split}.csv", index=False)
    result = pd.DataFrame(records)
    result.to_csv(dest / "holdout_metrics.csv", index=False)
    pd.DataFrame(estimated_records).to_csv(dest / "estimated_rank_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(dest / "holdout_summary.csv", index=False)
    return result


def perturbations(train_x, models, mix_fields, loss_fields, config, dest):
    threshold, train_nearest = support_reference(train_x, config["support_nn_quantile"])
    support = {"nearest_neighbor_95pct_distance": threshold, "reference": "training recipe leave-one-out Euclidean distance"}
    save_json(dest / "support_reference.json", support)
    rows = []
    for kind, fitted in models.items():
        baseline = predict(fitted, train_x, kind)
        for col, field in enumerate(mix_fields):
            for delta in config["perturbation_deltas"]:
                changed, feasible = single(train_x, col, delta)
                supported = feasible.copy()
                supported[feasible] = nearest_distance(changed[feasible], train_x) <= threshold
                eligible = supported & (train_nearest <= threshold)
                if eligible.any():
                    effect = predict(fitted, changed[eligible], kind) - baseline[eligible]
                    mean = effect.mean(axis=0)
                else:
                    mean = np.full(len(loss_fields), np.nan)
                for t, target in enumerate(loss_fields):
                    rows.append({"model": kind, "domain": field, "delta": delta, "target": target,
                                 "n_feasible": int(feasible.sum()), "n_supported": int(eligible.sum()),
                                 "AME": mean[t]})
    pd.DataFrame(rows).to_csv(dest / "single_domain_effects.csv", index=False)
    pair_rows = []
    delta = config["pair_delta"]
    for col in range(len(mix_fields)):
        for other in range(col + 1, len(mix_fields)):
            both, feasible = pair(train_x, col, other, delta, delta)
            solo_j, valid_j = single(train_x, col, delta)
            solo_k, valid_k = single(train_x, other, delta)
            eligible = feasible & valid_j & valid_k & (train_nearest <= threshold)
            for changed in (both, solo_j, solo_k):
                valid = np.flatnonzero(eligible)
                if len(valid):
                    eligible[valid] &= nearest_distance(changed[valid], train_x) <= threshold
            for kind, fitted in models.items():
                if eligible.any():
                    index = np.flatnonzero(eligible)
                    pred0 = predict(fitted, train_x[index], kind)
                    predj = predict(fitted, solo_j[index], kind)
                    predk = predict(fitted, solo_k[index], kind)
                    predb = predict(fitted, both[index], kind)
                    interaction = (predb - predj - predk + pred0).mean(axis=0)
                else:
                    interaction = np.full(len(loss_fields), np.nan)
                for t, target in enumerate(loss_fields):
                    pair_rows.append({"model": kind, "domain_j": mix_fields[col], "domain_k": mix_fields[other],
                                      "delta": delta, "target": target, "n_feasible": int(feasible.sum()),
                                      "n_supported": int(eligible.sum()), "interaction": interaction[t]})
    pd.DataFrame(pair_rows).to_csv(dest / "pair_interactions.csv", index=False)
    return support


def run(root, config, audit):
    _, out = paths(root)
    dest = out / "mixture"
    dest.mkdir(parents=True, exist_ok=True)
    data, mix_fields, loss_fields, inputs = load_all(root, audit, config)
    pd.DataFrame(inputs).to_csv(dest / "split_input_audit.csv", index=False)
    ids, x, y = data["train_1m"]
    selection = cv_models(x, y, config, dest)
    fitted = train_models(x, y, selection, config)
    models = {"Ridge": fitted[0], "LightGBM": fitted[1]}
    joblib.dump({"models": models, "selection": selection, "mix_fields": mix_fields,
                 "loss_fields": loss_fields, "seed": config["seed"]}, dest / "fitted_models.joblib", compress=3)
    holdout = evaluate_holdouts(data, models, loss_fields, y, dest, config)
    support = perturbations(x, models, mix_fields, loss_fields, config, dest)
    summary = {"training_rows": len(x), "mixture_fields": mix_fields, "loss_fields": loss_fields,
               "selection": selection, "split_audit": inputs, "support": support}
    save_json(dest / "mixture_summary.json", summary)
    return summary
