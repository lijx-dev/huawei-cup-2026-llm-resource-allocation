"""仅在 A4/A5 内部选择参数，再冻结多目标模型。"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


def ridge_x(x, reference_index):
    return np.delete(x, reference_index, axis=1)


def lgbm(params, seed):
    return LGBMRegressor(**params, random_state=seed, n_jobs=1, verbosity=-1, deterministic=True,
                         force_col_wise=True)


def select_and_fit(x, y, cfg, reference_index):
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    rx = ridge_x(x, reference_index)
    rows = []
    choices = {}
    for family, candidates in (("ridge", cfg["ridge_alphas"]), ("lightgbm", cfg["lightgbm_candidates"])):
        selected = []
        for target in range(y.shape[1]):
            scores = []
            for ci, candidate in enumerate(candidates):
                fold_errors = []
                for train, valid in cv.split(x):
                    model = Ridge(alpha=candidate) if family == "ridge" else lgbm(candidate, cfg["seed"])
                    xx = rx if family == "ridge" else x
                    model.fit(xx[train], y[train, target])
                    fold_errors.append(float(np.mean((model.predict(xx[valid]) - y[valid, target]) ** 2)))
                rmse = float(np.sqrt(np.mean(fold_errors)))
                scores.append(rmse)
                rows.append({"model": family, "target_index": target, "candidate_index": ci,
                             "params": str(candidate), "cv_rmse": rmse})
            selected.append(int(np.argmin(scores)))
        choices[family] = selected
    models = {"ridge": [], "lightgbm": []}
    for target in range(y.shape[1]):
        alpha = cfg["ridge_alphas"][choices["ridge"][target]]
        rm = Ridge(alpha=alpha).fit(rx, y[:, target])
        param = cfg["lightgbm_candidates"][choices["lightgbm"][target]]
        lm = lgbm(param, cfg["seed"]).fit(x, y[:, target])
        models["ridge"].append(rm)
        models["lightgbm"].append(lm)
    models["mean"] = y.mean(axis=0)
    return models, rows, choices


def predict(models, family, x, reference_index):
    if family == "mean":
        return np.broadcast_to(models["mean"], (len(x), len(models["mean"]))).copy()
    xx = ridge_x(x, reference_index) if family == "ridge" else x
    return np.column_stack([model.predict(xx) for model in models[family]])
