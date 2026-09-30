"""B1/B6 标度律的带约束多起点估计和组级验证。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


SEED = 7
POSITIVE = ["A", "B", "alpha", "beta"]


def predict(theta: dict, n, d, q=None, q0=None, form="base", h=0.0, lam=0.0):
    n, d = np.asarray(n, float), np.asarray(d, float)
    p = theta["A"] * n ** (-theta["alpha"])
    t = theta["B"] * d ** (-theta["beta"])
    if form == "base":
        return theta["E"] + p + t
    z = np.asarray(q0 - np.asarray(q, float), float)
    g = theta.get("gamma", 0.0) * z + theta.get("gamma2", 0.0) * z * z
    if form == "quality" or form == "A":
        return theta["E"] + p + t * np.exp(g + lam * h)
    if form == "B":
        return theta["E"] + (p + t * np.exp(g)) * np.exp(lam * h)
    raise ValueError(form)


def unpack(x, quality=False, quadratic=False):
    out = dict(E=float(x[0]), A=float(np.exp(x[1])), B=float(np.exp(x[2])), alpha=float(np.exp(x[3])), beta=float(np.exp(x[4])))
    if quality:
        out["gamma"] = float(x[5])
    if quadratic:
        out["gamma2"] = float(x[6])
    return out


def fit(frame, quality=False, q0=None, quadratic=False, starts=4, seed=SEED):
    """对正参数取 log，E 采用非负边界。初值仅取训练折。"""
    n = frame.N_params_B.to_numpy(float)
    d = frame.D_tokens_B.to_numpy(float)
    y = frame.val_loss.to_numpy(float)
    q = frame.Q_score.to_numpy(float) if quality else None
    if not (np.isfinite(n).all() and np.isfinite(d).all() and np.isfinite(y).all() and (n > 0).all() and (d > 0).all()):
        raise ValueError("invalid N/D/Loss")
    if quality and q0 is None:
        raise ValueError("Q reference must be fixed")
    rng = np.random.default_rng(seed)
    e0 = max(0.01, float(np.min(y) * 0.65))
    base = np.array([e0, np.log(max(.1, np.std(y))), np.log(max(.1, np.std(y))), np.log(.3), np.log(.3)] + ([.1] if quality else []) + ([0.] if quadratic else []))
    lo = [0, -12, -12, np.log(.01), np.log(.01)] + ([-10] if quality else []) + ([-20] if quadratic else [])
    hi = [max(10, float(np.max(y) * 2)), 5, 5, np.log(3), np.log(3)] + ([10] if quality else []) + ([20] if quadratic else [])
    runs = []
    for i in range(starts):
        x0 = base.copy()
        if i:
            x0[0] *= rng.uniform(.4, 1.4)
            x0[1:5] += rng.normal(0, .55, 4)
            if quality:
                x0[5] = rng.normal(0, .5)
        x0 = np.clip(x0, np.asarray(lo) + 1e-7, np.asarray(hi) - 1e-7)
        def residual(x):
            t = unpack(x, quality, quadratic)
            return predict(t, n, d, q, q0, "quality" if quality else "base") - y
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=2500, ftol=1e-10, xtol=1e-10)
        runs.append(dict(start=i, initial_values=json.dumps(unpack(x0, quality, quadratic)), final_values=json.dumps(unpack(res.x, quality, quadratic)), converged=bool(res.success), objective=float(np.sum(res.fun ** 2)), iterations=int(res.nfev), x=res.x))
    good = [r for r in runs if r["converged"]]
    if not good:
        raise RuntimeError("all nonlinear fits failed")
    best = min(good, key=lambda r: r["objective"])
    theta = unpack(best["x"], quality, quadratic)
    for r in runs:
        del r["x"]
    return theta, runs


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return dict(n=len(y), RMSE=float(np.sqrt(mean_squared_error(y, pred))), MAE=float(mean_absolute_error(y, pred)), R2=float(r2_score(y, pred)) if len(y) > 1 else None, Pearson=float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(y) and np.std(pred) else None, Spearman=float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(y) and np.std(pred) else None, mean_bias=float(np.mean(pred-y)))


def cv(frame, group, quality=False, q0=None, quadratic=False, n_splits=None):
    splitter = GroupKFold(n_splits=n_splits) if n_splits else LeaveOneGroupOut()
    rows, predictions = [], []
    for fold, (tr, te) in enumerate(splitter.split(frame, groups=group)):
        train, test = frame.iloc[tr], frame.iloc[te]
        theta, _ = fit(train, quality, q0, quadratic, seed=SEED + fold, starts=3)
        yhat = predict(theta, test.N_params_B, test.D_tokens_B, test.Q_score if quality else None, q0, "quality" if quality else "base")
        rows.append(dict(fold=fold, test_groups="|".join(sorted(set(np.asarray(group)[te].astype(str)))), model="MQ2" if quadratic else "MQ" if quality else "M0", **metrics(test.val_loss, yhat)))
        predictions.extend(dict(fold=fold, source_row=int(i), group=str(np.asarray(group)[i]), actual=float(test.loc[i].val_loss), prediction=float(v)) for i,v in zip(test.index,yhat))
    return rows, predictions


def grouped_bootstrap(frame, group, quality=False, q0=None, reps=60):
    groups = np.asarray(group).astype(str)
    unique = np.unique(groups)
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(reps):
        picks = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == g) for g in picks])
        try:
            theta, _ = fit(frame.iloc[positions], quality, q0, starts=2, seed=SEED + i)
            rows.append(dict(bootstrap=i, **theta))
        except RuntimeError:
            continue
    return rows


def save_csv(path: Path, rows, columns=None):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
