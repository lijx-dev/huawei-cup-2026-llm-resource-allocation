"""真实留出、跨规模和估算参考分开评价。"""
import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _corr(fn, a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(a, b).statistic)


def metrics(actual, predicted, names, top_k=(5, 10, 20)):
    if actual.shape != predicted.shape or actual.shape[1] != len(names):
        raise ValueError("评价矩阵形状不一致")
    records = []
    for col, name in enumerate([*names, "average_loss"]):
        a = actual[:, col] if col < len(names) else actual.mean(axis=1)
        b = predicted[:, col] if col < len(names) else predicted.mean(axis=1)
        rec = {"target": name, "n": len(a), "rmse": float(np.sqrt(mean_squared_error(a, b))),
               "mae": float(mean_absolute_error(a, b)), "r2": float(r2_score(a, b)),
               "pearson": _corr(pearsonr, a, b), "spearman": _corr(spearmanr, a, b),
               "kendall": _corr(kendalltau, a, b)}
        for k in top_k:
            kk = min(k, len(a))
            rec[f"overlap_{k}"] = float(len(set(np.argsort(a)[:kk]) & set(np.argsort(b)[:kk])) / kk)
        records.append(rec)
    return records
