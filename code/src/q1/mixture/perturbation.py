"""单纯形有限扰动及与训练配方的距离。"""
import numpy as np
from scipy.spatial.distance import cdist


def check_simplex(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < -1e-10).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-10):
        raise ValueError("配比必须是有限、非负、行和为 1 的矩阵")


def single(x, j, delta):
    check_simplex(x)
    if delta < 0 or delta > 1:
        raise ValueError("扰动幅度非法")
    donor = 1 - x[:, j]
    feasible = (donor > 1e-12) & (donor >= delta - 1e-12)
    result = np.full_like(x, np.nan)
    result[feasible] = x[feasible] * ((donor[feasible] - delta) / donor[feasible])[:, None]
    result[feasible, j] = x[feasible, j] + delta
    if feasible.any():
        check_simplex(result[feasible])
    return result, feasible


def pair(x, j, k, dj, dk):
    check_simplex(x)
    if j == k or min(dj, dk) < 0:
        raise ValueError("双领域或幅度非法")
    donor = 1 - x[:, j] - x[:, k]
    feasible = (donor > 1e-12) & (donor >= dj + dk - 1e-12)
    result = np.full_like(x, np.nan)
    result[feasible] = x[feasible] * ((donor[feasible] - dj - dk) / donor[feasible])[:, None]
    result[feasible, j] = x[feasible, j] + dj
    result[feasible, k] = x[feasible, k] + dk
    if feasible.any():
        check_simplex(result[feasible])
    return result, feasible


def support_reference(train, quantile):
    distance = cdist(train, train)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    return float(np.quantile(nearest, quantile)), nearest


def nearest_distance(x, train):
    return cdist(x, train).min(axis=1)


def interaction_value(pred_base, pred_j, pred_k, pred_both):
    return pred_both - pred_j - pred_k + pred_base
