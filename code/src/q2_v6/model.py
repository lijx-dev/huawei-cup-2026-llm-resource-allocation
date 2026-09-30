"""Q2 V6 独立重估的经典律和质量模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


BASE = ("E", "A", "B", "alpha", "beta")
MODELS = {"M0": (), "MD": ("rho_D",), "MN": ("rho_N",),
          "MND": ("rho_N", "rho_D"), "MCHANNEL_ADD": ("rho_N", "rho_D", "E1")}
LOW = {"E": 1e-8, "A": 1e-8, "B": 1e-8, "alpha": 1e-3, "beta": 1e-3,
       "rho_N": -10, "rho_D": -10, "E1": -10}
HIGH = {"E": 20, "A": 1e6, "B": 1e6, "alpha": 3, "beta": 3,
        "rho_N": 10, "rho_D": 10, "E1": 10}


def predict(model: str, p: dict, n: np.ndarray, d: np.ndarray,
            q: np.ndarray | float = 0.5, q0: float = 0.5) -> np.ndarray:
    if model not in MODELS:
        raise ValueError("未知质量模型")
    n, d, q = np.asarray(n, float), np.asarray(d, float), np.asarray(q, float)
    if np.any(n <= 0) or np.any(d <= 0) or not np.isfinite(n).all() or not np.isfinite(d).all() or not np.isfinite(q).all():
        raise ValueError("N、D、Q 非法")
    delta = q - q0
    nterm = p["A"] * n ** (-p["alpha"])
    dterm = p["B"] * d ** (-p["beta"])
    if "rho_N" in MODELS[model]:
        nterm *= np.exp(-p["rho_N"] * delta)
    if "rho_D" in MODELS[model]:
        dterm *= np.exp(-p["rho_D"] * delta)
    return p["E"] + nterm + dterm - (p["E1"] * delta if "E1" in MODELS[model] else 0)


@dataclass
class Fit:
    params: dict
    rss: float
    attempts: list[dict]


def fit(model: str, n: np.ndarray, d: np.ndarray, y: np.ndarray,
        q: np.ndarray | None = None, q0: float = 0.5, starts: int = 3) -> Fit:
    if model not in MODELS:
        raise ValueError("未知模型")
    n, d, y = np.asarray(n, float), np.asarray(d, float), np.asarray(y, float)
    q = np.full(len(y), q0) if q is None else np.asarray(q, float)
    if len(y) < 8 or len(np.unique(n)) < 2 or len(np.unique(d)) < 2:
        raise ValueError("拟合数据支持不足")
    if not np.isfinite(np.column_stack((n, d, q, y))).all() or (n <= 0).any() or (d <= 0).any() or (y <= 0).any():
        raise ValueError("拟合输入非法")
    names = BASE + MODELS[model]
    log_idx = [name in BASE for name in names]
    lower = np.array([np.log(LOW[name]) if log else LOW[name] for name, log in zip(names, log_idx)])
    upper = np.array([np.log(HIGH[name]) if log else HIGH[name] for name, log in zip(names, log_idx)])

    def decode(z: np.ndarray) -> dict:
        return {name: float(np.exp(value) if log else value) for name, log, value in zip(names, log_idx, z)}

    def encode(p: dict) -> np.ndarray:
        return np.array([np.log(p[name]) if log else p[name] for name, log in zip(names, log_idx)])

    floor = float(max(1e-5, min(y.min() * .7, 10)))
    seeds = [(floor, 1., 1., .3, .3, 0.), (max(1e-5, y.min() * .5), 2., 1., .5, .2, .2),
             (max(1e-5, y.min() * .3), 1., 2., .2, .5, -.2)]
    attempts = []
    candidates = []
    for index, seed in enumerate(seeds[:starts]):
        initial = dict(zip(BASE, seed[:5]))
        initial.update({name: seed[5] for name in MODELS[model]})
        z0 = np.clip(encode(initial), lower + 1e-8, upper - 1e-8)
        result = least_squares(lambda z: predict(model, decode(z), n, d, q, q0) - y,
                               z0, bounds=(lower, upper), max_nfev=2500,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        rss = float(result.fun @ result.fun)
        attempts.append({"start": index, "success": bool(result.success), "rss": rss,
                         "nfev": int(result.nfev), "initial": initial})
        if result.success and np.isfinite(rss):
            candidates.append((rss, decode(result.x)))
    if not candidates:
        raise RuntimeError("非线性拟合多起点未收敛")
    rss, params = min(candidates, key=lambda item: item[0])
    return Fit(params, rss, attempts)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if len(y) != len(pred) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("评测数据非法")
    residual = pred - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual**2) / ss) if ss > 0 else None,
            "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else None}
