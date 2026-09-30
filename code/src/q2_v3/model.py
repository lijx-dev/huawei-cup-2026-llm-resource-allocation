"""标度律的统一参数化、拟合和计算最优。N/D 单位为十亿。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .common import SEED

BASE = ("E", "A", "B", "alpha", "beta")
EXTRA = {"M0": (), "MD": ("rho_d",), "MN": ("rho_n",), "MND": ("rho_n", "rho_d"),
         "MCHANNEL_ADD": ("rho_n", "rho_d", "e1")}


def names(kind: str) -> tuple[str, ...]:
    return BASE + EXTRA[kind]


def predict(kind: str, params, n, d, q=0.0):
    p = dict(zip(names(kind), np.asarray(params, float)))
    n, d, q = np.asarray(n, float), np.asarray(d, float), np.asarray(q, float)
    if np.any(n <= 0) or np.any(d <= 0) or not np.isfinite(n).all() or not np.isfinite(d).all():
        raise ValueError("N,D 必须有限且大于零")
    en = np.exp(np.clip(-p.get("rho_n", 0) * q, -30, 30))
    ed = np.exp(np.clip(-p.get("rho_d", 0) * q, -30, 30))
    return p["E"] + p["A"] * n ** (-p["alpha"]) * en + p["B"] * d ** (-p["beta"]) * ed - p.get("e1", 0) * q


def bounds(kind: str):
    lo = [0, 1e-9, 1e-9, .001, .001]
    hi = [10, 100, 100, 3, 3]
    for name in EXTRA[kind]:
        lo.append(-5 if name != "e1" else -3)
        hi.append(5 if name != "e1" else 3)
    return np.array(lo), np.array(hi)


def fit(kind: str, df, starts: int = 1, seed: int = SEED, anchor: dict | None = None):
    n, d, q, y = (df["N_params_B"].to_numpy(float), df["D_tokens_B"].to_numpy(float),
                  df["Q_score"].to_numpy(float) if "Q_score" in df else np.zeros(len(df)), df["val_loss"].to_numpy(float))
    if len(df) < len(names(kind)) + 2 or not np.isfinite(np.c_[n, d, q, y]).all() or np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("拟合数据不足或含非法 N/D/Q/Loss")
    low, high = bounds(kind)
    base = np.array([max(.01, np.min(y) * .6), max(.01, np.ptp(y) * .3), max(.01, np.ptp(y) * .3), .3, .3] + [0.] * len(EXTRA[kind]))
    fixed = {names(kind).index(k): float(v) for k, v in (anchor or {}).items() if k in names(kind)}
    free = np.array([i for i in range(len(base)) if i not in fixed])
    for i, v in fixed.items(): base[i] = v
    rng = np.random.default_rng(seed)
    rows = []
    best = None
    for j in range(starts):
        start = base.copy()
        if j:
            start[:5] *= np.exp(rng.normal(0, .7, 5))
            for i in range(5, len(start)): start[i] = rng.normal(0, .2)
        start = np.clip(start, low + 1e-7, high - 1e-7)
        for i, v in fixed.items(): start[i] = v
        def unpack(x):
            p = start.copy(); p[free] = x
            return p
        def residual(x):
            return predict(kind, unpack(x), n, d, q) - y
        result = least_squares(residual, start[free], bounds=(low[free], high[free]), max_nfev=800, xtol=1e-9, ftol=1e-9)
        p = unpack(result.x)
        sse = float(np.sum(result.fun ** 2))
        rows.append({"start_id": j, "initial": start.tolist(), "parameters": p.tolist(), "success": bool(result.success), "nfev": result.nfev, "sse": sse})
        if best is None or sse < best[0]: best = (sse, p)
    return best[1], rows


def analytic_opt(kind: str, p, train_compute: float, q: float, hp: float = 0, lam: float = 0, structure: str = "P-D"):
    """C=6ND, N/D 参数均为十亿，故 C/6e18=N_b D_b。"""
    x = dict(zip(names(kind), p))
    c = train_compute / 6e18
    if c <= 0: raise ValueError("train_compute 必须大于零")
    a = x["A"] * np.exp(-x.get("rho_n", 0) * q)
    b = x["B"] * np.exp(-x.get("rho_d", 0) * q)
    if structure == "P-D": b *= np.exp(lam * hp)
    elif structure == "P-R": pass  # 公共倍率不改变 N/D 最优位置
    else: raise ValueError(structure)
    n = ((x["alpha"] * a) / (x["beta"] * b * c ** (-x["beta"]))) ** (1 / (x["alpha"] + x["beta"]))
    d = c / n
    loss = float(predict(kind, p, n, d, q))
    if structure == "P-D":
        raw_b = x["B"] * np.exp(-x.get("rho_d", 0) * q)
        loss += float(raw_b * d ** (-x["beta"]) * (np.exp(lam * hp) - 1))
    else:
        reducible = loss - x["E"] + x.get("e1", 0) * q
        loss += float((np.exp(lam * hp) - 1) * reducible)
    return float(n), float(d), float(loss)


def bounded_opt(kind, p, train_compute, q, limits, hp=0, lam=0, structure="P-D"):
    nmin, nmax, dmin, dmax = limits
    c = train_compute / 6e18
    low, high = max(nmin, c / dmax), min(nmax, c / dmin)
    if low > high: return None
    def objective(logn):
        n = np.exp(logn); d = c / n
        x = dict(zip(names(kind), p))
        base = float(predict(kind, p, n, d, q))
        if structure == "P-D": base += x["B"] * d ** (-x["beta"]) * np.exp(-x.get("rho_d", 0) * q) * (np.exp(lam * hp) - 1)
        else: base += (np.exp(lam * hp) - 1) * (base - x["E"] + x.get("e1", 0) * q)
        return base
    r = minimize_scalar(objective, bounds=(np.log(low), np.log(high)), method="bounded", options={"xatol": 1e-12})
    options = [(objective(np.log(low)), low), (objective(np.log(high)), high), (float(r.fun), float(np.exp(r.x)))]
    value, n = min(options)
    return float(n), float(c / n), float(value)
