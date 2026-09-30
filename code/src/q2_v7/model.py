"""经典律、双通道质量律、配比情景及解析导数。N、D 均以十亿计。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


BASE_NAMES = ("E", "A", "B", "alpha", "beta")
QUALITY_NAMES = BASE_NAMES + ("rho_N", "rho_D", "E1")


def predict(params: dict, n, d, q=None, hp=0.0, lam=0.0, form="A", qref=0.5, theta=0.0):
    """Form A 只调节数据项，Form B 调节两个可约项。"""
    n, d = np.asarray(n, float), np.asarray(d, float)
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(d)) or np.any(n <= 0) or np.any(d <= 0):
        raise ValueError("N、D 必须是正有限数")
    q = np.asarray(qref if q is None else q, float)
    hp = np.asarray(hp, float)
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(hp)) or not np.isfinite(lam) or not np.isfinite(theta):
        raise ValueError("Q、h_p、lambda 必须有限")
    if form not in ("A", "B"):
        raise ValueError("未知配比结构")
    dq = q - qref
    nt = params["A"] * n ** (-params["alpha"]) * np.exp(-params.get("rho_N", 0.0) * dq)
    dt = params["B"] * d ** (-params["beta"]) * np.exp(-params.get("rho_D", 0.0) * dq)
    factor = np.exp((lam + theta * dq) * hp)
    out = params["E"] - params.get("E1", 0.0) * dq + (nt + dt) * factor if form == "B" else params["E"] - params.get("E1", 0.0) * dq + nt + dt * factor
    return out


@dataclass
class Fit:
    params: dict
    attempts: list[dict]
    rss: float


def fit(n, d, loss, q=None, *, seed=7, starts=3, qref=0.5) -> Fit:
    """对正参数取 log，多个确定性起点拟合。q=None 表示经典律。"""
    n, d, y = (np.asarray(v, float) for v in (n, d, loss))
    if n.shape != d.shape or n.shape != y.shape or n.ndim != 1 or len(y) < 15:
        raise ValueError("拟合数据维度或数量不合格")
    if not np.isfinite(np.column_stack((n, d, y))).all() or (n <= 0).any() or (d <= 0).any() or (y <= 0).any():
        raise ValueError("拟合数据含非法 N、D、Loss")
    quality = q is not None
    q = np.full(len(y), qref) if q is None else np.asarray(q, float)
    if q.shape != y.shape or not np.isfinite(q).all():
        raise ValueError("Q_B 非法")
    names = QUALITY_NAMES if quality else BASE_NAMES
    lo = np.array([-15., -15., -15., np.log(.02), np.log(.02)] + ([-3., -3., -2.] if quality else []))
    hi = np.array([np.log(20.), np.log(100.), np.log(100.), np.log(2.), np.log(2.)] + ([3., 3., 2.] if quality else []))
    def decode(z):
        values = [float(np.exp(v)) for v in z[:5]] + [float(v) for v in z[5:]]
        return dict(zip(names, values))
    rng = np.random.default_rng(seed)
    attempts, good = [], []
    for i in range(starts):
        initial = np.array([max(.1, y.min() * (0.55 + .1 * i)), .5, 1.2, .28, .3] + ([.3, .15, .1] if quality else []), float)
        if i:
            initial[1:3] *= rng.uniform(.65, 1.45, 2)
            initial[3:5] *= rng.uniform(.8, 1.2, 2)
        z0 = np.array([*np.log(initial[:5]), *initial[5:]])
        z0 = np.clip(z0, lo + 1e-6, hi - 1e-6)
        res = least_squares(lambda z: predict(decode(z), n, d, q, qref=qref) - y,
                            z0, bounds=(lo, hi), max_nfev=1500, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        rss = float(res.fun @ res.fun)
        attempts.append({"start": i, "initial": decode(z0), "success": bool(res.success),
                         "nfev": int(res.nfev), "rss": rss})
        if res.success and np.isfinite(rss):
            good.append((rss, decode(res.x)))
    if not good:
        raise RuntimeError("多起点拟合全部失败")
    rss, params = min(good, key=lambda v: v[0])
    return Fit(params, attempts, rss)


def derivatives(params: dict, n: float, d: float, q: float, qref=0.5) -> dict:
    """导数及局部等损失替代率；以 L-E 为可约损失。"""
    nt = params["A"] * n ** (-params["alpha"]) * np.exp(-params.get("rho_N", 0.) * (q-qref))
    dt = params["B"] * d ** (-params["beta"]) * np.exp(-params.get("rho_D", 0.) * (q-qref))
    loss = float(predict(params, n, d, q, qref=qref))
    r = loss - params["E"]
    dn = -params["alpha"] * nt / n
    dd = -params["beta"] * dt / d
    dq = -params.get("rho_N", 0.) * nt - params.get("rho_D", 0.) * dt - params.get("E1", 0.)
    return {"L": loss, "R": r, "dL_dN": dn, "dL_dD": dd, "dL_dQ": dq,
            "epsilon_N_R": n*dn/r if r > 0 else None,
            "epsilon_D_R": d*dd/r if r > 0 else None,
            "semi_elasticity_Q_R": dq/r if r > 0 else None,
            "dN_dQ_at_fixed_L_D": -dq/dn,
            "dlogN_dQ_at_fixed_L_D": -dq/(n*dn)}


def compute_optimum(params: dict, budget_1e21: float, q: float, bounds=None, qref=0.5,
                    hp=0.0, lam=0.0, form="A") -> dict:
    """C≈6ND；N、D 用十亿计时 C[1e21]=0.006*N_B*D_B。"""
    if budget_1e21 <= 0:
        raise ValueError("算力预算必须为正")
    k = budget_1e21 / .006
    an = params["A"] * np.exp(-params.get("rho_N", 0.) * (q-qref))
    bd = params["B"] * np.exp(-params.get("rho_D", 0.) * (q-qref))
    if form == "A":
        bd *= np.exp(lam*hp)
    elif form == "B":
        an *= np.exp(lam*hp)
        bd *= np.exp(lam*hp)
    else:
        raise ValueError("未知配比结构")
    a, b = params["alpha"], params["beta"]
    n_analytic = ((a*an*k**b)/(b*bd))**(1/(a+b))
    d_analytic = k/n_analytic
    if bounds is None:
        n_bounded = n_analytic
    else:
        nlo, nhi = bounds["N_params_B"]
        dlo, dhi = bounds["D_tokens_B"]
        feasible_lo, feasible_hi = max(nlo, k/dhi), min(nhi, k/dlo)
        if feasible_lo > feasible_hi:
            raise ValueError("给定算力与校准边界没有可行解")
        n_bounded = float(np.clip(n_analytic, feasible_lo, feasible_hi))
    d_bounded = k/n_bounded
    return {"C_1e21": budget_1e21, "Q_B": q,
            "N_analytic_B": n_analytic, "D_analytic_B": d_analytic,
            "N_bounded_B": n_bounded, "D_bounded_B": d_bounded,
            "analytic_outside_bounds": bool(bounds is not None and not np.isclose(n_analytic, n_bounded)),
            "L_bounded": float(predict(params, n_bounded, d_bounded, q, hp=hp, lam=lam, form=form, qref=qref))}


def quality_incremental_cost_1e21(d_b: float, q_before: float, q_after: float, shape: str) -> float:
    """题面附录 B：C_Q=D[g(Q_after)-g(Q_before)]_+，结果单位 1e21 FLOPs。"""
    if d_b <= 0 or not 0 <= q_before <= 1 or not 0 <= q_after <= 1:
        raise ValueError("成本情景的 D 或 Q 非法")
    if shape == "exponential":
        g = lambda q: 1e7 * np.exp(6*q)
    elif shape == "power":
        g = lambda q: 5e9 * q**4
    elif shape == "logarithmic":
        g = lambda q: 2e9 * np.log1p(10*q)
    else:
        raise ValueError("未知质量成本函数")
    return float(d_b*1e9*max(0., g(q_after)-g(q_before))/1e21)
