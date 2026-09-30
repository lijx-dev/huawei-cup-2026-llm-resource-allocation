"""在 V5 已冻结质量参数上，替换为 M2 冻结配比响应的情景计算。"""

from __future__ import annotations

import numpy as np


def predict(params: dict, n: float, d: float, q: float, q0: float,
            h: np.ndarray, lam: float, form: str, model: str = "MCHANNEL_ADD") -> np.ndarray:
    if n <= 0 or d <= 0 or not np.isfinite([n, d, q, q0, lam]).all():
        raise ValueError("N、D、Q 或 lambda 非法")
    if form not in ("A", "B"):
        raise ValueError("未知 Form")
    from .model import MODELS

    if model not in MODELS:
        raise ValueError("未知质量模型")
    h = np.asarray(h, dtype=float)
    if not np.isfinite(h).all():
        raise ValueError("h_p 非法")
    dq = q - q0
    nterm = params["A"] * n ** (-params["alpha"]) * np.exp(-params.get("rho_N", 0) * dq)
    dterm = params["B"] * d ** (-params["beta"]) * np.exp(-params.get("rho_D", 0) * dq)
    factor = np.exp(lam * h)
    value = params["E"] + nterm + dterm * factor - params.get("E1", 0) * dq if form == "A" else (
        params["E"] + (nterm + dterm) * factor - params.get("E1", 0) * dq)
    if not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("情景预测 Loss 非法")
    return value
