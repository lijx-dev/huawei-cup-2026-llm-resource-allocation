"""百分制、归一化评分与等权/TOPSIS 对照。"""
import math


def score_row(normalized, weights):
    if len(normalized) != len(weights) or not weights:
        raise ValueError("指标与权重列数不一致")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in normalized):
        raise ValueError("标准化指标非法")
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(math.fsum(weights), 1, abs_tol=1e-9):
        raise ValueError("权重非法")
    q = math.fsum(z*w for z,w in zip(normalized, weights))
    return {"q":q, "Q":100*q, "Q_equal":100*math.fsum(normalized)/len(normalized)}


def topsis_closeness(normalized, weights):
    if len(normalized) != len(weights) or not weights:
        raise ValueError("指标与权重列数不一致")
    if any(not 0 <= x <= 1 for x in normalized):
        raise ValueError("标准化指标非法")
    plus = math.sqrt(math.fsum((w*(1-x))**2 for x,w in zip(normalized, weights)))
    minus = math.sqrt(math.fsum((w*x)**2 for x,w in zip(normalized, weights)))
    if plus + minus == 0:
        raise ValueError("理想解距离同时为零")
    return minus/(plus+minus)
