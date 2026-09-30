"""领域描述统计和平均分的固定 95% 正态近似区间。"""
import math
import statistics


def _quantile(sorted_values, p):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values)-1)*p
    left = int(position)
    right = min(left+1, len(sorted_values)-1)
    return sorted_values[left]*(right-position) + sorted_values[right]*(position-left)


def summarize_domain(scores):
    if not scores or any(not math.isfinite(x) or not 0 <= x <= 100 for x in scores):
        raise ValueError("领域评分为空或非法")
    ordered = sorted(scores)
    n = len(ordered)
    mean = statistics.fmean(ordered)
    std = statistics.stdev(ordered) if n > 1 else 0.0
    margin = 1.96*std/math.sqrt(n) if n > 1 else 0.0
    return {"n":n,"mean_Q":mean,"median_Q":statistics.median(ordered),"std_Q":std,
            "q25_Q":_quantile(ordered,0.25),"q75_Q":_quantile(ordered,0.75),
            "ci95_low_Q":max(0.0,mean-margin),"ci95_high_Q":min(100.0,mean+margin),
            "ci_method":"normal_approximation_iid_assumption"}
