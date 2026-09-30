"""A1 固定参考 Min–Max；对合法分布外值计数并按预定策略处理。"""
import json
import math


def fit_minmax(rows, directions):
    if not rows or not directions or any(x not in {"positive", "negative"} for x in directions):
        raise ValueError("缺少样本或指标方向")
    width = len(directions)
    minima = [math.inf] * width
    maxima = [-math.inf] * width
    for row in rows:
        if len(row) != width:
            raise ValueError("矩阵列数不一致")
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("非有限或非法指标值")
            minima[j] = min(minima[j], value)
            maxima[j] = max(maxima[j], value)
    if any(low == high for low, high in zip(minima, maxima)):
        raise ValueError("常量指标使 22 维主模型不可识别")
    return [{"minimum": low, "maximum": high, "direction": direction}
            for low, high, direction in zip(minima, maxima, directions)]


def transform_minmax(row, parameters, policy="clip_with_count"):
    if len(row) != len(parameters):
        raise ValueError("指标列数与拟合参数不一致")
    if policy not in {"clip_with_count", "reject_out_of_reference"}:
        raise ValueError("未知分布外策略")
    result, outside = [], []
    for value, item in zip(row, parameters):
        low, high, direction = item["minimum"], item["maximum"], item["direction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("非有限或非法指标值")
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("无效的 Min–Max 参考范围")
        flag = value < low or value > high
        if flag and policy == "reject_out_of_reference":
            raise ValueError("合法但超出 A1 参考范围的指标值")
        bounded = min(max(value, low), high) if flag else value
        z = (bounded - low) / (high - low) if direction == "positive" else (high - bounded) / (high - low)
        if direction not in {"positive", "negative"} or not 0 <= z <= 1:
            raise ValueError("归一化结果或方向非法")
        result.append(z)
        outside.append(flag)
    return result, outside


def save_parameters(path, parameters):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(parameters, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def load_parameters(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)
