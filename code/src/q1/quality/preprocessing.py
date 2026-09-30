"""指标模式审核与已确认列表的标量化。"""
import json
import math


class SchemaNotReadyError(ValueError):
    """22 维指标规则尚未核定。"""


def load_schema(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def schema_blockers(schema, expected_scalar, expected_list):
    specs = schema["indicators"]
    names = [item["name"] for item in specs]
    expected = [*expected_scalar, *expected_list]
    blockers = []
    if len(names) != 22 or len(names) != len(set(names)) or set(names) != set(expected):
        blockers.append("指标名称未精确覆盖审计确认的 22 项")
    for item in specs:
        name = item["name"]
        if item["raw_type"] != ("list" if name in expected_list else "scalar"):
            blockers.append(f"{name}: 原始类型与 P0 不符")
        if item["scalarization"] is None:
            blockers.append(f"{name}: 缺少标量化规则")
        if item["direction"] not in {"positive", "negative"}:
            blockers.append(f"{name}: 缺少经确认的评价方向")
        if item.get("decision"):
            blockers.append(f"{name}: {item['decision']}")
    return blockers


def require_schema_ready(schema, expected_scalar, expected_list):
    blockers = schema_blockers(schema, expected_scalar, expected_list)
    if blockers:
        raise SchemaNotReadyError("; ".join(blockers))


def _finite_list(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"列表长度应为 {length}")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("列表含非有限值或非数值元素")
    return [float(x) for x in value]


def stable_softmax(logits):
    values = _finite_list(logits, len(logits)) if isinstance(logits, list) else None
    if values is None or len(values) < 2:
        raise ValueError("Softmax 至少需要两个有限 logits")
    peak = max(values)
    shifted = [math.exp(x - peak) for x in values]
    total = math.fsum(shifted)
    return [x / total for x in shifted]


def ordered_probability_expectation(probabilities):
    values = _finite_list(probabilities, len(probabilities)) if isinstance(probabilities, list) else None
    if values is None or len(values) < 2 or any(x < 0 for x in values):
        raise ValueError("有序概率列表非法")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("概率列表之和必须为 1")
    return math.fsum(i * p / (len(values) - 1) for i, p in enumerate(values))


def scalarize(value, spec):
    method = spec.get("scalarization")
    length = spec.get("expected_length")
    if method == "identity":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("标量值非法")
        return float(value)
    if method == "log1p":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("log1p 输入必须是非负有限数")
        return math.log1p(value)
    if method == "interval_desirability":
        low, high = spec["interval"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("区间得分输入必须是非负有限数")
        if not 0 < low < high:
            raise ValueError("参考区间非法")
        if value < low:
            return value / low
        if value > high:
            return high / value
        return 1.0
    if method == "singleton":
        return _finite_list(value, 1)[0]
    if method == "list_element_index_3":
        return _finite_list(value, 4)[3]
    if method == "list_equal_mean_four":
        return math.fsum(_finite_list(value, 4)) / 4
    if method == "ordered_logits_expectation":
        return ordered_probability_expectation(stable_softmax(_finite_list(value, length)))
    if method == "ordered_probability_expectation":
        return ordered_probability_expectation(_finite_list(value, length))
    if method == "binary_logits_probability_index_1":
        return stable_softmax(_finite_list(value, 2))[1]
    raise SchemaNotReadyError(f"{spec['name']}: 未确认的标量化方法")
