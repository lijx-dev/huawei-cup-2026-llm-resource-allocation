"""熵权与 Spearman 冗余修正，输入必须是同一批完整样本。"""
import math


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        mean_rank = (pos + 1 + end) / 2
        for k in range(pos, end):
            ranks[order[k]] = mean_rank
        pos = end
    return ranks


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("相关系数样本数非法")
    mean_left = math.fsum(left) / len(left)
    mean_right = math.fsum(right) / len(right)
    a = [x - mean_left for x in left]
    b = [x - mean_right for x in right]
    denominator = math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
    if denominator == 0:
        raise ValueError("常量列的相关系数未定义")
    result = math.fsum(x*y for x, y in zip(a, b)) / denominator
    if not math.isfinite(result) or abs(result) > 1 + 1e-10:
        raise ValueError("相关系数非法")
    return max(-1.0, min(1.0, result))


def spearman_matrix(columns):
    ranked = [average_ranks(column) for column in columns]
    m = len(ranked)
    result = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            result[i][j] = result[j][i] = pearson(ranked[i], ranked[j])
    return result


def entropy_spearman_weights(rows):
    if len(rows) < 2 or not rows[0] or len(rows[0]) < 2:
        raise ValueError("熵权至少需要两个样本与两个指标")
    n, m = len(rows), len(rows[0])
    columns = [[] for _ in range(m)]
    for row in rows:
        if len(row) != m:
            raise ValueError("矩阵列数不一致")
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("标准化矩阵必须是有限的 [0,1] 数值")
            columns[j].append(value)
    entropies, differences = [], []
    for column in columns:
        if min(column) == max(column):
            raise ValueError("常量列不能计算熵权和 Spearman")
        total = math.fsum(column)
        if total <= 0:
            raise ValueError("指标概率分母为零")
        entropy = -math.fsum((z/total)*math.log(z/total) for z in column if z > 0) / math.log(n)
        if entropy < -1e-10 or entropy > 1+1e-10:
            raise ValueError("熵值超出 [0,1]")
        entropy = max(0.0, min(1.0, entropy))
        entropies.append(entropy)
        differences.append(1 - entropy)
    correlation = spearman_matrix(columns)
    independence = [math.fsum(1 - abs(correlation[j][k]) for k in range(m) if k != j)/(m-1)
                    for j in range(m)]
    information = [differences[j]*independence[j] for j in range(m)]
    denominator = math.fsum(information)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("修正信息量之和为零或非法")
    weights = [value/denominator for value in information]
    if any(value < 0 for value in weights) or not math.isclose(math.fsum(weights), 1, abs_tol=1e-10):
        raise ValueError("权重非法")
    return {"entropy":entropies, "difference":differences, "spearman":correlation,
            "independence":independence, "information":information, "weights":weights,
            "sample_count":n, "indicator_count":m}
