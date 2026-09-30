"""A17 审计、统计距离与显式低可信度映射。"""
import json
import lzma
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def audit_a17(root, domains):
    base = Path(root) / "data/real_attachments/A_data_value"
    table = pd.read_csv(base / "regmix_domain_summary.csv")
    needed = ["domain", "source_path", "sample_rows", "sample_bytes_requested", "avg_text_chars"]
    if list(table.columns) != needed or table.domain.duplicated().any() or set(table.domain) != set(domains):
        raise ValueError("A17 字段、重复项或领域覆盖异常")
    if ((table.sample_rows <= 0) | (table.avg_text_chars <= 0) | (table.sample_bytes_requested <= 0)).any():
        raise ValueError("A17 统计量非正")
    counts = defaultdict(int)
    lengths = defaultdict(int)
    sources = defaultdict(set)
    with lzma.open(base / "regmix_domain_sample.jsonl.xz", "rt", encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            row = json.loads(raw)
            domain = row.get("_source_domain")
            if domain not in domains or not isinstance(row.get("text"), str):
                raise ValueError(f"A17 原始抽样第 {line} 行无效")
            counts[domain] += 1
            lengths[domain] += len(row["text"])
            sources[domain].add(row.get("_source_path"))
    report = []
    for row in table.itertuples(index=False):
        observed_mean = lengths[row.domain] / counts[row.domain]
        ok = counts[row.domain] == row.sample_rows and abs(observed_mean - row.avg_text_chars) <= 0.011 and sources[row.domain] == {row.source_path}
        report.append({"domain": row.domain, "reported_rows": row.sample_rows, "observed_rows": counts[row.domain],
                       "reported_avg_text_chars": row.avg_text_chars, "observed_avg_text_chars": observed_mean,
                       "source_path_match": sources[row.domain] == {row.source_path}, "verified": ok})
    return table.set_index("domain").loc[domains].reset_index(), pd.DataFrame(report)


def features_and_similarity(a17, domains):
    raw = np.log(a17.set_index("domain").loc[domains, ["sample_rows", "avg_text_chars"]].to_numpy(float))
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    if (std <= 0).any():
        raise ValueError("A17 特征常量，无法标准化")
    z = (raw - mean) / std
    distance = cdist(z, z)
    return z, distance, np.exp(-distance), {"mean": mean.tolist(), "std": std.tolist()}


def build_mapping(a16, domains, quality_domains, distances, temperature=1.0, assisted=True):
    if temperature <= 0:
        raise ValueError("映射温度必须为正")
    guide = a16.set_index("mixture_domain")
    if guide.index.duplicated().any() or set(guide.index) != set(domains):
        raise ValueError("A16 覆盖或唯一性异常")
    matrix = np.full((len(domains), len(quality_domains)), np.nan)
    status = []
    representatives = {}
    for i, name in enumerate(domains):
        row = guide.loc[name]
        typ, quality = row.mapping_type, row.quality_domain
        if typ in {"direct", "near_direct"} and quality in quality_domains:
            matrix[i] = 0
            matrix[i, quality_domains.index(quality)] = 1
            representatives.setdefault(quality, []).append(i)
            state = typ
        elif typ == "inferred" and quality == "(none)":
            state = "unmapped"
        else:
            raise ValueError(f"A16 {name} 映射值异常")
        status.append({"mixture_domain": name, "a16_type": typ, "a16_quality_domain": quality, "mapping_status": state})
    if assisted:
        represented = [q for q in quality_domains if q in representatives]
        for i, row in enumerate(status):
            if row["mapping_status"] != "unmapped":
                continue
            d = np.array([min(distances[i, representatives[q]]) for q in represented])
            w = np.exp(-(d - min(d)) / temperature)
            w /= w.sum()
            matrix[i] = 0
            matrix[i, [quality_domains.index(q) for q in represented]] = w
            row["mapping_status"] = "inferred_a17_low_confidence"
    return matrix, pd.DataFrame(status), representatives


def project(x, matrix, q7):
    x = np.asarray(x, float)
    mapped = np.isfinite(matrix).all(axis=1)
    coverage = x[:, mapped].sum(axis=1)
    complete = np.all(x[:, ~mapped] <= 1e-12, axis=1)
    q17 = np.full(len(matrix), np.nan)
    q17[mapped] = matrix[mapped] @ q7
    qmix = np.full(len(x), np.nan)
    if complete.any():
        qmix[complete] = x[complete][:, mapped] @ q17[mapped]
    return q17, qmix, coverage, complete
