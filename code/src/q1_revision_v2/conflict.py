"""固定 P1 尺度和权重后的指标聚类、样本分歧及跨集验证。"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from .common import paths, save_json


def sample_conflict(z, w, i, j):
    pair_weights = w[i] * w[j]
    denominator = pair_weights.sum()
    if denominator <= 0:
        raise ValueError("231 对指标权重分母为零")
    return (np.abs(z[:, i] - z[:, j]) * pair_weights).sum(axis=1) / denominator


def run(root, config):
    _, out = paths(root)
    dest = out / "conflict"
    dest.mkdir(parents=True, exist_ok=True)
    qdir = out / "quality"
    meta = pd.read_csv(qdir / "sample_scores.csv.gz", keep_default_na=False)
    z = np.load(qdir / "normalized_22.npy", mmap_mode="r")
    names = config["quality_fields"]
    params = pd.read_csv(qdir / "indicator_parameters.csv")
    w = params.w_main.to_numpy()
    rho = pd.read_csv(qdir / "spearman_signed.csv", index_col=0).to_numpy()
    dist = np.sqrt(np.maximum(0, (1 - rho) / 2))
    np.fill_diagonal(dist, 0)
    i, j = np.triu_indices(len(w), k=1)
    linkage_rows = []
    for method in ("average", "complete", "single"):
        tree = linkage(squareform(dist), method=method)
        for k in config["cluster_candidate_k"]:
            labels = fcluster(tree, k, criterion="maxclust")
            silhouette = (silhouette_score(dist, labels, metric="precomputed")
                          if 1 < len(set(labels)) < len(labels) else np.nan)
            linkage_rows.append({"linkage": method, "requested_k": k,
                                 "actual_k": len(set(labels)), "silhouette": silhouette})
            if method == config["cluster_primary_linkage"] and k == config["cluster_primary_k"]:
                selected = labels
    pd.DataFrame(linkage_rows).to_csv(dest / "cluster_sensitivity.csv", index=False)
    pd.DataFrame({"field": names, "cluster": selected, "weight": w}).to_csv(dest / "clusters.csv", index=False)
    for cluster in sorted(set(selected)):
        idx = np.flatnonzero(selected == cluster)
        den = w[idx].sum()
        meta[f"cluster_{cluster}"] = (np.asarray(z[:, idx]) @ w[idx]) / den if den > 0 else np.nan
    c = np.empty(len(meta))
    step = config["chunk_size"]
    for start in range(0, len(meta), step):
        c[start:start+step] = sample_conflict(np.asarray(z[start:start+step]), w, i, j)
    meta["C"] = c
    a1 = meta.attachment == "A1"
    thresholds = {str(q): float(np.quantile(c[a1], q)) for q in config["conflict_threshold_quantiles"]}
    for q, threshold in thresholds.items():
        meta[f"high_{q}"] = c > threshold
    cluster_cols = [f"cluster_{x}" for x in sorted(set(selected))]
    cluster_pairs = [(x, y) for x in range(len(cluster_cols)) for y in range(x+1, len(cluster_cols))]
    values = meta[cluster_cols].to_numpy()
    difference = np.column_stack([np.abs(values[:, x] - values[:, y]) for x, y in cluster_pairs])
    best = difference.argmax(axis=1)
    meta["max_cluster_pair"] = [cluster_cols[cluster_pairs[x][0]] + "+" + cluster_cols[cluster_pairs[x][1]] for x in best]
    meta["max_cluster_gap"] = difference[np.arange(len(meta)), best]
    meta.to_csv(dest / "sample_conflict.csv.gz", index=False)
    rates = []
    for (attachment, domain), sub in meta.groupby(["attachment", "domain"]):
        row = {"attachment": attachment, "domain": domain, "n": len(sub), "C_mean": sub.C.mean()}
        for q in thresholds:
            row[f"rate_{q}"] = sub[f"high_{q}"].mean()
        rates.append(row)
    pd.DataFrame(rates).to_csv(dest / "domain_conflict_rates.csv", index=False)
    contributions = []
    for attachment in ("A1", "A2", "A3"):
        mask = ((meta.attachment == attachment) & meta["high_0.9"]).to_numpy()
        selected_rows = np.flatnonzero(mask)
        numerator = np.zeros(len(i))
        den = np.sum(w[i] * w[j])
        for start in range(0, len(selected_rows), step):
            block = np.asarray(z[selected_rows[start:start+step]])
            numerator += (np.abs(block[:, i] - block[:, j]) * w[i] * w[j] / den).sum(axis=0)
        h = numerator / len(selected_rows) if len(selected_rows) else numerator
        contributions.extend({"attachment": attachment, "field_j": names[a], "field_k": names[b],
                              "H_jk": float(value), "n_high": len(selected_rows)}
                             for a, b, value in zip(i, j, h))
    pd.DataFrame(contributions).to_csv(dest / "indicator_pair_contributions.csv", index=False)
    pd.crosstab(meta.loc[meta["high_0.9"], "attachment"],
                meta.loc[meta["high_0.9"], "max_cluster_pair"], normalize="index").to_csv(dest / "high_conflict_cluster_pairs.csv")
    summary = {"thresholds_from_A1": thresholds, "pair_count": len(i),
               "primary_cluster_method": config["cluster_primary_linkage"],
               "primary_cluster_requested_k": config["cluster_primary_k"],
               "high_conflict_is_corrupt": False}
    save_json(dest / "conflict_summary.json", summary)
    return summary
