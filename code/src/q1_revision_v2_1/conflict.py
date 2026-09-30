"""同一标准化矩阵上的加权、等权和固定语义组三种样本分歧。"""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from q1_revision_v2.conflict import sample_conflict
from .common import paths, save_json


METHODS = ("C_weighted_hierarchical", "C_equal_indicator", "C_semantic_group")


def three_disagreements(z, global_weight, groups):
    m = z.shape[1]
    i, j = np.triu_indices(m, k=1)
    weighted = sample_conflict(z, global_weight, i, j)
    equal = np.abs(z[:, i] - z[:, j]).mean(axis=1)
    g = np.column_stack([groups[name] for name in groups])
    semantic = (np.abs(g[:,0]-g[:,1]) + np.abs(g[:,0]-g[:,2]) + np.abs(g[:,1]-g[:,2])) / 3
    return np.column_stack([weighted, equal, semantic])


def run(root, config):
    _, out, _ = paths(root)
    dest = out / "conflict"
    dest.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(out / "quality/sample_quality_scores.csv.gz", keep_default_na=False)
    z = np.load(out / "quality/normalized_22.npy", mmap_mode="r")
    params = pd.read_csv(out / "quality/hierarchical_weights.csv")
    names = pd.read_csv(out / "quality/normalization_parameters.csv").field.tolist()
    weight_map = params.set_index("field").global_weight
    w = np.array([weight_map[field] for field in names])
    group_scores = {name: scores["S_"+name].to_numpy() for name in config["groups"]}
    result = np.empty((len(scores),3), dtype=float)
    for start in range(0, len(scores), config["chunk_size"]):
        stop = min(start + config["chunk_size"], len(scores))
        result[start:stop] = three_disagreements(np.asarray(z[start:stop]), w,
                                                  {name: a[start:stop] for name,a in group_scores.items()})
    if not np.isfinite(result).all() or result.min() < -1e-10 or result.max() > 1+1e-10:
        raise RuntimeError("三种分歧指标不在 [0,1]")
    for col, method in enumerate(METHODS):
        scores[method] = result[:,col]
    group_names = list(config["groups"])
    pair_names = [(group_names[0],group_names[1]), (group_names[0],group_names[2]), (group_names[1],group_names[2])]
    for a,b in pair_names:
        scores[f"gap_{a}_vs_{b}"] = np.abs(group_scores[a] - group_scores[b])
    thresholds = {}
    a1 = scores.attachment == "A1"
    for method in METHODS:
        thresholds[method] = {str(q): float(np.quantile(scores.loc[a1, method], q)) for q in config["disagreement_quantiles"]}
        for q, threshold in thresholds[method].items():
            scores[f"high_{method}_{q}"] = scores[method] > threshold
    scores[["attachment", "line_number", "record_id", "domain", "status", *METHODS,
            *[f"gap_{a}_vs_{b}" for a,b in pair_names],
            *[f"high_{method}_{q}" for method in METHODS for q in thresholds[method]]]].to_csv(dest / "sample_disagreement_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    save_json(dest / "disagreement_thresholds.json", {"reference": "A1 accepted source view", "thresholds": thresholds})
    union = scores.status == "valid"
    correlation = scores.loc[union, list(METHODS)].corr(method="spearman")
    correlation.to_csv(dest / "method_spearman.csv")
    comparison = []
    for first in METHODS:
        for second in METHODS:
            if first >= second:
                continue
            subset = scores[a1]
            high_a = set(subset.loc[subset[f"high_{first}_0.9"], "record_id"])
            high_b = set(subset.loc[subset[f"high_{second}_0.9"], "record_id"])
            comparison.append({"first": first, "second": second,
                               "union_unique_spearman": correlation.loc[first,second],
                               "A1_high90_overlap_fraction": len(high_a & high_b) / min(len(high_a),len(high_b))})
    pd.DataFrame(comparison).to_csv(dest / "conflict_method_comparison.csv", index=False)
    domain_rows, gap_rows = [], []
    views = [("union_unique", scores[union]), ("A1", scores[scores.attachment=="A1"]),
             ("A2", scores[scores.attachment=="A2"]), ("A3", scores[scores.attachment=="A3"])]
    gap_cols = [f"gap_{a}_vs_{b}" for a,b in pair_names]
    for view, frame in views:
        for domain, sub in frame.groupby("domain"):
            for method in METHODS:
                domain_rows.append({"view": view, "domain": domain, "method": method, "n": len(sub),
                                    "mean_C": sub[method].mean(), "median_C": sub[method].median(),
                                    **{f"rate_{q}": sub[f"high_{method}_{q}"].mean() for q in thresholds[method]}})
            for high_method in ("all", *METHODS):
                selection = sub if high_method == "all" else sub[sub[f"high_{high_method}_0.9"]]
                if len(selection):
                    gap_rows.extend({"view": view, "domain": domain, "selection": high_method,
                                     "group_pair": col, "n": len(selection), "mean_gap": selection[col].mean()}
                                    for col in gap_cols)
    pd.DataFrame(domain_rows).to_csv(dest / "domain_disagreement_summary.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(dest / "semantic_group_disagreement.csv", index=False)
    rho = pd.read_csv(out / "quality/spearman_signed.csv", index_col=0).to_numpy()
    distance = np.sqrt(np.maximum(0, (1-rho)/2))
    np.fill_diagonal(distance, 0)
    clusters = []
    for method in config["cluster_linkages"]:
        tree = linkage(squareform(distance), method=method)
        for k in config["cluster_candidate_k"]:
            labels = fcluster(tree, k, criterion="maxclust")
            clusters.append({"linkage": method, "requested_k": k, "actual_k": len(set(labels)),
                             "silhouette": (silhouette_score(distance, labels, metric="precomputed")
                                            if 1 < len(set(labels)) < len(labels) else np.nan)})
            if method == config["cluster_primary_linkage"] and k == config["cluster_primary_k"]:
                chosen = labels
    pd.DataFrame(clusters).to_csv(dest / "cluster_sensitivity.csv", index=False)
    pd.DataFrame({"field": names, "cluster": chosen}).to_csv(dest / "exploratory_clusters.csv", index=False)
    np.save(dest / "primary_linkage.npy", linkage(squareform(distance), method=config["cluster_primary_linkage"]))
    i,j = np.triu_indices(len(names), k=1)
    denominator = np.sum(w[i]*w[j])
    weighted_rows, equal_rows = [], []
    for attachment in ("A1","A2","A3"):
        for selection_method in ("C_weighted_hierarchical", "C_equal_indicator"):
            mask = ((scores.attachment==attachment) & scores[f"high_{selection_method}_0.9"]).to_numpy()
            indices = np.flatnonzero(mask)
            raw_sum = np.zeros(len(i))
            weighted_sum = np.zeros(len(i))
            for start in range(0, len(indices), config["chunk_size"]):
                block = np.asarray(z[indices[start:start+config["chunk_size"]]])
                diff = np.abs(block[:,i] - block[:,j])
                raw_sum += diff.sum(axis=0)
                weighted_sum += (diff * w[i] * w[j] / denominator).sum(axis=0)
            if len(indices):
                raw_sum /= len(indices)
                weighted_sum /= len(indices)
            for a,b,raw_value,weighted_value in zip(i,j,raw_sum,weighted_sum):
                common = {"attachment": attachment, "high_set_method": selection_method,
                          "field_j": names[a], "field_k": names[b], "n_high": len(indices)}
                weighted_rows.append({**common, "H_weighted": weighted_value})
                equal_rows.append({**common, "H_equal_raw": raw_value})
    pd.DataFrame(weighted_rows).to_csv(dest / "weighted_pair_contributions.csv", index=False)
    pd.DataFrame(equal_rows).to_csv(dest / "equal_pair_disagreement.csv", index=False)
    numeric = "rps_lines_numerical_chars_fraction"
    dominance = []
    for label, frame, col in (("weighted",pd.DataFrame(weighted_rows),"H_weighted"),
                              ("equal",pd.DataFrame(equal_rows),"H_equal_raw")):
        for (attachment, high_method), sub in frame.groupby(["attachment","high_set_method"]):
            incident = sub[(sub.field_j==numeric)|(sub.field_k==numeric)]
            dominance.append({"measure": label, "attachment": attachment, "high_set_method": high_method,
                              "incident_pair_share": incident[col].sum()/sub[col].sum() if sub[col].sum()>0 else np.nan,
                              "incident_pairs_in_top10": int(((sub.nlargest(10,col).field_j==numeric)|
                                                               (sub.nlargest(10,col).field_k==numeric)).sum())})
    pd.DataFrame(dominance).to_csv(dest / "numerical_chars_dominance.csv", index=False)
    return {"thresholds": thresholds, "pair_count": len(i), "accepted_rows": len(scores),
            "method_spearman": correlation.to_dict()}
