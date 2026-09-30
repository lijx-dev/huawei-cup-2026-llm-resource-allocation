"""固定三语义组：组间等权，组内熵信息差异与组内 Spearman 冗余。"""
import json
import lzma

import numpy as np
import pandas as pd

from q1.audit.inventory import expected_files
from q1_revision_v2.quality import scalarize, entropy_weights
from .common import paths, save_json


def validate_groups(fields, groups):
    members = [field for names in groups.values() for field in names]
    if len(groups) != 3 or len(members) != 22 or len(set(members)) != 22 or set(members) != set(fields):
        raise ValueError("三个固定语义组必须恰好覆盖 22 个字段且无重叠")


def group_weights(z_reference, rho, fields, groups, excluded=()):
    """每组单独算熵差异及组内绝对 Spearman 冗余；退化组记录 fallback。"""
    excluded = set(excluded)
    global_weight = np.zeros(len(fields), dtype=float)
    rows, fallback = [], []
    score_indices = {}
    for group, names in groups.items():
        indices = np.array([fields.index(name) for name in names if name not in excluded], dtype=int)
        if not len(indices):
            raise RuntimeError(f"{group} 排除后无指标")
        block = z_reference[:, indices]
        variable = np.ptp(block, axis=0) > 1e-12
        if not variable.any():
            raise RuntimeError(f"{group} 无非退化有效指标，无法执行确定性 fallback")
        try:
            d, _ = entropy_weights(block)
        except RuntimeError:
            d = np.zeros(len(indices))
        subset_rho = rho[np.ix_(indices, indices)]
        if len(indices) > 1:
            redundancy = (np.abs(subset_rho).sum(axis=1) - 1) / (len(indices) - 1)
        else:
            redundancy = np.zeros(len(indices))
        nonredundancy = np.clip(1 - redundancy, 0, 1)
        information = d * nonredundancy
        if information.sum() <= 1e-15:
            within = variable.astype(float) / variable.sum()
            fallback.append({"group": group, "reason": "all_entropy_redundancy_product_zero",
                             "valid_nonconstant_count": int(variable.sum()), "policy": "equal_nonconstant"})
        else:
            within = information / information.sum()
        global_weight[indices] = within / len(groups)
        score_indices[group] = indices
        for local, index in enumerate(indices):
            rows.append({"field": fields[index], "group": group, "entropy": d[local],
                         "redundancy": redundancy[local], "nonredundancy": nonredundancy[local],
                         "within_group_weight": within[local],
                         "global_weight": global_weight[index], "nondegenerate": bool(variable[local])})
    if not np.isclose(global_weight.sum(), 1, atol=1e-10) or np.any(global_weight < -1e-12):
        raise RuntimeError("层级权重非负或归一化约束失败")
    return global_weight, pd.DataFrame(rows), fallback, score_indices


def reconstruct_matrix(root, out, old, config, old_config):
    fields = old_config["quality_fields"]
    accepted = pd.read_csv(old / "audit/source_ledger.csv.gz", usecols=["attachment", "line_number", "record_id", "domain", "status"], keep_default_na=False)
    accepted = accepted[accepted.status.isin(("valid", "duplicate_exact"))]
    lookup = {(r.attachment, int(r.line_number)): (r.record_id, r.domain, r.status)
              for r in accepted.itertuples()}
    values, qparts, identifiers = [], [], []
    files = expected_files(root)
    qindex = fields.index("qurater")
    for attachment in ("A1", "A2", "A3"):
        with lzma.open(files[attachment][0], "rb") as stream:
            for line, raw in enumerate(stream, 1):
                entry = lookup.get((attachment, line))
                if entry is None:
                    continue
                record = json.loads(raw)
                if record["id"] != entry[0]:
                    raise RuntimeError("评分输入 ID 与审计账本不一致")
                scalar = scalarize(record, old_config)
                qparts.append(scalar[qindex])
                scalar[qindex] = np.nan
                values.append(scalar)
                identifiers.append((attachment, line, entry[0], entry[1], entry[2]))
    meta = pd.DataFrame(identifiers, columns=["attachment", "line_number", "record_id", "domain", "status"])
    old_scores = pd.read_csv(old / "quality/sample_scores.csv.gz")
    for col in ("attachment", "line_number", "record_id", "domain", "status"):
        if not meta[col].equals(old_scores[col]):
            raise RuntimeError(f"v2.1 与 v2 样本顺序/视图不一致: {col}")
    raw = np.asarray(values, dtype=float)
    qr = np.asarray(qparts, dtype=float)
    union = (meta.status == "valid").to_numpy()
    qlo, qhi = np.quantile(qr[union], old_config["winsor_quantiles"], axis=0)
    qr_normalized = np.divide(np.clip(qr, qlo, qhi) - qlo, qhi - qlo,
                              out=np.zeros_like(qr), where=qhi > qlo)
    raw[:, qindex] = qr_normalized.mean(axis=1)
    lo, hi = np.quantile(raw[union], old_config["winsor_quantiles"], axis=0)
    z = np.divide(np.clip(raw, lo, hi) - lo, hi - lo,
                  out=np.zeros_like(raw), where=hi > lo)
    for name in old_config["negative_after_scalarization"]:
        index = fields.index(name)
        if hi[index] > lo[index]:
            z[:, index] = 1 - z[:, index]
    stored_z = z.astype(np.float32)
    old_z = np.load(old / "quality/normalized_22.npy", mmap_mode="r")
    max_difference = float(np.max(np.abs(stored_z - old_z)))
    if stored_z.shape != old_z.shape or max_difference > 2e-6 or not np.isfinite(z).all():
        raise RuntimeError(f"重建 22 维矩阵未通过 v2 一致性核验: max_diff={max_difference}")
    old_params = pd.read_csv(old / "quality/indicator_parameters.csv")
    if not np.allclose(lo, old_params.q01) or not np.allclose(hi, old_params.q99):
        raise RuntimeError("重新拟合 q01/q99 与 v2 固定尺度不一致")
    np.save(out / "quality/normalized_22.npy", stored_z)
    pd.DataFrame({"field": fields, "q01": lo, "q99": hi,
                  "direction": ["negative" if f in old_config["negative_after_scalarization"] else
                                "direction_provisional" if f in config["direction_provisional"] else "positive"
                                for f in fields]}).to_csv(out / "quality/normalization_parameters.csv", index=False)
    return meta, old_scores, z, union, max_difference


def run(root, config, old_config):
    root, out, old = paths(root)
    dest = out / "quality"
    dest.mkdir(parents=True, exist_ok=True)
    fields = old_config["quality_fields"]
    validate_groups(fields, config["groups"])
    meta, old_scores, z, union, max_difference = reconstruct_matrix(root, out, old, config, old_config)
    ref = np.asarray(z[union], dtype=float)
    rho = pd.DataFrame(ref, columns=fields).corr(method="spearman").fillna(0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1)
    old_rho = pd.read_csv(old / "quality/spearman_signed.csv", index_col=0).to_numpy()
    if not np.allclose(rho, old_rho, atol=1e-8):
        raise RuntimeError("重算 Spearman 与 v2 不一致")
    pd.DataFrame(rho, index=fields, columns=fields).to_csv(dest / "spearman_signed.csv")
    wmain, weight_rows, fallbacks, indices = group_weights(ref, rho, fields, config["groups"])
    wdrop, drop_rows, drop_fallbacks, _ = group_weights(ref, rho, fields, config["groups"], config["direction_provisional"])
    wequal = np.zeros(len(fields))
    for group, idx in indices.items():
        nondegenerate = idx[np.ptp(ref[:, idx], axis=0) > 1e-12]
        wequal[nondegenerate] = 1 / (len(config["groups"]) * len(nondegenerate))
    if not np.isclose(wequal.sum(), 1):
        raise RuntimeError("H-equal-within 权重和不为 1")
    old_weights = pd.read_csv(old / "quality/indicator_parameters.csv").set_index("field")["w_main"]
    weight_rows["old_v2_global_weight"] = weight_rows.field.map(old_weights)
    weight_rows["direction_status"] = weight_rows.field.map(lambda f: "direction_provisional" if f in config["direction_provisional"] else "fixed")
    weight_rows.to_csv(dest / "hierarchical_weights.csv", index=False)
    drop_rows.to_csv(dest / "drop4_weights.csv", index=False)
    pd.DataFrame(fallbacks + drop_fallbacks, columns=["group", "reason", "valid_nonconstant_count", "policy"]).to_csv(dest / "weight_fallbacks.csv", index=False)
    group_scores = {}
    for group, idx in indices.items():
        group_scores[group] = np.asarray(z[:, idx]) @ (wmain[idx] * len(config["groups"]))
    scores = {"Q_hierarchical_balanced": 100 * np.asarray(z) @ wmain,
              "Q_hierarchical_drop4": 100 * np.asarray(z) @ wdrop,
              "Q_hierarchical_equal_within": 100 * np.asarray(z) @ wequal}
    if not np.allclose(scores["Q_hierarchical_equal_within"], old_scores["Q_group_equal"], atol=3e-5):
        raise RuntimeError("H-equal-within 与 v2 Q_group_equal 不一致")
    if any(np.nanmin(a) < -1e-8 or np.nanmax(a) > 100+1e-8 for a in scores.values()):
        raise RuntimeError("Q 越出 [0,100]")
    for key, values in group_scores.items():
        meta["S_" + key] = values
    for key, values in scores.items():
        meta[key] = values
    meta["q_hierarchical_balanced"] = scores["Q_hierarchical_balanced"] / 100
    for key in ("Q_entropy", "Q_entropy_redundancy", "Q_equal22", "Q_group_equal", "Q_TOPSIS", "Q_direction_drop4"):
        meta[key] = old_scores[key].to_numpy()
    meta.to_csv(dest / "sample_quality_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    meta[["attachment", "line_number", "record_id", "domain", "status", *["S_" + g for g in config["groups"]]]].to_csv(dest / "semantic_group_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    rng = np.random.default_rng(config["seed"])
    score_names = [*scores, "Q_entropy_redundancy", "Q_group_equal", "Q_TOPSIS"]
    views = [("union_unique", meta[union]), ("A1", meta[meta.attachment == "A1"]),
             ("A2", meta[meta.attachment == "A2"]), ("A3", meta[meta.attachment == "A3"])]
    domain_rows, group_distribution = [], []
    for view, frame in views:
        for domain, sub in frame.groupby("domain"):
            row = {"view": view, "domain": domain, "n": len(sub)}
            matrix = sub[score_names].to_numpy(dtype=float)
            row.update({key: float(matrix[:, col].mean()) for col, key in enumerate(score_names)})
            row["Q_hierarchical_balanced_sd"] = float(matrix[:, 0].std(ddof=1)) if len(sub) > 1 else None
            bootstrap = np.empty((config["bootstrap_repetitions"], 3))
            for b in range(len(bootstrap)):
                bootstrap[b] = matrix[rng.integers(0, len(sub), len(sub)), :3].mean(axis=0)
            for col, key in enumerate(list(scores)):
                row[key + "_ci025"], row[key + "_ci975"] = np.quantile(bootstrap[:, col], [.025, .975])
            domain_rows.append(row)
            for group in config["groups"]:
                a = sub["S_" + group].to_numpy()
                group_distribution.append({"view": view, "domain": domain, "group": group, "n": len(a),
                                           "mean": a.mean(), "sd": a.std(ddof=1) if len(a)>1 else None,
                                           "q05": np.quantile(a, .05), "median": np.median(a), "q95": np.quantile(a, .95)})
    domains = pd.DataFrame(domain_rows)
    domains.to_csv(dest / "domain_quality_summary.csv", index=False)
    pd.DataFrame(group_distribution).to_csv(dest / "semantic_group_distribution.csv", index=False)
    reference = meta[union]
    stability = []
    for method in score_names:
        row = {"method": method, "spearman_vs_hierarchical": reference[method].corr(reference["Q_hierarchical_balanced"], method="spearman")}
        for fraction in config["top_fractions"]:
            k = max(1, int(np.ceil(fraction * len(reference))))
            a = set(reference.nlargest(k, "Q_hierarchical_balanced").record_id)
            b = set(reference.nlargest(k, method).record_id)
            row[f"top_{int(100*fraction)}pct_overlap"] = len(a & b) / k
        ranked = domains[domains.view == "union_unique"].sort_values(method, ascending=False).domain.tolist()
        row["domain_ranking"] = ">".join(ranked)
        stability.append(row)
    pd.DataFrame(stability).to_csv(dest / "quality_method_comparison.csv", index=False)
    main_order = domains[domains.view == "union_unique"].sort_values("Q_hierarchical_balanced", ascending=False).domain.tolist()
    drop_order = domains[domains.view == "union_unique"].sort_values("Q_hierarchical_drop4", ascending=False).domain.tolist()
    pd.DataFrame({"domain": main_order, "main_rank": range(1,8),
                  "drop4_rank": [drop_order.index(d)+1 for d in main_order],
                  "rank_shift": [drop_order.index(d)+1-i for i,d in enumerate(main_order,1)]}).to_csv(dest / "quality_sensitivity.csv", index=False)
    summary = {"experiment_version": config["experiment_version"], "seed": config["seed"],
               "valid_unique": int(union.sum()), "accepted_source_rows": len(meta),
               "new_matrix_vs_v2_max_abs_diff": max_difference,
               "main_method": "Q_hierarchical_balanced", "fallbacks": fallbacks,
               "drop4_fallbacks": drop_fallbacks,
               "max_global_weight": float(wmain.max()),
               "numerical_chars_weight_v2_1": float(weight_rows.set_index("field").loc["rps_lines_numerical_chars_fraction", "global_weight"]),
               "numerical_chars_weight_v2": float(old_weights["rps_lines_numerical_chars_fraction"])}
    save_json(dest / "quality_summary.json", summary)
    return summary
