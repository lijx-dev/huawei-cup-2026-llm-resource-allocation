"""从原始质量记录重新拟合全量去重参考尺度与五种 Q。"""
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from q1.audit.inventory import expected_files
from .common import paths, save_json


def softmax(values):
    a = np.asarray(values, dtype=float)
    e = np.exp(a - np.max(a))
    return e / e.sum()


def scalarize(row, config):
    ans = []
    for field in config["quality_fields"]:
        value = row[field]
        if field.startswith("modernbert_"):
            value = float(softmax(value) @ (np.arange(6) / 5))
        elif field == "fluency_en":
            value = float(softmax(value)[1])
        elif field == "ad_en":
            value = float(softmax(value)[1])
        elif field == "fineweb_edu":
            value = float(value[0])
        elif field == "qurater":
            value = np.asarray(value, dtype=float)
        elif field in config["log1p_fields"]:
            value = float(np.log1p(value))
        ans.append(value)
    return ans


def entropy_weights(z):
    n = len(z)
    sums = z.sum(axis=0)
    p = np.divide(z, sums, out=np.zeros_like(z), where=sums > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0)
    d = np.clip(1 + terms.sum(axis=0) / np.log(n), 0, 1)
    d[np.ptp(z, axis=0) < 1e-12] = 0
    if d.sum() == 0:
        raise RuntimeError("熵权无可区分指标")
    return d, d / d.sum()


def score_topsis(z, weights):
    v = z * weights
    ideal_plus, ideal_minus = weights, np.zeros_like(weights)
    dplus = np.linalg.norm(v - ideal_plus, axis=1)
    dminus = np.linalg.norm(v - ideal_minus, axis=1)
    return 100 * np.divide(dminus, dplus + dminus, out=np.full(len(z), .5), where=(dplus + dminus) > 0)


def run(root, config):
    root, out = paths(root)
    dest = out / "quality"
    dest.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(out / "audit/source_ledger.csv.gz", keep_default_na=False)
    accepted = ledger[ledger.status.isin(["valid", "duplicate_exact"])]
    lookup = {(r.attachment, int(r.line_number)): (r.status, r.domain, r.record_id)
              for r in accepted.itertuples()}
    files = expected_files(root)
    records, values, qurater = [], [], []
    for attachment in ("A1", "A2", "A3"):
        with lzma.open(files[attachment][0], "rb") as stream:
            for line, raw in enumerate(stream, 1):
                key = attachment, line
                if key not in lookup:
                    continue
                status, domain, rid = lookup[key]
                scalar = scalarize(json.loads(raw), config)
                qurater.append(scalar[20])
                scalar[20] = np.nan
                values.append(scalar)
                records.append((attachment, line, rid, domain, status))
    meta = pd.DataFrame(records, columns=["attachment", "line_number", "record_id", "domain", "status"])
    raw = np.asarray(values, dtype=float)
    qr = np.asarray(qurater, dtype=float)
    union = (meta.status == "valid").to_numpy()
    if union.sum() != (ledger.status == "valid").sum():
        raise RuntimeError("评分记录数与审计账本不一致")
    qlo, qhi = np.quantile(qr[union], config["winsor_quantiles"], axis=0)
    qden = qhi - qlo
    qparts = np.divide(np.clip(qr, qlo, qhi) - qlo, qden,
                       out=np.zeros_like(qr), where=qden > 0)
    raw[:, 20] = qparts.mean(axis=1)
    lo, hi = np.quantile(raw[union], config["winsor_quantiles"], axis=0)
    z = np.divide(np.clip(raw, lo, hi) - lo, hi - lo,
                  out=np.zeros_like(raw), where=hi > lo)
    for name in config["negative_after_scalarization"]:
        index = config["quality_fields"].index(name)
        if hi[index] > lo[index]:
            z[:, index] = 1 - z[:, index]
    np.save(dest / "normalized_22.npy", z.astype(np.float32))
    pd.DataFrame(qr, columns=[f"qurater_raw_{x}" for x in ("writing", "expertise", "facts", "education")]).to_csv(dest / "qurater_raw_components.csv.gz", index=False)
    pd.DataFrame(qparts, columns=[f"qurater_{x}" for x in ("writing", "expertise", "facts", "education")]).to_csv(dest / "qurater_components.csv.gz", index=False)
    ref = z[union]
    d, wentropy = entropy_weights(ref)
    rho = pd.DataFrame(ref, columns=config["quality_fields"]).corr(method="spearman").to_numpy()
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1)
    u = 1 - (np.abs(rho).sum(axis=1) - 1) / (len(d) - 1)
    wmain = d * u
    wmain /= wmain.sum()
    wequal = np.where(hi > lo, 1.0, 0.0)
    wequal /= wequal.sum()
    wgroup = np.zeros(len(d))
    for group in config["groups"].values():
        indices = np.array([config["quality_fields"].index(f) for f in group])
        valid = indices[hi[indices] > lo[indices]]
        wgroup[valid] = 1 / (len(config["groups"]) * len(valid))
    # 方向敏感性：保持主尺度与方向，仅剔除四项待定单调方向。
    wsensitive = wmain.copy()
    wsensitive[[config["quality_fields"].index(f) for f in config["direction_sensitive"]]] = 0
    wsensitive /= wsensitive.sum()
    scores = {"Q_entropy": 100 * z @ wentropy,
              "Q_entropy_redundancy": 100 * z @ wmain,
              "Q_equal22": 100 * z @ wequal,
              "Q_group_equal": 100 * z @ wgroup,
              "Q_TOPSIS": score_topsis(z, wmain),
              "Q_direction_drop4": 100 * z @ wsensitive}
    for name, values in scores.items():
        meta[name] = values
    meta.to_csv(dest / "sample_scores.csv.gz", index=False)
    pd.DataFrame({"field": config["quality_fields"], "q01": lo, "q99": hi, "entropy_difference": d,
                  "nonredundancy": u, "w_entropy": wentropy, "w_main": wmain,
                  "w_equal22": wequal, "w_group_equal": wgroup,
                  "w_direction_drop4": wsensitive}).to_csv(dest / "indicator_parameters.csv", index=False)
    raw_audit = pd.read_csv(out / "audit/indicator_raw_audit.csv")
    methods = {}
    for field in config["quality_fields"]:
        methods[field] = ("stable_softmax_ordinal_expectation_0_to_5" if field.startswith("modernbert_") else
                          "stable_softmax_positive_class_1" if field in ("fluency_en", "ad_en") else
                          "singleton_value" if field == "fineweb_edu" else
                          "componentwise_q01_q99_then_equal_mean" if field == "qurater" else
                          "log1p" if field in config["log1p_fields"] else "identity")
    conversion = raw_audit.assign(conversion_method=raw_audit.field.map(methods),
                                  transformed_min=np.nanmin(raw, axis=0), transformed_max=np.nanmax(raw, axis=0),
                                  direction=["negative" if f in config["negative_after_scalarization"] else "positive_provisional" if f in config["direction_sensitive"] else "positive" for f in config["quality_fields"]],
                                  experiment_version=config["experiment_version"])
    conversion.to_csv(dest / "indicator_schema_and_conversion.csv", index=False)
    pd.DataFrame(rho, index=config["quality_fields"], columns=config["quality_fields"]).to_csv(dest / "spearman_signed.csv")
    save_json(dest / "qurater_parameters.json", {"q01": qlo.tolist(), "q99": qhi.tolist(), "component_order": ["writing", "expertise", "facts", "education"]})
    rng = np.random.default_rng(config["seed"])
    domain_rows = []
    for view, frame in [("union_unique", meta[union]), ("A1", meta[meta.attachment == "A1"]),
                        ("A2", meta[meta.attachment == "A2"]), ("A3", meta[meta.attachment == "A3"])]:
        for domain, sub in frame.groupby("domain"):
            row = {"view": view, "domain": domain, "n": len(sub)}
            for key in scores:
                a = sub[key].to_numpy()
                row[key] = float(a.mean())
                row[key + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else np.nan
                means = np.empty(config["bootstrap_repetitions"])
                for b in range(len(means)):
                    means[b] = a[rng.integers(0, len(a), size=len(a))].mean()
                row[key + "_ci025"], row[key + "_ci975"] = np.quantile(means, [.025, .975])
            domain_rows.append(row)
    domains = pd.DataFrame(domain_rows)
    domains.to_csv(dest / "domain_scores.csv", index=False)
    stability = []
    refmeta = meta[union]
    main = refmeta["Q_entropy_redundancy"]
    k = max(1, int(np.ceil(config["quality_top_fraction"] * len(refmeta))))
    main_top = set(refmeta.nlargest(k, "Q_entropy_redundancy").record_id)
    for key in scores:
        top = set(refmeta.nlargest(k, key).record_id)
        stability.append({"method": key, "spearman_vs_main": float(main.corr(refmeta[key], method="spearman")),
                          "top_1pct_overlap": len(main_top & top) / k,
                          "domain_order": ">".join(domains[domains.view == "union_unique"].sort_values(key, ascending=False).domain)})
    pd.DataFrame(stability).to_csv(dest / "score_stability.csv", index=False)
    summary = {"valid_union": int(union.sum()), "source_rows_scored": len(meta),
               "main_score": "Q_entropy_redundancy", "reference": "valid union_unique",
               "quarantine_excluded": int(len(ledger) - len(meta))}
    save_json(dest / "quality_summary.json", summary)
    return summary
