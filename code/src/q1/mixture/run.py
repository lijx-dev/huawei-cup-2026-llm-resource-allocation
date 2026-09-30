"""P3/M3 可复现实验入口：先冻结模型，再读取所有留出和估算数据。"""
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm
import sklearn

from q1.mixture.dataset import PAIRS, fields, load_pair, sha256
from q1.mixture.models import select_and_fit, predict
from q1.mixture.evaluation import metrics
from q1.mixture.perturbation import single, pair, support_reference, nearest_distance, interaction_value
from q1.integration.mapping import audit_a17, features_and_similarity, build_mapping, project


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.12g")


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prediction_table(ids, actual, pred, targets, split, role):
    frame = pd.DataFrame({"index": ids, "split": split, "data_role": role,
                          "observed_average_loss" if role != "estimated_reference" else "estimated_reference_average_loss": actual.mean(axis=1),
                          "predicted_average_loss": pred.mean(axis=1)})
    for j, name in enumerate(targets):
        frame[f"observed:{name}" if role != "estimated_reference" else f"estimated_reference:{name}"] = actual[:, j]
        frame[f"predicted:{name}"] = pred[:, j]
    return frame


def _effect_table(x, train, model, ref, cfg, domains, targets):
    threshold, nn_train = support_reference(train, cfg["support_quantile"])
    baseline = predict(model, "lightgbm", x, ref)
    rows, detail = [], []
    for j, domain in enumerate(domains):
        for delta in cfg["perturbations"]:
            modified, feasible = single(x, j, delta)
            if not feasible.any():
                continue
            idx = np.flatnonzero(feasible)
            dist = nearest_distance(modified[feasible], train)
            support = (dist <= threshold) & np.all(modified[feasible] <= train.max(axis=0) + 1e-12, axis=1)
            change = predict(model, "lightgbm", modified[feasible], ref) - baseline[feasible]
            for target_index, target in enumerate([*targets, "average_loss"]):
                value = change[:, target_index] if target_index < len(targets) else change.mean(axis=1)
                rows.append({"domain": domain, "delta": delta, "target": target, "feasible_n": len(idx),
                             "supported_n": int(support.sum()), "support_fraction": float(support.mean()),
                             "mean_delta_loss_supported": float(np.mean(value[support])) if support.any() else np.nan,
                             "mean_delta_loss_all_feasible_exploratory": float(value.mean())})
            for n, source in enumerate(idx):
                detail.append({"index": int(source), "domain": domain, "delta": delta,
                               "nearest_train_distance": dist[n], "support_status": "supported" if support[n] else "outside_support",
                               "predicted_average_delta_loss": float(change[n].mean())})
    return pd.DataFrame(rows), pd.DataFrame(detail), threshold, nn_train


def _interaction_table(x, train, model, ref, cfg, domains, targets, threshold):
    base = predict(model, "lightgbm", x, ref)
    delta = cfg["interaction_delta"]
    rows = []
    surface = []
    for j in range(len(domains)):
        for k in range(j + 1, len(domains)):
            xj, fj = pair(x, j, k, delta, 0)
            xk, fk = pair(x, j, k, 0, delta)
            xb, fb = pair(x, j, k, delta, delta)
            feasible = fj & fk & fb
            if not feasible.any():
                continue
            max_share = train.max(axis=0)
            sj = (nearest_distance(xj[feasible], train) <= threshold) & np.all(xj[feasible] <= max_share + 1e-12, axis=1)
            sk = (nearest_distance(xk[feasible], train) <= threshold) & np.all(xk[feasible] <= max_share + 1e-12, axis=1)
            sb = (nearest_distance(xb[feasible], train) <= threshold) & np.all(xb[feasible] <= max_share + 1e-12, axis=1)
            support = sj & sk & sb
            pj = predict(model, "lightgbm", xj[feasible], ref)
            pk = predict(model, "lightgbm", xk[feasible], ref)
            pb = predict(model, "lightgbm", xb[feasible], ref)
            effect = interaction_value(base[feasible], pj, pk, pb)
            for t, target in enumerate([*targets, "average_loss"]):
                value = effect[:, t] if t < len(targets) else effect.mean(axis=1)
                rows.append({"domain_j": domains[j], "domain_k": domains[k], "target": target,
                             "delta_j": delta, "delta_k": delta, "feasible_n": int(feasible.sum()),
                             "supported_n": int(support.sum()), "support_fraction": float(support.mean()),
                             "mean_interaction_supported": float(value[support].mean()) if support.any() else np.nan,
                             "mean_interaction_all_feasible_exploratory": float(value.mean())})
            if j == 0 and k == 1:
                # 同一个可行基线上的局部二维响应，不外推无支持的格点。
                for a in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
                    for b in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
                        xx, ok = pair(x, j, k, a, b)
                        if ok.any():
                            d = nearest_distance(xx[ok], train)
                            mask = (d <= threshold) & np.all(xx[ok] <= train.max(axis=0) + 1e-12, axis=1)
                            if mask.any():
                                pred = predict(model, "lightgbm", xx[ok][mask], ref).mean(axis=1)
                                surface.append({"domain_j": domains[j], "domain_k": domains[k], "delta_j": a,
                                                "delta_k": b, "supported_n": int(mask.sum()),
                                                "mean_predicted_average_loss": float(pred.mean())})
    return pd.DataFrame(rows), pd.DataFrame(surface)


def _association(q, loss, label, split, role):
    from scipy.stats import pearsonr, spearmanr
    good = np.isfinite(q) & np.isfinite(loss)
    if good.sum() < 3 or np.std(q[good]) == 0 or np.std(loss[good]) == 0:
        return {"scenario": label, "split": split, "loss_role": role, "n": int(good.sum()), "pearson": np.nan, "spearman": np.nan}
    return {"scenario": label, "split": split, "loss_role": role, "n": int(good.sum()),
            "pearson": float(pearsonr(q[good], loss[good]).statistic),
            "spearman": float(spearmanr(q[good], loss[good]).statistic)}


def run(root):
    root = Path(root).resolve()
    cfg_path, map_cfg_path = root / "configs/q1/mixture.json", root / "configs/q1/domain_mapping.json"
    cfg, map_cfg = json.loads(cfg_path.read_text()), json.loads(map_cfg_path.read_text())
    mixfields, targets = fields(root)
    domains = [s.removeprefix("train_the_pile_") for s in mixfields]
    ref = domains.index(cfg["ridge_reference_domain"])
    out, integ = root / "results/q1/mixture", root / "results/q1/integration"
    out.mkdir(parents=True, exist_ok=True)
    integ.mkdir(parents=True, exist_ok=True)
    # 冻结前仅打开训练数据；所有 CV 只用这 512 行。
    ids, x, y, train_audit = load_pair(root, "train_1m", mixfields, targets, cfg["sum_tolerance"])
    model, cv_rows, choices = select_and_fit(x, y, cfg, ref)
    save_csv(out / "ridge_cv_results.csv", [r for r in cv_rows if r["model"] == "ridge"])
    save_csv(out / "lightgbm_cv_results.csv", [r for r in cv_rows if r["model"] == "lightgbm"])
    joblib.dump({"models": model, "reference_index": ref, "mixture_fields": mixfields, "loss_fields": targets,
                 "config_version": cfg["version"]}, out / "frozen_models.joblib")
    frozen_sha = sha256(out / "frozen_models.joblib")
    logging.info("M3-A 已冻结模型；开始读取独立留出集")
    split_data = {"train_1m": (ids, x, y, train_audit)}
    split_audits = [train_audit]
    all_metrics = []
    for split in PAIRS:
        if split == "train_1m":
            continue
        current = load_pair(root, split, mixfields, targets, cfg["sum_tolerance"])
        split_data[split] = current
        ids_s, xs, ys, audit = current
        split_audits.append(audit)
        for family in ("mean", "ridge", "lightgbm"):
            pred = predict(model, family, xs, ref)
            role = audit["role"]
            for row in metrics(ys, pred, targets, cfg["top_k"]):
                all_metrics.append({"split": split, "data_role": role, "model": family, **row})
            prediction_table(ids_s, ys, pred, targets, split, role).to_csv(out / f"predictions_{split}_{family}.csv", index=False)
    save_csv(out / "mixture_integrity_report.csv", split_audits)
    metric_frame = pd.DataFrame(all_metrics)
    for split, file in (("test_1m", "test_1m_metrics.csv"), ("test_60m", "test_60m_metrics.csv"),
                        ("test_1b", "test_1b_metrics.csv"), ("est_10b", "estimated_10b_comparison.csv"),
                        ("est_70b", "estimated_70b_comparison.csv")):
        metric_frame[metric_frame.split == split].to_csv(out / file, index=False)
    test_average = metric_frame[(metric_frame.split == "test_1m") & (metric_frame.target == "average_loss")]
    logging.info("1M 综合 Loss 检验: %s", test_average[["model", "rmse", "r2", "spearman"]].to_dict("records"))
    # 最近邻阈值只由训练配方计算；所有结果标明模型预测和支持状态。
    effect, detail, threshold, nn_train = _effect_table(x, x, model, ref, cfg, domains, targets)
    effect.to_csv(out / "marginal_effects.csv", index=False)
    detail.to_csv(out / "marginal_support.csv", index=False)
    interactions, surface = _interaction_table(x, x, model, ref, cfg, domains, targets, threshold)
    interactions.to_csv(out / "interaction_effects.csv", index=False)
    surface.to_csv(out / "representative_response_surface.csv", index=False)
    # A17 由 A18 原始文本抽样独立复核行数、来源与文本长度；若不一致则停止辅助映射。
    a17, a17_report = audit_a17(root, domains)
    a17_report.to_csv(integ / "a17_audit_report.csv", index=False)
    if not a17_report.verified.all():
        raise RuntimeError("A17 与原始抽样统计不一致，已写审计报告，停止映射")
    z, distance, similarity, scaler = features_and_similarity(a17, domains)
    features = a17.copy()
    features["z_log_sample_rows"], features["z_log_avg_text_chars"] = z[:, 0], z[:, 1]
    features.to_csv(integ / "domain_features.csv", index=False)
    pd.DataFrame(distance, index=domains, columns=domains).to_csv(integ / "domain_distance_matrix.csv")
    pd.DataFrame(similarity, index=domains, columns=domains).to_csv(integ / "domain_similarity_matrix.csv")
    quality_table = pd.read_csv(root / "results/q1/quality/domain_quality_summary.csv")
    quality_table = quality_table[quality_table.view == map_cfg["quality_view"]].sort_values("domain")
    q_domains = quality_table.domain.tolist()
    if len(q_domains) != 7 or quality_table.mean_Q.isna().any() or quality_table.mean_equal_Q.isna().any():
        raise RuntimeError("P1 七域质量评分缺失")
    p1 = json.loads((root / "results/q1/quality/quality_model_metadata.json").read_text())
    p2 = json.loads((root / "results/q1/conflict/conflict_model_metadata.json").read_text())
    if p1["model_version"] != map_cfg["quality_model_version"] or p2["quality_model_version"] != p1["model_version"]:
        raise RuntimeError("P1/P2 质量版本不一致")
    q_main = quality_table.mean_Q.to_numpy(float) / 100
    q_equal = quality_table.mean_equal_Q.to_numpy(float) / 100
    a16_path = root / "data/real_attachments/A_data_value/domain_mapping_guide.csv"
    a16 = pd.read_csv(a16_path)
    matrix, status, representatives = build_mapping(a16, domains, q_domains, distance, map_cfg["primary_temperature"])
    pd.DataFrame(matrix, index=domains, columns=q_domains).to_csv(integ / "domain_mapping_matrix.csv")
    q17, _, _, _ = project(x, matrix, q_main)
    q17_eq, _, _, _ = project(x, matrix, q_equal)
    status["quality_proxy_main"] = q17
    status["quality_proxy_equal"] = q17_eq
    status.to_csv(integ / "projected_domain_quality.csv", index=False)
    scenarios = [("a16_only_main", False, 1.0, q_main, distance),
                 ("a17_assisted_main", True, 1.0, q_main, distance),
                 ("a17_assisted_equal", True, 1.0, q_equal, distance),
                 ("a17_length_only", True, 1.0, q_main, abs(z[:, 1, None] - z[None, :, 1])),
                 ("a17_count_only", True, 1.0, q_main, abs(z[:, 0, None] - z[None, :, 0]))]
    for temperature in map_cfg["sensitivity_temperatures"]:
        scenarios.append((f"a17_assisted_temp_{temperature}", True, temperature, q_main, distance))
    sensitivity, proxy_rows, association, coverage_rows = [], [], [], []
    for label, assisted, temperature, q7, scenario_distance in scenarios:
        mm, ss, _ = build_mapping(a16, domains, q_domains, scenario_distance, temperature, assisted)
        mapped_domains = int(np.isfinite(mm).all(axis=1).sum())
        for split, (split_ids, xs, ys, audit) in split_data.items():
            q17_s, qmix, coverage, complete = project(xs, mm, q7)
            pred = predict(model, "lightgbm", xs, ref)
            coverage_rows.append({"scenario": label, "split": split, "mapped_domains": mapped_domains,
                                  "complete_recipes": int(complete.sum()), "total_recipes": len(xs),
                                  "mean_share_covered": float(coverage.mean()),
                                  "min_share_covered": float(coverage.min())})
            association.append(_association(qmix, pred.mean(axis=1), label, split, "model_prediction"))
            association.append(_association(qmix, ys.mean(axis=1), label, split, audit["role"]))
            for n, key in enumerate(split_ids):
                proxy_rows.append({"scenario": label, "split": split, "data_role": audit["role"], "index": key,
                                   "coverage_share": coverage[n], "complete": bool(complete[n]),
                                   "quality_proxy": qmix[n], "predicted_average_loss": float(pred[n].mean()),
                                   "reference_average_loss": float(ys[n].mean())})
        sensitivity.append({"scenario": label, "temperature": temperature, "assisted": assisted,
                            "mapped_domains": mapped_domains, "quality_min": float(np.nanmin(q17_s)),
                            "quality_max": float(np.nanmax(q17_s))})
    save_csv(integ / "mapping_coverage.csv", coverage_rows)
    save_csv(integ / "mixture_quality_proxy.csv", proxy_rows)
    save_csv(integ / "quality_loss_association.csv", association)
    save_csv(integ / "mapping_sensitivity.csv", sensitivity)
    metadata = {"model_version": cfg["version"], "mapping_version": map_cfg["version"],
                "quality_model_version": p1["model_version"], "p2_model_version": p2["model_version"],
                "input_sha256": {a["split"]: {"mixture": a["mixture_sha256"], "loss": a["loss_sha256"]} for a in split_audits},
                "a16_sha256": sha256(a16_path),
                "a17_sha256": sha256(root / "data/real_attachments/A_data_value/regmix_domain_summary.csv"),
                "a18_sha256": sha256(root / "data/real_attachments/A_data_value/regmix_domain_sample.jsonl.xz"),
                "p1_domain_quality_sha256": sha256(root / "results/q1/quality/domain_quality_summary.csv"),
                "config_sha256": sha256(cfg_path), "mapping_config_sha256": sha256(map_cfg_path),
                "design_sha256": sha256(root / "question1-3(1)(1).md"),
                "code_sha256": {str(path.relative_to(root)): sha256(path) for directory in (root / "src/q1/mixture", root / "src/q1/integration") for path in sorted(directory.glob("*.py"))},
                "frozen_model_sha256": frozen_sha, "selected_candidates": choices,
                "reference_domain": domains[ref], "train_rows": len(x), "support_threshold": threshold,
                "train_nn_median": float(np.median(nn_train)), "support_rule": cfg["support_rule"],
                "train_share_max": dict(zip(domains, map(float, x.max(axis=0)))), "versions": {"numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__}}
    save_json(out / "mixture_model_metadata.json", metadata)
    save_json(integ / "mapping_metadata.json", {"config_version": map_cfg["version"], "feature_scaler": scaler,
              "quality_domains": q_domains, "mixture_domains": domains, "representatives": representatives,
              "source_sha256": {"a16": metadata["a16_sha256"], "a17": metadata["a17_sha256"], "a18": metadata["a18_sha256"]}})
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.root)
    from q1.mixture.reporting import finalize
    finalize(args.root)
    from q1.mixture.verify import verify
    verify(args.root)
