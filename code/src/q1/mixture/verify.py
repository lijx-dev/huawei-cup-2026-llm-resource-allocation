"""对正式结果进行独立结构与代数核查。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from q1.mixture.dataset import PAIRS, fields, load_pair, sha256


def verify(root):
    root = Path(root)
    mix, integ, figures = root / "results/q1/mixture", root / "results/q1/integration", root / "results/q1/figures"
    meta = json.loads((mix / "mixture_model_metadata.json").read_text())
    cfg = json.loads((root / "configs/q1/mixture.json").read_text())
    fields17, fields13 = fields(root)
    checks = {}
    checks["dimensions_17_13"] = len(fields17) == 17 and len(fields13) == 13
    checks["code_hashes_current"] = all(sha256(root / name) == digest for name, digest in meta["code_sha256"].items())
    checks["config_hash_current"] = sha256(root / "configs/q1/mixture.json") == meta["config_sha256"]
    checks["frozen_model_hash_current"] = sha256(mix / "frozen_models.joblib") == meta["frozen_model_sha256"]
    for split, (_, _, role) in PAIRS.items():
        ids, x, y, _ = load_pair(root, split, fields17, fields13, cfg["sum_tolerance"])
        checks[f"{split}_simplex"] = bool(np.isfinite(x).all() and (x >= 0).all() and np.allclose(x.sum(axis=1), 1))
        checks[f"{split}_input_hashes"] = (meta["input_sha256"][split]["mixture"] == sha256(root / "data/real_attachments/A_data_value/regmix_tables" / PAIRS[split][0]) and
                                            meta["input_sha256"][split]["loss"] == sha256(root / "data/real_attachments/A_data_value/regmix_tables" / PAIRS[split][1]))
        if split != "train_1m":
            source = (mix / (f"estimated_{split.removeprefix('est_')}_comparison.csv" if role == "estimated_reference" else f"{split}_metrics.csv"))
            table = pd.read_csv(source)
            checks[f"{split}_metrics_complete"] = len(table) == 42 and set(table.model) == {"mean", "ridge", "lightgbm"} and table.target.nunique() == 14 and set(table.data_role) == {role}
        proxy = pd.read_csv(integ / "mixture_quality_proxy.csv")
        primary = proxy[(proxy.scenario == "a17_assisted_main") & (proxy.split == split)]
        checks[f"{split}_proxy_role_and_count"] = len(primary) == len(ids) and set(primary.data_role) == {role}
    matrix = pd.read_csv(integ / "domain_mapping_matrix.csv", index_col=0)
    q = pd.read_csv(root / "results/q1/quality/domain_quality_summary.csv")
    q = q[q.view == "A1"].set_index("domain").loc[matrix.columns]
    projected = pd.read_csv(integ / "projected_domain_quality.csv").set_index("mixture_domain").loc[matrix.index]
    m = matrix.to_numpy(float)
    checks["mapping_17_7_simplex"] = m.shape == (17, 7) and np.isfinite(m).all() and (m >= 0).all() and np.allclose(m.sum(axis=1), 1)
    checks["mapped_quality_identity"] = np.allclose(m @ (q.mean_Q.to_numpy() / 100), projected.quality_proxy_main.to_numpy())
    checks["a17_all_verified"] = bool(pd.read_csv(integ / "a17_audit_report.csv").verified.all())
    checks["interactions_all_pairs_and_targets"] = len(pd.read_csv(mix / "interaction_effects.csv")) == 136 * 14
    checks["figures_traced"] = len(pd.read_csv(figures / "m3_figure_manifest.csv")) >= 14
    overlap = pd.read_csv(mix / "cross_split_overlap.csv")
    checks["train_test_no_identical_recipe"] = bool((overlap[(overlap.left_split == "train_1m") & overlap.right_split.str.startswith("test_")].identical_mixture_vectors == 0).all())
    checks["estimated_training_subset"] = bool((overlap[(overlap.left_split == "train_1m") & overlap.right_split.str.startswith("est_")].identical_mixture_vectors == 63).all())
    result = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values())}
    (mix / "verification_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not result["all_passed"]:
        raise RuntimeError(f"P3/M3 结果核查未通过: {[k for k,v in checks.items() if not v]}")
    return result


if __name__ == "__main__":
    print(verify(Path.cwd()))
