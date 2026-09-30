"""v2 的数学约束、来源隔离与实际运行产物检查；微型记录只用于测试。"""
import json
import lzma
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedKFold

from q1.audit.inventory import expected_files
from q1.mixture.dataset import load_pair
from q1.mixture.perturbation import single, pair
from q1_revision_v2.audit import audit_quality, indicator_hash
from q1_revision_v2.common import load_config, sha256
from q1_revision_v2.conflict import sample_conflict
from q1_revision_v2.integration import mapped_quality
from q1_revision_v2.mixture import predict, ridge_fit
from q1_revision_v2.quality import scalarize, softmax, entropy_weights, score_topsis


ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT)
OUT = ROOT / "results/q1_revision_v2"


def _record(rid="one"):
    row = {"id": rid, "_source_domain": "arxiv"}
    for field in CFG["quality_fields"]:
        row[field] = ([0.0] * CFG["list_lengths"][field] if field in CFG["list_lengths"] else 1.0)
    return row


def _mini_audit(tmp_path, rows):
    sources = []
    for a, records in zip(("A1", "A2", "A3"), rows):
        p = tmp_path / f"{a}.jsonl.xz"
        with lzma.open(p, "wb") as stream:
            for row in records:
                stream.write((json.dumps(row) + "\n").encode())
        sources.append((a, p))
    return audit_quality(tmp_path, tmp_path / "results", CFG, sources)


def test_duplicate_id_same_fields(tmp_path):
    a = _record()
    result = _mini_audit(tmp_path, ([a], [a], []))
    assert result["status_counts"] == {"valid": 1, "duplicate_exact": 1}


def test_duplicate_id_conflicting_fields(tmp_path):
    a, b = _record(), _record()
    b["dsir_books"] = 2.0
    result = _mini_audit(tmp_path, ([a], [b], []))
    assert result["status_counts"] == {"duplicate_conflict": 2}


def test_eight_list_decoders():
    r = _record()
    r["fineweb_edu"] = [2.5]
    r["fluency_en"] = [0, 2]
    r["ad_en"] = [2, 0]
    for name in ("modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning", "modernbert_professionalism"):
        r[name] = [0, 0, 0, 0, 0, 10]
    r["qurater"] = [1, 2, 3, 4]
    s = scalarize(r, CFG)
    assert s[14] == 2.5
    assert s[15] == pytest.approx(softmax([0, 2])[1])
    assert all(s[CFG["quality_fields"].index(n)] > .99 for n in CFG["quality_fields"] if n.startswith("modernbert_"))
    assert s[21] == pytest.approx(softmax([2, 0])[1])


def test_qurater_keeps_four_separate_values():
    assert list(scalarize(_record(), CFG)[20]) == [0, 0, 0, 0]
    r = _record()
    r["qurater"] = [1, 2, 3, 4]
    assert list(scalarize(r, CFG)[20]) == [1, 2, 3, 4]


def test_winsor_boundaries_in_saved_matrix():
    p = pd.read_csv(OUT / "quality/indicator_parameters.csv")
    z = np.load(OUT / "quality/normalized_22.npy", mmap_mode="r")
    assert len(p) == 22 and z.shape[1] == 22
    assert z.min() >= 0 and z.max() <= 1


def test_direction_and_log_conversion():
    r = _record()
    r["rps_doc_word_count"] = 9
    s = scalarize(r, CFG)
    assert s[3] == pytest.approx(math.log1p(9))
    schema = pd.read_csv(OUT / "quality/indicator_schema_and_conversion.csv")
    assert set(schema[schema.direction == "negative"].field) == set(CFG["negative_after_scalarization"])


def test_entropy_zero_log_zero_and_sum():
    d, w = entropy_weights(np.array([[0., 1., 1.], [0., 0., 1.], [1., 0., 1.]]))
    assert np.isfinite(d).all() and w.sum() == pytest.approx(1)
    assert w[2] == 0


def test_redundancy_weight_formula():
    p = pd.read_csv(OUT / "quality/indicator_parameters.csv")
    expected = p.entropy_difference * p.nonredundancy
    expected /= expected.sum()
    assert np.allclose(expected, p.w_main)


def test_topsis_endpoints():
    a = np.array([[0., 0.], [1., 1.]])
    assert np.allclose(score_topsis(a, np.array([.5, .5])), [0, 100])


def test_signed_cluster_distance_bounds():
    rho = pd.read_csv(OUT / "quality/spearman_signed.csv", index_col=0).to_numpy()
    distance = np.sqrt(np.maximum(0, (1-rho)/2))
    assert 0 <= distance.min() and distance.max() <= 1 + 1e-10


def test_conflict_bounds_and_pair_decomposition():
    z = np.array([[0., .5, 1.]])
    w = np.array([.2, .3, .5])
    i, j = np.triu_indices(3, k=1)
    c = sample_conflict(z, w, i, j)[0]
    terms = w[i]*w[j]*np.abs(z[0, i]-z[0, j]) / np.sum(w[i]*w[j])
    assert 0 <= c <= 1 and c == pytest.approx(terms.sum())


def test_a1_thresholds_frozen_for_extensions():
    summary = json.loads((OUT / "conflict/conflict_summary.json").read_text())
    sample = pd.read_csv(OUT / "conflict/sample_conflict.csv.gz", usecols=["attachment", "C", "high_0.9"])
    tau = summary["thresholds_from_A1"]["0.9"]
    assert tau == pytest.approx(sample.loc[sample.attachment == "A1", "C"].quantile(.9))
    assert (sample["high_0.9"] == (sample.C > tau)).all()


def test_seventeen_mixture_simplex_and_join():
    summary = json.loads((OUT / "audit/audit_summary.json").read_text())
    ids, x, y, _ = load_pair(ROOT, "train_1m", summary["tables"]["mixture_fields"], summary["tables"]["loss_fields"], CFG["mixture_sum_tolerance"])
    assert x.shape == (512, 17) and np.allclose(x.sum(axis=1), 1)
    assert len(set(ids)) == len(ids)


def test_thirteen_target_order():
    summary = json.loads((OUT / "audit/audit_summary.json").read_text())
    fields = summary["tables"]["loss_fields"]
    assert len(fields) == 13 and len(set(fields)) == 13
    for p in summary["tables"]["pairs"]:
        assert p["one_to_one"]


def test_ridge_reference_and_fold_scaling():
    x = np.array([[.1,.9], [.2,.8], [.4,.6]])
    fit = ridge_fit(x, np.array([1.,2.,3.]), 1.)
    assert fit[0].n_features_in_ == 1
    assert fit[0].mean_[0] == pytest.approx(np.mean(x[:, 0]))
    assert np.isfinite(predict([fit], x, "Ridge")).all()


def test_repeated_cv_membership_no_leakage():
    fold = pd.read_csv(OUT / "mixture/cv_fold_membership.csv")
    assert fold.fold.nunique() == 50
    for _, group in fold.groupby("fold"):
        assert len(group) == 512 and group.row.nunique() == 512
        assert set(group.role) == {"train", "validation"}


def test_lightgbm_reload_prediction_consistency():
    artifact = joblib.load(OUT / "mixture/fitted_models.joblib")
    summary = json.loads((OUT / "audit/audit_summary.json").read_text())
    ids, x, _, _ = load_pair(ROOT, "test_1m", summary["tables"]["mixture_fields"], summary["tables"]["loss_fields"], CFG["mixture_sum_tolerance"])
    model = artifact["models"]["LightGBM"][0]
    output = pd.read_csv(OUT / "mixture/predictions_test_1m.csv")
    target = summary["tables"]["loss_fields"][0]
    assert np.array_equal(ids, output["index"].astype(str).to_numpy())
    assert np.allclose(model.predict(x), output[f"predicted_LightGBM/{target}"].to_numpy())


def test_single_perturbation_conservation():
    x = np.array([[.2, .3, .5]])
    out, okay = single(x, 0, .05)
    assert okay[0] and out.sum() == pytest.approx(1.) and out[0,0] == pytest.approx(.25)


def test_pair_perturbation_conservation():
    x = np.array([[.2, .3, .5]])
    out, okay = pair(x, 0, 1, .03, .03)
    assert okay[0] and out.sum() == pytest.approx(1.)
    assert out[0,0] == pytest.approx(.23) and out[0,1] == pytest.approx(.33)


def test_qmapped_zero_denominator_missing():
    mapping = pd.DataFrame([{"mixture_domain": "a", "quality_domain": "x", "mapping_type": "direct"}])
    q, mass, _ = mapped_quality(np.array([[0.,1.],[.2,.8]]), ["train_the_pile_a", "train_the_pile_b"], mapping, {"x": 70.})
    assert np.isnan(q[0]) and q[1] == pytest.approx(70.) and mass[0] == 0


def test_estimates_not_in_observed_metrics():
    observed = pd.read_csv(OUT / "mixture/holdout_metrics.csv")
    estimate = pd.read_csv(OUT / "mixture/estimated_rank_metrics.csv")
    assert all(not x.startswith("est_") for x in observed.split)
    assert all(x.startswith("est_") for x in estimate.split)
    assert set(estimate.evidence_type) == {"estimated_reference"}


def test_common_supported_subset_is_not_larger_than_each_domain():
    common = pd.read_csv(OUT / "mixture/common_subset_ame.csv")
    own = pd.read_csv(OUT / "mixture/single_domain_effects.csv")
    supported = common[common.subset == "common_supported"]
    merged = supported.merge(own, on=["model", "domain", "delta", "target"], suffixes=("_common", "_own"))
    assert len(merged) == len(supported)
    assert (merged.n <= merged.n_supported).all()


def test_raw_files_unchanged():
    inventory = pd.read_csv(OUT / "audit/file_inventory.csv")
    for row in inventory.itertuples():
        assert sha256(ROOT / row.source_file) == row.sha256
