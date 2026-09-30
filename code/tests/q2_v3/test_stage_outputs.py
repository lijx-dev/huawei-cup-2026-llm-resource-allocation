"""正式产物的只读门控；数值必须来自实际运行，不能以样例替代。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.q2_v3.common import OUT, ROOT, sha


def test_p0_inventory_and_overlap():
    inv = pd.read_csv(OUT / "audit/b_attachment_inventory.csv")
    assert set(inv.attachment_id) == {f"B{i}" for i in range(1, 13)}
    for row in inv.itertuples():
        assert sha(ROOT / row.relative_path) == row.sha256
    ov = pd.read_csv(OUT / "audit/b6_b7_overlap.csv")
    assert not (ov.role == "duplicate_conflict").any()
    assert (ov.role == "b7_new").any()
    q1 = json.loads((OUT / "audit/q1_interface.json").read_text())
    assert q1["n_inputs"] == 17 and q1["n_outputs"] == 13
    assert len(q1["p0"]) == 17 and abs(sum(q1["p0"]) - 1) < 1e-8


def test_p1_group_cv_and_positive_parameters():
    folds = pd.read_csv(OUT / "b1_baseline/group_cv_metrics.csv")
    assert len(folds) == folds.held_out_N.nunique()
    assert folds.n.sum() == json.loads((OUT / "b1_baseline/final_parameters.json").read_text())["n"]
    assert all(x > 0 for x in json.loads((OUT / "b1_baseline/final_parameters.json").read_text())["parameters"].values())
    assert (OUT / "figures/b1_residual_diagnostics.png").is_file()
    assert set(pd.read_csv(OUT / "b1_baseline/exclusions.csv").columns) == {"source_file", "row_id_or_experiment_id", "reason", "evidence", "stage", "action"}


def test_p2_source_roles():
    d = pd.read_csv(OUT / "loss_comparability/zero_dof_metrics.csv")
    assert d.set_index("source").loc["B3", "evidence"] == "interpolation_consistency"
    assert set(d.source) == {"B2", "B3", "B4", "B5"}
    literature = pd.read_csv(OUT / "loss_comparability/per_literature_metrics.csv")
    assert literature.source.eq("B5").all()


def test_p3_fixed_bins_and_bootstrap():
    d = pd.read_csv(OUT / "quality_position/amplitude_by_group.csv")
    assert d.fixed_low_Q_cut.nunique() == d.fixed_high_Q_cut.nunique() == 1
    assert d.fixed_low_Q_cut.iloc[0] == .3 and d.fixed_high_Q_cut.iloc[0] == .8
    assert len(pd.read_csv(OUT / "quality_position/amplitude_bootstrap.csv")) >= 2500


def test_p4_freeze_and_common_folds():
    path = OUT / "quality_models/selected_model.json"
    assert sha(path) == (OUT / "quality_models/model_freeze.sha256").read_text().split()[0]
    d = pd.read_csv(OUT / "quality_models/candidate_model_metrics.csv")
    for fold, group in d.groupby("fold"):
        assert group.n.nunique() == 1
    boot = pd.read_csv(OUT / "quality_models/parameter_bootstrap.csv")
    assert len(boot) >= 400
    diff = pd.read_csv(OUT / "quality_models/bootstrap_model_differences.csv")
    assert "parametric_gaussian_null_by_group" in set(diff.method)


def test_p5_locked_validation_and_refit():
    pre = json.loads((OUT / "locked_validation/pre_validation_parameters.json").read_text())
    freeze = json.loads((OUT / "quality_models/selected_model.json").read_text())
    assert pre["parameters"] == freeze["parameters"]
    assert pre["fit_sample_hash"] == freeze["fit_sample_hash"]
    assert len(pd.read_csv(OUT / "locked_validation/post_validation_bootstrap.csv")) >= 400
    summary = json.loads((OUT / "locked_validation/validation_summary.json").read_text())
    assert summary["validation_type"] == "new-Q interpolation validation"
    assert summary["new_N_D_groups"] == 0


def test_p6_provenance_separation():
    d = pd.read_csv(OUT / "stress_extrapolation/b8_model_errors.csv")
    assert set(d.data_type) == {"calibrated", "extrapolated"}
    assert pd.read_csv(OUT / "stress_extrapolation/b10_consistency.csv").role.iloc[0] == "estimated_consistency"


def test_p7_q1_predictions_and_scenarios():
    d = pd.read_csv(OUT / "q1_interface/hp_by_target.csv")
    assert len(d) == 13 and (d[["p0_loss", "candidate_loss"]] > 0).all().all()
    assert np.allclose(d.h_p_v_p0, 0)
    s = pd.read_csv(OUT / "q1_interface/mixture_structure_scenarios.csv")
    assert set(s.lambda_p) == {0, .5, 1, 1.5}


def test_p8_analytic_numeric_and_bounds():
    d = pd.read_csv(OUT / "compute_opt/numerical_verification.csv")
    assert (d.relative_gap < 1e-5).all()
    b = json.loads((OUT / "compute_opt/support_bounds.json").read_text())
    opt = pd.read_csv(OUT / "compute_opt/bounded_optima.csv")
    assert opt.N_billions.between(*b["N_billions"]).all()
    assert opt.D_billions.between(*b["D_billions"]).all()
    uncertainty = pd.read_csv(OUT / "compute_opt/uncertainty_summary.csv")
    assert {"N_billions", "D_billions", "pred_loss"}.issubset(set(uncertainty.quantity))


def test_p9_report_and_evidence_grades():
    report = (OUT / "report/question2_v3_experiment_report.md").read_text()
    for i in range(1, 19): assert f"## {i}." in report
    assert len(pd.read_csv(OUT / "report/evidence_grade_table.csv")) == 7
