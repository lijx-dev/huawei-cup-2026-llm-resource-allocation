"""问题二阶段验收。微型样例仅在内存中构造，不进入正式结果。"""
import json

import numpy as np
import pandas as pd
import pytest

from src.q2_v2 import stages
from src.q2_v2.model import fit, predict


OUT = stages.OUT


def test_p0_inventory_contract_and_hashes():
    inventory=pd.read_csv(OUT/"audit/b_attachment_inventory.csv")
    meta=json.loads((OUT/"audit/audit_metadata.json").read_text())
    assert set(inventory.attachment_id)=={f"B{i}" for i in range(1,13)}
    assert inventory.readable.all() and meta["source_hash_unchanged"]
    assert all(stages.filehash(stages.ROOT/p)==h for p,h in meta["input_sha256"].items())
    contract=json.loads((OUT/"audit/field_contract.yaml").read_text())
    assert {"Q_B_raw","Q_B_normalized","Q_B_direction"} <= set(contract["standard_fields"])
    assert "uncertain" in contract["standard_fields"]["Q_B_direction"]
    quality=pd.read_csv(OUT/"audit/quality_semantics.csv")
    assert set(quality.attachment)=={"Q_A","B6","B7","B8"}
    assert quality.loc[quality.attachment.eq("Q_A"),"same_definition_as_B6"].item()=="unproven"
    role=pd.read_csv(OUT/"audit/dataset_role_matrix.csv").set_index("attachment")
    assert not role.loc[["B3","B9","B10","B12"],"can_fit"].any()


def test_p0_overlap_and_exclusions():
    overlap=pd.read_csv(OUT/"audit/b6_b7_overlap.csv")
    assert overlap.loc[overlap.in_b6,"fully_identical"].all()
    assert not overlap.conflicting_duplicate.any()
    invalid=pd.read_csv(OUT/"audit/invalid_rows.csv")
    assert set(invalid.attachment)=={"B9"} and set(invalid.reason_code)=={"invalid_D"}
    q1=json.loads((OUT/"audit/q1_interface.json").read_text())
    assert q1["ready_for_h_v"] and q1["input_count"]==17 and q1["output_count"]==13


def test_p1_loso_and_positive_multistart():
    folds=pd.read_csv(OUT/"b1_baseline/group_cv_metrics.csv")
    preds=pd.read_csv(OUT/"b1_baseline/group_cv_predictions.csv")
    assert len(folds)==8 and folds.n.eq(147).all() and len(preds)==1176
    assert preds.source_row.is_unique
    params=json.loads((OUT/"b1_baseline/final_parameters.json").read_text())
    assert params["E"]>=0 and all(params[k]>0 for k in ("A","B","alpha","beta"))
    starts=pd.read_csv(OUT/"b1_baseline/multistart_runs.csv")
    assert len(starts)>=3 and starts.converged.any()


def test_p1_synthetic_multistart_reproducible():
    n=np.repeat([.1,.3,1,3],8)
    d=np.tile(np.geomspace(1,100,8),4)
    theta=dict(E=1.2,A=.7,B=.9,alpha=.4,beta=.3)
    f=pd.DataFrame(dict(N_params_B=n,D_tokens_B=d,val_loss=predict(theta,n,d)))
    a,_=fit(f,starts=3,seed=7)
    b,_=fit(f,starts=3,seed=7)
    assert a==b and np.max(np.abs(predict(a,n,d)-f.val_loss))<1e-6


def test_p2_frozen_transfer_roles():
    v=pd.read_csv(OUT/"transfer_validation/validation_matrix.csv")
    assert {"B2","B3","B4","B5"}<=set(v.attachment)
    assert v.loc[v.attachment.eq("B3"),"provenance"].eq("interpolated").all()
    assert v.absolute_comparable_to_B1.eq("uncertain").all()
    assert len(v.loc[v.attachment.eq("B4")])>1 and len(v.loc[v.attachment.eq("B5")])>1


def test_p3_identical_group_folds_and_holdout():
    cv=pd.read_csv(OUT/"quality_model/b6_cv_metrics.csv")
    for model in ("M0","MQ","MQ2"):
        assert len(cv[cv.model==model])==5
        assert cv.loc[cv.model==model,"n"].sum()==360
    base=cv[cv.model=="M0"].set_index("fold").test_groups
    assert base.equals(cv[cv.model=="MQ"].set_index("fold").test_groups)
    b6,b7=stages.load("B6"),stages.load("B7")
    new=b7[~b7.experiment_id.isin(b6.experiment_id)]
    assert len(new)==len(pd.read_csv(OUT/"quality_model/b7_new_predictions_MQ.csv"))
    assert len(set(zip(new.N_params_B,new.D_tokens_B)) & set(zip(b6.N_params_B,b6.D_tokens_B)))==45


def test_p3_quality_reference_and_b8_separation():
    q=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    t=q["MQ"]
    y=predict(t,[1],[100],[q["q0"]],q["q0"],"quality")
    assert np.allclose(y,predict(t,[1],[100],form="base"))
    t0=dict(t,gamma=0)
    assert np.allclose(predict(t0,[1],[100],[.1],q["q0"],"quality"),predict(t0,[1],[100],form="base"))
    stress=pd.read_csv(OUT/"quality_model/b8_stress_test.csv")
    assert set(stress.data_type)=={"calibrated","extrapolated"}
    assert len(pd.read_csv(OUT/"quality_model/b6_bootstrap_parameters.csv"))>40


def test_p4_frozen_interface_and_thirteen_targets():
    ref=json.loads((OUT/"q1_interface/reference_mixture.json").read_text())
    h=pd.read_csv(OUT/"q1_interface/h_by_target.csv")
    assert len(ref["fields"])==17 and np.isclose(sum(ref["values"]),1)
    assert {f"h_{i}" for i in range(1,14)}<=set(h)
    assert np.allclose(h.iloc[0][[f"h_{i}" for i in range(1,14)]].to_numpy(float),0)
    assert stages.filehash(stages.ROOT/json.loads((OUT/"audit/q1_interface.json").read_text())["model_path"])==ref["model_sha256"]


def test_p5_scenarios_simplex_and_forms():
    grid=pd.read_csv(OUT/"generalized_law/scenario_grid.csv")
    assert set(grid.lambda_p)=={0,.5,1,1.5} and set(grid.form)=={"A","B"}
    zero=grid[grid.lambda_p==0].pivot(index=["workpoint","Q_B","recipe"],columns="form",values="loss_prediction")
    assert np.allclose(zero.A,zero.B)
    p=np.array([.2,.3,.5])
    v=stages.transfer_share(p,0,2,.1)
    assert np.isclose(v.sum(),1) and (v>=0).all()
    with pytest.raises(ValueError): stages.transfer_share(p,0,2,.6)
    pair=pd.read_csv(OUT/"generalized_law/path_conditioned_pair_response.csv")
    assert len(pair)>0 and pair.max_nearest_distance.notna().all()


def test_p6_analytic_numeric_and_bounds():
    t=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    a=pd.read_csv(OUT/"compute_opt/analytic_optima.csv")
    b=pd.read_csv(OUT/"compute_opt/bounded_optima.csv")
    n=pd.read_csv(OUT/"compute_opt/numerical_verification.csv")
    assert np.max(np.abs(a.alpha_P_minus_beta_T))<1e-8
    assert n.relative_difference.max()<1e-4
    assert np.allclose(a.N_star_billions*a.D_star_billions*6,a.C_1e18_FLOPs)
    assert b.loc[b.feasible,"N_star_bounded_billions"].between(b.bounds_N_min.min()-1e-8,b.bounds_N_max.max()+1e-8).all()
    assert b.loc[b.feasible,"D_star_bounded_billions"].between(b.bounds_D_min.min()-1e-8,b.bounds_D_max.max()+1e-8).all()
    scen=pd.read_csv(OUT/"compute_opt/mixture_scenario_optima.csv")
    assert set(scen.lambda_p)=={0,.5,1,1.5}


def test_p7_estimates_are_reference_only():
    v=pd.read_csv(OUT/"extrapolation/b10_estimated_consistency.csv")
    assert set(v.provenance)=={"estimated_reference"}
    assert v.n.eq(128).all()
    assert len(pd.read_csv(OUT/"extrapolation/b9_valid_metadata.csv"))==128


def test_p8_report_and_evidence():
    d=OUT/"report"
    report=(d/"question2_experiment_report.md").read_text()
    assert "COMPLETE_WITH_WARNINGS" in report and "B7-new" in report
    assert all((d/n).exists() for n in ("key_results_table.csv","evidence_grade_table.csv","figure_manifest.csv","reproducibility_summary.json"))
    grades=pd.read_csv(d/"evidence_grade_table.csv")
    assert set(grades.evidence_grade)=={"direct_fit","held_out_validation","semi_synthetic_calibration","imported_Q1","scenario_assumption","extrapolation_reference"}


def test_p8_detailed_report_is_traceable_and_repeatable():
    import re
    from src.q2_v2.detailed_report import build_report

    path=OUT/"report/question2_experiment_report.md"
    report=path.read_text()
    assert report==build_report(OUT)
    assert len(re.findall(r"^## 46\.\d+ ",report,flags=re.M))==18
    assert all(name in report for name in json.loads((OUT/"q1_interface/interface_metadata.json").read_text())["output_order"])
    expected=pd.read_csv(OUT/"b1_baseline/group_cv_metrics.csv").RMSE.mean()
    assert f"{expected:.6g}" in report
    assert "B7-new" in report and "半合成" in report and "情景" in report
    assert all(ord(char)>=32 or char in "\n\t" for char in report)
    for target in re.findall(r"\]\(([^)]+)\)",report):
        if not target.startswith(("http:","https:")):
            assert (path.parent/target).exists(), target
