"""v2.1 约束和最终产物验证；所有微型数组仅作人工测试。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from q1.mixture.dataset import load_pair
from q1_revision_v2.common import load_config as load_old_config
from q1_revision_v2_1.audit import invalid_count, summarize_field
from q1_revision_v2_1.common import load_config, file_manifest
from q1_revision_v2_1.quality import validate_groups, group_weights
from q1_revision_v2_1.conflict import three_disagreements
from q1_revision_v2_1.mixture_validation import nearest_rows, recipe_clusters, load_data
from q1_revision_v2_1.q_mapping import mapping_columns, mapped_q, nonidentity_permutations, cv_ridge, placebo


ROOT=Path(__file__).resolve().parents[2]
CFG=load_config(ROOT)
OLD=load_old_config(ROOT)
OUT=ROOT / "results/q1_revision_v2_1"


def test_invalid_elements_not_rows():
    assert invalid_count({"x":[float("nan")]*6},"x",6)==6


def test_unique_invalid_ids_traced():
    row=summarize_field("x",[("id1","rejected_corrupt",6)],6)
    assert row["raw_invalid_rows"]==1 and row["unique_invalid_ids"]==1


def test_duplicate_invalid_rows_not_double_subtracted():
    row=summarize_field("x",[("id1","rejected_corrupt",6),("id1","rejected_corrupt",6)],12)
    assert row["duplicate_invalid_rows"]==1 and row["unique_invalid_ids"]==1


def test_invalid_scored_row_flagged():
    row=summarize_field("x",[("id1","valid",1)],1)
    assert row["used_in_scoring"] and row["remaining_invalid_after_dedup"]==1


def test_actual_invalid_rows_excluded():
    x=pd.read_csv(OUT / "audit/field_invalid_resolution.csv")
    assert x.remaining_invalid_after_dedup.sum()==0 and not x.used_in_scoring.any()
    reason=x.set_index("field").loc["modernbert_reasoning"]
    professional=x.set_index("field").loc["modernbert_professionalism"]
    assert (reason.raw_invalid_rows,reason.raw_invalid_elements)==(13,78)
    assert (professional.raw_invalid_rows,professional.raw_invalid_elements)==(6,36)


def test_groups_cover_exactly_22():
    validate_groups(OLD["quality_fields"],CFG["groups"])
    assert sum(map(len,CFG["groups"].values()))==22


def test_group_overlap_rejected():
    groups={k:list(v) for k,v in CFG["groups"].items()}
    groups["B_dsir"][0]=groups["A_model_ratings"][0]
    with pytest.raises(ValueError):
        validate_groups(OLD["quality_fields"],groups)


def test_every_group_within_weight_sums_one():
    x=pd.read_csv(OUT / "quality/hierarchical_weights.csv")
    assert np.allclose(x.groupby("group").within_group_weight.sum(),1)


def test_each_group_global_weight_one_third():
    x=pd.read_csv(OUT / "quality/hierarchical_weights.csv")
    assert np.allclose(x.groupby("group").global_weight.sum(),1/3)


def test_global_weight_sums_one():
    x=pd.read_csv(OUT / "quality/hierarchical_weights.csv")
    assert x.global_weight.sum()==pytest.approx(1)


def test_hierarchical_q_in_bounds():
    x=pd.read_csv(OUT / "quality/sample_quality_scores.csv.gz",usecols=["Q_hierarchical_balanced","q_hierarchical_balanced"])
    assert x.Q_hierarchical_balanced.between(0,100).all()
    assert np.allclose(x.q_hierarchical_balanced,x.Q_hierarchical_balanced/100)


def test_drop4_rebalances_group_c():
    x=pd.read_csv(OUT / "quality/drop4_weights.csv")
    assert not set(CFG["direction_provisional"]) & set(x.field)
    assert x[x.group=="C_language_structure"].within_group_weight.sum()==pytest.approx(1)


def test_degenerate_group_deterministic_fallback():
    z=np.array([[0.,0.,0.,1.],[1.,1.,1.,0.],[0.,0.,1.,1.]])
    rho=np.corrcoef(z,rowvar=False)
    w,_,fallback,_=group_weights(z,rho,["a","b","c","d"],{"A":["a","b"],"B":["c"],"C":["d"]})
    assert fallback and w[0]==pytest.approx(1/6) and w[1]==pytest.approx(1/6)


def test_new_normalized_matrix_matches_frozen_preprocessing():
    a=np.load(OUT / "quality/normalized_22.npy",mmap_mode="r")
    b=np.load(ROOT / "results/q1_revision_v2/quality/normalized_22.npy",mmap_mode="r")
    assert a.shape==b.shape==(272475,22) and np.array_equal(a,b)


def test_three_disagreements_in_unit_interval():
    z=np.array([[0.,.5,1.],[1.,.2,.2]])
    c=three_disagreements(z,np.array([.2,.3,.5]),{"A":z[:,0],"B":z[:,1],"C":z[:,2]})
    assert c.shape==(2,3) and c.min()>=0 and c.max()<=1


def test_all_equal_signals_zero_disagreement():
    z=np.full((2,3),.4)
    c=three_disagreements(z,np.array([.2,.3,.5]),{"A":z[:,0],"B":z[:,1],"C":z[:,2]})
    assert np.allclose(c,0)


def test_equal_indicator_ignores_q_weights():
    z=np.array([[0.,.5,1.]])
    groups={"A":z[:,0],"B":z[:,1],"C":z[:,2]}
    a=three_disagreements(z,np.array([.2,.3,.5]),groups)
    b=three_disagreements(z,np.array([.8,.1,.1]),groups)
    assert a[0,1]==pytest.approx(b[0,1])


def test_weighted_pair_decomposition():
    z=np.array([0.,.3,1.])
    w=np.array([.2,.3,.5])
    i,j=np.triu_indices(3,k=1)
    manual=(w[i]*w[j]*abs(z[i]-z[j])).sum()/(w[i]*w[j]).sum()
    observed=three_disagreements(z[None,:],w,{"A":z[None,0],"B":z[None,1],"C":z[None,2]})[0,0]
    assert observed==pytest.approx(manual)


def test_extensions_use_fixed_a1_threshold():
    thresholds=json.loads((OUT / "conflict/disagreement_thresholds.json").read_text())["thresholds"]
    x=pd.read_csv(OUT / "conflict/sample_disagreement_scores.csv.gz")
    for method,levels in thresholds.items():
        a1=x[x.attachment=="A1"]
        assert levels["0.9"]==pytest.approx(a1[method].quantile(.9))
        assert (x[f"high_{method}_0.9"]==(x[method]>levels["0.9"])).all()


def test_unweighted_pair_output_has_raw_difference():
    x=pd.read_csv(OUT / "conflict/equal_pair_disagreement.csv")
    assert "H_equal_raw" in x and "global_weight" not in x
    assert x.H_equal_raw.between(0,1).all()


def _mini_data():
    train=np.array([[1.,0.],[0.,1.],[.6,.4]])
    test=np.array([[1.,0.],[.5,.5]])
    return {"train_1m":(np.array(["a","b","c"]),train,np.zeros((3,1))),
            **{name:(np.array(["d","e"]),test,np.zeros((2,1))) for name in ("test_1m","test_60m","test_1b","est_10b","est_70b")}}


def test_train_nearest_excludes_self():
    rows=nearest_rows(_mini_data())
    assert (rows[rows.split=="train_1m"].nearest_distance>0).all()


def test_test_nearest_searches_train_only():
    rows=nearest_rows(_mini_data())
    test=rows[rows.split=="test_1m"]
    assert test.iloc[0].nearest_distance==pytest.approx(0)
    assert test.iloc[0].nearest_train_index=="a"


def test_distance_splits_preserved():
    rows=nearest_rows(_mini_data())
    assert rows.groupby("split").size().to_dict()["train_1m"]==3
    assert set(rows.split)==set(_mini_data())


def test_recipe_clustering_only_accepts_x():
    import inspect
    assert "y" not in inspect.signature(recipe_clusters).parameters
    x=np.random.default_rng(7).random((30,17))
    a,_=recipe_clusters(x,7,10,20)
    b,_=recipe_clusters(x,7,10,20)
    assert np.array_equal(a,b)


def test_cluster_cv_no_group_leakage():
    x=pd.read_csv(OUT / "mixture/cluster_cv_fold_membership.csv")
    for _,fold in x.groupby("fold"):
        tr=set(fold.loc[fold.role=="train","cluster"])
        va=set(fold.loc[fold.role=="validation","cluster"])
        assert not tr & va


def test_cluster_cv_frozen_hyperparameters():
    ref=json.loads((OUT / "mixture/inherited_v2_model_reference.json").read_text())
    old=json.loads((ROOT / "results/q1_revision_v2/mixture/model_selection.json").read_text())
    assert ref["frozen_params"]==old["lightgbm_params"]
    assert ref["prediction_max_abs_diff"]["LightGBM"]<1e-10


def test_coverage_equals_six_proportions():
    data,fields,_=load_data(ROOT,ROOT / "results/q1_revision_v2",OLD)
    mapping,cols=mapping_columns(ROOT,fields)
    ids,x,_=data["train_1m"]
    result=pd.read_csv(OUT / "q_mapping/qmapped_coverage.csv")
    a=result[result.split=="train_1m"]
    assert len(cols)==6 and len(mapping)==6
    assert np.allclose(a.coverage,x[:,cols].sum(axis=1))


def test_coverage_unit_interval():
    x=pd.read_csv(OUT / "q_mapping/qmapped_coverage.csv")
    assert x.coverage.between(0,1+1e-10).all()


def test_coverage_filter_does_not_mutate_input():
    x=np.array([[.2,.8],[.8,.2]])
    original=x.copy()
    q,mass=mapped_q(x,[0],np.array([70.]))
    _=x[mass>=.4]
    assert np.array_equal(x,original)


def test_placebo_only_permutations_and_seed_reproducible():
    a=nonidentity_permutations(7,1000)
    b=nonidentity_permutations(7,1000)
    assert np.array_equal(a,b)
    assert all(sorted(row)==list(range(6)) and not np.array_equal(row,np.arange(6)) for row in a)


def test_placebo_keeps_cv_folds_fixed():
    x=np.random.default_rng(7).random((20,3))
    a=list(KFold(5,shuffle=True,random_state=7).split(x))
    b=list(KFold(5,shuffle=True,random_state=7).split(x))
    assert all(np.array_equal(i[1],j[1]) for i,j in zip(a,b))


def test_placebo_function_needs_only_training_split(tmp_path):
    rng=np.random.default_rng(7)
    x=rng.dirichlet(np.ones(7),size=10)
    y=rng.normal(size=(10,2))
    config={**CFG,"placebo_repetitions":1,"q_ridge_alphas":[.1,1.]}
    result=placebo({"train_1m":(np.arange(10),x,y)},list(range(6)),{"Q_hierarchical_balanced":np.arange(6.)},config,tmp_path)
    assert len(result)==1 and result[0]["repetitions"]==1


def test_actual_placebo_has_1000_draws_per_q():
    x=pd.read_csv(OUT / "q_mapping/qmapped_placebo.csv")
    assert x.groupby("quality_method").size().eq(1000).all()
    assert all(p!="0-1-2-3-4-5" for p in x.permutation)


def test_v2_history_hashes_unchanged():
    old=ROOT / "results/q1_revision_v2"
    expected=json.loads((OUT / "audit/v2_result_hashes_before.json").read_text())
    assert file_manifest(old)==expected
