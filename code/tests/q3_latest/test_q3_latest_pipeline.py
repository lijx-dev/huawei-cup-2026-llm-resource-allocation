import json
import numpy as np
import pandas as pd

from src.q3_latest.pipeline import OUT, Q1, Q2, equivalent_legacy_parameters, loss, run


def test_q3_latest_outputs_and_upstream():
    meta=run()
    assert meta["latest"] and meta["status"]=="PASS_WITH_WARNINGS"
    assert meta["upstream"]["q2"]=="q2-fusion-rerun-v1"
    assert meta["checks"]["formal_q1_interface"]
    assert meta["checks"]["parameter_transform_max_abs_error"]<1e-12
    assert meta["checks"]["kkt_local_pass"]
    assert meta["checks"]["mechanism_epsilon_finite"]
    grid=pd.read_csv(OUT/"scenario_grid_all.csv")
    assert len(grid)==135 and set(grid.eta)=={1e-4,2e-4,4e-4}
    assert grid.loc[grid.feasible,"budget_used_fraction"].le(1+1e-8).all()


def test_q2_formula_transform_and_mixture_identity():
    q2=json.loads((Q2/"quality_model/b6_parameters.json").read_text())
    p=dict(q2["MQ"],q0=q2["q0"]);t=equivalent_legacy_parameters(p)
    n,d,q=1.3,160,.7
    old=t["E"]+t["A"]*n**(-t["alpha"])+t["B"]*d**(-t["beta"])*np.exp(-t["rho_D"]*q)
    assert np.isclose(old,loss(p,n,d,q))
    mix=pd.read_csv(OUT/"mixture_scenarios.csv")
    p0=mix[(mix.recipe_id=="p0") & (mix.lambda_p==1)].iloc[0]
    assert p0.h_agg==0
    assert mix.groupby("lambda_p").size().nunique()==1
