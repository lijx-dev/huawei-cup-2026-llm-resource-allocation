"""人工构造的微型样例，仅验证 V7 数学和完整性门控。"""

import io
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from q2_v7.model import compute_optimum, derivatives, fit, predict, quality_incremental_cost_1e21
from q2_v7.pipeline import INTERFACE, INTERFACE_MANIFEST, PACKAGE_MANIFEST, digest, group_cv, load_package
from q2_v7.extensions import paired_scale_analysis


P = {"E": 1.7, "A": .55, "B": 1.3, "alpha": .3, "beta": .28,
     "rho_N": .35, "rho_D": .15, "E1": .1}


def test_model_reference_forms_and_derivatives():
    base = predict(P, 1., 150., .5)
    assert predict(P, 1., 150., .5, hp=0, lam=1.5, form="A") == pytest.approx(base)
    assert predict(P, 1., 150., .5, hp=0, lam=1.5, form="B") == pytest.approx(base)
    assert predict(P, 1., 150., .5, hp=.2, lam=0) == pytest.approx(base)
    assert predict(P, 1., 150., .5, hp=.2, lam=1, theta=.5) == pytest.approx(predict(P, 1., 150., .5, hp=.2, lam=1))
    assert predict(P, 1., 150., .7, hp=.2, lam=1, theta=.5) > predict(P, 1., 150., .7, hp=.2, lam=1, theta=0)
    assert predict(P, 1., 150., .5, hp=.2, lam=1, form="B") > predict(P, 1., 150., .5, hp=.2, lam=1, form="A")
    values = derivatives(P, 1., 150., .5)
    eps = 1e-5
    central = (predict(P, 1., 150., .5+eps)-predict(P, 1., 150., .5-eps))/(2*eps)
    assert values["dL_dQ"] == pytest.approx(central, rel=1e-8)
    assert values["dN_dQ_at_fixed_L_D"] < 0
    for bad in (0, -1, np.nan):
        with pytest.raises(ValueError):
            predict(P, bad, 150, .5)


def test_compute_analytic_matches_numeric_and_bounded():
    budget = .9
    result = compute_optimum(P, budget, .5, {"N_params_B": [.07, 11.97], "D_tokens_B": [10, 600]})
    k = budget/.006
    numeric = minimize_scalar(lambda n: predict(P, n, k/n, .5), bounds=(.25, 11.97), method="bounded")
    assert result["N_analytic_B"] == pytest.approx(numeric.x, rel=1e-5)
    assert .07 <= result["N_bounded_B"] <= 11.97
    assert 10 <= result["D_bounded_B"] <= 600
    assert .006*result["N_bounded_B"]*result["D_bounded_B"] == pytest.approx(budget)
    form_b = compute_optimum(P, budget, .5, hp=.2, lam=1., form="B")
    assert form_b["N_analytic_B"] == pytest.approx(result["N_analytic_B"])


def test_appendix_b_quality_cost_units():
    c = quality_incremental_cost_1e21(150., .5, .6, "exponential")
    expected = 150e9 * 1e7 * (np.exp(6*.6)-np.exp(6*.5))/1e21
    assert c == pytest.approx(expected)
    assert c/.006/150 == pytest.approx(0.0275, rel=.03)
    assert quality_incremental_cost_1e21(150., .6, .5, "exponential") == 0
    with pytest.raises(ValueError):
        quality_incremental_cost_1e21(150., .5, .6, "unknown")


def test_fit_reproducible_and_positive():
    n = np.repeat([.1, .3, 1., 3., 10.], 8)
    d = np.tile(np.geomspace(10, 300, 8), 5)
    y = predict(P, n, d)
    a = fit(n, d, y, seed=7, starts=2)
    b = fit(n, d, y, seed=7, starts=2)
    assert a.params == b.params
    assert a.rss < 1e-10
    assert all(a.params[x] > 0 for x in ("E", "A", "B", "alpha", "beta"))


def test_zip_hash_gate(tmp_path):
    coords = [f"c{i}" for i in range(13)]
    targets = [f"t{i}" for i in range(13)]
    cols = ["split", "recipe_id"] + [f"p_{x}" for x in coords] + [f"h_{x}" for x in targets] + ["h_p_eq", "hull13", "hull14", "hull17"]
    row = ["p0 (A4 mean)", -1] + [.07]*13 + [0.]*13 + [0., 1, 1, 1]
    buffer = io.BytesIO()
    pd.DataFrame([row], columns=cols).to_csv(buffer, index=False, compression="gzip")
    payload = buffer.getvalue()
    package = {"files": [{"path": "03_对照材料/Q1_M2响应接口.csv.gz", "bytes": len(payload), "sha256": "0"*64}]}
    manifest = {"interface_sha256": digest(payload), "coordinate_space": coords, "loss_targets": targets,
                "n_rows": 1, "p0_13coords": [.07]*13, "source_coefficients_sha256": "claimed"}
    path = tmp_path / "package.zip"
    with ZipFile(path, "w") as z:
        z.writestr(PACKAGE_MANIFEST, json.dumps(package))
        z.writestr(INTERFACE, payload)
        z.writestr(INTERFACE_MANIFEST, json.dumps(manifest))
    with pytest.raises(ValueError, match="完整性失败"):
        load_package(path)


def test_quality_cv_keeps_grid_together():
    n = np.repeat([.1, .3, 1., 3., 10.], 3*3)
    d = np.tile(np.repeat([10., 50., 150.], 3), 5)
    q = np.tile([.2, .5, .8], 15)
    y = predict(P, n, d, q)
    table = pd.DataFrame({"N_params_B": n, "D_tokens_B": d, "Q_score": q, "val_loss": y})
    folds, predictions = group_cv(table, True, 7)
    assert len(folds) == 10
    mq = predictions.loc[predictions.model=="MQ"].copy()
    mq["grid"] = [f"{n[i]}|{d[i]}" for i in mq.row_number.to_numpy()-2]
    assert mq.groupby("grid").fold.nunique().max() == 1


def test_regmix_joins_by_index_and_recovers_slope(tmp_path):
    base = tmp_path / "data/real_attachments/A_data_value/regmix_tables"
    base.mkdir(parents=True)
    ids = [10, 30, 20, 40]
    mixcols = [f"train_the_pile_d{i}" for i in range(17)]
    mix = pd.DataFrame({"index": ids, **{col: [float(i==j) for i in range(4)] for j,col in enumerate(mixcols)}})
    mix.to_csv(base / "test_mixture_1m.csv", index=False)
    mix.iloc[::-1].to_csv(base / "test_mixture_60m.csv", index=False)
    targets = [f"d{i}" for i in range(13)]
    losscols = [f"metric/the_pile_{t}_val_loss" for t in targets]
    loss1 = pd.DataFrame({"index": ids, **{c: [3., 4., 5., 6.] for c in losscols}})
    loss60 = pd.DataFrame({"index": ids, **{c: [2+.75*v for v in [3.,4.,5.,6.]] for c in losscols}})
    loss1.to_csv(base / "test_pile_loss_1m.csv", index=False)
    loss60.iloc[::-1].to_csv(base / "test_pile_loss_60m.csv", index=False)
    result = paired_scale_analysis(tmp_path, tmp_path / "out", targets, 7)
    assert result["n_paired_recipes"] == 4
    assert result["absolute_slope_aggregate"] == pytest.approx(.75)
    assert result["paired_recipe_max_abs_difference"] == 0
