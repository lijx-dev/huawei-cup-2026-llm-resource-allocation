"""问题三移植脚本的输入口径和约束微型校验。"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/q3"))

import q3_model_core as core
import q3_bootstrap_transitions as boot
import q3_fused_grid as grid
import q3_p_subproblem_fix as mix
from q3_paths import A_DIR, C7, INPUT_DIR


def test_frozen_interface_and_reference_recipe():
    manifest = json.loads((INPUT_DIR / "Q1_M2接口_manifest.json").read_text())
    for name, key in [
        ("mix_final_model_quad.csv", "source_coefficients_sha256"),
        ("Q1_M2响应接口.csv.gz", "interface_sha256"),
    ]:
        actual = hashlib.sha256((INPUT_DIR / name).read_bytes()).hexdigest()
        assert actual == manifest[key]
    a4 = pd.read_csv(A_DIR / "train_mixture_1m.csv")
    cols = [f"train_the_pile_{domain}" for domain in core.DOM17]
    assert len(a4) == 512
    np.testing.assert_allclose(a4[cols].mean().to_numpy(), core.P0_17_RAW, atol=1e-12)
    assert abs(core.h17(core.P0_17)) < 1e-10


def test_contexts_and_loss_parameterization():
    observed = sorted(pd.read_csv(C7)["max_position_embeddings"].dropna().unique())
    assert observed == core.LCTX_C7
    assert np.isclose(boot.rec(core.PAR_DEF, 1.0, 100.0, 0.5),
                      core.loss(1.0, 100.0, 0.5))
    q2 = json.loads((ROOT / "results/q2_v8/05_results/q2_v8_generalized_model.json").read_text())
    names = ["E_tilde", "A_tilde", "alpha", "B_tilde", "beta", "rho_N", "rho_D", "E1"]
    np.testing.assert_allclose(core.PAR_DEF, [q2["parameters"][name] for name in names], atol=5e-5)
    assert q2["status"] == "PARTIAL_CANDIDATE_INTERFACE"


def test_box_feasibility_and_budget():
    context = 32768
    minimum = grid.c_min(context, 2e-4)
    assert grid.solve_box(minimum * 0.99, "exp", 0.5, context) is None
    result = grid.solve_box(1e22, "exp", 0.5, context, nd=120, nq=100, refine=1)
    assert result is not None
    assert core.BOX["N"][0] <= result["N"] <= core.BOX["N"][1]
    assert core.BOX["D"][0] <= result["D"] <= core.BOX["D"][1]
    assert 0.5 <= result["Q"] <= 1
    assert result["idle_C18"] >= -1e-8
    assert np.isclose(result["s_train"] + result["s_attn"] + result["s_Q"]
                      + result["idle_frac"], 1.0, atol=1e-8)


def test_pure_scale_stationarity():
    point = core.pure_scale(1e22, 0.5, 4096)
    lhs = core._foc(point["N_B"], 0.5, 1e4, point["K"], 0.0)
    assert abs(lhs) < 1e-9


def test_strict_l1_optimizer_returns_feasible_points():
    radius = 0.1781
    score, recipe, count, rejected = mix.strict_slsqp_max(radius, nstart=2)
    assert count > 0 and rejected >= 0 and np.isfinite(score)
    assert np.isclose(recipe.sum(), 1.0, atol=1e-9)
    assert np.abs(recipe - core.P0_17).sum() <= radius + 1e-9
