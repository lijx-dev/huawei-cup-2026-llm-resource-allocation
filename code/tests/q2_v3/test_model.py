"""人工构造微型样例，用于数学及阶段约束回归。"""
import numpy as np
import pandas as pd
import pytest

from src.q2_v3.model import analytic_opt, bounded_opt, fit, predict
from src.q2_v3.stages import _amplitudes, _base_group, _marginals


def test_positive_nd_and_quality_reference():
    p = np.array([1.0, 2.0, 3.0, .4, .3, .2])
    n, d = 2., 5.
    assert predict("MD", p, n, d, 0) == pytest.approx(predict("M0", p[:5], n, d))
    with pytest.raises(ValueError): predict("MD", p, 0, d, .5)


def test_analytic_numeric_and_bounds():
    p = np.array([1.0, 2.0, 3.0, .4, .3])
    c = 6e20
    a = analytic_opt("M0", p, c, .5)
    n = bounded_opt("M0", p, c, .5, (.01, 1000, .01, 1000))
    assert a[0] == pytest.approx(n[0], rel=1e-6)
    assert a[1] == pytest.approx(n[1], rel=1e-6)
    limited = bounded_opt("M0", p, c, .5, (1, 2, 1, 100))
    assert 1 <= limited[0] <= 2
    assert limited[1] == pytest.approx(c / 6e18 / limited[0])


def test_group_amplitude_and_marginal_sign():
    d = pd.DataFrame({"N_params_B": [1.] * 5, "D_tokens_B": [10.] * 5,
                      "Q_score": [.1, .3, .5, .7, .9], "val_loss": [3., 2.8, 2.6, 2.4, 2.2]})
    amp = _amplitudes(d).iloc[0]
    assert amp.A_max == pytest.approx(.8)
    assert amp.dL_dQ == pytest.approx(-1.)
    m = _marginals("MD", np.array([1., 2., 3., .4, .3, .2]), 1., 10., .5)
    assert m["M_N"] > 0 and m["M_D"] > 0 and m["M_Q"] > 0


def test_fit_reproducible_and_group_holdout_contract():
    n = np.repeat(np.array([.2, .5, 1., 2., 5.]), 6)
    d = np.tile(np.array([1., 2., 4., 8., 16., 32.]), 5)
    y = predict("M0", np.array([1., .8, 1.2, .3, .4]), n, d)
    frame = pd.DataFrame({"N_params_B": n, "D_tokens_B": d, "val_loss": y})
    a, _ = fit("M0", frame, starts=3)
    b, _ = fit("M0", frame, starts=3)
    assert np.allclose(a, b)
    assert np.all(a[:5] > 0)


def test_base_group_canonicalizes_csv_numeric_types():
    a = pd.DataFrame({"N_params_B": [0.07], "D_tokens_B": [10]})
    b = pd.DataFrame({"N_params_B": [0.07], "D_tokens_B": [10.0]})
    assert _base_group(a).iloc[0] == _base_group(b).iloc[0]
