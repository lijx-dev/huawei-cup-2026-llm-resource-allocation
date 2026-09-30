"""P10 局部替代符号与非线性误差核验。"""

import numpy as np
import pandas as pd

from q2_v5.common import stage_dir


def test_substitution_identity() -> None:
    out = stage_dir("P10")
    marg = pd.read_csv(out / "marginal_effects.csv")
    marg = marg.loc[(marg.p_scenario == "p0") & (marg.lambda_p == 0)].set_index("workpoint")
    n = pd.read_csv(out / "q_n_substitution.csv").set_index("workpoint")
    d = pd.read_csv(out / "q_d_substitution.csv").set_index("workpoint")
    for key in marg.index:
        assert np.isclose(n.at[key, "dN_dQ"], -marg.at[key, "dL_dQ"] / marg.at[key, "dL_dN"])
        assert np.isclose(d.at[key, "dD_dQ"], -marg.at[key, "dL_dQ"] / marg.at[key, "dL_dD"])
    nonlinear = pd.read_csv(out / "substitution_nonlinearity.csv")
    assert set(nonlinear.delta_Q) == {.01, .05, .1}
    assert np.isfinite(nonlinear.loc[nonlinear.feasible, "local_linearization_error"]).all()
