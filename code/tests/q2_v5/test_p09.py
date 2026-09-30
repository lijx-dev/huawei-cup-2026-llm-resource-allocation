"""P09 解析导数对数值差分核验。"""

import numpy as np
import pandas as pd

from q2_v5.common import stage_dir


def test_derivative_checks_and_signs() -> None:
    checks = pd.read_csv(stage_dir("P09") / "derivative_checks.csv")
    effects = pd.read_csv(stage_dir("P09") / "marginal_effects.csv")
    assert checks.relative_error.max() < 1e-5
    assert (effects.M_N > 0).all() and (effects.M_D > 0).all()
    assert np.isfinite(effects.predicted_loss).all()
