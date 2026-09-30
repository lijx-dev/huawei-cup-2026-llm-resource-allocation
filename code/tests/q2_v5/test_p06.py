"""P06 证据等级和支持距离测试。"""

import numpy as np
import pandas as pd

from q2_v5.common import stage_dir
from q2_v5.robustness import log_support_distance


def test_support_distance_zero_on_reference() -> None:
    ref = pd.DataFrame({"N_params_B": [1., 2.], "D_tokens_B": [3., 4.]})
    result = log_support_distance(np.array([1.]), np.array([3.]), ref)
    assert np.isclose(result[0], 0)


def test_stress_evidence() -> None:
    out = stage_dir("P06")
    b8 = pd.read_csv(out / "b8_predictions.csv")
    b9 = pd.read_csv(out / "b9_support_distance.csv")
    b10 = pd.read_csv(out / "b10_estimated_consistency.csv")
    assert b8.evidence_type.eq("stress_test").all()
    assert b9.evidence_type.eq("metadata_only").all()
    assert b10.evidence_type.eq("estimated_reference").all()
    assert "val_loss" not in b9
