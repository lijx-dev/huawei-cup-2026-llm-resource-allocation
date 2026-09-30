"""P03 冻结迁移输出核验。"""

import json

import numpy as np
import pandas as pd

from q2_v5.baseline import predict
from q2_v5.common import stage_dir


def test_frozen_predictions_recalculate() -> None:
    p = json.loads((stage_dir("P02") / "final_parameters.json").read_text())["parameters"]
    frame = pd.read_csv(stage_dir("P03") / "transfer_predictions.csv")
    expected = predict(p, frame.N_params_B, frame.D_tokens_B)
    assert np.max(np.abs(expected - frame.prediction)) < 1e-10
    assert set(frame.attachment) == {"B2", "B3", "B4", "B5"}
    assert (frame.loc[frame.attachment == "B3", "evidence_type"] == "interpolation").all()


def test_metric_matrix_roles() -> None:
    matrix = pd.read_csv(stage_dir("P03") / "validation_matrix.csv")
    assert set(matrix.interpretation_level) == {
        "trend_only", "interpolation_consistency", "cross_family", "literature_cross_source"}
    assert matrix.n.gt(0).all()
