"""P05 冻结一致性与 pre-validation 预测复核。"""

import json

import numpy as np
import pandas as pd

from q2_v5.common import stage_dir
from q2_v5.locked_validation import verify_freeze
from q2_v5.quality_models import quality_predict


def test_pre_validation_predictions() -> None:
    prior = verify_freeze()
    pred = pd.read_csv(stage_dir("P05") / "b7_new_predictions.csv")
    actual = quality_predict(prior["selected_model"], prior["parameters"],
                             pred.N_params_B, pred.D_tokens_B, pred.Q_score, prior["q0"])
    assert len(pred) == 90
    assert np.max(abs(actual - pred.selected_prediction)) < 1e-10
    assert pred.validation_type.eq("new-Q validation").all()


def test_postfit_not_used_for_holdout() -> None:
    post = json.loads((stage_dir("P05") / "post_validation_parameters.json").read_text())
    summary = json.loads((stage_dir("P05") / "validation_summary.json").read_text())
    assert post["used_for_holdout_metrics"] is False
    assert summary["uses_pre_validation_parameters"] is True
