"""P02 数学约束与分组预测测试。"""

import json

import numpy as np
import pandas as pd
import pytest

from q2_v5.baseline import fit, predict
from q2_v5.common import metrics, stage_dir


def test_positive_prediction_and_invalid_support() -> None:
    p = {"E": 1, "A": 2, "B": 3, "alpha": 0.5, "beta": 0.25}
    assert np.isclose(predict(p, np.array([4]), np.array([16]))[0], 1 + 1 + 1.5)
    with pytest.raises(ValueError, match="正"):
        predict(p, np.array([0]), np.array([1]))


def test_fit_synthetic_micro_example() -> None:
    n, d = np.meshgrid(np.array([0.1, 0.5, 2, 10]), np.array([1, 5, 20, 100]))
    truth = {"E": 1.3, "A": 0.8, "B": 0.6, "alpha": 0.4, "beta": 0.3}
    y = predict(truth, n.ravel(), d.ravel())
    fitted = fit(n.ravel(), d.ravel(), y)
    assert np.max(np.abs(predict(fitted.parameters, n.ravel(), d.ravel()) - y)) < 1e-6
    assert all(value > 0 for value in fitted.parameters.values())


def test_real_oof_has_no_own_scale() -> None:
    out = stage_dir("P02")
    oof = pd.read_csv(out / "oof_predictions.csv")
    folds = pd.read_csv(out / "group_cv_metrics.csv")
    pooled = json.loads((out / "pooled_metrics.json").read_text())
    macro = json.loads((out / "macro_metrics.json").read_text())
    assert len(folds) == oof.N_params_B.nunique()
    assert np.isfinite(oof.prediction).all()
    assert np.isclose(pooled["rmse"], metrics(oof.val_loss, oof.prediction)["rmse"])
    assert np.isclose(macro["rmse_macro"], folds.rmse.mean())
