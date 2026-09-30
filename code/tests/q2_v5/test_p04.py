"""P04 模型选择、组隔离与 B7 屏蔽。"""

import json

import numpy as np
import pandas as pd
import pytest

from q2_v5.common import stage_dir
from q2_v5.quality_models import MODELS, freeze_digest, quality_predict, train_loader


def test_quality_model_reduces_at_q0() -> None:
    p = {"E": 1, "A": 2, "B": 3, "alpha": 0.5, "beta": .25,
         "rho_N": .7, "rho_D": -.2, "E1": .1}
    base = quality_predict("M0", p, np.array([4]), np.array([16]), np.array([.5]), .5)
    for model in MODELS:
        assert np.isclose(quality_predict(model, p, np.array([4]), np.array([16]), np.array([.5]), .5), base).all()


def test_p04_loader_rejects_other_stage() -> None:
    with pytest.raises(PermissionError):
        train_loader("P05")


def test_frozen_selection_and_group_isolation() -> None:
    out = stage_dir("P04")
    assert (out / "model_freeze.sha256").read_text().strip() == freeze_digest(out)
    decision = json.loads((out / "selection_decision.json").read_text())
    assert decision["selected_model"] in MODELS
    assert decision["B7_loss_accessed"] is False
    frame = pd.read_csv(out / "oof_predictions.csv")
    assert frame.groupby(["model", "group"]).fold.nunique().max() == 1
    assert frame.groupby(["model", "experiment_id"]).size().eq(1).all()
