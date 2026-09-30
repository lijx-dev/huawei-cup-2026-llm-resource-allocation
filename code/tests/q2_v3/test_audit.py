"""仅使用人工构造微型样例；不进入正式结果。"""
import pandas as pd
import pytest

from src.q2_v3.audit import overlap
from src.q2_v3.common import valid_ndl


def test_overlap_identifies_exact_and_new():
    a = pd.DataFrame({"experiment_id": ["a"], "N_params_B": [1.], "D_tokens_B": [10.], "Q_score": [.4], "val_loss": [2.]})
    b = pd.concat([a, a.assign(experiment_id="b")], ignore_index=True)
    result = overlap(a, b)
    assert list(result.role) == ["duplicate_exact", "b7_new"]
    b.loc[0, "val_loss"] = 3.
    assert overlap(a, b).iloc[0].role == "duplicate_conflict"


def test_invalid_rows_are_excluded_with_evidence():
    d = pd.DataFrame({"N_params_B": [1., -1.], "D_tokens_B": [2., 2.], "val_loss": [3., 3.]})
    valid, excluded = valid_ndl(d, "P0", "artificial.csv")
    assert len(valid) == len(excluded) == 1
    assert excluded[0]["action"] == "exclude"
