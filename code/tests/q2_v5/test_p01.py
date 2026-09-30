"""P01 数据合同与重叠测试。"""

import pandas as pd
import pytest

from q2_v5.audit import overlap_keys
from q2_v5.common import BFILES, EVIDENCE, stage_dir


def test_contract_and_evidence() -> None:
    assert set(BFILES) == {f"B{i}" for i in range(1, 13)}
    assert EVIDENCE["B3"] == "interpolation"
    assert EVIDENCE["B6"] == "semi_synthetic"
    assert EVIDENCE["B10"] == "estimated_reference"


def test_overlap_uses_identifying_fields() -> None:
    b6 = pd.DataFrame({"experiment_id": ["a", "b"], "N_params_B": [1, 2],
                       "D_tokens_B": [3, 4], "Q_score": [0.2, 0.3],
                       "val_loss": [2.1, 2.2]})
    b7 = pd.concat([b6, pd.DataFrame({"experiment_id": ["c"], "N_params_B": [1],
                                       "D_tokens_B": [3], "Q_score": [0.4],
                                       "val_loss": [2.0]})], ignore_index=True)
    overlap, fresh = overlap_keys(b6, b7)
    assert len(overlap) == 2 and len(fresh) == 1
    assert fresh.iloc[0].experiment_id == "c"
    b7.loc[0, "val_loss"] = 9
    with pytest.raises(ValueError, match="Loss 冲突"):
        overlap_keys(b6, b7)


def test_real_audit_outputs() -> None:
    out = stage_dir("P01")
    inventory = pd.read_csv(out / "attachment_inventory.csv")
    fresh = pd.read_csv(out / "b7_new.csv")
    overlap = pd.read_csv(out / "b6_b7_overlap.csv")
    assert set(inventory.attachment) == set(BFILES)
    assert len(fresh) == 90 and len(overlap) == 360
    assert not set(fresh.experiment_id) & set(overlap.experiment_id)
