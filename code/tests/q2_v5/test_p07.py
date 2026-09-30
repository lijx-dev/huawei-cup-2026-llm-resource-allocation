"""P07 冻结产物、simplex 与 h(p0)=0。"""

import json

import numpy as np
import pandas as pd
import pytest

from q2_v5.common import sha, stage_dir
from q2_v5.q1_bridge import frozen_dir, hp, load_frozen_model, mapping_scenarios, predict_mixture


def test_frozen_copy_integrity() -> None:
    hashes = json.loads((frozen_dir() / "frozen_hashes.json").read_text())
    assert all(sha(frozen_dir() / name) == value["sha256"] for name, value in hashes.items())


def test_hp_reference_and_13_targets() -> None:
    artifact = load_frozen_model()
    p0 = np.array(json.loads((frozen_dir() / "reference_mixture.json").read_text())["values"])
    assert np.isclose(p0.sum(), 1)
    assert hp(p0, p0, artifact).shape == (1, 13)
    assert np.max(np.abs(hp(p0, p0, artifact))) < 1e-12
    assert (predict_mixture(p0, artifact) > 0).all()
    with pytest.raises(ValueError):
        predict_mixture(p0 * 2, artifact)


def test_mapping_is_scenario() -> None:
    mappings = pd.read_csv(stage_dir("P07") / "qa_qb_mapping_scenarios.csv")
    assert mappings.mapping.nunique() == 3
    assert mappings.evidence_type.eq("scenario_assumption").all()


def test_rank_mapping_uses_empirical_percentile() -> None:
    qa = pd.DataFrame({"domain": ["a", "b", "c"], "Q_A": [10., 20., 100.]})
    qb = pd.Series([.1, .5, .9])
    mapped = mapping_scenarios(qa, qb)
    rank = mapped.loc[mapped.mapping == "rank_percentile"].set_index("domain")
    assert rank.at["b", "Q_B_scenario"] == .5
