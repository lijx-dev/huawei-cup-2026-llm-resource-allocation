"""P11 simplex、目标维度与联合响应恒等式。"""

import numpy as np
import pandas as pd
import pytest

from q2_v5.common import stage_dir
from q2_v5.mixture_effects import transfer


def test_transfer_simplex() -> None:
    p0 = np.full(17, 1 / 17)
    p = transfer(p0, 0, 1, .02)
    assert np.isclose(p.sum(), 1) and np.all(p >= 0)
    with pytest.raises(ValueError):
        transfer(p0, 0, 1, .1)


def test_pair_and_joint_outputs() -> None:
    out = stage_dir("P11")
    pair = pd.read_csv(out / "pairwise_transfer_by_target.csv")
    joint = pd.read_csv(out / "joint_response_by_target.csv")
    assert len(pair) == 17 * 16 * 13
    assert len(joint) == 17 * 16 // 2 * 13
    assert np.allclose(joint.local_joint_response, joint.Ljk - joint.Lj - joint.Lk + joint.L0)
