"""P12 核心冻结与不确定性分类。"""

import pandas as pd

from q2_v5.common import stage_dir
from q2_v5.core_robustness import verify_core_freeze


def test_core_freeze_and_uncertainty_types() -> None:
    manifest = verify_core_freeze()
    assert len(manifest["output_sha256"]) > 50
    matrix = pd.read_csv(stage_dir("P12") / "robustness_matrix.csv")
    assert set(matrix.category) == {"statistical", "structural", "cross_source", "scenario"}
