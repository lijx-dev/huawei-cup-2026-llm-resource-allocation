"""P13 只读历史比较和核心冻结。"""

import pandas as pd

from q2_v5.common import stage_dir
from q2_v5.core_robustness import verify_core_freeze


def test_shadow_after_freeze() -> None:
    assert verify_core_freeze()["freeze_scope"] == "P01–P11 outputs and P08 model contract"
    table = pd.read_csv(stage_dir("P13") / "shadow_comparison.csv")
    assert {"B1_parameter", "quality_selection", "B7_new", "mixture", "effects", "stress"} <= set(table.topic)
