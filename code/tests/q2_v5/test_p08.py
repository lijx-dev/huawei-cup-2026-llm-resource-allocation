"""P08 退化式与参数身份测试。"""

import json

import numpy as np

from q2_v5.baseline import predict
from q2_v5.common import stage_dir
from q2_v5.generalized_law import generalized_predict


def test_reduction_at_reference() -> None:
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    p = contract["quality_parameters"]
    n, d = np.array([.1, 1., 10.]), np.array([10., 100., 1000.])
    expected = predict(p, n, d)
    for form in ("A", "B"):
        actual = generalized_predict(contract["quality_model"], p, n, d,
                                     contract["q0"], contract["q0"], 0, 1, form)
        assert np.allclose(actual, expected)
    assert "NOT_IDENTIFIED" in contract["lambda_p"]["status"]
