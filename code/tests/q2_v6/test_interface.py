"""只用人工构造微型接口检验哈希门控和参考点。"""

import json

import numpy as np
import pandas as pd
import pytest

from q2_v6.interface import load_interface, sha256
from q2_v6.scenarios import predict


def test_predict_reference_and_forms() -> None:
    params = {"E": 1.6, "A": .5, "B": 1.2, "alpha": .3, "beta": .2,
              "rho_N": .1, "rho_D": .2, "E1": .1}
    base = 1.6 + .5 + 1.2 * 150 ** (-.2)
    for form in ("A", "B"):
        assert predict(params, 1, 150, .5, .5, np.array([0.]), 1, form)[0] == pytest.approx(base)
    assert predict(params, 1, 150, .5, .5, np.array([.1]), 1, "B")[0] > predict(
        params, 1, 150, .5, .5, np.array([.1]), 1, "A")[0]
    with pytest.raises(ValueError):
        predict(params, 0, 150, .5, .5, np.array([0.]), 1, "A")


def test_manifest_hash_gate(tmp_path) -> None:
    path = tmp_path / "interface.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    columns = ["split", "recipe_id"] + [f"p_{i}" for i in range(13)] + [f"h_{i}" for i in range(13)] + ["h_p_eq", "hull13", "hull14", "hull17"]
    frame = pd.DataFrame([["p0 (A4 mean)", -1] + [0.07] * 13 + [0.] * 13 + [0., 1, 1, 1]], columns=columns)
    frame.to_csv(path, index=False, compression="gzip")
    manifest = {"interface_sha256": sha256(path), "coordinate_space": list(range(13)),
                "loss_targets": list(range(13)), "n_rows": 1, "p0_13coords": [0.07] * 13,
                "source_coefficients_sha256": "unknown", "self_checks": {"复读校验通过": True}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="分层行数"):
        load_interface(path, manifest_path)
    frame.loc[0, "h_0"] = 1
    frame.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="SHA-256"):
        load_interface(path, manifest_path)
