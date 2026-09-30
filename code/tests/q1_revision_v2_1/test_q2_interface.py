import json

import joblib
import numpy as np
import pandas as pd

from src.q1_revision_v2_1.export_q2_interface import DEFAULT_OUT, MODEL_PATH, ROOT, export, sha256


def test_formal_q2_interface_contract():
    manifest = export()
    assert manifest["status"] == "PASS"
    assert manifest["model"]["sha256"] == sha256(MODEL_PATH)
    assert len(manifest["input_order"]) == 17
    assert len(manifest["output_order"]) == 13
    data = pd.read_csv(ROOT / manifest["mixture_response_path"])
    assert len(data) == 1214
    assert data.groupby("split")["index"].apply(lambda x: x.is_unique).all()
    assert np.allclose(data[manifest["input_order"]].sum(axis=1), 1.0, atol=1e-12)
    assert (data["nearest_train_distance"] >= 0).all()


def test_reference_prediction_and_quality_policy():
    manifest = json.loads((DEFAULT_OUT / "manifest.json").read_text())
    reference = json.loads((DEFAULT_OUT / "reference_mixture.json").read_text())
    bundle = joblib.load(MODEL_PATH)
    p0 = np.asarray(reference["values"])[None, :]
    prediction = np.column_stack([m.predict(p0) for m in bundle["models"][manifest["model"]["name"]]])[0]
    assert np.allclose(prediction, reference["reference_prediction"])
    mapping = pd.read_csv(DEFAULT_OUT / "quality_mapping_interface.csv")
    assert len(mapping) == 6
    assert set(mapping.mapping_type) == {"direct", "near_direct"}
    assert "remain distinct" in manifest["quality_policy"]
