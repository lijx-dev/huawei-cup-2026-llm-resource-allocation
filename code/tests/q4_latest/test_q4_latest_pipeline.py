import json
import pandas as pd

from src.q4_latest.pipeline import OUT


def test_q4_latest_markers_and_upstream():
    meta=json.loads((OUT/"metadata.json").read_text())
    marker=json.loads((OUT/"LATEST_VERSION.json").read_text())
    assert marker["latest"] and meta["latest"]
    assert meta["status"]=="PASS_WITH_WARNINGS"
    assert meta["scripts_success"]==meta["scripts_total"]==20
    assert meta["counts"]["c_only_unchanged"]==meta["counts"]["c_only_checked"]==7
    assert meta["comparisons"]["forecast_changed_with_q3_mechanism"]
    assert meta["upstream"]=={"q1":"q1-to-q2-v1","q2":"q2-fusion-rerun-v1","q3":"q3-latest-20260926-v1"}


def test_q4_latest_compat_interface_and_q3_gate():
    compat=pd.read_csv(OUT/"reproduction/inputs/Q1_M2响应接口.csv.gz")
    assert "h_p_eq" in compat and len(compat)==1215
    assert compat.iloc[0].h_p_eq==0
    integration=json.loads((OUT/"upstream_integration.json").read_text())
    assert integration["q3_rows"]==135 and integration["q3_feasible"]==123
    assert "not converted" in integration["policy"]
