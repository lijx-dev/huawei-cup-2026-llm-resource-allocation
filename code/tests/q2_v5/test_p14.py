"""P14 终稿、图源和一致性审计。"""

import json

import pandas as pd

from q2_v5.common import stage_dir


def test_final_report_and_figures() -> None:
    out = stage_dir("P14")
    report = (out / "question2_v5_experiment_report.md").read_text()
    assert all(f"## {i} " in report for i in range(1, 20))
    checks = json.loads((out / "final_consistency_audit.json").read_text())
    assert checks["critical_failures"] == 0
    figures = pd.read_csv(out / "figure_manifest.csv")
    sources = pd.read_csv(out / "figure_sources.csv")
    assert len(figures) == 8
    assert set(figures.figure) == set(sources.figure)
    assert all((out / name).exists() for name in figures.path)


def test_key_results_traceability() -> None:
    out = stage_dir("P14")
    keys = pd.read_csv(out / "key_results.csv")
    assert keys.source.notna().all()
    assert keys.metric.is_unique


def test_shared_directory_stage_hash_boundaries() -> None:
    p09 = json.loads((stage_dir("P09") / "p09_metadata.json").read_text())
    p12 = json.loads((stage_dir("P12") / "p12_metadata.json").read_text())
    assert "p10_metadata.json" not in p09["output_sha256"]
    assert "p13_metadata.json" not in p12["output_sha256"]
