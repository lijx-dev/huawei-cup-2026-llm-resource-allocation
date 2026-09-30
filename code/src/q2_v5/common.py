"""V5 公共合同、阶段门和可复现元数据。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BROOT = ROOT / "data/real_attachments/B_scaling_laws"
OUT = ROOT / "results/q2_v5"
CONFIG = ROOT / "configs/q2_v5/pipeline.yaml"
STAGES = {
    "P01": "01_audit", "P02": "02_b1_baseline", "P03": "03_transfer",
    "P04": "04_quality_selection", "P05": "05_locked_validation",
    "P06": "06_stress_and_extrapolation", "P07": "07_q1_bridge",
    "P08": "08_generalized_law", "P09": "09_effects", "P10": "09_effects",
    "P11": "10_mixture_effects", "P12": "11_robustness",
    "P13": "11_robustness", "P14": "12_report",
}
BFILES = {
    "B1": ["pythia_training_log_existing.csv"],
    "B2": ["cerebras_training_log.csv"],
    "B3": ["training_trajectories"],
    "B4": ["scaling_baseline.csv"],
    "B5": ["published_scaling_data.csv"],
    "B6": ["supplementary_NQ_experiment.csv"],
    "B7": ["supplementary_NQ_experiment_expanded.csv"],
    "B8": ["supplementary_NQ_experiment_large.csv"],
    "B9": ["supplementary_large_models.csv"],
    "B10": ["supplementary_large_baseline.csv"],
    "B11": ["pythia_checkpoint_index.csv"],
    "B12": ["open_model_family_metadata.csv"],
}
EVIDENCE = {
    "B1": "direct_observation", "B2": "semi_synthetic",
    "B3": "interpolation", "B4": "direct_observation",
    "B5": "direct_observation", "B6": "semi_synthetic",
    "B7": "semi_synthetic", "B8": "stress_test",
    "B9": "metadata_only", "B10": "estimated_reference",
    "B11": "metadata_only", "B12": "metadata_only",
}


def config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_for(attachment: str) -> list[Path]:
    if attachment not in BFILES:
        raise ValueError(f"未授权的 B 编号: {attachment}")
    paths = []
    for name in BFILES[attachment]:
        path = BROOT / name
        paths.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    for path in paths:
        if not path.resolve(strict=True).is_relative_to(BROOT.resolve(strict=True)):
            raise ValueError("B 路径越界")
    return paths


def load_b(attachment: str) -> pd.DataFrame:
    frames = []
    for path in paths_for(attachment):
        frame = pd.read_csv(path)
        frame["_source_file"] = path.relative_to(ROOT).as_posix()
        frame["_source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def stage_dir(stage: str) -> Path:
    return OUT / STAGES[stage]


def require_pass(stage: str) -> dict[str, Any]:
    path = stage_dir(stage) / f"{stage.lower()}_metadata.json"
    if not path.exists():
        raise RuntimeError(f"前置阶段 {stage} 未完成")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["status"] not in ("PASS", "PASS_WITH_SCENARIO_LIMITATION"):
        raise RuntimeError(f"前置阶段 {stage} 未通过")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run_tests(stage: str) -> str:
    test = ROOT / "tests/q2_v5" / f"test_{stage.lower()}.py"
    if not test.exists():
        raise RuntimeError(f"阶段测试不存在: {test}")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(test)],
                          cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    if proc.returncode:
        raise RuntimeError(f"{stage} 测试失败:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def finalize(stage: str, inputs: list[Path], warnings: list[str],
             assumptions: list[str], evidence: list[str],
             *, status: str = "PASS") -> dict[str, Any]:
    if stage != "P01":
        require_pass(f"P{int(stage[1:])-1:02d}")
    out = stage_dir(stage)
    result = run_tests(stage)
    later_stage_names = {
        "P09": {"p10_metadata.json", "p10_report.md", "q_n_substitution.csv",
                "q_d_substitution.csv", "substitution_nonlinearity.csv"},
        "P12": {"p13_metadata.json", "shadow_comparison.csv",
                "historical_difference_report.md"},
    }
    excluded = later_stage_names.get(stage, set()) | {f"{stage.lower()}_metadata.json"}
    files = [p for p in out.rglob("*") if p.is_file() and p.name not in excluded]
    p00 = json.loads((OUT / "00_manifest/manifest.json").read_text(encoding="utf-8"))
    metadata = {
        "stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": p00["git"]["commit"], "config_sha256": sha(CONFIG),
        "config_snapshot": config(), "seed": config()["seed"],
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "output_sha256": {p.relative_to(out).as_posix(): sha(p) for p in files},
        "status": status, "warnings": warnings, "assumptions": assumptions,
        "evidence_type": evidence, "pytest": result,
    }
    write_json(out / f"{stage.lower()}_metadata.json", metadata)
    return metadata


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    from scipy.stats import pearsonr, spearmanr

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if not len(y):
        return {key: None for key in ("n", "rmse", "mae", "r2", "pearson", "spearman", "mean_bias", "median_bias")}
    err = pred - y
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean())**2)) if len(y) > 1 and np.var(y) > 0 else None,
        "pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0 else None,
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0 else None,
        "mean_bias": float(np.mean(err)), "median_bias": float(np.median(err)),
    }
