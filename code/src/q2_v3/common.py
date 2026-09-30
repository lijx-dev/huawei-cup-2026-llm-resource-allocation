"""路径、哈希、统计和阶段门控。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q2_v3"
BROOT = ROOT / "data/real_attachments/B_scaling_laws"
SEED = 7


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda v: float(v) if isinstance(v, np.number) else str(v)), encoding="utf-8")


def csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty and path.name == "exclusions.csv":
        frame = pd.DataFrame(columns=["source_file", "row_id_or_experiment_id", "reason", "evidence", "stage", "action"])
    frame.to_csv(path, index=False)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(BROOT / name)


def valid_ndl(df: pd.DataFrame, stage: str, source: str) -> tuple[pd.DataFrame, list[dict]]:
    cols = ["N_params_B", "D_tokens_B", "val_loss"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{source} 缺少关键字段：{sorted(missing)}")
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
    excluded = [{"source_file": source, "row_id_or_experiment_id": str(df.iloc[i].get("experiment_id", i)), "reason": "invalid_N_D_or_loss", "evidence": str(vals.iloc[i].to_dict()), "stage": stage, "action": "exclude"} for i in np.where(~ok)[0]]
    return df.loc[ok].copy(), excluded


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("n", "RMSE", "MAE", "R2", "Pearson", "Spearman", "mean_bias", "median_bias")}
    residual = pred - y
    return {"n": len(y), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": float(mean_absolute_error(y, pred)),
            "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
            "Pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(y) and np.std(pred) else np.nan,
            "Spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(y) and np.std(pred) else np.nan,
            "mean_bias": float(np.mean(residual)), "median_bias": float(np.median(residual))}


def finish(stage: int, status: str, inputs: list[Path], report: Path, warnings: list[str] | None = None) -> None:
    if status not in {"PASS", "PASS_WITH_WARNINGS", "BLOCKED"}:
        raise ValueError(status)
    output_files = [p for p in OUT.rglob("*") if p.is_file() and not p.name.endswith("metadata.json")]
    data = {"experiment_version": "q2_v3", "stage": stage, "status": status, "seed": SEED,
            "warnings": warnings or [], "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs if p.is_file()},
            "output_sha256": {str(p.relative_to(OUT)): sha(p) for p in output_files},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src/q2_v3").glob("*.py")}}
    dump(report.parent / ("audit_metadata.json" if stage == 0 else f"p{stage}_metadata.json"), data)
    print(f"Q2_V3-P{stage} STATUS: {status}", flush=True)
