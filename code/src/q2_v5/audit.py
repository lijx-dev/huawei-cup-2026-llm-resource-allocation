"""P01：B1–B12 数据合同、重叠和来源口径审计。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import (BFILES, EVIDENCE, OUT, ROOT, finalize, load_b,
                     paths_for, sha, stage_dir, write_json)


def overlap_keys(b6: pd.DataFrame, b7: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["experiment_id", "N_params_B", "D_tokens_B", "Q_score"]
    for name, frame in (("B6", b6), ("B7", b7)):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name} identifying fields 缺失或重复")
    joined = b7.merge(b6[keys + ["val_loss"]], how="left", on=keys,
                      indicator=True, suffixes=("", "_b6"), validate="one_to_one")
    overlap = joined.loc[joined["_merge"] == "both"].copy()
    fresh = joined.loc[joined["_merge"] == "left_only", b7.columns].copy()
    if not np.allclose(overlap["val_loss"], overlap["val_loss_b6"], rtol=0, atol=1e-12):
        raise ValueError("B6/B7 同一实验的 Loss 冲突")
    return overlap, fresh


def invalid_numeric(frame: pd.DataFrame, attachment: str) -> pd.DataFrame:
    rows = []
    rules = {"N_params_B": "positive", "D_tokens_B": "positive",
             "Q_score": "finite", "val_loss": "positive"}
    for column, rule in rules.items():
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = ~np.isfinite(numeric) | ((numeric <= 0) if rule == "positive" else False)
        for index in np.flatnonzero(bad):
            rows.append({"attachment": attachment, "source_file": frame.at[index, "_source_file"],
                         "source_row": int(frame.at[index, "_source_row"]),
                         "column": column, "reason": f"invalid_{rule}"})
    return pd.DataFrame(rows, columns=["attachment", "source_file", "source_row", "column", "reason"])


def direction_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, d), group in frame.groupby(["N_params_B", "D_tokens_B"], dropna=False):
        x, y = group["Q_score"].to_numpy(float), group["val_loss"].to_numpy(float)
        rho = float(spearmanr(x, y).statistic) if len(group) >= 3 and len(set(x)) > 1 else np.nan
        rows.append({"N_params_B": n, "D_tokens_B": d, "n": len(group),
                     "spearman_q_loss": rho,
                     "direction": "negative" if rho < 0 else "positive" if rho > 0 else "undetermined",
                     "data_type": ",".join(sorted(group["data_type"].dropna().astype(str).unique()))})
    return pd.DataFrame(rows)


def run() -> dict:
    out = stage_dir("P01")
    out.mkdir(parents=True, exist_ok=True)
    inventory, fields, invalid, duplicates, levels, semantics = [], [], [], [], [], []
    frames: dict[str, pd.DataFrame] = {}
    all_paths: list[Path] = []
    for attachment in BFILES:
        paths = paths_for(attachment)
        all_paths.extend(paths)
        frame = load_b(attachment)
        frames[attachment] = frame
        invalid.append(invalid_numeric(frame, attachment))
        keys = [c for c in frame if not c.startswith("_source_")]
        for index in np.flatnonzero(frame.duplicated(keys, keep=False)):
            duplicates.append({"attachment": attachment,
                               "source_file": frame.at[index, "_source_file"],
                               "source_row": int(frame.at[index, "_source_row"]),
                               "reason": "exact_row_duplicate"})
        for path in paths:
            subset = frame.loc[frame["_source_file"] == path.relative_to(ROOT).as_posix()]
            inventory.append({"attachment": attachment, "path": path.relative_to(ROOT).as_posix(),
                              "size_bytes": path.stat().st_size, "sha256": sha(path),
                              "rows": len(subset), "columns": len(keys),
                              "evidence_type": EVIDENCE[attachment],
                              "N_support": "N_params_B" in frame,
                              "D_support": "D_tokens_B" in frame,
                              "Q_support": "Q_score" in frame,
                              "Loss_support": "val_loss" in frame,
                              "N_unit": "billion parameters" if "N_params_B" in frame else "absent",
                              "D_unit": "billion tokens" if "D_tokens_B" in frame else "absent",
                              "Loss_semantics": "cross-source; compare conditionally" if attachment in ("B2", "B4", "B5") else
                                                "estimated reference" if attachment == "B10" else
                                                "validation cross-entropy" if "val_loss" in frame else "absent"})
        for column in keys:
            numeric = pd.api.types.is_numeric_dtype(frame[column])
            fields.append({"attachment": attachment, "column": column,
                           "dtype": str(frame[column].dtype), "missing": int(frame[column].isna().sum()),
                           "min": float(frame[column].min()) if numeric and frame[column].notna().any() else None,
                           "max": float(frame[column].max()) if numeric and frame[column].notna().any() else None})
        levels.append({"attachment": attachment, "evidence_type": EVIDENCE[attachment],
                       "modelling_role": {
                           "B1": "baseline_fit", "B2": "trend_validation", "B3": "interpolation_consistency",
                           "B4": "cross_family_validation", "B5": "literature_cross_source",
                           "B6": "quality_selection", "B7": "locked_validation", "B8": "stress_test",
                           "B9": "support_only", "B10": "estimated_consistency",
                           "B11": "checkpoint_metadata", "B12": "family_metadata"}[attachment]})
        semantics.append({"attachment": attachment, "has_loss": "val_loss" in frame,
                          "source_semantics": EVIDENCE[attachment],
                          "strictly_comparable_to_B1": attachment == "B1",
                          "note": "Loss 口径跨来源需条件解释" if attachment in ("B2", "B4", "B5") else ""})
    b6, b7 = frames["B6"], frames["B7"]
    overlap, fresh = overlap_keys(b6, b7)
    if len(b6) != 360 or len(b7) != 450 or len(overlap) != 360 or len(fresh) != 90:
        raise RuntimeError(f"CRITICAL: B6/B7 重叠不符: {len(b6)}/{len(b7)}/{len(overlap)}/{len(fresh)}")
    pd.DataFrame(inventory).to_csv(out / "attachment_inventory.csv", index=False)
    # JSON 语法是 YAML 的子集。
    (out / "field_contract.yaml").write_text(json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.concat(invalid, ignore_index=True).to_csv(out / "invalid_rows.csv", index=False)
    pd.DataFrame(duplicates, columns=["attachment", "source_file", "source_row", "reason"]).to_csv(out / "duplicate_rows.csv", index=False)
    overlap[["experiment_id", "N_params_B", "D_tokens_B", "Q_score", "val_loss", "_source_row"]].to_csv(out / "b6_b7_overlap.csv", index=False)
    fresh.to_csv(out / "b7_new.csv", index=False)
    directions = direction_by_group(frames["B8"])
    directions.to_csv(out / "b8_quality_direction.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "evidence_level.csv", index=False)
    pd.DataFrame(semantics).to_csv(out / "loss_semantics.csv", index=False)
    summary = {"B6_rows": len(b6), "B7_rows": len(b7), "overlap_rows": len(overlap),
               "B7_new_rows": len(fresh), "B8_positive_groups": int((directions.direction == "positive").sum()),
               "B8_negative_groups": int((directions.direction == "negative").sum()),
               "invalid_numeric_rows": int(sum(map(len, invalid))), "exact_duplicate_rows": len(duplicates)}
    write_json(out / "audit_summary.json", summary)
    (out / "audit_report.md").write_text(
        "# P01 B 数据审计\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items())
        + "\n\nB3 为插值轨迹；B6/B7/B8 为半合成；B10 为估算参考；B9 无 Loss。"
          "跨来源 Loss 严格可比性尚无充分证据。\n", encoding="utf-8")
    metadata = finalize("P01", all_paths, [],
                        ["N、D 文件单位分别为十亿参数和十亿 token；以数据说明为依据"],
                        sorted(set(EVIDENCE.values())))
    return {"summary": summary, "metadata": metadata}
