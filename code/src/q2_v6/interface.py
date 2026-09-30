"""冻结 M2 接口的数据合同；不读取附件 A 原始表。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_interface(csv_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256(csv_path)
    if actual_hash != manifest["interface_sha256"]:
        raise ValueError("Q1 M2 响应接口 SHA-256 与 manifest 不一致")
    coords = manifest["coordinate_space"]
    targets = manifest["loss_targets"]
    if len(coords) != 13 or len(set(coords)) != 13 or len(targets) != 13 or len(set(targets)) != 13:
        raise ValueError("接口坐标或目标顺序无效")
    expected = ["split", "recipe_id"] + [f"p_{k}" for k in coords] + [f"h_{t}" for t in targets] + ["h_p_eq", "hull13", "hull14", "hull17"]
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != expected or len(frame) != manifest["n_rows"]:
        raise ValueError("接口字段顺序或行数与 manifest 不一致")
    numeric = expected[2:]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("接口含非有限数值")
    if frame.duplicated(["split", "recipe_id"]).any():
        raise ValueError("接口存在重复配方键")
    counts = frame.groupby("split").size().to_dict()
    required_counts = {"A6-A7 (1M)": 256, "A8-A9 (60M)": 256, "A10-A11 (1B)": 64, "p0 (A4 mean)": 1}
    if counts != required_counts:
        raise ValueError(f"接口分层行数异常: {counts}")
    pcols = [f"p_{k}" for k in coords]
    hcols = [f"h_{t}" for t in targets]
    p = frame[pcols].to_numpy(dtype=float)
    h = frame[hcols].to_numpy(dtype=float)
    if (p < -1e-10).any() or (p.sum(axis=1) > 1.01).any():
        raise ValueError("M2 坐标超出允许范围")
    if not np.allclose(h.mean(axis=1), frame.h_p_eq.to_numpy(float), atol=1e-8, rtol=0):
        raise ValueError("h_p_eq 与 13 目标等权平均不一致")
    for name in ("hull13", "hull14", "hull17"):
        if not frame[name].isin((0, 1)).all():
            raise ValueError(f"{name} 应为 0/1")
    if ((frame.hull17 > frame.hull14) | (frame.hull14 > frame.hull13)).any():
        raise ValueError("凸包包含关系不成立")
    ref = frame.loc[frame.split == "p0 (A4 mean)"].iloc[0]
    if int(ref.recipe_id) != -1 or not np.allclose(ref[pcols].to_numpy(float), manifest["p0_13coords"], atol=1e-9, rtol=0):
        raise ValueError("p0 坐标与 manifest 不一致")
    if np.max(np.abs(ref[hcols].to_numpy(float))) > 1e-12 or abs(float(ref.h_p_eq)) > 1e-12:
        raise ValueError("p0 相对响应不是零")
    if not all(bool(v) for k, v in manifest["self_checks"].items() if "通过" in k):
        raise ValueError("manifest 自检未全部通过")
    hull_counts = {split: {name: int(group[name].sum()) for name in ("hull13", "hull14", "hull17")}
                   for split, group in frame.groupby("split") if split != "p0 (A4 mean)"}
    for split, claim in manifest.get("hull_counts", {}).items():
        if split not in hull_counts:
            raise ValueError("manifest 含未知凸包分层")
        parsed = re.findall(r"(?:13|14|17) 维(?:原始)?\s*(\d+)/(\d+)", claim)
        if len(parsed) != 3 or any(int(hit) != hull_counts[split][name] or int(total) != counts[split]
                                   for (hit, total), name in zip(parsed, ("hull13", "hull14", "hull17"))):
            raise ValueError("manifest 声称的凸包计数与响应表不一致")
    first = frame.loc[frame.split == "A6-A7 (1M)"].sort_values("recipe_id")
    second = frame.loc[frame.split == "A8-A9 (60M)"].sort_values("recipe_id")
    paired_equal = bool(np.array_equal(first.recipe_id.to_numpy(), second.recipe_id.to_numpy()) and
                        np.allclose(first[pcols + hcols].to_numpy(float), second[pcols + hcols].to_numpy(float), atol=1e-10, rtol=0))
    audit = {"status": "PASS_WITH_PROVENANCE_LIMITATION", "interface_sha256": actual_hash,
             "manifest_sha256": sha256(manifest_path), "source_coefficients_sha256_claimed": manifest["source_coefficients_sha256"],
             "source_coefficients_verified": False, "rows": len(frame), "split_counts": counts,
             "hull_counts": hull_counts, "one_m_sixty_m_same_recipe_and_m2_response": paired_equal,
             "target_count": len(targets), "coordinate_count": len(coords),
             "coordinate_sum_min": float(p.sum(axis=1).min()),
             "coordinate_sum_max": float(p.sum(axis=1).max()),
             "lineage": "M2 13-coordinate interface; distinct from Q1 v2.1 inherited 17-domain LightGBM"}
    return frame, manifest, audit
