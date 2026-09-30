"""严格按 index 连接经审计的配比和 Loss。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


PAIRS = {
    "train_1m": ("train_mixture_1m.csv", "train_pile_loss_1m.csv", "training"),
    "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv", "held_out"),
    "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv", "held_out"),
    "test_1b": ("test_mixture_1B.csv", "test_pile_loss_1B.csv", "held_out"),
    "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimated_reference"),
    "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimated_reference"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pair(root, name, mixture_fields, loss_fields, tolerance):
    if name not in PAIRS:
        raise ValueError(f"未知数据切分: {name}")
    a, b, role = PAIRS[name]
    base = Path(root) / "data/real_attachments/A_data_value/regmix_tables"
    pm, pl = base / a, base / b
    mix, loss = pd.read_csv(pm, dtype={"index": str}), pd.read_csv(pl, dtype={"index": str})
    if list(mix.columns) != ["index", *mixture_fields] or list(loss.columns) != ["index", *loss_fields]:
        raise ValueError(f"{name} 字段或顺序与 P0 模式不一致")
    if mix["index"].isna().any() or loss["index"].isna().any() or mix["index"].duplicated().any() or loss["index"].duplicated().any():
        raise ValueError(f"{name} index 缺失或重复")
    if set(mix["index"]) != set(loss["index"]):
        raise ValueError(f"{name} 配比与 Loss 的 index 不一一对应")
    frame = mix.merge(loss, on="index", validate="one_to_one", sort=False)
    x = frame[mixture_fields].to_numpy(dtype=float)
    y = frame[loss_fields].to_numpy(dtype=float)
    totals = x.sum(axis=1)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any():
        raise ValueError(f"{name} 存在缺失、非有限或负数")
    if (abs(totals - 1) > tolerance).any() or (totals <= 0).any():
        raise ValueError(f"{name} 配比行和越过舍入容限")
    normalized = x / totals[:, None]
    audit = {"split": name, "role": role, "rows": len(frame), "invalid": 0,
             "normalized_rows": int(np.count_nonzero(abs(totals - 1) > 1e-12)),
             "max_sum_error": float(max(abs(totals - 1))),
             "mixture_sha256": sha256(pm), "loss_sha256": sha256(pl),
             "mixture_file": str(pm.relative_to(root)), "loss_file": str(pl.relative_to(root))}
    return frame["index"].to_numpy(), normalized, y, audit


def fields(root):
    cfg = json.loads((Path(root) / "configs/q1/audit.json").read_text())
    return cfg["mixture_fields"], cfg["loss_fields"]
