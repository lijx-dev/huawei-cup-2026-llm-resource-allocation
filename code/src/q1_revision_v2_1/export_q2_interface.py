"""导出可审计的 Q1→Q2 冻结接口。

该模块只读取第一问已冻结模型、第一问结果和 A4--A15 原始附件；不重新拟合模型。
所有配比使用与第一问训练一致的逐行单纯形归一化，并始终按 ``index`` 连接。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/q1_revision_v2_1/q2_interface"
MODEL_PATH = ROOT / "results/q1_revision_v2/mixture/fitted_models.joblib"
SUPPORT_PATH = ROOT / "results/q1_revision_v2/mixture/support_reference.json"
QUALITY_PATH = ROOT / "results/q1_revision_v2_1/quality/domain_quality_summary.csv"
MAPPING_PATH = ROOT / "results/q1_revision_v2_1/q_mapping/reliable_mapping_used.csv"
REGMIX = ROOT / "data/real_attachments/A_data_value/regmix_tables"

SPLITS = {
    "train_1m": ("train_mixture_1m.csv", "train_pile_loss_1m.csv", "train_observation"),
    "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv", "held_out_observation"),
    "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv", "cross_scale_observation"),
    "test_1b": ("test_mixture_1B.csv", "test_pile_loss_1B.csv", "cross_scale_observation"),
    "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimated_reference"),
    "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimated_reference"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _predict(bundle: dict, x: np.ndarray) -> np.ndarray:
    models = bundle["models"][bundle["selection"]["main_model"]]
    prediction = np.column_stack([model.predict(x) for model in models])
    if prediction.shape != (len(x), 13) or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise RuntimeError("冻结模型产生非法预测")
    return prediction


def export(out: Path = DEFAULT_OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    mix_fields = list(bundle["mix_fields"])
    loss_fields = list(bundle["loss_fields"])
    if len(mix_fields) != 17 or len(loss_fields) != 13:
        raise RuntimeError("冻结模型输入/输出维度不是 17/13")

    support = float(json.loads(SUPPORT_PATH.read_text())["nearest_neighbor_95pct_distance"])
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (MODEL_PATH, SUPPORT_PATH, QUALITY_PATH, MAPPING_PATH)}
    frames: list[pd.DataFrame] = []
    normalized_train = None

    for split, (mixture_name, loss_name, role) in SPLITS.items():
        mixture_path, loss_path = REGMIX / mixture_name, REGMIX / loss_name
        source_hashes[str(mixture_path.relative_to(ROOT))] = sha256(mixture_path)
        source_hashes[str(loss_path.relative_to(ROOT))] = sha256(loss_path)
        mixture, loss = pd.read_csv(mixture_path), pd.read_csv(loss_path)
        if not mixture["index"].is_unique or not loss["index"].is_unique:
            raise RuntimeError(f"{split}: index 不是一一键")
        if set(mixture["index"]) != set(loss["index"]):
            raise RuntimeError(f"{split}: 配比与 Loss 的 index 集不一致")
        if [c for c in mixture.columns if c != "index"] != mix_fields:
            raise RuntimeError(f"{split}: 17 维输入顺序不一致")
        if [c for c in loss.columns if c != "index"] != loss_fields:
            raise RuntimeError(f"{split}: 13 维输出顺序不一致")
        joined = mixture.merge(loss, on="index", how="inner", validate="one_to_one", sort=False)
        raw = joined[mix_fields].to_numpy(float)
        row_sum = raw.sum(axis=1)
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(row_sum <= 0):
            raise RuntimeError(f"{split}: 配比包含非法值")
        x = raw / row_sum[:, None]
        if split == "train_1m":
            normalized_train = x.copy()
        prediction = _predict(bundle, x)
        frame = pd.DataFrame({"split": split, "data_role": role, "index": joined["index"], "raw_mixture_sum": row_sum})
        frame[mix_fields] = x
        frame[loss_fields] = joined[loss_fields].to_numpy(float)
        for i, field in enumerate(loss_fields):
            frame[f"predicted::{field}"] = prediction[:, i]
        frames.append(frame)

    if normalized_train is None:
        raise RuntimeError("缺少 train_1m")
    p0 = normalized_train.mean(axis=0)
    p0 /= p0.sum()
    y0 = _predict(bundle, p0[None, :])[0]

    combined = pd.concat(frames, ignore_index=True)
    all_x = combined[mix_fields].to_numpy(float)
    all_prediction = combined[[f"predicted::{field}" for field in loss_fields]].to_numpy(float)
    nearest = np.sqrt(((all_x[:, None, :] - normalized_train[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    combined["nearest_train_distance"] = nearest
    combined["support_status"] = np.where(nearest <= support, "within_support", "low_confidence_extrapolation")
    h = np.log(all_prediction / y0)
    for i, field in enumerate(loss_fields):
        combined[f"h::{field}"] = h[:, i]
    combined["h_agg"] = h.mean(axis=1)
    mixture_output = out / "mixture_response.csv.gz"
    # 固定 gzip mtime，保证相同输入生成相同接口哈希。
    combined.to_csv(mixture_output, index=False, compression={"method": "gzip", "mtime": 0})

    reference = {
        "fields": mix_fields,
        "values": p0.tolist(),
        "normalized_reference_sum": float(p0.sum()),
        "method": "mean of normalized train_1m recipes, then normalized to the simplex",
        "prediction_fields": loss_fields,
        "reference_prediction": y0.tolist(),
        "support_distance_threshold": support,
    }
    _write_json(out / "reference_mixture.json", reference)

    quality = pd.read_csv(QUALITY_PATH)
    quality = quality.loc[quality["view"].eq("union_unique")].copy()
    quality.to_csv(out / "quality_domain_interface.csv", index=False)
    mapping = pd.read_csv(MAPPING_PATH)
    mapping.to_csv(out / "quality_mapping_interface.csv", index=False)

    output_files = [mixture_output, out / "reference_mixture.json", out / "quality_domain_interface.csv", out / "quality_mapping_interface.csv"]
    checks = {
        "model_is_declared_lightgbm": bundle["selection"]["main_model"] == "LightGBM",
        "input_count_17": len(mix_fields) == 17,
        "output_count_13": len(loss_fields) == 13,
        "all_index_joins_one_to_one": True,
        "all_normalized_rows_on_simplex": bool(np.allclose(all_x.sum(axis=1), 1.0, atol=1e-12)),
        "all_predictions_positive_finite": bool(np.isfinite(all_prediction).all() and np.all(all_prediction > 0)),
        "reference_h_is_zero": bool(np.max(np.abs(np.log(_predict(bundle, p0[None, :])[0] / y0))) <= 1e-12),
        "quality_mapping_only_reliable_types": set(mapping["mapping_type"]) <= {"direct", "near_direct"},
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"
    manifest = {
        "interface_version": "q1-to-q2-v1",
        "status": status,
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH), "name": bundle["selection"]["main_model"]},
        "mixture_response_path": str(mixture_output.relative_to(ROOT)),
        "reference_mixture_path": str((out / "reference_mixture.json").relative_to(ROOT)),
        "quality_domain_path": str((out / "quality_domain_interface.csv").relative_to(ROOT)),
        "quality_mapping_path": str((out / "quality_mapping_interface.csv").relative_to(ROOT)),
        "input_order": mix_fields,
        "output_order": loss_fields,
        "split_rows": combined.groupby("split").size().astype(int).to_dict(),
        "source_sha256": source_hashes,
        "product_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in output_files},
        "support_rule": {"metric": "Euclidean distance on normalized 17-simplex", "threshold": support},
        "quality_policy": "Q_A and Q_B remain distinct; no empirical conversion is claimed",
        "mapping_policy": "only direct and near_direct mappings are exported; inferred mappings are excluded",
        "checks": checks,
    }
    _write_json(out / "manifest.json", manifest)
    report = [
        "# Q1→Q2 正式接口校验",
        "",
        f"冻结模型：{manifest['model']['name']}；SHA-256 `{manifest['model']['sha256']}`。",
        f"接口包含 {len(combined)} 个配方、17 维归一化输入和 13 维观测/预测 Loss；所有配比与 Loss 均按 `index` 一一连接。",
        "Q_A 与 Q_B 不作数值转换；质量域只输出 direct/near_direct 六条可靠映射。",
        "10B/70B 的 Loss 保持 `estimated_reference` 身份，不作为真实观测。",
        "",
        f"FINAL_STATUS = {status}",
        "",
    ]
    (out / "interface_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = export()
    print(f"Q1_TO_Q2_INTERFACE_STATUS: {result['status']}")
