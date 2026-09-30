"""读取并逐批核验 P1 已保存的标准化结果，不重新拟合 P1。"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from q1.audit.inventory import sha256_file


def load_p1_model(root, expected_version):
    base = root / "results/q1/quality"
    metadata_path = base / "quality_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model_version"] != expected_version:
        raise RuntimeError("P1 模型版本与 P2 配置不一致")
    for key, path in [("schema_sha256", root / "configs/q1/indicator_schema.json"),
                      ("config_sha256", root / "configs/q1/quality.json"),
                      ("design_sha256", root / "question1-1(1).md")]:
        if sha256_file(path) != metadata[key]:
            raise RuntimeError(f"P1 来源配置已变化：{path}")
    for relative, digest in metadata["code_sha256"].items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"P1 代码与模型元数据不一致：{relative}")
    with (base / "entropy_weights.csv").open(encoding="utf-8", newline="") as stream:
        weights_rows = list(csv.DictReader(stream))
    names = [row["name"] for row in weights_rows]
    weights = np.array([float(row["weight"]) for row in weights_rows], dtype=np.float64)
    if len(names) != 22 or len(set(names)) != 22 or np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-10):
        raise RuntimeError("P1 权重非法")
    with (base / "normalization_parameters.csv").open(encoding="utf-8", newline="") as stream:
        if [row["name"] for row in csv.DictReader(stream)] != names:
            raise RuntimeError("P1 归一化参数与权重列顺序不一致")
    with (base / "spearman_correlation.csv").open(encoding="utf-8", newline="") as stream:
        corr_rows = list(csv.DictReader(stream))
    if [row["name"] for row in corr_rows] != names:
        raise RuntimeError("P1 Spearman 矩阵与权重列顺序不一致")
    corr = np.array([[float(row[name]) for name in names] for row in corr_rows])
    if corr.shape != (22, 22) or not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, atol=1e-9) or not np.allclose(np.diag(corr), 1, atol=1e-9) or np.any(np.abs(corr) > 1+1e-9):
        raise RuntimeError("P1 Spearman 矩阵非法，不能聚类")
    paths = {name:base / name for name in ("quality_model_metadata.json", "entropy_weights.csv", "normalization_parameters.csv",
                                         "spearman_correlation.csv", "sample_quality_scores.csv.gz", "extended_quality_scores.csv.gz")}
    return {"names":names,"weights":weights,"correlation":corr,"metadata":metadata,
            "input_sha256":{name:sha256_file(path) for name,path in paths.items()},"paths":paths}


def iter_p1_batches(path, names, weights, model_version, chunk_size, attachments):
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["z_" + name for name in names]
        if [column for column in reader.fieldnames if column.startswith("z_")] != expected:
            raise RuntimeError(f"P1 标准化矩阵的列或顺序不一致：{path}")
        required = {"record_id","source_file","line_number","record_hash","attachment","domain","audit_status",
                    "model_version","union_membership","Q",*expected}
        if not required.issubset(reader.fieldnames):
            raise RuntimeError(f"P1 样本视图缺少必要列：{path}")
        rows = []
        for row in reader:
            if row["model_version"] != model_version or row["attachment"] not in attachments:
                raise RuntimeError("P1 样本模型版本或来源不一致")
            if row["attachment"] == "A1" and row["audit_status"] != "valid":
                raise RuntimeError("A1 正式视图含无效状态")
            if row["attachment"] in {"A2","A3"} and row["audit_status"] not in {"valid","duplicate_exact"}:
                raise RuntimeError("扩展视图含无效状态")
            if row["union_membership"] != str(row["audit_status"]=="valid"):
                raise RuntimeError("P1 联合去重标记与审计状态不一致")
            rows.append(row)
            if len(rows) == chunk_size:
                yield _checked_batch(rows,expected,weights)
                rows = []
        if rows:
            yield _checked_batch(rows,expected,weights)


def _checked_batch(rows, columns, weights):
    z = np.array([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    q = np.array([float(row["Q"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z > 1) or not np.all(np.isfinite(q)):
        raise RuntimeError("P1 标准化指标或质量分存在非法值")
    if not np.allclose(q, 100*(z @ weights), rtol=0, atol=1e-8):
        raise RuntimeError("P1 样本质量分与固定权重不一致")
    return rows,z,q
