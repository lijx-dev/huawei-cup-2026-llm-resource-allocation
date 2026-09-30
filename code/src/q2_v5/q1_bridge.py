"""P07：只从已保存 Q1 产物建立冻结配比与质量桥。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, config, finalize, load_b, require_pass, sha, stage_dir, write_json


Q1_V2 = ROOT / "results/q1_revision_v2"
Q1_V21 = ROOT / "results/q1_revision_v2_1"
Q1_FILES = {
    "fitted_models.joblib": Q1_V2 / "mixture/fitted_models.joblib",
    "model_selection.json": Q1_V2 / "mixture/model_selection.json",
    "mixture_summary.json": Q1_V2 / "mixture/mixture_summary.json",
    "support_reference.json": Q1_V2 / "mixture/support_reference.json",
    "inherited_v2_model_reference.json": Q1_V21 / "mixture/inherited_v2_model_reference.json",
    "domain_quality_summary.csv": Q1_V21 / "quality/domain_quality_summary.csv",
    "mapped_domain_quality.csv": Q1_V21 / "q_mapping/mapped_domain_quality.csv",
    "mapping_used.csv": Q1_V2 / "integration/mapping_used.csv",
    "predictions_test_1m.csv": Q1_V2 / "mixture/predictions_test_1m.csv",
}


def frozen_dir() -> Path:
    return stage_dir("P07") / "frozen_q1_interface"


def load_frozen_model() -> dict:
    path = frozen_dir() / "fitted_models.joblib"
    hashes = json.loads((frozen_dir() / "frozen_hashes.json").read_text())
    if sha(path) != hashes["fitted_models.joblib"]["sha256"]:
        raise RuntimeError("Q1 frozen model hash mismatch")
    artifact = joblib.load(path)
    if artifact["selection"]["main_model"] != "LightGBM":
        raise RuntimeError("Q1 主模型选择与冻结接口不一致")
    if len(artifact["mix_fields"]) != 17 or len(artifact["loss_fields"]) != 13:
        raise RuntimeError("Q1 冻结模型维度不符")
    return artifact


def predict_mixture(p: np.ndarray, artifact: dict | None = None) -> np.ndarray:
    artifact = artifact or load_frozen_model()
    p = np.atleast_2d(np.asarray(p, float))
    if p.shape[1] != 17 or np.any(p < -1e-12) or not np.allclose(p.sum(axis=1), 1, atol=1e-10):
        raise ValueError("配比必须处于 17 维 simplex")
    models = artifact["models"]["LightGBM"]
    result = np.column_stack([model.predict(p) for model in models])
    if result.shape[1] != 13 or not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("Q1 冻结模型预测 Loss 必须有限且严格为正")
    return result


def hp(p: np.ndarray, p0: np.ndarray, artifact: dict | None = None) -> np.ndarray:
    artifact = artifact or load_frozen_model()
    baseline = predict_mixture(p0, artifact)[0]
    return np.log(predict_mixture(p, artifact) / baseline)


def mapping_scenarios(qa: pd.DataFrame, qb: pd.Series) -> pd.DataFrame:
    qa_min, qa_max = float(qa.Q_A.min()), float(qa.Q_A.max())
    qb_min, qb_max = float(qb.min()), float(qb.max())
    qa_percentiles = (qa.Q_A.rank(method="average") - 1) / max(len(qa) - 1, 1)
    rows = []
    for index, row in qa.iterrows():
        q = float(row.Q_A)
        common = q / 100.0  # 共同 0–1 归一尺度上的恒等情景。
        affine = qb_min + (q - qa_min) / (qa_max - qa_min) * (qb_max - qb_min)
        percentile = float(np.quantile(qb, qa_percentiles.loc[index]))
        for name, value in (("common_normalized_identity", common),
                            ("support_affine", affine),
                            ("rank_percentile", percentile)):
            rows.append({"domain": row.domain, "Q_A": q, "mapping": name,
                         "Q_B_scenario": value, "evidence_type": "scenario_assumption"})
    return pd.DataFrame(rows)


def run() -> dict:
    require_pass("P06")
    out = stage_dir("P07")
    frozen = frozen_dir()
    frozen.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in Q1_FILES.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Q1 frozen interface 必要文件缺失: {missing}")
    copied = {}
    for name, source in Q1_FILES.items():
        destination = frozen / name
        shutil.copy2(source, destination)
        copied[name] = {"source": source.relative_to(ROOT).as_posix(),
                        "sha256": sha(destination), "size_bytes": destination.stat().st_size}
    write_json(frozen / "frozen_hashes.json", copied)
    artifact = load_frozen_model()
    source_reference = json.loads((frozen / "inherited_v2_model_reference.json").read_text())
    if source_reference["model_file_sha256"] != copied["fitted_models.joblib"]["sha256"]:
        raise RuntimeError("Q1 模型文件哈希与继承记录不一致")
    # 已保存结果没有数值参考配比；等权点是明确的 Q2 情景假设。
    p0 = np.full(17, 1 / 17)
    write_json(frozen / "reference_mixture.json", {
        "values": p0.tolist(), "fields": artifact["mix_fields"],
        "source": "Q2 scenario: equal-share simplex point; Q1 output has no saved numeric p0",
        "evidence_type": "scenario_assumption", "support_status": "unverifiable_without_saved_training_recipes"})
    points = [p0]
    labels = ["p0"]
    for j in range(17):
        for k in range(17):
            if j == k:
                continue
            p = p0.copy()
            p[j] += .01
            p[k] -= .01
            points.append(p)
            labels.append(f"{j}<-{k}@0.01")
    values = hp(np.vstack(points), p0, artifact)
    target_rows = []
    for label, vector in zip(labels, values):
        for target, value in zip(artifact["loss_fields"], vector):
            target_rows.append({"scenario": label, "target": target, "h_p_v": float(value),
                                "evidence_type": "q1_imported", "reference_evidence": "scenario_assumption"})
    pd.DataFrame(target_rows).to_csv(out / "hp_by_target.csv", index=False)
    pd.DataFrame({"scenario": labels, "h_p_aggregate_equal_weight": values.mean(axis=1),
                  "weight_rule": "13 targets equal weight", "evidence_type": "q1_imported"}).to_csv(
                      out / "hp_aggregate.csv", index=False)
    domain = pd.read_csv(frozen / "domain_quality_summary.csv")
    domain = domain.loc[domain.view == "union_unique", ["domain", "Q_hierarchical_balanced"]]
    domain = domain.rename(columns={"Q_hierarchical_balanced": "Q_A"})
    qb = pd.concat([load_b("B6").Q_score, pd.read_csv(stage_dir("P01") / "b7_new.csv").Q_score])
    mappings = mapping_scenarios(domain, qb)
    mappings.to_csv(out / "qa_qb_mapping_scenarios.csv", index=False)
    structure = {"form_A_identified": False, "form_B_identified": False,
                 "status": "NOT_IDENTIFIED_FROM_FROZEN_Q1_OUTPUT",
                 "primary_structure": config()["mixture_primary_form_if_unidentified"],
                 "other_structure": "B", "reason": "Q1 输出没有同一配比在多个 N,D 上的联合观测"}
    write_json(out / "form_structure_diagnostic.json", structure)
    audit = {"primary_q1_version": "q1-revision-v2.1 quality/mapping; inherited v2 frozen LightGBM",
             "selected_mixture_model": "LightGBM", "model_hash_anchored_in_v2_1_reference": True,
             "reference_mixture_saved_in_q1": False,
             "test_predictions_saved": True, "quality_domains": len(domain),
             "mapping_domains": int(pd.read_csv(frozen / "mapped_domain_quality.csv").shape[0]),
             "form_identification": structure["status"]}
    write_json(out / "bridge_audit.json", audit)
    (out / "report.md").write_text(
        "# P07 冻结 Q1 桥\n\n"
        "v2.1 继承 v2 冻结 LightGBM；已复制最小产物并记录 SHA-256。"
        "Q1 未保存数值 p0，因此等权 p0 是 Q2 情景假设，支持距离不能核实。"
        "Q_A 与 Q_B 无配对观测；三种映射仅为情景。"
        "Form A/B 从冻结 Q1 输出不可识别，预注册 Form A 主情景和 Form B 敏感性。\n",
        encoding="utf-8")
    metadata = finalize("P07", list(Q1_FILES.values()) + [stage_dir("P01") / "b7_new.csv"],
                        ["Q1 无已保存数值 p0；等权参考配比为情景假设",
                         "Q1 模型哈希在 v2.1 引用中验证，但原 v2 metadata 未锚定",
                         "Form A/B 未识别"],
                        ["QA/QB 映射无配对观测，三种方案均非数据识别"],
                        ["q1_imported", "scenario_assumption"], status="PASS_WITH_SCENARIO_LIMITATION")
    return {"audit": audit, "metadata": metadata}
