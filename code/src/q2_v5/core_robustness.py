"""P12：分类稳健性与核心结果冻结。"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .common import OUT, ROOT, STAGES, finalize, require_pass, sha, stage_dir, write_json


def core_digest(out) -> str:
    digest = hashlib.sha256()
    for name in ("core_model_contract.json", "core_results_manifest.json"):
        digest.update(name.encode())
        digest.update((out / name).read_bytes())
    return digest.hexdigest()


def verify_core_freeze() -> dict:
    out = stage_dir("P12")
    saved = (out / "core_freeze.sha256").read_text().strip()
    if saved != core_digest(out):
        raise RuntimeError("核心冻结摘要损坏")
    manifest = json.loads((out / "core_results_manifest.json").read_text())
    for relative, digest in manifest["output_sha256"].items():
        if sha(OUT / relative) != digest:
            raise RuntimeError(f"冻结后核心文件发生变化: {relative}")
    return manifest


def run() -> dict:
    require_pass("P11")
    out = stage_dir("P12")
    out.mkdir(parents=True, exist_ok=True)
    b1_multi = pd.read_csv(stage_dir("P02") / "multistart_runs.csv")
    b1_boot = pd.read_csv(stage_dir("P02") / "bootstrap_parameters.csv")
    b1_macro = json.loads((stage_dir("P02") / "macro_metrics.json").read_text())
    b1_pool = json.loads((stage_dir("P02") / "pooled_metrics.json").read_text())
    quality_boot = pd.read_csv(stage_dir("P04") / "parameter_bootstrap.csv")
    quality_corr = pd.read_csv(stage_dir("P04") / "parameter_correlation.csv")
    quality_profile = pd.read_csv(stage_dir("P04") / "parameter_profile.csv")
    quality_diff = pd.read_csv(stage_dir("P04") / "bootstrap_model_differences.csv")
    anchored = json.loads((stage_dir("P04") / "anchored_sensitivity.json").read_text())
    b7_boot = pd.read_csv(stage_dir("P05") / "bootstrap_validation.csv")
    p08 = pd.read_csv(stage_dir("P08") / "scenario_predictions.csv")
    support = pd.read_csv(stage_dir("P06") / "b9_support_distance.csv")
    delta = pd.read_csv(stage_dir("P11") / "delta_sensitivity.csv")
    matrix = [
        {"category": "statistical", "analysis": "B1 multistart convergence", "value": float(b1_multi.success.mean())},
        {"category": "statistical", "analysis": "B1 group bootstrap convergence", "value": float(b1_boot.success.mean())},
        {"category": "statistical", "analysis": "B1 macro RMSE", "value": b1_macro["rmse_macro"]},
        {"category": "statistical", "analysis": "B1 pooled RMSE", "value": b1_pool["rmse"]},
        {"category": "statistical", "analysis": "quality group bootstrap convergence", "value": float(quality_boot.success.mean())},
        {"category": "statistical", "analysis": "quality largest absolute parameter correlation", "value": float(quality_corr.select_dtypes("number").abs().where(lambda x: x < .999999).max().max())},
        {"category": "statistical", "analysis": "quality profile rows", "value": float(len(quality_profile))},
        {"category": "structural", "analysis": "B1 anchored quality RSS", "value": anchored["rss"]},
        {"category": "statistical", "analysis": "B7-new RMSE difference q025", "value": float(b7_boot.rmse_difference.quantile(.025))},
        {"category": "statistical", "analysis": "B7-new RMSE difference q975", "value": float(b7_boot.rmse_difference.quantile(.975))},
        {"category": "cross_source", "analysis": "QA-QB mapping variants", "value": float(p08.mapping.nunique())},
        {"category": "scenario", "analysis": "lambda variants", "value": float(p08.lambda_p.nunique())},
        {"category": "structural", "analysis": "Form variants", "value": float(p08.form.nunique())},
        {"category": "structural", "analysis": "B9 median log ND support distance", "value": float(support.log_nd_nearest_distance.median())},
        {"category": "scenario", "analysis": "pair transfer delta variants", "value": float(delta.delta.nunique())},
    ]
    pd.DataFrame(matrix).to_csv(out / "robustness_matrix.csv", index=False)
    spread = p08.groupby(["domain", "mapping", "lambda_p", "form"], as_index=False).predicted_loss.mean()
    spread.to_csv(out / "mapping_lambda_form_sensitivity.csv", index=False)
    quality_diff.groupby("model").rmse_minus_M0.quantile([.025, .5, .975]).unstack().to_csv(
        out / "quality_model_difference_ci.csv")
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    freeze = {"selected_model": contract["quality_model"], "contract": contract,
              "p04_freeze_sha256": (stage_dir("P04") / "model_freeze.sha256").read_text().strip(),
              "q1_frozen_hashes": json.loads((stage_dir("P07") / "frozen_q1_interface/frozen_hashes.json").read_text()),
              "status": "core_frozen_before_historical_comparison"}
    write_json(out / "core_model_contract.json", freeze)
    files = []
    for stage_num in range(1, 12):
        stage = f"P{stage_num:02d}"
        directory = stage_dir(stage)
        files.extend(path for path in directory.rglob("*") if path.is_file())
    manifest = {"stages": [f"P{i:02d}" for i in range(1, 12)],
                "output_sha256": {path.relative_to(OUT).as_posix(): sha(path)
                                  for path in sorted(set(files))},
                "freeze_scope": "P01–P11 outputs and P08 model contract"}
    write_json(out / "core_results_manifest.json", manifest)
    (out / "core_freeze.sha256").write_text(core_digest(out) + "\n", encoding="ascii")
    (out / "p12_report.md").write_text(
        "# P12 稳健性与核心冻结\n\n"
        "统计、结构、跨来源和情景不确定性分开保存。"
        "B1 起点与 bootstrap、质量参数 bootstrap/相关/剖面、B7 锁定验证 bootstrap、"
        "三种 QA→QB 映射、四种 lambda、Form A/B、支持距离和配比 delta 均已纳入。"
        "core_freeze.sha256 冻结 P01–P11 文件摘要，历史比较不得改变 V5 主模型。\n",
        encoding="utf-8")
    metadata = finalize("P12", [stage_dir("P08") / "model_contract.json",
                        stage_dir("P05") / "bootstrap_validation.csv",
                        stage_dir("P11") / "delta_sensitivity.csv"],
                        ["跨来源和情景不确定性不是单一统计置信区间"],
                        ["质量参数固定其余参数的 RSS 剖面仅作识别诊断"],
                        ["semi_synthetic", "q1_imported", "scenario_assumption"])
    return {"core_files": len(manifest["output_sha256"]), "metadata": metadata}
