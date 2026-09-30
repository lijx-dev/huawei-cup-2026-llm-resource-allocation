"""P0：附件与冻结接口的只读审计。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import BROOT, OUT, ROOT, csv, dump, finish, read, sha, valid_ndl

FILES = {
    "B1": "pythia_training_log_existing.csv", "B2": "cerebras_training_log.csv",
    "B3": "training_trajectories", "B4": "scaling_baseline.csv", "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv", "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv", "B9": "supplementary_large_models.csv",
    "B10": "supplementary_large_baseline.csv", "B11": "open_model_family_metadata.csv",
    "B12": "pythia_checkpoint_index.csv",
}
PROVENANCE = {"B1": "observed_public_trajectory", "B2": "semi_synthetic", "B3": "interpolated",
              "B4": "public_cross_family", "B5": "published_mixed_semantics", "B6": "semi_synthetic",
              "B7": "semi_synthetic", "B8": "semi_synthetic_calibrated_and_extrapolated", "B9": "public_metadata",
              "B10": "estimated_loss", "B11": "metadata", "B12": "checkpoint_index"}


def overlap(b6: pd.DataFrame, b7: pd.DataFrame) -> pd.DataFrame:
    if b6.experiment_id.duplicated().any() or b7.experiment_id.duplicated().any():
        raise ValueError("B6/B7 内部 experiment_id 重复")
    a = b6.set_index("experiment_id")
    b = b7.set_index("experiment_id")
    rows = []
    for key in b.index:
        in_b6 = key in a.index
        same = bool(np.allclose(a.loc[key, ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_numpy(float),
                                b.loc[key, ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].to_numpy(float), rtol=0, atol=1e-10)) if in_b6 else False
        rows.append({"experiment_id": key, "in_b6": in_b6, "same_values": same,
                     "role": "duplicate_exact" if in_b6 and same else "duplicate_conflict" if in_b6 else "b7_new"})
    return pd.DataFrame(rows)


def run() -> str:
    dest = OUT / "audit"
    dest.mkdir(parents=True, exist_ok=True)
    rows, source_paths = [], []
    for aid, rel in FILES.items():
        path = BROOT / rel
        paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        if not paths or any(not p.exists() for p in paths):
            rows.append({"attachment_id": aid, "actual_file": "MISSING", "relative_path": rel, "sha256": "", "raw_rows": 0, "raw_columns": "", "provenance": PROVENANCE[aid], "usable_for_fit": False, "usable_for_validation": False, "notes": "missing"})
            continue
        for p in paths:
            d = pd.read_csv(p)
            source_paths.append(p)
            rows.append({"attachment_id": aid, "actual_file": p.name, "relative_path": str(p.relative_to(ROOT)), "sha256": sha(p),
                         "raw_rows": len(d), "raw_columns": "|".join(d.columns), "provenance": PROVENANCE[aid],
                         "usable_for_fit": aid in {"B1", "B6"}, "usable_for_validation": aid in {"B2", "B3", "B4", "B5", "B7", "B8"}, "notes": "B3 files are one attachment" if aid == "B3" else ""})
    csv(dest / "b_attachment_inventory.csv", rows)
    contract = {"unit": "N_params_B/D_tokens_B 均为十亿；原始 N/D = 表值 × 10^9；训练算力 FLOPs ≈ 6×N_raw×D_raw",
                "standard_fields": {"N_params_raw": "N_params_B * 1e9", "N_billions": "N_params_B", "D_tokens_raw": "D_tokens_B * 1e9",
                                    "D_billions": "D_tokens_B", "val_loss": "val_loss", "Q_B_raw": "Q_score; B6/B7/B8 only",
                                    "model_family": "family or run_id source family", "model_scale": "N_params_B", "checkpoint": "steps/step when present",
                                    "experiment_id": "experiment_id or row locator", "provenance": "data description and attachment"}}
    (dest / "field_contract.yaml").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    b1 = read(FILES["B1"])
    b6, b7, b8 = (read(FILES[x]) for x in ("B6", "B7", "B8"))
    for aid, df in (("B1", b1), ("B6", b6), ("B7", b7), ("B8", b8)):
        valid_ndl(df, "P0", aid)
    ov = overlap(b6, b7)
    csv(dest / "b6_b7_overlap.csv", ov)
    sem = []
    for aid in ("B2", "B3", "B4", "B5"):
        sem.append({"attachment_id": aid, "absolute_comparable_to_B1": "unknown", "trend_comparable_to_B1": "diagnostic_only",
                    "evaluation_corpus_known": False, "tokenizer_known": False, "token_count_definition_known": aid == "B3",
                    "provenance": PROVENANCE[aid], "notes": "B3 从 B1 检查点插值；非独立验证" if aid == "B3" else "跨来源评估语义未有充分文件级证明"})
    csv(dest / "loss_semantics.csv", sem)
    dirs = []
    for (kind, n, d), g in b8.groupby(["data_type", "N_params_B", "D_tokens_B"]):
        rho = float(spearmanr(g.Q_score, g.val_loss).statistic) if len(g) >= 3 else np.nan
        dirs.append({"data_type": kind, "N_billions": n, "D_billions": d, "n": len(g), "spearman_Q_loss": rho,
                     "direction": "negative" if rho < 0 else "positive" if rho > 0 else "unresolved"})
    csv(dest / "b8_quality_direction.csv", dirs)
    model = ROOT / "results/q1_revision_v2/mixture/fitted_models.joblib"
    meta = ROOT / "results/q1/mixture/mixture_model_metadata.json"
    ref = ROOT / "results/q1_revision_v2_1/mixture/inherited_v2_model_reference.json"
    opt = ROOT / "results/q1/mixture/optimal_mixture.json"
    import joblib
    obj = joblib.load(model)
    expected = json.loads(ref.read_text())["model_file_sha256"]
    q1opt = json.loads(opt.read_text())
    p0 = np.array([q1opt["best_training_recipe_P"][f.replace("train_the_pile_", "")] for f in obj["mix_fields"]], float)
    if len(p0) != 17 or np.any(p0 < 0) or not np.isclose(p0.sum(), 1, atol=1e-8):
        raise ValueError("Q1 参考训练配方不满足 17 维单纯形")
    q1 = {"model_path": str(model.relative_to(ROOT)), "model_sha256": sha(model), "reference_sha256": expected,
          "hash_matches_v2_1_reference": sha(model) == expected, "historical_v2_hash_anchor": False,
          "n_inputs": len(obj["mix_fields"]), "n_outputs": len(obj["loss_fields"]),
          "input_columns": obj["mix_fields"], "output_columns": obj["loss_fields"],
          "support_rule": json.loads(meta.read_text())["support_rule"],
          "support_threshold": json.loads(meta.read_text())["support_threshold"],
          "candidate_p0": "best_training_recipe_P in Q1 optimal_mixture.json; original training recipe", "p0": p0.tolist(),
          "p0_source_sha256": sha(opt), "latest_QA_version": "q1-revision-v2.1"}
    dump(dest / "q1_interface.json", q1)
    checkpoint = read(FILES["B12"])
    # B12 仅为检查点索引。检查 B1 的 step 覆盖，不把不同规模仓库记录误当作同一训练轨迹。
    b1_audit = []
    index_steps = set(checkpoint.step.astype(int))
    for n, group in b1.groupby("N_params_B"):
        ordered = group.sort_values("steps")
        b1_audit.append({"N_billions": n, "n": len(group), "unique_steps": group.steps.nunique(),
                         "unique_D": group.D_tokens_B.nunique(), "duplicate_N_step": int(group.duplicated(["N_params_B", "steps"]).sum()),
                         "D_strictly_increasing_by_step": bool((ordered.D_tokens_B.diff().dropna() > 0).all()),
                         "steps_in_B12": int(group.steps.isin(index_steps).sum()), "n_B12_total": len(checkpoint)})
    csv(dest / "b1_trajectory_audit.csv", b1_audit)
    warnings = ["推荐方案和 Q2_V3_WORK_PLAN 缺失；本任务书提供替代规则", "B2/B4/B5 的绝对 Loss 同口径未获证明", "Q1 v2 模型缺历史哈希锚，但 v2.1 报告提供可核验哈希"]
    if not q1["hash_matches_v2_1_reference"]: warnings.append("Q1 冻结模型与 v2.1 参考哈希不一致")
    report = dest / "p0_audit_report.md"
    report.write_text("# P0 数据与接口审计\n\n" + "\n".join(f"- {x}" for x in warnings) +
                      f"\n\nB1 有 {len(b1)} 条、{b1.N_params_B.nunique()} 条规模轨迹；B12 检查点索引 {len(checkpoint)} 条。"
                      f"B6/B7 重叠 {int(ov.in_b6.sum())} 条，B7-new {int((~ov.in_b6).sum())} 条，冲突 {int((ov.role=='duplicate_conflict').sum())} 条。"
                      f"B8 provenance: {b8.data_type.value_counts().to_dict()}。B6–B8 的 Q_score 有 0–1 数值范围，但生成定义未在附件文件中独立核验。B9 无 val_loss；B10 是估算，存在循环生成风险。\n",
                      encoding="utf-8")
    status = "BLOCKED" if any(x["actual_file"] == "MISSING" for x in rows) or (ov.role == "duplicate_conflict").any() or not q1["hash_matches_v2_1_reference"] else "PASS_WITH_WARNINGS"
    finish(0, status, source_paths + [model, meta, ref, opt], report, warnings)
    return status
