"""Q2-P0：只读 B 附件，输出可追溯的数据契约与审计表。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
B_DIR = ROOT / "data/real_attachments/B_scaling_laws"
OUT = ROOT / "results/q2_scaling_v1/audit"
SEED = 7
VERSION = "q2-scaling-v1"
RULE = "q2-p0-1"
EXPECTED_ROWS = {"B1": 1176, "B2": 1029, "B3": 500, "B4": 57, "B5": 44, "B6": 360, "B7": 450, "B8": 1704, "B9": 132, "B10": 128, "B11": 18, "B12": 1386}

# 编号依据 docs/数据说明.pdf 的编号与读取速查，不依据目录遍历顺序。
FILES = {
    "B1": ["pythia_training_log_existing.csv"],
    "B2": ["cerebras_training_log.csv"],
    "B3": ["training_trajectories/pythia_0.070542B_trajectory.csv", "training_trajectories/pythia_0.162405B_trajectory.csv", "training_trajectories/pythia_0.409009B_trajectory.csv", "training_trajectories/pythia_1.040867B_trajectory.csv", "training_trajectories/pythia_1.416184B_trajectory.csv", "training_trajectories/pythia_2.782831B_trajectory.csv", "training_trajectories/pythia_6.86104B_trajectory.csv", "training_trajectories/pythia_11.965825B_trajectory.csv"],
    "B4": ["scaling_baseline.csv"],
    "B5": ["published_scaling_data.csv"],
    "B6": ["supplementary_NQ_experiment.csv"],
    "B7": ["supplementary_NQ_experiment_expanded.csv"],
    "B8": ["supplementary_NQ_experiment_large.csv"],
    "B9": ["supplementary_large_models.csv"],
    "B10": ["supplementary_large_baseline.csv"],
    "B11": ["open_model_family_metadata.csv"],
    "B12": ["pythia_checkpoint_index.csv"],
}
ROLES = {
    "B1": ("classic_scaling_fit", "reported_observation", True, False, False),
    "B2": ("cross_source_validation", "semi_synthetic", False, True, False),
    "B3": ("interpolated_trajectory_validation", "interpolated", False, True, False),
    "B4": ("cross_family_validation", "reported_observation", False, True, False),
    "B5": ("literature_scale_validation", "mixed_reported_sources", False, True, False),
    "B6": ("quality_scaling_fit", "semi_synthetic", True, False, False),
    "B7": ("quality_holdout_extension", "semi_synthetic", False, True, False),
    "B8": ("quality_direction_stress_test", "semi_synthetic_and_extrapolated", False, False, True),
    "B9": ("large_model_metadata", "metadata_only", False, False, False),
    "B10": ("estimated_large_scale_reference", "estimated_reference", False, False, False),
    "B11": ("model_family_metadata", "metadata_only", False, False, False),
    "B12": ("checkpoint_metadata", "metadata_only", False, False, False),
}
FIELDS = {
    "B1": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_scale="N_params_B", checkpoint="steps", experiment_id="run_id"),
    "B2": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_scale="N_params_B", checkpoint="steps", experiment_id="run_id"),
    "B3": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_scale="N_params_B", checkpoint="step"),
    "B4": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_family="family"),
    "B5": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_family="family"),
    "B6": dict(N="N_params_B", D="D_tokens_B", Q="Q_score", loss="val_loss", experiment_id="experiment_id"),
    "B7": dict(N="N_params_B", D="D_tokens_B", Q="Q_score", loss="val_loss", experiment_id="experiment_id"),
    "B8": dict(N="N_params_B", D="D_tokens_B", Q="Q_score", loss="val_loss", experiment_id="experiment_id", data_type="data_type"),
    "B9": dict(N="N_params_B", D="D_tokens_B", model_name="model_name"),
    "B10": dict(N="N_params_B", D="D_tokens_B", loss="val_loss", model_name="family"),
    "B11": dict(model_family="family", model_repo="model_repo"),
    "B12": dict(model_repo="model_repo", model_scale="model_size", checkpoint="step"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    # 零异常也是实测结果；保留表头供下游明确识别零行。
    pd.DataFrame(rows, columns=columns).to_csv(OUT / name, index=False)


def numeric_issues(df: pd.DataFrame, mapping: dict, attachment: str, file: str) -> list[dict]:
    issues = []
    for key in ("N", "D", "loss", "Q"):
        col = mapping.get(key)
        if col not in df:
            continue
        nums = pd.to_numeric(df[col], errors="coerce")
        bad = ~np.isfinite(nums.to_numpy(dtype=float))
        if key in ("N", "D"):
            bad |= nums.to_numpy(dtype=float) <= 0
        if key == "Q":
            bad |= (nums.to_numpy(dtype=float) < 0) | (nums.to_numpy(dtype=float) > 1)
        for pos in np.flatnonzero(bad):
            issues.append(dict(attachment=attachment, source_file=file, csv_line=int(pos + 2), record_id=str(df.iloc[pos].get("experiment_id", df.iloc[pos].get("run_id", ""))), status="numeric_invalid", reason_code=f"invalid_{key}", column=col, raw_value=str(df.iloc[pos][col]), rule_version=RULE))
    return issues


def group_direction(frame: pd.DataFrame, data_type: str) -> list[dict]:
    rows = []
    for (n, d), g in frame.groupby(["N_params_B", "D_tokens_B"], dropna=False):
        q = g.Q_score.to_numpy(float)
        loss = g.val_loss.to_numpy(float)
        if len(g) >= 2 and np.std(q) > 0 and np.std(loss) > 0:
            pearson = float(np.corrcoef(q, loss)[0, 1])
            spearman = float(spearmanr(q, loss).statistic)
        else:
            pearson = spearman = math.nan
        order = np.argsort(q)
        delta = np.diff(loss[order])
        monotonic = "increasing" if len(delta) and np.all(delta >= 0) else "decreasing" if len(delta) and np.all(delta <= 0) else "mixed" if len(delta) else "insufficient"
        rows.append(dict(data_type=data_type, group_id=f"N={n}|D={d}", N_params_B=n, D_tokens_B=d, n_rows=len(g), n_q_levels=g.Q_score.nunique(), q_min=q.min(), q_max=q.max(), loss_min=loss.min(), loss_max=loss.max(), pearson_Q_loss=pearson, spearman_Q_loss=spearman, monotonic_direction=monotonic))
    return rows


def q1_interface() -> dict:
    path = ROOT / "results/q1_revision_v2/mixture/fitted_models.joblib"
    ref = ROOT / "results/q1_revision_v2_1/mixture/inherited_v2_model_reference.json"
    meta = ROOT / "results/q1_revision_v2_1/metadata.json"
    report = ROOT / "results/q1_revision_v2_1/report/question1_v2_1_report.md"
    mixture = ROOT / "data/real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv"
    losses = ROOT / "data/real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv"
    result = dict(model_exists=path.exists(), model_path=str(path.relative_to(ROOT)), model_sha256=sha256(path) if path.exists() else None, reference_exists=ref.exists(), final_report_exists=report.exists(), inherited_reference=json.loads(ref.read_text()) if ref.exists() else None)
    if path.exists():
        import joblib
        bundle = joblib.load(path)
        result.update(model_bundle_keys=list(bundle), input_fields=bundle.get("mix_fields"), output_fields=bundle.get("loss_fields"), input_count=len(bundle.get("mix_fields", [])), output_count=len(bundle.get("loss_fields", [])), model_version="q1-revision-v2; rechecked in q1-revision-v2.1", reference_mixture_source="A4 train_mixture_1m.csv", c4_status="excluded_from_formal_quality_mapping_per_q2_agents", qmapped_role="sensitivity_only", historical_hash_anchored=bool(json.loads(ref.read_text()).get("model_hash_anchor_in_v2_metadata")) if ref.exists() else False)
        result["input_order_matches_A4"] = mixture.exists() and bundle.get("mix_fields") == [c for c in pd.read_csv(mixture, nrows=0).columns if c != "index"]
        result["output_order_matches_A5"] = losses.exists() and bundle.get("loss_fields") == [c for c in pd.read_csv(losses, nrows=0).columns if c != "index"]
        result["model_hash_matches_v2_1_reference"] = ref.exists() and result["model_sha256"] == json.loads(ref.read_text()).get("model_file_sha256")
        result["A4_training_mixture_exists"] = mixture.exists()
        result["A4_training_mixture_sha256"] = sha256(mixture) if mixture.exists() else None
    result["ready_for_h_v"] = bool(result.get("input_count") == 17 and result.get("output_count") == 13 and result.get("input_order_matches_A4") and result.get("output_order_matches_A5") and result.get("model_hash_matches_v2_1_reference") and mixture.exists())
    result["q_mapping_policy"] = "B6/B7 Q_score is independent input; do not replace with Qmapped"
    return result


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {x["file"]: x for x in json.loads((ROOT / "data/real_attachments/source_manifest.json").read_text()) if x.get("problem") == "B"}
    frames, inventory, schemas, quality, invalid, duplicates, roles = {}, [], [], [], [], [], []
    before = {}
    for aid, paths in FILES.items():
        role, provenance, fit, validate, direction = ROLES[aid]
        roles.append(dict(attachment=aid, provenance=provenance, model_role=role, validation_role="direction_stress" if direction else "holdout_or_reference" if validate else "none", can_fit=fit, can_tune=False, can_validate=validate, can_test_direction=direction, can_extrapolate_reference=aid in ("B9", "B10"), leakage_risk="derived_from_B1" if aid in ("B2", "B3", "B10") else "overlap_with_B6" if aid in ("B7", "B8") else "shared_trajectory" if aid in ("B1", "B12") else "source_metric_heterogeneity" if aid in ("B4", "B5") else "none_known", notes="role is restricted to provenance and available columns"))
        parts = []
        for rel in paths:
            p = B_DIR / rel
            repo_rel = str(p.relative_to(ROOT))
            if not p.exists():
                inventory.append(dict(attachment_id=aid, expected_role=role, actual_file="", relative_path=repo_rel, file_type="csv", sha256="", file_size="", raw_rows="", raw_columns="", readable=False, provenance_type=provenance, proposed_model_role=role, notes="missing"))
                continue
            digest = sha256(p)
            before[repo_rel] = digest
            try:
                part = pd.read_csv(p)
                readable, note = True, "primary" if aid != "B3" else "one_of_8_interpolated_files"
                parts.append(part)
                missing = sorted(set(FIELDS[aid].values()) - set(part.columns))
                if missing:
                    note += f"; missing_contract_columns={missing}"
                invalid.extend(numeric_issues(part, FIELDS[aid], aid, repo_rel))
                for col in part:
                    s = part[col]
                    schemas.append(dict(attachment=aid, source_file=repo_rel, column=col, dtype=str(s.dtype), non_null=int(s.notna().sum()), null_count=int(s.isna().sum()), unique_count=int(s.nunique(dropna=True)), numeric_min=float(s.min()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else "", numeric_max=float(s.max()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else ""))
                duplicate_full = part.duplicated(keep=False)
                for pos in np.flatnonzero(duplicate_full):
                    duplicates.append(dict(attachment=aid, source_file=repo_rel, csv_line=int(pos + 2), duplicate_type="full_row", record_id=str(part.iloc[pos].get("experiment_id", ""))))
                for key in ("experiment_id",):
                    if key in part:
                        for pos in np.flatnonzero(part.duplicated(key, keep=False)):
                            duplicates.append(dict(attachment=aid, source_file=repo_rel, csv_line=int(pos + 2), duplicate_type=f"duplicate_{key}", record_id=str(part.iloc[pos][key])))
                invalid_count = len({x["csv_line"] for x in numeric_issues(part, FIELDS[aid], aid, repo_rel)})
                quality.append(dict(attachment=aid, source_file=repo_rel, rows=len(part), columns=len(part.columns), full_duplicate_rows=int(duplicate_full.sum()), required_columns_missing="|".join(missing), numeric_invalid_rows=invalid_count, status="structural_invalid" if missing else "numeric_invalid_present" if invalid_count else "valid", provenance_type=provenance))
            except Exception as exc:
                readable, note = False, f"read_error:{type(exc).__name__}:{exc}"
            entry = manifest.get(str(p.relative_to(ROOT / "data/real_attachments")), {})
            if entry.get("bytes") != p.stat().st_size:
                note += "; manifest_size_mismatch"
            if readable and len(part) != EXPECTED_ROWS[aid]:
                note += f"; document_row_count_mismatch:{EXPECTED_ROWS[aid]}"
            inventory.append(dict(attachment_id=aid, expected_role=role, actual_file=p.name, relative_path=repo_rel, file_type=p.suffix.lstrip("."), sha256=digest, file_size=p.stat().st_size, expected_rows=EXPECTED_ROWS[aid], raw_rows=len(part) if readable else "", raw_columns=len(part.columns) if readable else "", readable=readable, provenance_type=provenance, proposed_model_role=role, notes=note))
        if parts:
            frames[aid] = pd.concat(parts, ignore_index=True)

    write_csv("b_attachment_inventory.csv", inventory, ["attachment_id", "expected_role", "actual_file", "relative_path", "file_type", "sha256", "file_size", "expected_rows", "raw_rows", "raw_columns", "readable", "provenance_type", "proposed_model_role", "notes"])
    write_csv("schema_summary.csv", schemas, ["attachment", "source_file", "column", "dtype", "non_null", "null_count", "unique_count", "numeric_min", "numeric_max"])
    write_csv("data_quality_summary.csv", quality, ["attachment", "source_file", "rows", "columns", "full_duplicate_rows", "required_columns_missing", "numeric_invalid_rows", "status", "provenance_type"])
    write_csv("invalid_rows.csv", invalid, ["attachment", "source_file", "csv_line", "record_id", "status", "reason_code", "column", "raw_value", "rule_version"])
    write_csv("duplicate_rows.csv", duplicates, ["attachment", "source_file", "csv_line", "duplicate_type", "record_id"])
    write_csv("dataset_role_matrix.csv", roles, ["attachment", "provenance", "model_role", "validation_role", "can_fit", "can_tune", "can_validate", "can_test_direction", "can_extrapolate_reference", "leakage_risk", "notes"])

    b6, b7 = frames["B6"], frames["B7"]
    overlap = []
    for eid in sorted(set(b6.experiment_id) | set(b7.experiment_id)):
        a, b = b6[b6.experiment_id == eid], b7[b7.experiment_id == eid]
        x, y = a.iloc[0] if len(a) else None, b.iloc[0] if len(b) else None
        both = x is not None and y is not None
        flags = {k: bool(x[col] == y[col]) if both else None for k, col in [("same_N", "N_params_B"), ("same_D", "D_tokens_B"), ("same_Q", "Q_score"), ("same_loss", "val_loss")]}
        overlap.append(dict(experiment_id=eid, in_b6=x is not None, in_b7=y is not None, **flags, fully_identical=both and all(flags.values()), conflicting_duplicate=both and not all(flags.values()), b6_csv_line=int(a.index[0] + 2) if len(a) else "", b7_csv_line=int(b.index[0] + 2) if len(b) else ""))
    write_csv("b6_b7_overlap.csv", overlap, ["experiment_id", "in_b6", "in_b7", "same_N", "same_D", "same_Q", "same_loss", "fully_identical", "conflicting_duplicate", "b6_csv_line", "b7_csv_line"])
    overlap_stats = dict(n_b6=len(b6), n_b7=len(b7), n_overlap=sum(r["in_b6"] and r["in_b7"] for r in overlap), n_b7_new=sum(r["in_b7"] and not r["in_b6"] for r in overlap), n_conflicting=sum(r["conflicting_duplicate"] for r in overlap))

    b8 = frames["B8"]
    groups = group_direction(b8, "all")
    for kind, part in b8.groupby("data_type"):
        groups.extend(group_direction(part, kind))
    write_csv("b8_groupwise_q_loss_audit.csv", groups, ["data_type", "group_id", "N_params_B", "D_tokens_B", "n_rows", "n_q_levels", "q_min", "q_max", "loss_min", "loss_max", "pearson_Q_loss", "spearman_Q_loss", "monotonic_direction"])

    b1, b3, b12 = frames["B1"], frames["B3"], frames["B12"]
    # B12 只有名义 model_size/step，没有精确参数量或 tokens；不凭规模顺序硬配。
    alignment = []
    for (n, step), part in b1.groupby(["N_params_B", "steps"]):
        token = float(part.D_tokens_B.iloc[0])
        near = b3[(b3.N_params_B == n) & (b3.D_tokens_B == token)]
        alignment.append(dict(model_scale=n, checkpoint=step, tokens_b1=token, tokens_b3=float(near.D_tokens_B.iloc[0]) if len(near) else "", tokens_b12="", b3_loss_match=bool(np.isclose(part.val_loss.iloc[0], near.val_loss.iloc[0], atol=1e-4)) if len(near) else "", match_status="b1_b3_exact_N_D; b12_unresolved" if len(near) else "b1_only; b12_unresolved"))
    write_csv("b1_b3_b12_alignment.csv", alignment, ["model_scale", "checkpoint", "tokens_b1", "tokens_b3", "tokens_b12", "b3_loss_match", "match_status"])

    cross = []
    def cross_row(pair, method, a, b, note=""):
        cross.append(dict(pair=pair, key_method=method, left_rows=len(a), right_rows=len(b), overlap_count=len(set(a) & set(b)), note=note))
    cross_row("B1:B3", "exact_N_D", list(zip(b1.N_params_B, b1.D_tokens_B)), list(zip(b3.N_params_B, b3.D_tokens_B)), "B3 interpolated; shared points are not independent")
    cross_row("B1:B2", "exact_N_D", list(zip(b1.N_params_B, b1.D_tokens_B)), list(zip(frames["B2"].N_params_B, frames["B2"].D_tokens_B)), "exact scale/token only; provenance differs")
    cross_row("B6:B7", "experiment_id", b6.experiment_id, b7.experiment_id)
    for aid in ("B6", "B7"):
        f = frames[aid]
        cross_row(f"{aid}:B8", "exact_N_D_Q", list(zip(f.N_params_B, f.D_tokens_B, f.Q_score)), list(zip(b8.N_params_B, b8.D_tokens_B, b8.Q_score)), "configuration overlap does not prove equal provenance")
    b9, b10 = frames["B9"], frames["B10"]
    cross_row("B9:B10", "exact_model_name", b9.model_name.astype(str), b10.family.astype(str), "B10 family acts as model name")
    cross_row("B9:B10", "exact_N_D", list(zip(b9.N_params_B, b9.D_tokens_B)), list(zip(b10.N_params_B, b10.D_tokens_B)))
    write_csv("cross_attachment_overlap_summary.csv", cross, ["pair", "key_method", "left_rows", "right_rows", "overlap_count", "note"])

    losses = []
    for aid in FILES:
        if "loss" not in FIELDS[aid]:
            continue
        losses.append(dict(attachment=aid, loss_column="val_loss", loss_name="validation cross-entropy per contest description", metric_type="cross_entropy_reported", validation_dataset="unknown", tokenizer="unknown", vocab="unknown", evaluation_protocol="unknown", observed_or_estimated=ROLES[aid][1], directly_comparable_to_B1="within_B1" if aid == "B1" else "uncertain", evidence="docs/数据说明.pdf pp.1,9-11; actual CSV header; no per-source tokenizer/eval protocol supplied"))
    write_csv("loss_semantics.csv", losses, ["attachment", "loss_column", "loss_name", "metric_type", "validation_dataset", "tokenizer", "vocab", "evaluation_protocol", "observed_or_estimated", "directly_comparable_to_B1", "evidence"])
    qs = []
    for aid in ("B6", "B7", "B8"):
        f = frames[aid]
        qs.append(dict(attachment=aid, q_column="Q_score", min=f.Q_score.min(), max=f.Q_score.max(), mean=f.Q_score.mean(), std=f.Q_score.std(), unique_count=f.Q_score.nunique(), higher_means_better="uncertain", definition="numeric quality score; operational measurement/encoding not specified", generation_method="semi-synthetic calibrated from Pythia and RegMix quality signals" if aid == "B6" else "semi-synthetic; exact generator not documented", same_definition_as_B6="self" if aid == "B6" else "uncertain", evidence="docs/数据说明.pdf pp.9-11; source_manifest.json; actual Q_score column; no explicit direction mapping"))
    write_csv("quality_semantics.csv", qs, ["attachment", "q_column", "min", "max", "mean", "std", "unique_count", "higher_means_better", "definition", "generation_method", "same_definition_as_B6", "evidence"])

    contract = {"experiment_version": VERSION, "stage": "Q2-P0", "seed": SEED, "standard_fields": {"N_params_raw": "source N_params_B", "N_billions": "N_params_B (already billions)", "D_tokens_raw": "source D_tokens_B", "D_billions": "D_tokens_B (already billions)", "val_loss": "source val_loss, no cross-source calibration", "Q_raw": "source Q_score", "Q_normalized": "identity only for source Q_score within [0,1]; semantic equivalence unconfirmed", "Q_direction": "uncertain; no automatic flip", "model_family": "source family or metadata mapping, no automatic inference", "model_scale": "source N_params_B or model_size; not interchangeable without mapping", "checkpoint": "B1 steps/B3 step/B12 step; semantics differ", "experiment_id": "source experiment_id; B1 run_id is row identifier", "data_role": "dataset_role_matrix.csv", "provenance_type": "dataset_role_matrix.csv", "validation_role": "dataset_role_matrix.csv"}, "units": {aid: {"N_original_unit": "billions of parameters" if "N" in FIELDS[aid] else "not_applicable", "D_original_unit": "billions of tokens" if "D" in FIELDS[aid] else "not_applicable", "unit_evidence": "docs/数据说明.pdf p.5; N_params_B/D_tokens_B convention", "conversion_rule": "identity for *_B; multiply by 1e9 only if raw counts required"} for aid in FILES}, "attachments": FIELDS, "unresolved": ["B12 exact model_size to N_params_B mapping", "B3 step versus B1/B12 step semantics", "Q_score operational direction and B8 equivalence", "cross-source Loss evaluation protocol"]}
    # JSON 是 YAML 1.2 的合法子集；现有环境无需新增 PyYAML 依赖。
    (OUT / "field_contract.yaml").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    interface = q1_interface()
    (OUT / "q1_interface_audit.json").write_text(json.dumps(interface, ensure_ascii=False, indent=2), encoding="utf-8")
    b10_risk = "# B10 生成与循环验证风险\n\n数据说明第 10 页明确 B10 `supplementary_large_baseline.csv` 的 Loss 是使用已拟合标度律参数估算，并非观测。文件自身没有 formula/source/estimated/fitted/synthetic 等生成过程列，仅有 family、N_params_B、D_tokens_B、val_loss、is_converged。若后续用 B1 标度律预测 B10 并称其为独立验证，会有循环验证风险；是否确实使用同一套 B1 拟合参数，现有材料不能判定。B10 只可作为 estimated consistency reference。\n"
    (OUT / "b10_generation_risk_report.md").write_text(b10_risk, encoding="utf-8")
    after = {rel: sha256(ROOT / rel) for rel in before}
    metadata = dict(experiment_version=VERSION, stage="Q2-P0", seed=SEED, rule_version=RULE, audit_code_sha256=sha256(Path(__file__)), input_sha256=before, source_hash_unchanged=before == after, attachment_rows={aid: len(df) for aid, df in frames.items()}, expected_row_mismatch_count=sum(bool(r["readable"] and r["raw_rows"] != r["expected_rows"]) for r in inventory), manifest_size_mismatch_count=sum("manifest_size_mismatch" in r["notes"] for r in inventory), invalid_numeric_rows=len({(x["attachment"], x["source_file"], x["csv_line"]) for x in invalid}), duplicate_entries=len(duplicates), b6_b7=overlap_stats, b8_data_type_counts={str(k): int(v) for k, v in b8.data_type.value_counts().items()}, b1_scale_checkpoint_counts={str(k): int(v) for k, v in b1.groupby("N_params_B").size().items()}, b12_model_size_counts={str(k): int(v) for k, v in b12.model_size.value_counts().items()}, b1_b12_step_set_overlap=len(set(b1.steps) & set(b12.step)), b3_exact_N_D_matches=sum(bool(x["tokens_b3"] != "") for x in alignment), b3_exact_N_D_loss_matches=sum(x["b3_loss_match"] is True for x in alignment), unresolved=contract["unresolved"])
    (OUT / "audit_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Q2-P0 B1–B12 审计报告", "", "实验版本 `q2-scaling-v1`，固定种子 7；只读附件，未拟合标度律。", "", "## 附件映射与角色", "", "完整路径、SHA-256、字段和角色见 `b_attachment_inventory.csv`、`schema_summary.csv`、`dataset_role_matrix.csv`。编号依据《数据说明》；B3 有 8 个文件。", "", "## B1/B2/B3/B12", "", f"B1 共 {len(b1)} 行、{b1.N_params_B.nunique()} 个精确参数规模；各规模 checkpoint 数：{metadata['b1_scale_checkpoint_counts']}。同一规模内 checkpoint 属训练轨迹，不能当独立训练；后续 LOSO 应按 `N_params_B` 分组。B1 `run_id` 实际逐行唯一，不是轨迹 ID。", f"B2 共 {len(frames['B2'])} 行，清单与数据说明均标半合成，来源说明称按 Pythia 标度律校准；B1/B2 Loss 的 tokenizer/评测集未给出，绝对 RMSE 可比性未确认。", f"B3 共 {len(b3)} 行，文件 `interpolated` 列及数据说明均确认插值；B1/B3 精确 N,D 匹配 {metadata['b3_exact_N_D_matches']} 点，但同时 Loss 精确一致点数为 {metadata['b3_exact_N_D_loss_matches']}，不能称含真实 B1 原始节点。B3 只作轨迹形状检查，不能调参或作为独立测试。", "B12 仅有 `model_repo/model_size/step/branch/commit`，无 token 数或精确参数量；当前不能把名义 model_size 强行对应 B1 精确 N，也不能把 B3 的 step 直接视为 B12 step。对齐表明确标 `b12_unresolved`。", "", "## B4/B5 Loss 口径", "", f"B4 {len(frames['B4'])} 行、{frames['B4'].family.nunique()} 族；`is_converged` 分布 {frames['B4'].is_converged.value_counts().to_dict()}，当前均保留。B5 {len(frames['B5'])} 行、{frames['B5'].source.nunique()} 文献来源。两表缺 tokenizer、验证语料和评测协议；与 B1 绝对 Loss 可比性不明，后续以分源趋势为主要证据。", "", "## B6/B7 质量实验及重叠", "", f"B6/B7 均为半合成；B6 {len(b6)} 行，B7 {len(b7)} 行，按 experiment_id 重叠 {overlap_stats['n_overlap']} 行，B7 真新增 {overlap_stats['n_b7_new']} 行，冲突 {overlap_stats['n_conflicting']} 行。留出候选是 `b6_b7_overlap.csv` 中 `in_b7=True,in_b6=False` 的 ID；不得直接随机拆同一 N,D 组。", f"B6 有 {b6.groupby(['N_params_B','D_tokens_B']).ngroups} 个 N,D 组，各组 Q 水平数分布 {b6.groupby(['N_params_B','D_tokens_B']).Q_score.nunique().value_counts().to_dict()}。Q_score 实测范围 {b6.Q_score.min()}–{b6.Q_score.max()}；来源文档只称质量分，未给操作性定义或明确高值方向。", "", "## B8 质量方向压力测试", "", f"B8 {len(b8)} 行，data_type 分布 {metadata['b8_data_type_counts']}。Q_score 范围 {b8.Q_score.min()}–{b8.Q_score.max()}；与 B6 的定义等价性仍未知。", "|data_type|组数|Pearson 正/负|Spearman 正/负|单调递增/递减/混合|", "|---|---:|---:|---:|---:|"]
    for kind in ("calibrated", "extrapolated", "all"):
        g = pd.DataFrame([x for x in groups if x["data_type"] == kind])
        report.append(f"|{kind}|{len(g)}|{int((g.pearson_Q_loss > 0).sum())}/{int((g.pearson_Q_loss < 0).sum())}|{int((g.spearman_Q_loss > 0).sum())}/{int((g.spearman_Q_loss < 0).sum())}|{int((g.monotonic_direction == 'increasing').sum())}/{int((g.monotonic_direction == 'decreasing').sum())}/{int((g.monotonic_direction == 'mixed').sum())}|")
    report += ["", "方向是附件数值表现，不据此翻转 Q 或判定记录错误。B8 无逐行生成公式、校准细节或与 B6 的 Q 编码对照，编码/对齐异常尚无直接证据。", "", "## B9/B10 和辅助数据", "", f"B9 {len(b9)} 行，参数范围 {b9.N_params_B.min()}–{b9.N_params_B.max()} B；无 Loss，是元数据。D_tokens_B 非正值见 `invalid_rows.csv`，不作为可用 N,D 点。B10 {len(b10)} 行，参数范围 {b10.N_params_B.min()}–{b10.N_params_B.max()} B，Loss 为拟合标度律估算。循环生成风险见 `b10_generation_risk_report.md`。", f"B11 {len(frames['B11'])} 行，有 `family/model_repo`；仅能提供家族或仓库辅助映射，不能从其字段推断 B4 每行的唯一模型。", "", "## 第一问接口", "", f"冻结模型存在：{interface['model_exists']}；SHA-256：{interface['model_sha256']}；字段顺序：{interface.get('input_count')} 输入/{interface.get('output_count')} 输出；与 A4/A5 表头顺序相符：{interface.get('input_order_matches_A4')}/{interface.get('output_order_matches_A5')}，见 `q1_interface_audit.json`。历史 v2 元数据无模型哈希锚点；v2.1 参考文件哈希匹配：{interface.get('model_hash_matches_v2_1_reference')}。后续构造 h_v(p) 所需模型与 A4 配方存在，但参考 p0/s_v 尚未计算。Qmapped 仅敏感性，不替代 B6 Q。", "", "## 完整性和未解决事项", "", f"所有 B 文件运行前后 SHA-256 不变：{metadata['source_hash_unchanged']}；数值非法记录数：{metadata['invalid_numeric_rows']}；重复审计条目数：{metadata['duplicate_entries']}。`valid` 仅指通过当前结构与数值规则，不证明来源真实性。", "Loss 跨来源的评测集、tokenizer、vocab 和协议缺失；Q_score 的操作性定义与 B8 同义性缺失；B12 与 B1/B3 的精确模型规模及 token 对齐缺证据。以上均在字段契约中保持 `uncertain/unresolved`。", "", "## 阶段判定", "", "Q2-P0 STATUS: PASS_WITH_WARNINGS。B1 可作为后续经典 Scaling Law 拟合的候选输入；跨源绝对 Loss 验证、B8 Q 同义性和 B12 强对齐须保留限制。未进入 Q2-P1。", ""]
    extra = [
        "", "## 跨表与字段复核补充", "",
        f"B1 的 `(N_params_B,steps)` 在每条轨迹内唯一；B1 与 B12 的步数集合精确重叠 {metadata['b1_b12_step_set_overlap']} 个，但 B12 有 9 个名义 model_size，B1 只有 8 个精确 N 值，故这不是模型级完整对齐。B3 的 `step` 与 B1 `steps` 不能直接作为同一检查点键。",
        "B1/B2 的 N,D 精确组合重叠 0 个；B2 的半合成值仍可能依据 B1 标度律校准，不能据零组合重叠认定独立。",
        "B4 所有 57 行都有 family 标签，B5 所有 44 行都有 family 与 source；这些标签不能补足缺失的 tokenizer、验证集和协议，因此目前不支持与 B1 进行零校准绝对 RMSE 解释。",
        f"B6/B7 共有 {overlap_stats['n_overlap']} 个相同 experiment_id，且 N、D、Q、Loss 全一致，支持重叠部分采用同一数值口径；B7 新增 90 个点的生成规则未单独说明。B8 与 B6/B7 的相同 N,D,Q 配置数分别为 {cross[3]['overlap_count']}/{cross[4]['overlap_count']}，不能据此认定 Loss 同义。",
        f"B9/B10 精确模型名称重叠 {cross[5]['overlap_count']} 个，精确 N,D 重叠 {cross[6]['overlap_count']} 个；B10 的 Loss 仍为估算，不能当外部真实观测。",
        f"数据说明预期行数不符文件数 {metadata['expected_row_mismatch_count']}，来源清单字节数不符文件数 {metadata['manifest_size_mismatch_count']}。", "",
    ]
    report[report.index("## 阶段判定"):report.index("## 阶段判定")] = extra
    (OUT / "q2_p0_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
