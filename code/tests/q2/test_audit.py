"""Q2-P0 审计测试；微型数据仅在内存中构造，不进入正式结果。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.q2 import audit


OUTPUT = audit.OUT


def test_all_mapped_files_exist_and_hashes_unchanged():
    meta = json.loads((OUTPUT / "audit_metadata.json").read_text())
    assert len(audit.FILES) == 12
    assert all((audit.B_DIR / rel).is_file() for paths in audit.FILES.values() for rel in paths)
    assert meta["source_hash_unchanged"] is True
    assert all(audit.sha256(audit.ROOT / rel) == value for rel, value in meta["input_sha256"].items())


def test_required_fields_and_types_from_actual_data():
    for aid, paths in audit.FILES.items():
        for rel in paths:
            df = pd.read_csv(audit.B_DIR / rel)
            assert set(audit.FIELDS[aid].values()) <= set(df.columns)
            for key in ("N", "D", "loss", "Q"):
                col = audit.FIELDS[aid].get(key)
                if col:
                    assert pd.api.types.is_numeric_dtype(df[col]), (aid, col)


def test_numeric_rule_rejects_invalid_n_d_loss_q():
    # 明确标记的人工微型样例。
    df = pd.DataFrame({"N": [0, 1, 1], "D": [1, -1, 1], "L": [1, 1, np.inf], "Q": [0.5, 1.2, np.nan]})
    issues = audit.numeric_issues(df, {"N": "N", "D": "D", "loss": "L", "Q": "Q"}, "test", "synthetic_fixture")
    assert {x["reason_code"] for x in issues} == {"invalid_N", "invalid_D", "invalid_loss", "invalid_Q"}


def test_actual_numeric_invalid_rows_are_isolated():
    invalid = pd.read_csv(OUTPUT / "invalid_rows.csv")
    assert set(invalid.status) == {"numeric_invalid"}
    assert set(invalid.attachment) == {"B9"}
    assert set(invalid.reason_code) == {"invalid_D"}
    assert len(invalid) == 4


def test_b1_checkpoint_unit_and_group_structure():
    b1 = pd.read_csv(audit.B_DIR / audit.FILES["B1"][0])
    assert b1.run_id.is_unique
    assert b1.groupby("N_params_B").steps.nunique().eq(147).all()
    assert not b1.duplicated(["N_params_B", "steps"]).any()
    assert b1.N_params_B.gt(0).all() and b1.D_tokens_B.gt(0).all()
    assert np.isfinite(b1.val_loss).all()


def test_b6_b7_overlap_and_uniqueness():
    b6 = pd.read_csv(audit.B_DIR / audit.FILES["B6"][0])
    b7 = pd.read_csv(audit.B_DIR / audit.FILES["B7"][0])
    overlap = pd.read_csv(OUTPUT / "b6_b7_overlap.csv")
    assert b6.experiment_id.is_unique and b7.experiment_id.is_unique
    assert set(b6.experiment_id) <= set(b7.experiment_id)
    assert overlap.in_b6.sum() == len(b6)
    assert overlap.in_b7.sum() == len(b7)
    assert not overlap.conflicting_duplicate.any()
    assert overlap.loc[overlap.in_b6, ["same_N", "same_D", "same_Q", "same_loss"]].all().all()
    assert len(overlap.loc[overlap.in_b7 & ~overlap.in_b6]) == 90


def test_b6_factorial_and_q_range():
    b6 = pd.read_csv(audit.B_DIR / audit.FILES["B6"][0])
    levels = b6.groupby(["N_params_B", "D_tokens_B"]).Q_score.nunique()
    assert len(levels) == 45
    assert levels.eq(8).all()
    assert b6.Q_score.between(0, 1).all() and np.isfinite(b6.Q_score).all()


def test_b8_group_direction_is_calculated_not_recoded():
    b8 = pd.read_csv(audit.B_DIR / audit.FILES["B8"][0])
    rows = audit.group_direction(b8, "all")
    assert len(rows) == b8.groupby(["N_params_B", "D_tokens_B"]).ngroups
    assert all(row["spearman_Q_loss"] > 0 for row in rows)
    assert b8.data_type.value_counts().to_dict() == {"calibrated": 984, "extrapolated": 720}
    assert b8.experiment_id.is_unique


def test_interpolated_estimated_metadata_roles_never_fit():
    roles = pd.read_csv(OUTPUT / "dataset_role_matrix.csv").set_index("attachment")
    for aid in ("B3", "B9", "B10", "B11", "B12"):
        assert not roles.loc[aid, "can_fit"]
        assert not roles.loc[aid, "can_tune"]
    assert roles.loc["B3", "provenance"] == "interpolated"
    assert roles.loc["B10", "provenance"] == "estimated_reference"
    assert roles.loc["B2", "provenance"] == "semi_synthetic"


def test_alignment_preserves_unresolved_b12():
    align = pd.read_csv(OUTPUT / "b1_b3_b12_alignment.csv")
    b12 = pd.read_csv(audit.B_DIR / audit.FILES["B12"][0])
    assert len(align) == 1176
    assert align.tokens_b3.notna().sum() == 16
    assert align.b3_loss_match.dropna().eq(False).all()
    assert align.tokens_b12.isna().all()
    assert align.match_status.str.contains("b12_unresolved").all()
    assert {"model_repo", "model_size", "step", "branch", "commit"} == set(b12.columns)
    assert b12.groupby(["model_repo", "step"]).size().eq(1).all()


def test_model_family_link_is_only_metadata():
    b11 = pd.read_csv(audit.B_DIR / audit.FILES["B11"][0])
    assert b11.model_repo.is_unique
    assert set(b11.family) == {"pythia", "cerebras_gpt", "olmo"}
    assert "N_params_B" not in b11.columns


def test_first_question_frozen_interface():
    q1 = json.loads((OUTPUT / "q1_interface_audit.json").read_text())
    assert q1["model_exists"] and q1["reference_exists"]
    assert q1["input_count"] == 17 and q1["output_count"] == 13
    assert q1["model_sha256"] == audit.sha256(audit.ROOT / q1["model_path"])
    assert q1["historical_hash_anchored"] is False
    assert q1["qmapped_role"] == "sensitivity_only"


def test_contract_does_not_claim_unverified_semantics():
    # 该文件使用 YAML 1.2 合法的 JSON 子集编码。
    contract = json.loads((OUTPUT / "field_contract.yaml").read_text())
    assert contract["standard_fields"]["Q_direction"].startswith("uncertain")
    assert "already billions" in contract["standard_fields"]["N_billions"]
    assert contract["attachments"]["B8"]["data_type"] == "data_type"
    quality = pd.read_csv(OUTPUT / "quality_semantics.csv")
    assert set(quality.higher_means_better) == {"uncertain"}
    assert quality.loc[quality.attachment.eq("B8"), "same_definition_as_B6"].item() == "uncertain"
