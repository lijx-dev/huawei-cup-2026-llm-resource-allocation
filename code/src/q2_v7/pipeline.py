"""Q2 V7 独立入口：仅读取附件 B 和本次交接 ZIP 的冻结接口。"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from .model import compute_optimum, derivatives, fit, predict, quality_incremental_cost_1e21


INTERFACE = "团队交接包_问题一接口/03_对照材料/Q1_M2响应接口.csv.gz"
INTERFACE_MANIFEST = "团队交接包_问题一接口/03_对照材料/Q1_M2接口_manifest.json"
PACKAGE_MANIFEST = "团队交接包_问题一接口/文件清单_含SHA256.json"
B_FILES = {
    "B1": "pythia_training_log_existing.csv", "B2": "cerebras_training_log.csv",
    "B4": "scaling_baseline.csv", "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv", "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv", "B9": "supplementary_large_models.csv",
    "B10": "supplementary_large_baseline.csv", "B11": "open_model_family_metadata.csv",
    "B12": "pythia_checkpoint_index.csv",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, float_format="%.12g")


def valid_rows(frame: pd.DataFrame, source: str, q=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["N_params_B", "D_tokens_B", "val_loss"] + (["Q_score"] if q else [])
    missing = set(cols) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} 缺少字段: {sorted(missing)}")
    numbers = frame[cols].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numbers.to_numpy()).all(axis=1) & (numbers[["N_params_B", "D_tokens_B", "val_loss"]] > 0).all(axis=1).to_numpy()
    if q:
        mask &= numbers.Q_score.between(0, 1).to_numpy()
    rejected = pd.DataFrame({"source_file": source, "row_number": np.flatnonzero(~mask)+2,
                             "status": "rejected_corrupt", "reason_code": "invalid_N_D_L_or_Q"})
    return frame.loc[mask].copy(), rejected


def load_package(path: Path) -> tuple[pd.DataFrame, dict, dict]:
    """在 ZIP 内逐文件核验清单；禁止把包内脚本当执行指令。"""
    with ZipFile(path) as archive:
        package = json.loads(archive.read(PACKAGE_MANIFEST))
        for entry in package["files"]:
            name = "团队交接包_问题一接口/" + entry["path"]
            payload = archive.read(name)
            if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
                raise ValueError(f"交接包文件完整性失败: {name}")
        raw = archive.read(INTERFACE)
        manifest = json.loads(archive.read(INTERFACE_MANIFEST))
    if digest(raw) != manifest["interface_sha256"]:
        raise ValueError("Q1 冻结接口与 manifest SHA-256 不符")
    frame = pd.read_csv(io.BytesIO(raw), compression="gzip")
    coords, targets = manifest["coordinate_space"], manifest["loss_targets"]
    expected = ["split", "recipe_id"] + [f"p_{x}" for x in coords] + [f"h_{x}" for x in targets] + ["h_p_eq", "hull13", "hull14", "hull17"]
    if len(coords) != 13 or len(targets) != 13 or len(set(coords)) != 13 or len(set(targets)) != 13:
        raise ValueError("冻结接口坐标或目标不完整")
    if list(frame.columns) != expected or len(frame) != manifest["n_rows"] or frame.duplicated(["split", "recipe_id"]).any():
        raise ValueError("冻结接口字段、行数或配方键异常")
    numeric = frame[expected[2:]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("冻结接口存在非有限数")
    p = frame[[f"p_{x}" for x in coords]].to_numpy(float)
    h = frame[[f"h_{x}" for x in targets]].to_numpy(float)
    # 包内检验配方坐标保留三位小数；13 个坐标的行和实测可到 1.002。
    # 这里只给 13 次舍入的最坏上界留容差，不改写冻结的坐标或 h_p。
    if (p < -1e-10).any() or (p.sum(axis=1) > 1 + 13*.0005 + 1e-10).any():
        raise ValueError("冻结接口配比坐标非法")
    if not np.allclose(h.mean(axis=1), frame.h_p_eq.to_numpy(float), atol=1e-9, rtol=0):
        raise ValueError("等权 h_p 与 13 目标不一致")
    ref = frame.loc[frame.split == "p0 (A4 mean)"]
    if len(ref) != 1 or not np.allclose(ref.iloc[0][[f"p_{x}" for x in coords]].to_numpy(float), manifest["p0_13coords"], atol=1e-9):
        raise ValueError("p0 与 manifest 不一致")
    if np.max(np.abs(ref[[f"h_{x}" for x in targets]+["h_p_eq"]].to_numpy(float))) > 1e-12:
        raise ValueError("p0 的相对响应不为零")
    for col in ("hull13", "hull14", "hull17"):
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} 非二值")
    if ((frame.hull17 > frame.hull14) | (frame.hull14 > frame.hull13)).any():
        raise ValueError("凸包包含关系错误")
    for split, claim in manifest.get("hull_counts", {}).items():
        part = frame.loc[frame.split == split]
        parsed = re.findall(r"(\d+)/(\d+)", claim)
        actual = [(int(part[col].sum()), len(part)) for col in ("hull13", "hull14", "hull17")]
        if len(parsed) != 3 or [(int(a),int(b)) for a,b in parsed] != actual:
            raise ValueError(f"{split} 凸包计数与 manifest 不一致")
    return frame, manifest, {"zip_sha256": digest(path.read_bytes()), "interface_sha256": digest(raw),
                              "interface_rows": len(frame), "checked_package_files": len(package["files"]),
                              "coordinate_sum_max": float(p.sum(axis=1).max()),
                              "coordinate_rounding_tolerance": 13*.0005,
                              "source_coefficients_sha256_claimed": manifest["source_coefficients_sha256"],
                              "source_coefficients_independently_verified": False}


def score(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(np.column_stack((y, p))).all():
        raise ValueError("评测输入非法")
    ss = float(np.sum((y-y.mean())**2))
    rho = spearmanr(y, p).statistic if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {"n": len(y), "mae": float(np.mean(np.abs(p-y))), "rmse": float(np.sqrt(np.mean((p-y)**2))),
            "r2": float(1-np.sum((p-y)**2)/ss) if ss > 0 else None,
            "spearman": float(rho) if np.isfinite(rho) else None,
            "mean_bias": float(np.mean(p-y))}


def group_cv(frame: pd.DataFrame, q: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if q:
        groups = frame.N_params_B.astype(str) + "|" + frame.D_tokens_B.astype(str)
    else:
        groups = frame.N_params_B.astype(str)
    folds = GroupKFold(n_splits=5 if q else groups.nunique())
    rows, preds = [], []
    for k, (train, test) in enumerate(folds.split(frame, groups=groups)):
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise AssertionError("分组交叉验证泄漏")
        tr, te = frame.iloc[train], frame.iloc[test]
        models = ["M0", "MQ"] if q else ["M0"]
        for name in models:
            fitted = fit(tr.N_params_B, tr.D_tokens_B, tr.val_loss,
                         tr.Q_score if name == "MQ" else None, seed=seed+k, starts=2)
            pred = predict(fitted.params, te.N_params_B.to_numpy(), te.D_tokens_B.to_numpy(),
                           te.Q_score.to_numpy() if q else None)
            rows.append({"fold": k, "model": name, "train_n": len(tr), "test_n": len(te),
                         "train_groups": groups.iloc[train].nunique(), "test_groups": groups.iloc[test].nunique(),
                         **score(te.val_loss, pred)})
            preds.extend({"row_number": int(idx)+2, "fold": k, "model": name, "observed": float(obs), "predicted": float(est)}
                         for idx, obs, est in zip(te.index, te.val_loss, pred))
    return pd.DataFrame(rows), pd.DataFrame(preds)


def bootstrap(frame: pd.DataFrame, q: bool, nrep: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 N 或 (N,D) 独立设计单元重抽样；不按 checkpoint / Q 点 iid 抽样。"""
    keys = frame.N_params_B.astype(str) + ("|" + frame.D_tokens_B.astype(str) if q else "")
    unique = keys.unique()
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(nrep):
        picked = rng.choice(unique, size=len(unique), replace=True)
        sample = pd.concat([frame.loc[keys == key] for key in picked], ignore_index=True)
        try:
            result = fit(sample.N_params_B, sample.D_tokens_B, sample.val_loss,
                         sample.Q_score if q else None, seed=seed+rep, starts=1)
            rows.append({"replicate": rep, "status": "success", **result.params})
        except (ValueError, RuntimeError) as exc:
            rows.append({"replicate": rep, "status": "failed", "reason": str(exc)})
    table = pd.DataFrame(rows)
    success = table.loc[table.status == "success"]
    if len(success) < max(20, int(.8*nrep)):
        raise RuntimeError("分组 bootstrap 有效次数不足")
    ci = pd.DataFrame([{"parameter": name, "q025": float(success[name].quantile(.025)),
                        "median": float(success[name].median()), "q975": float(success[name].quantile(.975)),
                        "successful_replicates": len(success)} for name in result.params])
    return table, ci


def prepare_b(root: Path, out: Path) -> tuple[dict, dict]:
    tables, inventory, rejects = {}, [], []
    source_manifest_path = root.parent / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_bytes = {item["file"].removeprefix("B_scaling_laws/"): item["bytes"]
                      for item in source_manifest if item.get("problem") == "B"}
    paths = {**{key: root / rel for key, rel in B_FILES.items()},
             **{f"B3_{p.stem}": p for p in sorted((root / "training_trajectories").glob("*.csv"))}}
    if len([k for k in paths if k.startswith("B3_")]) != 8:
        raise ValueError("B3 插值轨迹文件数量异常")
    if set(expected_bytes) != {str(path.relative_to(root)) for path in paths.values()}:
        raise ValueError("B 附件实物与 source_manifest 文件集合不一致")
    for key, path in paths.items():
        if path.stat().st_size != expected_bytes[str(path.relative_to(root))]:
            raise ValueError(f"{path} 字节数与 source_manifest 不符")
        data = pd.read_csv(path)
        tables[key] = data
        inventory.append({"attachment": key, "source_file": str(path), "bytes": path.stat().st_size,
                          "sha256": digest(path.read_bytes()), "rows": len(data), "columns": list(data.columns)})
        if key in {"B1", "B2", "B4", "B5", "B6", "B7", "B8", "B10"} or key.startswith("B3_"):
            clean, rejected = valid_rows(data, key, q=key in {"B6", "B7", "B8"})
            tables[key] = clean
            rejects.append(rejected)
    write_json(out / "audit/inventory.json", inventory)
    write_csv(out / "audit/rejected_rows.csv", pd.concat(rejects, ignore_index=True))
    return tables, {"inventory": inventory, "rejected_rows": int(sum(len(v) for v in rejects)),
                    "source_manifest_sha256": digest(source_manifest_path.read_bytes())}


def run(root: Path) -> dict:
    from .extensions import frozen_m2_transfers, paired_scale_analysis, qa_qb_scenarios
    from scipy.optimize import brentq
    root = root.resolve()
    config_path = root / "configs/q2_v7/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = (root / config["output_dir"]).resolve()
    if out != root / "results/q2_v7":
        raise ValueError("q2_v7 输出路径必须严格隔离")
    out.mkdir(parents=True, exist_ok=True)
    qref, seed = config["quality_reference"], config["seed"]
    interface, q1_manifest, interface_audit = load_package(root / config["input_zip"])
    write_json(out / "audit/q1_interface.json", interface_audit)
    tables, audit = prepare_b(root / config["input_b_dir"], out)
    b1, b6, b7, b8 = (tables[key] for key in ("B1", "B6", "B7", "B8"))
    if b6.experiment_id.duplicated().any() or b7.experiment_id.duplicated().any():
        raise ValueError("B6/B7 实验 ID 不唯一")
    common = b6.merge(b7, on="experiment_id", suffixes=("_b6", "_b7"))
    check_cols = ["N_params_B", "D_tokens_B", "Q_score", "val_loss"]
    if any(not np.allclose(common[f"{c}_b6"], common[f"{c}_b7"], rtol=0, atol=1e-10) for c in check_cols):
        raise ValueError("B6/B7 重叠 ID 内容冲突")
    b7_new = b7.loc[~b7.experiment_id.isin(b6.experiment_id)].copy()
    if len(b7_new) == 0:
        raise ValueError("B7 无真正新增留出记录")
    write_csv(out / "audit/b7_new_ids.csv", b7_new[["experiment_id"]])
    evidence_levels = {"B1": "direct_fit", "B2": "semi_synthetic_transfer", "B4": "public_reported_cross_family",
                       "B5": "published_cross_source", "B6": "semi_synthetic_calibration", "B7": "semi_synthetic_holdout_after_dedup",
                       "B8": "semi_synthetic_stress_by_provenance", "B9": "large_model_metadata_only",
                       "B10": "estimated_extrapolation_reference", "B11": "model_family_metadata_only",
                       "B12": "checkpoint_metadata_only"}
    write_csv(out / "audit/evidence_levels.csv", pd.DataFrame([{"attachment": key,
              "evidence": "interpolated_consistency" if key.startswith("B3_") else evidence_levels[key],
              "rows_valid": len(frame)} for key,frame in tables.items()]))
    write_json(out / "audit/summary.json", {"valid_rows": {k: len(v) for k,v in tables.items()},
               "rejected_rows": audit["rejected_rows"], "b6_b7_exact_overlap": len(common),
               "b7_new": len(b7_new), "b8_provenance": b8.data_type.value_counts().to_dict(),
               "Q_A_Q_B_relation": "unidentified; no paired quality anchors"})

    # B1：按完整模型规模轨迹留出，不随机分 checkpoint。
    b1_cv, b1_oof = group_cv(b1, False, seed)
    write_csv(out / "baseline/group_cv.csv", b1_cv)
    write_csv(out / "baseline/oof_predictions.csv", b1_oof)
    b1_fit = fit(b1.N_params_B, b1.D_tokens_B, b1.val_loss, seed=seed)
    write_json(out / "baseline/fit.json", {"params": b1_fit.params, "attempts": b1_fit.attempts,
               "in_sample": score(b1.val_loss, predict(b1_fit.params, b1.N_params_B, b1.D_tokens_B)),
               "evidence": "direct_fit_B1"})
    boot_b1, ci_b1 = bootstrap(b1, False, config["bootstrap_replicates"], seed)
    write_csv(out / "baseline/bootstrap.csv", boot_b1)
    write_csv(out / "baseline/parameter_ci.csv", ci_b1)
    residual = b1[["N_params_B", "D_tokens_B", "val_loss"]].copy()
    residual["prediction"] = predict(b1_fit.params, b1.N_params_B, b1.D_tokens_B)
    residual["residual"] = residual.prediction - residual.val_loss
    write_csv(out / "baseline/residuals.csv", residual)

    # B2/B3/B4/B5 仅外部比较；B3 为插值轨迹。来源偏移仅作诊断，不回写 B1 参数。
    transfers = []
    for key in ["B2", *sorted(k for k in tables if k.startswith("B3_")), "B4", "B5"]:
        data = tables[key]
        pred = predict(b1_fit.params, data.N_params_B, data.D_tokens_B)
        evidence = "interpolated_consistency" if key.startswith("B3_") else "semi_synthetic_transfer" if key == "B2" else "cross_source_reference"
        groups = data.groupby("family") if "family" in data else [("all", data)]
        for family, part in groups:
            ix = part.index.to_numpy()
            p = predict(b1_fit.params, part.N_params_B, part.D_tokens_B)
            offset = float((part.val_loss.to_numpy()-p).mean())
            transfers.append({"source": key, "family": family, "evidence": evidence,
                              "source_offset_diagnostic": offset, **score(part.val_loss, p),
                              "r2_after_source_offset": score(part.val_loss, p+offset)["r2"]})
    write_csv(out / "transfer/metrics.csv", pd.DataFrame(transfers))

    # B6 同样的分组折比较 M0 与 MQ；B7-new 只在模型冻结后使用。
    quality_cv, quality_oof = group_cv(b6, True, seed)
    write_csv(out / "quality/group_cv.csv", quality_cv)
    write_csv(out / "quality/oof_predictions.csv", quality_oof)
    qfits = {name: fit(b6.N_params_B, b6.D_tokens_B, b6.val_loss,
                       b6.Q_score if name == "MQ" else None, seed=seed)
             for name in ("M0", "MQ")}
    write_json(out / "quality/fits.json", {name: {"params": f.params, "attempts": f.attempts,
               "in_sample": score(b6.val_loss, predict(f.params, b6.N_params_B, b6.D_tokens_B, b6.Q_score)),
               "evidence": "semi_synthetic_calibration_B6"} for name, f in qfits.items()})
    heldout = []
    heldout_pred = {}
    for name, fitted in qfits.items():
        p = predict(fitted.params, b7_new.N_params_B, b7_new.D_tokens_B, b7_new.Q_score)
        heldout_pred[name] = np.asarray(p, float)
        heldout.append({"model": name, "split": "B7_new", "evidence": "semi_synthetic_held_out", **score(b7_new.val_loss, p)})
    write_csv(out / "quality/b7_new_metrics.csv", pd.DataFrame(heldout))
    b7_groups = (b7_new.N_params_B.astype(str)+"|"+b7_new.D_tokens_B.astype(str)).to_numpy()
    unique_groups = np.unique(b7_groups)
    rng = np.random.default_rng(seed)
    diff = []
    truth = b7_new.val_loss.to_numpy(float)
    for rep in range(1000):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        positions = np.concatenate([np.flatnonzero(b7_groups == key) for key in sampled])
        rmse0 = float(np.sqrt(np.mean((heldout_pred["M0"][positions]-truth[positions])**2)))
        rmseq = float(np.sqrt(np.mean((heldout_pred["MQ"][positions]-truth[positions])**2)))
        diff.append({"replicate": rep, "rmse_M0_minus_MQ": rmse0-rmseq})
    write_csv(out / "quality/b7_new_difference_bootstrap.csv", pd.DataFrame(diff))
    boot_q, ci_q = bootstrap(b6, True, config["bootstrap_replicates"], seed)
    write_csv(out / "quality/bootstrap.csv", boot_q)
    write_csv(out / "quality/parameter_ci.csv", ci_q)
    b8_rows = []
    for provenance, part in b8.groupby("data_type"):
        p = predict(qfits["MQ"].params, part.N_params_B, part.D_tokens_B, part.Q_score)
        b8_rows.append({"provenance": provenance, "evidence": "semi_synthetic_stress_test", **score(part.val_loss, p)})
    write_csv(out / "quality/b8_stress.csv", pd.DataFrame(b8_rows))
    direction = []
    for source, data in (("B6", b6), ("B8_calibrated", b8.loc[b8.data_type=="calibrated"]),
                         ("B8_extrapolated", b8.loc[b8.data_type=="extrapolated"])):
        for (n, d), part in data.groupby(["N_params_B", "D_tokens_B"]):
            if part.Q_score.nunique() < 2:
                continue
            x, y = part.Q_score.to_numpy(float), part.val_loss.to_numpy(float)
            slope = float(np.polyfit(x, y, 1)[0])
            direction.append({"source": source, "N_params_B": n, "D_tokens_B": d,
                              "Q_L_slope_within_grid": slope, "n_quality_levels": len(part)})
    direction_table = pd.DataFrame(direction)
    write_csv(out / "quality/q_direction_by_grid.csv", direction_table)
    write_json(out / "quality/q_direction_summary.json", {s: {"grids": len(g),
               "positive_slopes": int((g.Q_L_slope_within_grid>0).sum()),
               "negative_slopes": int((g.Q_L_slope_within_grid<0).sum())}
               for s,g in direction_table.groupby("source")})
    amplitudes = b6.groupby(["N_params_B", "D_tokens_B"]).agg(q_amplitude=("val_loss", lambda s: s.max()-s.min())).reset_index()
    write_csv(out / "quality/q_amplitudes.csv", amplitudes)

    # 冻结接口 13 目标逐一保留；等权只作预设汇总。λ 未由 B 拟合。
    valid_interface = interface.loc[interface.split != "p0 (A4 mean)"].copy()
    n0, d0, q0 = (config["workpoint"][k] for k in ("N_params_B", "D_tokens_B", "Q_B"))
    scenario_rows = []
    for form in ("A", "B"):
        for lam in config["lambda_scenarios"]:
            for theta in config["theta_scenarios"]:
                for qs in config["scenario_quality_levels"]:
                    estimates = predict(qfits["MQ"].params, n0, d0, qs, hp=valid_interface.h_p_eq.to_numpy(), lam=lam, form=form, theta=theta)
                    base = float(predict(qfits["MQ"].params, n0, d0, qs))
                    scenario_rows.extend({"split": row.split, "recipe_id": int(row.recipe_id), "form": form,
                                          "lambda_p_assumed": lam, "theta_Qp_assumed": theta, "Q_B": qs,
                                          "h_p_eq": float(row.h_p_eq), "hull13": int(row.hull13), "hull17": int(row.hull17),
                                          "L_scenario": float(est), "relative_change": float(est/base-1),
                                          "evidence": "imported_Q1_plus_unidentified_transfer_scenario"}
                                         for row, est in zip(valid_interface.itertuples(), estimates))
    scenario = pd.DataFrame(scenario_rows)
    write_csv(out / "scenario/predictions.csv", scenario)
    envelope = scenario.groupby(["split", "recipe_id", "Q_B"]).agg(L_min=("L_scenario", "min"), L_max=("L_scenario", "max"),
                           relative_min=("relative_change", "min"), relative_max=("relative_change", "max"),
                           hull17=("hull17", "first")).reset_index()
    envelope["evidence"] = "scenario_range_not_confidence_interval"
    write_csv(out / "scenario/envelope.csv", envelope)
    target_cols = [f"h_{x}" for x in q1_manifest["loss_targets"]]
    write_csv(out / "scenario/q1_target_responses.csv", valid_interface[["split", "recipe_id", *target_cols, "h_p_eq", "hull13", "hull14", "hull17"]])
    loo = []
    for target in q1_manifest["loss_targets"]:
        subset = [c for c in target_cols if c != f"h_{target}"]
        v = valid_interface[subset].mean(axis=1)
        loo.append({"omitted_target": target, "pearson_with_equal_weight": float(np.corrcoef(v, valid_interface.h_p_eq)[0,1]),
                    "p95": float(v.quantile(.95))})
    write_csv(out / "scenario/leave_one_target_out.csv", pd.DataFrame(loo))
    m2_transfer_audit = frozen_m2_transfers(root / config["input_zip"], q1_manifest, interface, out,
                                             qfits["MQ"].params, n0, d0, q0)
    regmix = paired_scale_analysis(root, out, q1_manifest["loss_targets"], seed)
    qa_audit = qa_qb_scenarios(root, out)

    # 局部导数、等损失替代与固定算力配置，统一使用 B6 MQ 参数。
    params = qfits["MQ"].params
    effects = []
    for n in (.07, .1, .3, 1., 3., 10., 30.):
        for d in (10., 50., 150., 500., 2000.):
            for q in (.2, .35, .5, .65, .8):
                effects.append({"N_params_B": n, "D_tokens_B": d, "Q_B": q,
                                "support": "within_B6_axis_ranges" if (b6.N_params_B.min() <= n <= b6.N_params_B.max() and b6.D_tokens_B.min() <= d <= b6.D_tokens_B.max()) else "outside_B6_axis_ranges",
                                "evidence": "model_local_semi_synthetic", **derivatives(params, n, d, q)})
    effect_table = pd.DataFrame(effects)
    write_csv(out / "effects/workpoint_grid.csv", effect_table)
    write_json(out / "effects/grid_summary.json", {"workpoints": len(effect_table),
               "all_dL_dN_negative": bool((effect_table.dL_dN<0).all()),
               "all_dL_dD_negative": bool((effect_table.dL_dD<0).all()),
               "all_dL_dQ_negative": bool((effect_table.dL_dQ<0).all()),
               "quality_marginal_larger_than_N": int((effect_table.dL_dQ.abs()*effect_table.Q_B > effect_table.dL_dN.abs()*effect_table.N_params_B).sum()),
               "within_B6_axis_ranges": int((effect_table.support=="within_B6_axis_ranges").sum())})
    target_loss = float(predict(params, n0, d0, q0))
    q1 = q0+.1
    n_equivalent = float(brentq(lambda n: float(predict(params,n,d0,q1))-target_loss, 1e-5, n0))
    write_json(out / "effects/finite_quality_substitution.json", {"baseline_N_B": n0, "D_tokens_B": d0,
               "Q_before": q0, "Q_after": q1, "N_equivalent_B": n_equivalent,
               "N_reduction_B": n0-n_equivalent, "target_L": target_loss,
               "evidence": "conditional_model_substitution_from_B6_semi_synthetic"})
    opt_rows = []
    for budget in config["compute_budgets_1e21"]:
        for q in (.3, .5, .7):
            opt_rows.append({"evidence": "model_analytic_with_calibration_bounds",
                             **compute_optimum(params, budget, q, config["compute_bounds"])})
    write_csv(out / "compute/optima.csv", pd.DataFrame(opt_rows))
    mix_opt = []
    ordered = valid_interface.sort_values("h_p_eq").reset_index(drop=True)
    for quantile in (.05, .5, .95):
        ix = int(round(quantile*(len(ordered)-1)))
        chosen = ordered.iloc[ix]
        for form in ("A", "B"):
            for lam in config["lambda_scenarios"]:
                for budget in config["compute_budgets_1e21"]:
                    for qs in config["scenario_quality_levels"]:
                        mix_opt.append({"recipe_split": chosen["split"], "recipe_id": int(chosen.recipe_id),
                                        "h_quantile_reference": quantile, "h_p_eq": float(chosen.h_p_eq),
                                        "hull17": int(chosen.hull17), "form": form, "lambda_p_assumed": lam,
                                        "evidence": "imported_Q1_compute_allocation_scenario",
                                        **compute_optimum(params, budget, qs, config["compute_bounds"],
                                                          hp=float(chosen.h_p_eq), lam=lam, form=form)})
    write_csv(out / "compute/mixture_scenarios.csv", pd.DataFrame(mix_opt))
    quality_gain = target_loss - float(predict(params, n0, d0, q1))
    equal_cost = []
    for shape in ("exponential", "power", "logarithmic"):
        extra_c = quality_incremental_cost_1e21(d0, q0, q1, shape)
        extra_n = extra_c/(.006*d0)
        scale_gain = target_loss - float(predict(params, n0+extra_n, d0, q0))
        equal_cost.append({"quality_cost_shape_assumed": shape, "Q_before": q0, "Q_after": q1,
                           "D_tokens_B": d0, "N_baseline_B": n0, "additional_compute_1e21": extra_c,
                           "same_compute_extra_N_B": extra_n, "quality_L_reduction": quality_gain,
                           "scale_L_reduction": scale_gain, "quality_to_scale_gain_ratio": quality_gain/scale_gain,
                           "evidence": "appendix_B_compute_cost_scenario_not_money_or_joint_validation"})
    write_csv(out / "compute/equal_flops_quality_vs_scale.csv", pd.DataFrame(equal_cost))
    # Bootstrap 参数的工作点输出区间，不把情景跨度误作 CI。
    uncertainty = []
    for row in boot_q.loc[boot_q.status == "success"].itertuples(index=False):
        bp = {name: getattr(row, name) for name in params}
        v = derivatives(bp, n0, d0, q0)
        uncertainty.append({"replicate": row.replicate, "dlogN_dQ": v["dlogN_dQ_at_fixed_L_D"],
                            "dL_dQ": v["dL_dQ"]})
    write_csv(out / "effects/bootstrap.csv", pd.DataFrame(uncertainty))
    opt_uncertainty = []
    for row in boot_q.loc[boot_q.status == "success"].itertuples(index=False):
        bp = {name: getattr(row, name) for name in params}
        for budget in config["compute_budgets_1e21"]:
            for q in (.3, .5, .7):
                try:
                    opt_uncertainty.append({"replicate": row.replicate, **compute_optimum(bp, budget, q, config["compute_bounds"])})
                except ValueError:
                    pass
    write_csv(out / "compute/bootstrap_optima.csv", pd.DataFrame(opt_uncertainty))

    # B9 无 Loss；B10 是估算值，只作外推参考。
    b10 = tables["B10"]
    b10_pred = predict(b1_fit.params, b10.N_params_B, b10.D_tokens_B)
    write_json(out / "extrapolation/b9_b10.json", {"B9_metadata_rows": len(tables["B9"]),
               "B10_estimated_rows": len(b10), "B10_vs_B1_baseline": score(b10.val_loss, b10_pred),
               "evidence": "estimated_consistency_only_not_independent_validation"})

    report = ["# 问题二 Q2 V7 独立实验报告", "", "## 数据与证据", "",
              "本实验从原始 B 附件、本次 ZIP 的 Q1 冻结 M2 接口读取主输入；另为文档第 4.5 节单独读取四个附件 A RegMix 表作跨规模配对诊断，并只读 Q1 已有域级质量输出。未读取旧版 Q2 结果。",
              f"B1 有效 {len(b1)} 行，按 {b1.N_params_B.nunique()} 个模型规模轨迹分组留出；B6 有效 {len(b6)} 行。",
              f"B6/B7 同 ID 且内容一致 {len(common)} 行；B7 新增留出 {len(b7_new)} 行。损坏隔离 {audit['rejected_rows']} 行。",
              "B2 与 B6–B8 为半合成，B3 为插值，B10 为估算；这些不能写成独立真实实验。", "",
              "N_params_B 与 D_tokens_B 均以十亿计，C_1e21≈0.006×N_B×D_B。所有来源的证据等级逐文件列于 `audit/evidence_levels.csv`。", "",
              "## 模型估计与验证", "",
              f"B1 经典律参数：`{json.dumps(b1_fit.params, ensure_ascii=False)}`。", 
              f"B1 分组留出 RMSE 中位数：{b1_cv.rmse.median():.6f}。",
              f"B6 双通道质量律参数：`{json.dumps(params, ensure_ascii=False)}`。",
              f"B6 分组 CV 平均 RMSE：M0={quality_cv.loc[quality_cv.model=='M0','rmse'].mean():.6f}；MQ={quality_cv.loc[quality_cv.model=='MQ','rmse'].mean():.6f}。",
              f"B7-new 留出 RMSE：M0={heldout[0]['rmse']:.6f}；MQ={heldout[1]['rmse']:.6f}。", "",
              f"B7-new 按 (N,D) 网格重抽样的 RMSE 改善 95% 区间：{np.quantile([v['rmse_M0_minus_MQ'] for v in diff],[.025,.975]).tolist()}。", "",
              "B1 参数与 B6 参数各自有按独立组重抽样的区间，见 `baseline/parameter_ci.csv` 和 `quality/parameter_ci.csv`；二者不混用。",
              "B2/B3/B4/B5 的分来源误差、秩相关和仅作诊断的来源偏移后 R² 见 `transfer/metrics.csv`。B3 插值轨迹不算独立验证，B2 为半合成。",
              f"B8 方向压力测试：B6 有 {int((direction_table.loc[direction_table.source=='B6','Q_L_slope_within_grid']<0).sum())} 个负斜率网格；B8 calibrated / extrapolated 分别有 {int((direction_table.loc[direction_table.source=='B8_calibrated','Q_L_slope_within_grid']>0).sum())} / {int((direction_table.loc[direction_table.source=='B8_extrapolated','Q_L_slope_within_grid']>0).sum())} 个正斜率网格。保留原始 Q_B 方向，不翻转 B8。",
              f"B8 calibrated / extrapolated 的 MQ RMSE 分别为 {b8_rows[0]['rmse']:.4f} / {b8_rows[1]['rmse']:.4f}；B8 不参与 B6 模型选择。", "",
              "## 接口与广义律", "",
              f"Q1 冻结接口 SHA-256：`{interface_audit['interface_sha256']}`；13 目标与 p0 零点已核验。",
              "主情景 Form A：`L=E−E1(Q_B−Qref)+A N^(−α) exp[−ρN(Q_B−Qref)]+B D^(−β) exp[−ρD(Q_B−Qref)+λp h_p]`。",
              "Form B 对两个可约项均施加配比因子，只作结构敏感性。λp=0/0.5/1/1.5、θQp=−0.5/0/+0.5 均为预设情景，并未由附件 B 估计。",
              f"独立 RegMix 配对诊断：{regmix['n_paired_recipes']} 个 1M/60M 同配方，聚合绝对 Loss 斜率 {regmix['absolute_slope_aggregate']:.4f}，log Loss 斜率 {regmix['log_slope_aggregate']:.4f}；只有两个代理规模点，不用于估计跨源 λp。",
              f"只读 M2 系数重建接口最大误差 {m2_transfer_audit['interface_reconstruction_max_abs_error']:.3g}；生成 {m2_transfer_audit['feasible_directed_transfers']} 个可行 14 组份额转移情景，新转移点的 A4 训练支持未验证。",
              f"冻结接口原始 17 维凸包计数：{valid_interface.groupby('split').hull17.agg(['sum','count']).to_dict('index')}。凸包内外还受坐标维度影响，三档标记均保留在接口响应表。",
              f"13 目标等权汇总的留一目标 Pearson 最低为 {min(v['pearson_with_equal_weight'] for v in loo):.4f}，对应目标 {min(loo, key=lambda v:v['pearson_with_equal_weight'])['omitted_target']}；目标权重会影响情景量级。",
              f"按 Q_B={config['scenario_quality_levels']}、λp、θQp、Form A/B 扫描得到 {len(scenario)} 行情景，逐配方包络见 `scenario/envelope.csv`；包络不是置信区间。",
              f"Q1 域级质量只给 {qa_audit.get('mapped_direct_or_near_direct', 0)} 个 direct/near_direct 映射做量尺情景；{qa_audit.get('excluded_inferred', '未知')} 个 inferred 映射不算完整配方质量。Q_A 与 Q_B 缺少同一样本锚点，映射不可识别；质量×配比交互及来源偏移亦不可联合估计。",
              "冻结接口中 A6/A7 和 A8/A9 的 17 维原始凸包内点数需结合情景表 `hull17` 解释，不把凸包外直接判为数据错误。", "",
              "## 边际效应、资源配置与外推", "",
              f"175 点网格局部导数、等损失替代率及 C≈6ND 的解析与有界最优见 CSV；Q_B 从 {q0} 提高至 {q1}、固定 D 和 Loss 时，N 从 {n0:.3f}B 降至 {n_equivalent:.3f}B。其数值依赖 B6 半合成校准及所列工作点、边界。",
              f"175 点网格中 N、D、Q 的边际 Loss 导数均为负的点数分别是 {int((effect_table.dL_dN<0).sum())}/{len(effect_table)}、{int((effect_table.dL_dD<0).sum())}/{len(effect_table)}、{int((effect_table.dL_dQ<0).sum())}/{len(effect_table)}；其中部分点超出 B6 的 N/D 轴范围，按 `support` 单列。",
              f"固定算力无配比解析/有界解共 {len(opt_rows)} 点，含配比 Form A/B 情景共 {len(mix_opt)} 点；所设边界为配置中的 B6 校准轴范围近似，不代表联合设计凸包。",
              "按赛题附录 B 给出的指数、幂函数、对数三种质量成本函数，已在代表点计算等 FLOPs 提质与扩模比较，见 `compute/equal_flops_quality_vs_scale.csv`；成本函数参数是题设情景，不是实测货币价格。",
              f"口径核对：建模文档第 8 节给出的三项‘提质/扩模改善率’为 11.2、5.4、4.1；用同节的质量 Loss 降幅和题面附录 B 成本式，本版直接复算为 {[round(x['quality_to_scale_gain_ratio'],3) for x in equal_cost]}。原表比值与其列出的损失降幅不自洽，本版采用可复算结果。",
              f"B9 有 {len(tables['B9'])} 条规模元数据；B10 有 {len(b10)} 条估算 Loss，与 B1 基准的 R²={score(b10.val_loss,b10_pred)['r2']:.6f}，只属估算一致性，不能作为独立验证。", "",
              "## 复现与限制", "",
              "运行 `PYTHONPATH=src .venv/bin/python -m q2_v7.pipeline --root .`。配置、输入哈希、各产物哈希见 `metadata.json`。",
              "Q1 包内 M2 系数哈希与 manifest 一致，且可复算冻结接口；未以附件 A 重新拟合来证明来源真实性。",
              "交接说明规定 Q2 只读接口和 manifest；为实现可控份额转移，这里额外只读同一 ZIP 内已冻结、经 SHA-256 核验的 M2 系数，并单列审计。跨规模诊断按建模文档第 4.5 节的例外独立读取附件 A 四张小表，输入哈希单列。",
              "本仓库 Q1 域级质量输出不在交接 ZIP 内；其来源由文件 SHA-256 标记，未重新审计 Q1 原始文本。只有 direct/near_direct 域用于量尺情景。",
              "附录 B 成本函数参数来自赛题正文，未用实际货币账单校准；等 FLOPs 效率比不能解释为每元收益。",
              "没有 A/B 联合观测，广义律整体预测、λp、Q_A→Q_B 映射与结构位置不能称作已验证事实。", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    import numpy, pandas
    artifacts = {str(p.relative_to(out)): digest(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "metadata.json"}
    metadata = {"version": "q2_v7", "seed": seed, "config_sha256": digest(config_path.read_bytes()),
                "design_document_sha256": digest((root/"问题二完整建模思路_最终版.md").read_bytes()),
                "problem_statement_sha256": digest((root/"docs/算力约束下提升大语言模型能力的资源配置建模.docx").read_bytes()),
                "data_description_sha256": digest((root/"docs/数据说明.pdf").read_bytes()),
                "model_source_sha256": {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")},
                "python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "input_zip_sha256": interface_audit["zip_sha256"],
                "q1_m2_coefficients_sha256": m2_transfer_audit["coefficient_sha256"],
                "input_a_regmix_sha256": regmix["input_sha256"],
                "input_q1_quality_sha256": qa_audit.get("source_sha256"),
                "input_b_sha256": {v["attachment"]: v["sha256"] for v in audit["inventory"]},
                "source_manifest_sha256": audit["source_manifest_sha256"],
                "artifact_sha256": artifacts,
                "fit_sources": {"B1": ["B1"], "B6_M0_MQ": ["B6"]},
                "heldout_sources": ["B7_new"], "scenario_only_sources": ["Q1_M2", "B8", "B9", "B10"]}
    write_json(out / "metadata.json", metadata)
    return {"output": str(out), "B1": len(b1), "B6": len(b6), "B7_new": len(b7_new),
            "B6_B7_overlap": len(common), "rejected": audit["rejected_rows"], "artifacts": len(artifacts)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="问题二 V7 独立复现实验")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
