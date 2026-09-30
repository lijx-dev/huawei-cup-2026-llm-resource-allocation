"""附件 A 独立跨规模诊断与交接包只读 M2 份额转移情景。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .model import predict
from .pipeline import digest, write_csv, write_json


COEFF = "团队交接包_问题一接口/03_对照材料/mix_final_model_quad.csv"


def paired_scale_analysis(root: Path, out: Path, targets: list[str], seed: int) -> dict:
    """A 端新增分析：按 index 严格配对 1M/60M，不参与 B 端模型拟合。"""
    base = root / "data/real_attachments/A_data_value/regmix_tables"
    files = {key: base / f"{key}.csv" for key in ("test_mixture_1m", "test_mixture_60m", "test_pile_loss_1m", "test_pile_loss_60m")}
    tables = {key: pd.read_csv(path) for key, path in files.items()}
    mixcols = [c for c in tables["test_mixture_1m"].columns if c.startswith("train_the_pile_")]
    losscols = [f"metric/the_pile_{t}_val_loss" for t in targets]
    if len(mixcols) != 17 or len(losscols) != 13:
        raise ValueError("RegMix 字段数量异常")
    for key, table in tables.items():
        if table["index"].duplicated().any() or table["index"].isna().any():
            raise ValueError(f"{key} 的 index 不能一一连接")
        required = mixcols if "mixture" in key else losscols
        if not set(required).issubset(table.columns):
            raise ValueError(f"{key} 缺少配比或 Loss 字段")
        a = table[required].to_numpy(float)
        invalid = not np.isfinite(a).all() or ((a < 0).any() if "mixture" in key else (a <= 0).any())
        if invalid:
            raise ValueError(f"{key} 数值非法")
    m1 = tables["test_mixture_1m"].sort_values("index")
    m60 = tables["test_mixture_60m"].sort_values("index")
    if not np.array_equal(m1["index"], m60["index"]) or not np.array_equal(m1[mixcols].to_numpy(), m60[mixcols].to_numpy()):
        raise ValueError("1M/60M 配比不能精确配对")
    y1 = m1[["index"]].merge(tables["test_pile_loss_1m"][["index", *losscols]], on="index", validate="one_to_one")
    y60 = m60[["index"]].merge(tables["test_pile_loss_60m"][["index", *losscols]], on="index", validate="one_to_one")
    if len(y1) != len(m1) or len(y60) != len(m60):
        raise ValueError("RegMix 配方与 Loss 有缺失连接")
    x = y1[losscols].to_numpy(float)
    z = y60[losscols].to_numpy(float)
    def slope(a, b):
        aa, bb = a-a.mean(), b-b.mean()
        denominator = float(aa@aa)
        return float((aa@bb)/denominator) if denominator > 1e-14 else float("nan")
    xeq, zeq = x.mean(axis=1), z.mean(axis=1)
    rng = np.random.default_rng(seed)
    slopes, logslopes = [], []
    for _ in range(500):
        idx = rng.integers(0, len(xeq), size=len(xeq))
        slopes.append(slope(xeq[idx], zeq[idx]))
        logslopes.append(slope(np.log(xeq[idx]), np.log(zeq[idx])))
    slopes = np.asarray(slopes, float)
    logslopes = np.asarray(logslopes, float)
    if np.isfinite(slopes).sum() < 400 or np.isfinite(logslopes).sum() < 400:
        raise ValueError("配方分组 bootstrap 有效重复数不足")
    rows = []
    for i, target in enumerate(targets):
        rows.append({"target": target, "L_1m_mean": float(x[:,i].mean()), "L_60m_mean": float(z[:,i].mean()),
                     "absolute_slope": slope(x[:,i], z[:,i]),
                     "log_slope": slope(np.log(x[:,i]), np.log(z[:,i])),
                     "sd_ratio": float(z[:,i].std()/x[:,i].std())})
    write_csv(out / "regmix/target_slopes.csv", pd.DataFrame(rows))
    summary = {"evidence": "A_regmix_same_recipe_two_scale_observations",
               "n_paired_recipes": len(xeq), "n_targets": len(targets),
               "absolute_slope_aggregate": slope(xeq, zeq),
               "absolute_slope_recipe_bootstrap_95": list(np.nanquantile(slopes, [.025,.975])),
               "log_slope_aggregate": slope(np.log(xeq), np.log(zeq)),
               "log_slope_recipe_bootstrap_95": list(np.nanquantile(logslopes, [.025,.975])),
               "bootstrap_valid": int(np.isfinite(slopes).sum()),
               "mean_loss_ratio": float(zeq.mean()/xeq.mean()),
               "paired_recipe_max_abs_difference": float(np.max(np.abs(m1[mixcols].to_numpy()-m60[mixcols].to_numpy()))),
               "limitation": "仅 1M/60M 两个代理规模档；表内无 N、D，不能估计跨 A/B lambda_p 或连续衰减规律",
               "input_sha256": {key: digest(path.read_bytes()) for key, path in files.items()}}
    write_json(out / "regmix/summary.json", summary)
    return summary


def frozen_m2_transfers(zip_path: Path, manifest: dict, interface: pd.DataFrame,
                        out: Path, b_params: dict, n: float, d: float, q: float) -> dict:
    """只读 ZIP 内经校验系数，重算接口响应后在 p0 做可行的有向份额转移。"""
    with ZipFile(zip_path) as archive:
        payload = archive.read(COEFF)
    if digest(payload) != manifest["source_coefficients_sha256"]:
        raise ValueError("冻结 M2 系数 SHA-256 不符")
    c = pd.read_csv(io.BytesIO(payload), index_col=0)
    coords, targets, quad = manifest["coordinate_space"], manifest["loss_targets"], manifest["quadratic_domains"]
    if list(c.columns) != targets or set(c.index) != {"intercept", *coords, *(x+"^2" for x in quad)}:
        raise ValueError("M2 系数合同不一致")
    p0 = np.array(manifest["p0_13coords"], float)
    beta = c.loc[coords, targets].to_numpy(float)
    gamma = c.loc[[x+"^2" for x in quad], targets].to_numpy(float)
    qidx = [coords.index(x) for x in quad]
    def h(p):
        p = np.atleast_2d(np.asarray(p, float))
        return (p-p0)@beta + (p[:,qidx]**2-p0[qidx]**2)@gamma
    hp_interface = h(interface[[f"p_{x}" for x in coords]].to_numpy(float))
    expected = interface[[f"h_{x}" for x in targets]].to_numpy(float)
    error = float(np.max(np.abs(hp_interface-expected)))
    if error > 1e-8:
        raise ValueError(f"冻结系数未能复算接口响应，最大误差 {error}")
    p14 = np.r_[p0, 1-p0.sum()]
    names = [*coords, manifest["reference_domain"]]
    delta = .01
    rows = []
    for donor in range(14):
        for receiver in range(14):
            if donor == receiver:
                continue
            candidate = p14.copy()
            candidate[donor] -= delta
            candidate[receiver] += delta
            if candidate.min() < -1e-10 or abs(candidate.sum()-1) > 1e-10:
                continue
            response = h(candidate[:13])[0]
            hp_eq = float(response.mean())
            base = float(predict(b_params, n, d, q))
            scenario = float(predict(b_params, n, d, q, hp=hp_eq, lam=1))
            rows.append({"donor": names[donor], "receiver": names[receiver], "delta": delta,
                         "h_p_eq": hp_eq, "delta_L_form_A_lambda1": scenario-base,
                         "support_status": "training_design_support_unverified",
                         "evidence": "imported_Q1_M2_coefficient_scenario"})
    write_csv(out / "scenario/directed_transfers.csv", pd.DataFrame(rows))
    result = {"coefficient_sha256": digest(payload), "interface_reconstruction_max_abs_error": error,
              "p0_14_sum": float(p14.sum()), "feasible_directed_transfers": len(rows),
              "delta": delta, "warning": "14 组含 other；未对新转移点做 A4 训练设计支持验证，均为情景"}
    write_json(out / "scenario/m2_transfer_audit.json", result)
    return result


def qa_qb_scenarios(root: Path, out: Path) -> dict:
    """只读 Q1 已有域级输出；不把无共同锚点的 Q_A/Q_B 映射当估计。"""
    path = root / "results/q1/integration/projected_domain_quality.csv"
    if not path.exists():
        result = {"status": "unavailable", "reason": "Q1 domain quality output missing"}
        write_json(out / "scenario/qa_qb_audit.json", result)
        return result
    frame = pd.read_csv(path)
    required = {"mixture_domain", "a16_type", "quality_proxy_main"}
    if not required.issubset(frame.columns) or frame.mixture_domain.duplicated().any():
        raise ValueError("Q1 域级评分接口不满足合同")
    acceptable = frame.loc[frame.a16_type.isin(["direct", "near_direct"])].copy()
    if acceptable.empty or not np.isfinite(acceptable.quality_proxy_main.to_numpy(float)).all():
        raise ValueError("Q1 可用域级评分缺失或非法")
    lo, hi = acceptable.quality_proxy_main.min(), acceptable.quality_proxy_main.max()
    if hi <= lo:
        raise ValueError("Q_A 场景映射参考尺度为常数")
    acceptable["Q_A"] = acceptable.quality_proxy_main
    acceptable["Q_A_reference_rank"] = (acceptable.Q_A-lo)/(hi-lo)
    rows = []
    for gamma in (.5, 1., 2.):
        for row in acceptable.itertuples():
            u = row.Q_A_reference_rank
            rows.append({"mixture_domain": row.mixture_domain, "mapping_type": row.a16_type,
                         "Q_A": row.Q_A, "u": u, "gamma_assumed": gamma,
                         "Q_B_scenario_unit_interval": u**gamma,
                         "Q_B_scenario_within_B6_range": .1+.9*u**gamma,
                         "evidence": "unidentified_quality_scale_mapping_scenario"})
    write_csv(out / "scenario/qa_qb_scenarios.csv", pd.DataFrame(rows))
    result = {"status": "partial_scenarios_only", "source_file": str(path.relative_to(root)),
              "source_sha256": digest(path.read_bytes()), "mapped_direct_or_near_direct": len(acceptable),
              "excluded_inferred": int((frame.a16_type=="inferred").sum()),
              "complete_recipe_quality": False,
              "reason": "11 个推断映射域未作为正式质量观测；A/B 没有共同质量锚点，gamma 不能估计"}
    write_json(out / "scenario/qa_qb_audit.json", result)
    return result
