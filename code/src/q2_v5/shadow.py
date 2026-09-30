"""P13：核心冻结后只读历史结果，解释差异。"""

from __future__ import annotations

import json

import pandas as pd

from .common import ROOT, finalize, require_pass, stage_dir
from .core_robustness import verify_core_freeze


OLD_V2 = ROOT / "results/q2_scaling_v2"
OLD_V3 = ROOT / "results/q2_v3"


def run() -> dict:
    require_pass("P12")
    verify_core_freeze()  # 首次访问旧结果之前核验核心冻结。
    out = stage_dir("P13")
    out.mkdir(parents=True, exist_ok=True)
    p_v5 = json.loads((stage_dir("P02") / "final_parameters.json").read_text())["parameters"]
    p_v2 = json.loads((OLD_V2 / "b1_baseline/final_parameters.json").read_text())
    p_v3 = json.loads((OLD_V3 / "b1_baseline/final_parameters.json").read_text())["parameters"]
    v5_macro = json.loads((stage_dir("P02") / "macro_metrics.json").read_text())["rmse_macro"]
    v5_pooled = json.loads((stage_dir("P02") / "pooled_metrics.json").read_text())["rmse"]
    v3_macro = pd.read_csv(OLD_V3 / "b1_baseline/macro_group_metrics.csv").iloc[0].RMSE
    v3_folds = pd.read_csv(OLD_V3 / "b1_baseline/group_cv_metrics.csv")
    v3_pooled = float((v3_folds.RMSE.pow(2) * v3_folds.n).sum() / v3_folds.n.sum()) ** .5
    v5_choice = json.loads((stage_dir("P04") / "selection_decision.json").read_text())["selected_model"]
    v3_choice = json.loads((OLD_V3 / "quality_models/selected_model.json").read_text())["model_name"]
    v5_q = pd.read_csv(stage_dir("P04") / "candidate_model_metrics.csv")
    v3_q = json.loads((OLD_V3 / "quality_models/selected_model.json").read_text())["selection_evidence"]["pooled_CV"]
    v5_b7 = pd.read_csv(stage_dir("P05") / "b7_new_metrics.csv")
    v3_b7 = pd.read_csv(OLD_V3 / "locked_validation/b7_new_metrics.csv")
    v2_b7 = pd.read_csv(OLD_V2 / "quality_model/b7_new_metrics.csv")
    rows = []
    for name in p_v5:
        rows.append({"topic": "B1_parameter", "metric": name, "v5": p_v5[name],
                     "v3": p_v3.get(name), "v2": p_v2.get(name),
                     "attribution": "numerical_implementation", "comparison_limit": "same B1 units"})
    rows.extend([
        {"topic": "B1_validation", "metric": "macro_RMSE", "v5": v5_macro,
         "v3": v3_macro, "v2": None, "attribution": "metric_definition", "comparison_limit": "LOSO fold average"},
        {"topic": "B1_validation", "metric": "pooled_RMSE", "v5": v5_pooled,
         "v3": v3_pooled, "v2": None, "attribution": "metric_definition", "comparison_limit": "pooled OOF"},
        {"topic": "quality_selection", "metric": "selected_structure", "v5": v5_choice,
         "v3": v3_choice, "v2": "MQ", "attribution": "model_structure",
         "comparison_limit": "V5 and V3 use centered vs raw Q parameterization"},
        {"topic": "quality_selection", "metric": "B6_pooled_CV_RMSE", "v5": float(v5_q.set_index("model").at[v5_choice, "rmse"]),
         "v3": v3_q[v3_choice]["RMSE"], "v2": None, "attribution": "data_split",
         "comparison_limit": "fold definitions may differ"},
        {"topic": "B7_new", "metric": "selected_RMSE", "v5": float(v5_b7.iloc[0].rmse),
         "v3": float(v3_b7.iloc[0].RMSE), "v2": float(v2_b7.loc[v2_b7.model == "MQ", "RMSE"].iloc[0]),
         "attribution": "model_structure", "comparison_limit": "pre-validation frozen models"},
        {"topic": "mixture", "metric": "p0_source", "v5": "equal-share scenario; Q1 p0 absent",
         "v3": "Q1 reported best training recipe", "v2": "historical reference",
         "attribution": "q1_interface", "comparison_limit": "do not revise V5 p0 after freeze"},
        {"topic": "mixture", "metric": "Form_attachment", "v5": "Form A primary assumption; B sensitivity",
         "v3": "scenario comparison", "v2": "D-channel primary assumption",
         "attribution": "scenario_assumption", "comparison_limit": "not jointly identified"},
        {"topic": "effects", "metric": "elasticity_substitution", "v5": "observed B6/B7 workpoints",
         "v3": "historical workpoints", "v2": "historical workpoints",
         "attribution": "metric_definition", "comparison_limit": "numbers require matched N,D,Q,p"},
        {"topic": "stress", "metric": "B8_Q_direction", "v5": "150 positive groups, no Q flip",
         "v3": "B8 direction audited", "v2": "B8 stress only",
         "attribution": "evidence_reclassification", "comparison_limit": "different grouping summaries"},
        {"topic": "data_semantics", "metric": "B3_B10_B7", "v5": "interpolation / estimated reference / 90 new-Q",
         "v3": "historical labels", "v2": "historical labels",
         "attribution": "evidence_reclassification", "comparison_limit": "B3 and B10 are not independent direct validation"},
    ])
    pd.DataFrame(rows).to_csv(out / "shadow_comparison.csv", index=False)
    (out / "historical_difference_report.md").write_text(
        "# P13 历史影子比较\n\n"
        "本阶段在 V5 核心冻结后首次读取旧 Q2。B1 数值接近，说明同一输入下独立实现一致；"
        "这不参与 V5 模型选择。V3 与 V5 质量参数的 Q 中心化口径不同，不能直接逐系数比较。"
        "历史 p0 在旧 Q2 输出中可见，但不回填到 V5 冻结接口。"
        "旧版本边际效应工作点不同，数值并排解释需匹配 N,D,Q,p。"
        "差异分类见 shadow_comparison.csv。\n", encoding="utf-8")
    metadata = finalize("P13", [OLD_V2 / "b1_baseline/final_parameters.json",
                        OLD_V3 / "b1_baseline/final_parameters.json",
                        OLD_V3 / "quality_models/selected_model.json",
                        OLD_V3 / "locked_validation/b7_new_metrics.csv",
                        stage_dir("P12") / "core_freeze.sha256"],
                        ["旧版工作点和 Q 参数口径不同，部分结果仅作定性对照"],
                        ["历史比较不得回写 V5 主模型选择"],
                        ["metadata_only"])
    return {"rows": len(rows), "metadata": metadata}
