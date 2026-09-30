"""P14：最终可追溯实验报告与一致性审计。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (BROOT, OUT, ROOT, config, finalize, require_pass,
                     sha, stage_dir, write_json)
from .core_robustness import verify_core_freeze
from .figures import create_figures
from .quality_models import freeze_digest


def markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    view = frame[columns].head(limit) if limit else frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = []
    for _, row in view.iterrows():
        values = [f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)
                  for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def consistency_audit(report_path: Path) -> dict:
    checks = {}
    p00 = json.loads((OUT / "00_manifest/input_hashes.json").read_text())
    checks["p00_input_hashes_unchanged"] = all(sha(ROOT / path) == value["sha256"] for path, value in p00.items())
    for stage_num in range(1, 15):
        if stage_num == 14:
            continue
        stage = f"P{stage_num:02d}"
        checks[f"{stage}_passed"] = require_pass(stage)["status"] in ("PASS", "PASS_WITH_SCENARIO_LIMITATION")
    inputs = []
    for stage_num in range(1, 14):
        metadata = require_pass(f"P{stage_num:02d}")
        inputs.extend(metadata["input_sha256"])
    checks["no_A_raw_direct_inputs"] = not any(path.startswith("data/real_attachments/A_data_value/") for path in inputs)
    p04inputs = require_pass("P04")["input_sha256"]
    checks["P04_no_B7_target_input"] = not any("supplementary_NQ_experiment_expanded" in path or "b7_new" in path for path in p04inputs)
    p04_source = ast.parse((ROOT / "src/q2_v5/quality_models.py").read_text())
    load_calls = [node for node in ast.walk(p04_source) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "load_b"]
    checks["P04_static_loader_only_B6"] = all(node.args and isinstance(node.args[0], ast.Constant)
                                                 and node.args[0].value == "B6" for node in load_calls)
    overlap = pd.read_csv(stage_dir("P01") / "b6_b7_overlap.csv")
    new = pd.read_csv(stage_dir("P01") / "b7_new.csv")
    checks["B7_new_90_no_overlap"] = len(new) == 90 and len(overlap) == 360 and not set(new.experiment_id) & set(overlap.experiment_id)
    evidence = pd.read_csv(stage_dir("P01") / "evidence_level.csv").set_index("attachment")
    checks["evidence_classification"] = (evidence.at["B3", "evidence_type"] == "interpolation"
                                           and all(evidence.at[x, "evidence_type"] == "semi_synthetic" for x in ("B6", "B7"))
                                           and evidence.at["B8", "evidence_type"] == "stress_test"
                                           and evidence.at["B10", "evidence_type"] == "estimated_reference"
                                           and evidence.at["B9", "evidence_type"] == "metadata_only")
    checks["macro_pooled_distinguished"] = (stage_dir("P02") / "macro_metrics.json").exists() and (stage_dir("P02") / "pooled_metrics.json").exists()
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    checks["QA_QB_and_lambda_unidentified"] = "NOT_IDENTIFIED" in contract["lambda_p"]["status"] and "NOT_IDENTIFIED" in contract["QA_to_QB"]
    checks["model_freeze_intact"] = (stage_dir("P04") / "model_freeze.sha256").read_text().strip() == freeze_digest(stage_dir("P04"))
    summary = json.loads((stage_dir("P05") / "validation_summary.json").read_text())
    post = json.loads((stage_dir("P05") / "post_validation_parameters.json").read_text())
    checks["B7_prevalidation_only"] = summary["uses_pre_validation_parameters"] and not post["used_for_holdout_metrics"]
    checks["core_freeze_intact"] = len(verify_core_freeze()["output_sha256"]) > 0
    checks["analytic_derivative_verified"] = pd.read_csv(stage_dir("P09") / "derivative_checks.csv").relative_error.max() < 1e-5
    checks["local_substitution_nonlinearity_saved"] = set(pd.read_csv(stage_dir("P10") / "substitution_nonlinearity.csv").delta_Q) == {.01, .05, .1}
    joint = pd.read_csv(stage_dir("P11") / "joint_response_by_target.csv")
    checks["joint_response_identity"] = bool(np.allclose(joint.local_joint_response, joint.Ljk - joint.Lj - joint.Lk + joint.L0))
    checks["report_19_sections"] = all(f"## {i} " in report_path.read_text(encoding="utf-8") for i in range(1, 20))
    checks = {name: bool(passed) for name, passed in checks.items()}
    critical = [name for name, passed in checks.items() if not passed]
    return {"checks": checks, "critical_failures": len(critical),
            "critical_failure_names": critical,
            "interpretation_limit": "文件系统无法证明其他进程的历史读取；本审计仅检查 V5 代码输入与产物"}


def run() -> dict:
    require_pass("P13")
    verify_core_freeze()
    out = stage_dir("P14")
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((stage_dir("P01") / "audit_summary.json").read_text())
    inventory = pd.read_csv(stage_dir("P01") / "attachment_inventory.csv")
    evidence = pd.read_csv(stage_dir("P01") / "evidence_level.csv")
    b1 = json.loads((stage_dir("P02") / "final_parameters.json").read_text())["parameters"]
    b1_macro = json.loads((stage_dir("P02") / "macro_metrics.json").read_text())
    b1_pool = json.loads((stage_dir("P02") / "pooled_metrics.json").read_text())
    b1_ci = pd.read_csv(stage_dir("P02") / "parameter_ci.csv")
    transfer = pd.read_csv(stage_dir("P03") / "validation_matrix.csv")
    candidate = pd.read_csv(stage_dir("P04") / "candidate_model_metrics.csv")
    decision = json.loads((stage_dir("P04") / "selection_decision.json").read_text())
    quality_ci = pd.read_csv(stage_dir("P04") / "parameter_ci.csv")
    b7 = pd.read_csv(stage_dir("P05") / "b7_new_metrics.csv")
    validation = json.loads((stage_dir("P05") / "validation_summary.json").read_text())
    stress = json.loads((stage_dir("P06") / "stress_summary.json").read_text())
    bridge = json.loads((stage_dir("P07") / "bridge_audit.json").read_text())
    contract = json.loads((stage_dir("P08") / "model_contract.json").read_text())
    marg = pd.read_csv(stage_dir("P09") / "marginal_effects.csv")
    elast = pd.read_csv(stage_dir("P09") / "elasticities.csv")
    sub_n = pd.read_csv(stage_dir("P10") / "q_n_substitution.csv")
    sub_d = pd.read_csv(stage_dir("P10") / "q_d_substitution.csv")
    nonlin = pd.read_csv(stage_dir("P10") / "substitution_nonlinearity.csv")
    pair = pd.read_csv(stage_dir("P11") / "delta_sensitivity.csv")
    robustness = pd.read_csv(stage_dir("P12") / "robustness_matrix.csv")
    mid = marg.loc[(marg.workpoint == "mid") & (marg.p_scenario == "p0") & (marg.lambda_p == 0)].iloc[0]
    emid = elast.loc[(elast.workpoint == "mid") & (elast.p_scenario == "p0") & (elast.lambda_p == 0)].iloc[0]
    npair = pair.loc[np.isclose(pair.delta, .02)]
    p = contract["quality_parameters"]
    param_rows = [{"parameter": key, "value": value,
                   "source": "B6+B7-new post-validation estimate", "status": "estimated_semi_synthetic",
                   "unit_or_note": "N,D in billions; Q_B raw B scale"} for key, value in p.items()]
    param_rows.extend([{"parameter": "lambda_p", "value": value,
                        "source": "scenario grid", "status": "NOT_IDENTIFIED",
                        "unit_or_note": "dimensionless"} for value in contract["lambda_p"]["values"]])
    pd.DataFrame(param_rows).to_csv(out / "parameter_table.csv", index=False)
    evidence.to_csv(out / "evidence_matrix.csv", index=False)
    key_rows = [
        {"metric": "B1_macro_LOSO_RMSE", "value": b1_macro["rmse_macro"], "source": "02_b1_baseline/macro_metrics.json"},
        {"metric": "B1_pooled_OOF_RMSE", "value": b1_pool["rmse"], "source": "02_b1_baseline/pooled_metrics.json"},
        {"metric": "B6_selected_model", "value": decision["selected_model"], "source": "04_quality_selection/selection_decision.json"},
        {"metric": "B6_selected_grouped_CV_RMSE", "value": float(candidate.set_index("model").at[decision["selected_model"], "rmse"]), "source": "04_quality_selection/candidate_model_metrics.csv"},
        {"metric": "B7_new_selected_RMSE", "value": float(b7.iloc[0].rmse), "source": "05_locked_validation/b7_new_metrics.csv"},
        {"metric": "B7_new_M0_RMSE", "value": float(b7.iloc[1].rmse), "source": "05_locked_validation/b7_new_metrics.csv"},
        {"metric": "B8_positive_Q_Loss_groups", "value": stress["b8_positive_groups"], "source": "06_stress_and_extrapolation/stress_summary.json"},
        {"metric": "mid_dlogN_dQ", "value": float(sub_n.set_index("workpoint").at["mid", "dlogN_dQ"]), "source": "09_effects/q_n_substitution.csv"},
        {"metric": "mid_dlogD_dQ", "value": float(sub_d.set_index("workpoint").at["mid", "dlogD_dQ"]), "source": "09_effects/q_d_substitution.csv"},
    ]
    pd.DataFrame(key_rows).to_csv(out / "key_results.csv", index=False)
    figures, figure_sources = create_figures(out)
    figures.to_csv(out / "figure_manifest.csv", index=False)
    figure_sources.to_csv(out / "figure_sources.csv", index=False)
    formula = ("L=E+A N^{-α}exp[-ρ_N(Q_B-Q_0)] + "
               "B D^{-β}exp[-ρ_D(Q_B-Q_0)]exp[λ_p h_p(p)] - E_1(Q_B-Q_0)")
    report = f"""# 问题二 Q2 V5：广义标度律实验报告

> 版本：Q2 V5 clean-room；seed=7；核心在 P12 冻结，P13 历史比较未参与模型选择。数值以本目录和各阶段 CSV/JSON 为准。

## 1 数据与证据边界

B1–B12 文件清单见 [P01 inventory](../01_audit/attachment_inventory.csv)。B1 为直接观测轨迹；B2、B6、B7 为半合成；B3 为检查点插值；B8 为半合成压力测试；B9/B11/B12 为元数据；B10 为估算参考。B4/B5 来源虽为公开资料，Loss 口径与 B1 严格可比性未获证明。B6={audit['B6_rows']}、B7={audit['B7_rows']}，按实验标识和 N/D/Q 自动核对得到 {audit['overlap_rows']} 个完全重叠实验及 B7-new={audit['B7_new_rows']}。B9 有 {audit['invalid_numeric_rows']} 个非法/缺失数值条目，详见 [P01 审计](../01_audit/audit_report.md)。

## 2 经典 Scaling Law

B1 拟合 `L₀(N,D)=E+A N^(-α)+B D^(-β)`，N、D 分别以十亿参数和十亿 token 计。多起点正参数拟合得到 E={b1['E']:.6g}、A={b1['A']:.6g}、B={b1['B']:.6g}、α={b1['alpha']:.6g}、β={b1['beta']:.6g}。按模型规模 Leave-One-Scale-Out 的 macro fold RMSE={b1_macro['rmse_macro']:.6g}，pooled OOF RMSE={b1_pool['rmse']:.6g}；前者对规模等权、后者对检查点等权，故可能不同。参数按模型规模重抽样的区间见 [P02 CI](../02_b1_baseline/parameter_ci.csv)，残差见 [P02 residuals](../02_b1_baseline/residuals.csv)。

## 3 跨来源验证

冻结 B1 参数后得到：

{markdown_table(transfer, ['attachment', 'interpretation_level', 'n', 'rmse', 'spearman', 'mean_bias'])}

B2 仅作趋势验证；B3 高一致性属于插值自洽，不是独立外部泛化；B4/B5 的偏差须结合不同 Loss 口径解释。详见 [P03](../03_transfer/validation_matrix.csv)。

## 4 数据质量作用位置诊断

B6 在相同 N,D 的不同 Q 层级上按基础实验分组五折。预先固定的 1-SE 简约规则选中 **{decision['selected_model']}**；候选比较如下：

{markdown_table(candidate, ['model', 'n_parameters', 'macro_fold_rmse', 'rmse', 'mae', 'r2'])}

选择规则、AIC/BIC、组 bootstrap、参数相关及剖面均在 B7 Loss 读取前写入 [P04](../04_quality_selection/model_selection_rule.json)，并由 `model_freeze.sha256` 锁定。质量系数符号由数据估计，未人为翻转 Q。

## 5 含质量的 Scaling Law

当前入选结构为 `E+A N^-α exp[-ρ_N(Q_B-Q₀)]+B D^-β exp[-ρ_D(Q_B-Q₀)]-E₁(Q_B-Q₀)`，Q₀={contract['q0']:.6g}。验证后对 B6∪B7-new 的同结构重拟合参数为 E={p['E']:.6g}、A={p['A']:.6g}、B={p['B']:.6g}、α={p['alpha']:.6g}、β={p['beta']:.6g}、ρ_N={p['rho_N']:.6g}、ρ_D={p['rho_D']:.6g}、E₁={p['E1']:.6g}。这些参数仅用于最终公式及局部效应；B7 留出精度使用事先冻结参数。B6 组 bootstrap 参数分位数见 [P04 CI](../04_quality_selection/parameter_ci.csv)，不可直接当成验证后参数的 CI。

## 6 B7-new 锁定验证

B7-new 含新的 Q={validation['support_novelty']['new_Q']}，没有新的 N 或 D。使用 P04 冻结参数，{decision['selected_model']} 的 RMSE={b7.iloc[0].rmse:.6g}、MAE={b7.iloc[0].mae:.6g}；M0 RMSE={b7.iloc[1].rmse:.6g}。配对组 bootstrap 的 RMSE 差区间为 {validation['bootstrap_rmse_difference_ci']}。这是半合成的新 Q 留出，不是新规模外推。见 [P05](../05_locked_validation/validation_summary.json)。

## 7 B8 压力测试与外推边界

B8 固定 N,D 的 {stress['b8_positive_groups']} 组 Q–Loss 为正向、{stress['b8_negative_groups']} 组为负向；这与 B6/B7 的入选质量作用方向冲突，构成跨附件结构不一致或压力测试失败。我们保留原 Q 符号和模型。B9 无 Loss，仅用于 N/D 支持距离；B10 的估算 Loss 与模型比较仅为 estimated-reference consistency，不能算真实外部验证。见 [P06](../06_stress_and_extrapolation/stress_summary.json)。

## 8 问题一冻结接口

Q1 v2.1 指向 v2 冻结的 17 域、13 目标 LightGBM；模型 SHA-256 已核对。每目标 `h_{{p,v}}(p)=log[f_v(p)/f_v(p₀)]`，并保留 13 目标等权聚合。Q1 输出没有保存数值 p₀，因此 V5 预先采用等权 simplex 点作情景参考；Q1 模型对该点的训练支持距离无法核验。对应的 [冻结接口](../07_q1_bridge/frozen_q1_interface/frozen_hashes.json) 和 [目标效应](../07_q1_bridge/hp_by_target.csv) 可复算。

## 9 Q_A 与 Q_B 的跨尺度关系

Q_A 是 Q1 质量评分，Q_B 是 B6/B7 的质量变量，二者没有配对实验。共同归一恒等、支持区间仿射、分位保持三种映射均是 scenario assumption，不能视为唯一识别的真实换算。见 [映射情景](../07_q1_bridge/qa_qb_mapping_scenarios.csv)。

## 10 配比结构

冻结 Q1 输出缺少同一配比在多种 N,D 上的联合观测，Form A（配比调节 D 项）和 Form B（调节全部可约项）均未被识别。按预注册规则以 Form A 为主情景，Form B 作结构敏感性；不能写成已证实的作用机制。

## 11 最终广义 Scaling Law

主情景：`{formula}`，其中 `h_p` 为 13 目标等权的冻结 Q1 对数 Loss 比值。令 Q_B=Q₀、p=p₀ 则退化为经典 N-D 形式。`λ_p∈{{0,0.5,1,1.5}}` 为未识别情景参数；Q_A→Q_B 与 p₀ 也属于情景。模型合同见 [P08](../08_generalized_law/model_contract.json)。

## 12 参数估计与可识别性

E、A、B、α、β、ρ_N、ρ_D、E₁ 来自 B6+B7-new 半合成数据的验证后估计；`h_p` 及 Q_A 量来自 Q1 冻结输出。`λ_p`、Q_A→Q_B、p₀ 及 Form 附加位置不可由当前独立数据源联合识别。参数分类见 [parameter_table.csv](parameter_table.csv)。

## 13 边际效用

在 Form A，记 U_N=A N^-α exp[-ρ_N(Q_B-Q₀)]、U_D=B D^-β exp[-ρ_D(Q_B-Q₀)]exp[λh]。解析式为 `M_N=αU_N/N`、`M_D=βU_D/D`、`M_Q=ρ_N U_N+ρ_D U_D+E₁`。mid 观测支持工作点 N={mid.N_params_B:g}、D={mid.D_tokens_B:g}、Q={mid.Q_B:g}、p=p₀、λ=0 时，三者分别为 {mid.M_N:.6g}、{mid.M_D:.6g}、{mid.M_Q:.6g}。解析导数与中心差分已核对，详见 [P09](../09_effects/derivative_checks.csv)。

## 14 弹性

`ε_N=L_N N/L`、`ε_D=L_D D/L`、`∂log L/∂Q=L_Q/L`。同一 mid 工作点分别为 {emid.epsilon_N:.6g}、{emid.epsilon_D:.6g}、{emid.dlogL_dQ:.6g}。Q_B 的比例零点无明确物理意义，因此质量报告半弹性。见 [P09 elasticities](../09_effects/elasticities.csv)。

## 15 数据质量与模型/数据规模替代

固定 L,D,p 时 `dN/dQ=-L_Q/L_N`；固定 L,N,p 时 `dD/dQ=-L_Q/L_D`。mid 点 `dlogN/dQ={sub_n.set_index('workpoint').at['mid', 'dlogN_dQ']:.6g}`、`dlogD/dQ={sub_d.set_index('workpoint').at['mid', 'dlogD_dQ']:.6g}`。ΔQ=0.1 的一阶替代只是局部近似；ΔQ=0.01/0.05/0.1 重新代入完整模型的误差保留在 [非线性检查](../09_effects/substitution_nonlinearity.csv)，不能概括为全局恒定等价。

## 16 领域配比替代和联合响应

配比转移 `p(δ)=p₀+δ(e_j-e_k)` 始终保持 17 维 simplex。冻结 LightGBM 的极小中心差分可能落在树叶平台而给出零斜率；δ=0.005/0.02 的有限变化更能显示局部情景响应。δ=0.02 时 {len(npair)} 个有向转移的 aggregate Loss 变化绝对值中位数为 {npair.aggregate_loss_change.abs().median():.6g}。13 目标与四角 `local_joint_response` 单独保存于 [P11](../10_mixture_effects/report.md)，不解释为因果互补。

## 17 稳健性分析

统计不确定性包括 B1/质量参数组 bootstrap 和 B7 留出差异；结构不确定性包括质量模型及 Form A/B；跨来源不确定性是三种 Q_A→Q_B 映射；情景不确定性包括 λ 与配比扰动幅度。它们不能合成一个 CI。分类表见 [P12](../11_robustness/robustness_matrix.csv)。V5 核心在 P12 冻结后才与历史版本比较，见 [影子比较](../11_robustness/historical_difference_report.md)。

## 18 局限性

B6/B7/B8 为半合成；B8 与主质量趋势显著冲突；B2/B4/B5 Loss 口径跨来源；B3 为插值；B10 为估算；Q_A/Q_B 无配对观测；λ 与 Form 无联合识别；等权 p₀ 的 Q1 训练支持距离无法核验；树模型局部导数受叶节点平台影响。外推结论只能在证据边界内陈述。

## 19 问题二结论

B1 支持 N、D 增大时 Loss 降低的经典经验关系。B6 的五折比较选中 {decision['selected_model']}，在半合成 B7-new 上相较 M0 有较低误差，但 B8 压力测试显示跨生成机制不稳定。广义 N,D,Q,p 公式可给出明确定义的局部边际效应、弹性及单纯形配比响应；Q_A→Q_B、λ、p₀ 与 Form 属于情景或结构假设。该式可供问题三作为候选目标函数，完整算力预算优化不在本问实施。

### 复现与 AI 辅助说明

本报告由 Q2 V5 Python 流水线生成，输入及产物 SHA-256、配置和阶段测试见 [reproducibility.json](reproducibility.json)。AI 辅助了代码实现、调试和文字整理；竞赛提交前应由参赛队伍独立核验公式、数据来源和论证，并按竞赛要求披露使用范围。
"""
    report_path = out / "question2_v5_experiment_report.md"
    report_path.write_text(report, encoding="utf-8")
    checks = consistency_audit(report_path)
    write_json(out / "final_consistency_audit.json", checks)
    if checks["critical_failures"]:
        raise RuntimeError(f"最终一致性审计失败: {checks['critical_failure_names']}")
    reproducibility = {"version": "q2_v5", "seed": config()["seed"],
                       "P00_git": json.loads((OUT / "00_manifest/manifest.json").read_text())["git"],
                       "config_sha256": sha(ROOT / "configs/q2_v5/pipeline.yaml"),
                       "code_sha256": {path.relative_to(ROOT).as_posix(): sha(path)
                                       for path in sorted((ROOT / "src/q2_v5").glob("*.py"))},
                       "test_sha256": {path.relative_to(ROOT).as_posix(): sha(path)
                                       for path in sorted((ROOT / "tests/q2_v5").glob("test_*.py"))},
                       "core_freeze_sha256": (stage_dir("P12") / "core_freeze.sha256").read_text().strip(),
                       "stage_metadata": {f"P{i:02d}": sha(stage_dir(f"P{i:02d}") / f"p{i:02d}_metadata.json")
                                          for i in range(1, 14)},
                       "status": "COMPLETE_WITH_WARNINGS"}
    write_json(out / "reproducibility.json", reproducibility)
    write_json(out / "pipeline_status.json", {
        "status": "COMPLETE_WITH_WARNINGS", "critical_failures": 0,
        "stage_status": {**{f"P{i:02d}": require_pass(f"P{i:02d}")["status"] for i in range(1, 14)},
                         "P14": "PASS"},
        "warnings": ["B8 质量方向与主模型冲突", "Q1 p0、QA→QB、lambda 与 Form 未识别",
                     "Git commit/status 不可用"]})
    (out / "p14_report.md").write_text(
        "# P14 终验\n\n最终报告、8 幅图、证据矩阵、参数表与一致性审计已生成。"
        "critical_failures=0；整体 COMPLETE_WITH_WARNINGS。\n", encoding="utf-8")
    metadata = finalize("P14", [stage_dir("P12") / "core_freeze.sha256",
                        stage_dir("P13") / "shadow_comparison.csv",
                        stage_dir("P08") / "model_contract.json"],
                        ["B8 压力测试冲突", "QA/QB 与配比桥存在情景限制", "Git 不可用"],
                        ["报告所有数字来自阶段 CSV/JSON；图源及 SHA 单独列示"],
                        ["direct_observation", "semi_synthetic", "interpolation",
                         "estimated_reference", "q1_imported", "scenario_assumption"])
    return {"status": "COMPLETE_WITH_WARNINGS", "critical_failures": 0,
            "figure_count": len(figures), "metadata": metadata}
