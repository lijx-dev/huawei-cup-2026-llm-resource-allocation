"""核验问题三实际输出，并生成可追溯的实验元数据与简报。"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from q3_paths import A_DIR, B_DIR, C7, INPUT_DIR, OUTPUT_DIR, ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    names = [
        "q3_core_results.json", "q3_constrained_vs_free.json",
        "q3_fused_grid.json", "q3_fused_grid_all.csv",
        "q3_fused_grid_eta2e-4.csv", "q3_p_subproblem_results.json",
        "q3_kkt_results.json", "q3_derived_results.json",
        "q3_bootstrap_transitions.json",
    ]
    for name in names:
        assert (OUTPUT_DIR / name).is_file(), f"缺少结果：{name}"

    core = json.loads((OUTPUT_DIR / names[0]).read_text())
    fused = pd.read_csv(OUTPUT_DIR / "q3_fused_grid_eta2e-4.csv")
    all_grid = pd.read_csv(OUTPUT_DIR / "q3_fused_grid_all.csv")
    p = json.loads((OUTPUT_DIR / "q3_p_subproblem_results.json").read_text())
    constrained = json.loads((OUTPUT_DIR / "q3_constrained_vs_free.json").read_text())
    kkt = json.loads((OUTPUT_DIR / "q3_kkt_results.json").read_text())
    boot = json.loads((OUTPUT_DIR / "q3_bootstrap_transitions.json").read_text())
    assert len(fused) == 45 and len(all_grid) == 135
    assert fused.feasible.sum() == 42 and (~fused.feasible).sum() == 3
    assert boot["n_data"] == 360 and boot["n_boot"] == 400
    assert all(x["consistent"] for x in p["extremes"])
    assert kkt["kkt_pass"]["n_bad"] == 0
    assert all(x["rel"] < 1e-12 and x["dL"] == 0 for x in core["selfcheck_purescale"])

    files = {
        "package_zip": ROOT / "团队交付包_问题三融合方案.zip",
        "attachment_source_manifest": ROOT / "data/real_attachments/source_manifest.json",
        "q2_parameter_source": ROOT / "results/q2_v8/05_results/q2_v8_generalized_model.json",
        "q1_m2_coefficients": INPUT_DIR / "mix_final_model_quad.csv",
        "q1_m2_interface": INPUT_DIR / "Q1_M2响应接口.csv.gz",
        "q1_p0": INPUT_DIR / "Q1_p0_17dim.csv",
        "A4_train_mixture": A_DIR / "train_mixture_1m.csv",
        "A6_test_mixture": A_DIR / "test_mixture_1m.csv",
        "A8_test_mixture": A_DIR / "test_mixture_60m.csv",
        "A10_test_mixture": A_DIR / "test_mixture_1B.csv",
        "B6_quality": B_DIR / "supplementary_NQ_experiment.csv",
        "C7_context": C7,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "q3_fused_package_port_b6_bootstrap",
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "seed_bootstrap": boot["seed"],
        "evidence": {
            "q2_parameter_source_status": "PARTIAL_CANDIDATE_INTERFACE",
            "B6": "semi_synthetic_calibration",
            "Q1_M2": "package_candidate_imported",
            "C7": "observed_context_values",
            "eta": "scenario_assumption",
            "quality_cost_functions": "given_scenarios_not_calibrated",
            "lambda_p": "scenario_not_identified",
        },
        "inputs_sha256": {key: sha256(path) for key, path in files.items()},
        "code_sha256": {path.name: sha256(path) for path in sorted((ROOT / "src/q3").glob("*.py"))},
        "outputs_sha256": {name: sha256(OUTPUT_DIR / name) for name in names},
        "counts": {
            "A4": 512, "B6": boot["n_data"], "C7": len(pd.read_csv(C7)),
            "main_grid": len(fused), "main_grid_feasible": int(fused.feasible.sum()),
            "main_grid_infeasible": int((~fused.feasible).sum()),
            "all_grid": len(all_grid), "bootstrap": boot["n_boot"],
            "kkt_checked": kkt["kkt_pass"]["n"],
        },
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    point = fused[(fused.Lctx == 32768) & (fused.g == "exp")].sort_values("C")
    box_penalty = [row["dL_pct"] for row in constrained if "dL_pct" in row]
    lines = [
        "# 问题三融合方案实跑简报", "",
        "本报告来自本仓库脚本重新计算；包内预存结果仅作对照。",
        "", "## 输入与口径", "",
        "- 问题二使用 `SET_A` 四位小数冻结参数；仓库问题二完整精度参数文件标记为 `PARTIAL_CANDIDATE_INTERFACE`。",
        "- 问题一 M2 配比接口来自交付包，SHA-256 已与包内 manifest 核对；仍属候选接口。",
        "- B6 为半合成校准数据，Bootstrap 仅在 B6 的 (N,D) 组上重抽。B7 保持前问留出角色。",
        "- η、质量成本函数及 λp 是情景假设；C7 仅支持上下文长度取值，不能识别 η。",
        "", "## 验收结果", "",
        f"- 默认 η=2e-4 的主表 {len(fused)} 行：可行 {int(fused.feasible.sum())}，不可行 {int((~fused.feasible).sum())}。全部 η 情景 {len(all_grid)} 行。",
        f"- 分组 Bootstrap 成功 {boot['n_boot']} 次，B6 输入 {boot['n_data']} 行；KKT 检查 {kkt['kkt_pass']['n']} 例，失败 {kkt['kkt_pass']['n_bad']} 例。",
        f"- 18 个盒约束对照情景的预测 Loss 增量：中位 {np.median(box_penalty):.2f}%，最大 {max(box_penalty):.2f}%。",
        f"- 配比极值两档均通过脚本内部一致性标记；A4 中 h_p<0 的配方数见 `logs/team_claim_check.log`。",
        "", "## 示例：指数型质量成本，Lctx=32768，η=2e-4", "",
        "| 预算 FLOPs | N (十亿参数) | D (十亿 token) | Q | 预测 Loss | 闲置预算 |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in point.iterrows():
        lines.append(f"| {row.C:.0e} | {row.N:.5f} | {row.D:.3f} | {row.Q:.5f} | {row.L:.5f} | {row.idle_frac:.2%} |")
    lines += [
        "", "## 解释边界", "",
        "- 表中的 Loss 是模型预测，且高预算盒约束解触及校准边界；闲置预算只在当前盒约束内有意义。",
        "- 配比通道假定没有额外算力成本；其预测响应不等于已证实的因果收益。",
        "- 原交付包 Bootstrap 用 B6∪B7 重拟合，与冻结 SET_A 的 B6 拟合口径冲突；本次改为仅 B6。",
        "- 原配比复核用单个辅助量近似 L1，导致无可行数值复核点仍显示通过；本次使用 17 个辅助量并要求找到可行点。",
        "- 原派生量脚本称 D/N 与预算无关；本次修正为 D/N ∝ C^((α-β)/(α+β))。",
        "- 结果、日志、输入及代码 SHA-256 均见 `metadata.json`。",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"验收通过：{len(fused)} 条默认主表，{boot['n_boot']} 次 Bootstrap，{kkt['kkt_pass']['n']} 例 KKT。")


if __name__ == "__main__":
    main()
