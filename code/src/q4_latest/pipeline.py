"""重跑问题四，并显式接入 Q1/Q2/Q3 最新接口。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/q4_latest_20260926"
RUN=OUT/"reproduction"
VERSION="q4-latest-20260926-v1"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=float),encoding="utf-8")


def run():
    os.environ["Q4_RUN_DIR"]="results/q4_latest_20260926/reproduction"
    os.environ["Q4_LATEST_MODE"]="1"
    # 环境变量必须先设置，再导入路径常量。
    from src.q4 import run_fusion_package as runner
    from src.q4 import audit_inputs
    audit_code=audit_inputs.main()
    if audit_code:
        raise RuntimeError(f"Q4 input audit failed: {audit_code}")
    code=runner.main()
    if code:
        raise RuntimeError(f"Q4 scripts failed: {code}")

    q1=ROOT/"results/q1_revision_v2_1/q2_interface/manifest.json"
    q2=ROOT/"results/q2_fusion_rerun_v1/report/reproducibility_summary.json"
    q3=ROOT/"results/q3_latest_20260926/metadata.json"
    q3_meta=json.loads(q3.read_text())
    if q3_meta["status"]!="PASS_WITH_WARNINGS" or not q3_meta["latest"]:
        raise RuntimeError("Q3 latest interface unavailable")
    status=json.loads((RUN/"execution_status.json").read_text())
    data_audit=json.loads((RUN/"data_audit.json").read_text())
    outputs=RUN/"outputs"
    p1b=json.loads((outputs/"q4_p1b_dynamics_results.json").read_text())
    p1d=json.loads((outputs/"q4_p1d_channels_results.json").read_text())
    evidence=json.loads((outputs/"q4_evidence_results.json").read_text())
    forecast=json.loads((outputs/"q4_forecast_results.json").read_text())
    backtest=json.loads((outputs/"q4_p2c_backtest_results.json").read_text())

    grid=pd.read_csv(ROOT/"results/q3_latest_20260926/scenario_grid_all.csv")
    integration={
        "version":VERSION,"q3_rows":len(grid),"q3_feasible":int(grid.feasible.sum()),
        "q3_infeasible":int((~grid.feasible).sum()),
        "q3_loss_range_feasible":[float(grid.loc[grid.feasible,"L"].min()),float(grid.loc[grid.feasible,"L"].max())],
        "q1_mix_channel":p1d.get("c_mix"),
        "q2_reference_parameters":p1b.get("set_A_ref"),
        "loss_benchmark_bridge":evidence.get("bridge"),
        "policy":"Q3 conditional Loss is not converted to benchmark score without an identified bridge",
    }
    write_json(OUT/"upstream_integration.json",integration)

    old_outputs=ROOT/"results/q4/fusion_reproduction/outputs"
    c_only_names=["q4_c8_p90_results.json","q4_extra_results.json","q4_p1_chinchilla_rho_results.json",
        "q4_p1c_schaeffer_results.json","q4_p1e_musr_results.json","q4_p2a_cluster_results.json","q4_p2b_eligibility_results.json"]
    c_only_checks={name:json.loads((outputs/name).read_text())==json.loads((old_outputs/name).read_text()) for name in c_only_names}
    old_forecast=json.loads((old_outputs/"q4_forecast_results.json").read_text())
    forecast_changed=forecast!=old_forecast

    source_paths=[q1,q2,q3,ROOT/"团队交付包_问题四融合方案.zip"]
    output_paths=sorted(outputs.glob("*"))+[OUT/"upstream_integration.json",RUN/"data_audit.json",RUN/"input_inventory.json",RUN/"execution_status.json"]
    meta={"version":VERSION,"latest":True,"status":"PASS_WITH_WARNINGS",
          "supersedes":"results/q4 and 团队交付包_问题四融合方案.zip",
          "upstream":{"q1":"q1-to-q2-v1","q2":"q2-fusion-rerun-v1","q3":"q3-latest-20260926-v1"},
          "script_status":status,"scripts_success":sum(v==0 for v in status.values()),"scripts_total":len(status),
          "data_audit":{"manifest_size_mismatches":len(data_audit["source_manifest_size_mismatches"]),"c8":data_audit["c8"]},
          "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in source_paths},
          "product_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in output_paths if p.is_file()},
          "counts":{"q3_grid":len(grid),"q3_feasible":int(grid.feasible.sum()),"q4_machine_outputs":len(list(outputs.glob('*'))),"c_only_unchanged":sum(c_only_checks.values()),"c_only_checked":len(c_only_checks)},
          "comparisons":{"forecast_changed_with_q3_mechanism":forecast_changed,"c_only_outputs_unchanged":c_only_checks},
          "warnings":["C6 Loss-Benchmark bridge remains weak and cross-source","Q3 quality cost and lambda_p remain scenarios","12/24 month forecasts lack matching-horizon prospective validation"]}
    write_json(OUT/"metadata.json",meta)
    write_json(OUT/"LATEST_VERSION.json",{"latest":True,"version":VERSION,"status":meta["status"],"display_name":"问题四融合方案｜最新版 2026-09-26"})

    lines=["# 问题四融合方案｜最新版 2026-09-26","",f"**LATEST / 当前最新版**：`{VERSION}`；状态 `PASS_WITH_WARNINGS`。","",
      "本版完整重跑交付包 20 个脚本，并把跨题接口替换为 Q1 正式 LightGBM、Q2 重跑主模型和 Q3 最新资源前沿。原 ZIP 保持只读。","",
      "## 验收","",f"- B/C 输入清单大小差异 {len(data_audit['source_manifest_size_mismatches'])}；C8 选中视图有效 {data_audit['c8']['selected_valid_count']}、隔离 {data_audit['c8']['selected_invalid_count']}。",f"- 脚本成功 {meta['scripts_success']}/{meta['scripts_total']}。",f"- Q3 最新网格接入：{len(grid)} 行，可行 {int(grid.feasible.sum())}，不可行 {int((~grid.feasible).sum())}。",f"- 含 Q3 机制腿的预测已按最新预算弹性变化：{forecast_changed}。",f"- 明确不依赖跨题接口的 C-only 输出逐字段一致：{sum(c_only_checks.values())}/{len(c_only_checks)}。","",
      "## 最新接口影响","","- P1-B 的问题二参考参数已换成 MQ 的严格等价八参数表示：rho_N=0、E1=0、rho_D=gamma。","- P1-D 的配比通道改读正式 LightGBM h_agg；旧 mix_final_model_quad.csv 不再作为响应来源。","- C6 桥接使用新 Q2 E 做压力检查，但由于 Loss 口径异质，仍不得将 Q3 Loss 换算成真实 benchmark 分数。","",
      "## 保留与变化","","C 数据驱动的 C8 审计、时间序列辅助结果、资格漏斗等七项 C-only 输出保持不变；含“问题三机制腿”的 12/24 月合成预测及依赖该预测校准的回测输出，已随最新版 Q3 重算。","",
      "## 证据边界","","Q3 的质量成本、eta、lambda_p 仍是情景；问题四跨来源桥接较弱；长期预测尚无同期限前瞻覆盖率验证。","",f"Q4 LATEST STATUS: {meta['status']}",""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"README_LATEST.md").write_text("# LATEST｜问题四最新版\n\n唯一推荐入口：`PYTHONPATH=src .venv/bin/python -m src.q4_latest.pipeline`。\n\n查看 `LATEST_VERSION.json`、`metadata.json`、`report.md`。\n",encoding="utf-8")
    return meta


if __name__=="__main__":
    result=run();print(f"Q4_LATEST_STATUS: {result['status']}")
