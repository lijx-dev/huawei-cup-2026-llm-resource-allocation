"""复用已审查的 B 附件审计逻辑，结果隔离到 v2。"""
import json
import os
from pathlib import Path

import pandas as pd

from src.q2 import audit as base


ROOT = base.ROOT
OUTPUT_ROOT = Path(os.environ.get("Q2_OUTPUT_DIR", "results/q2_scaling_v2"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = ROOT / OUTPUT_ROOT
OUT = OUTPUT_ROOT / "audit"
VERSION = os.environ.get("Q2_EXPERIMENT_VERSION", "q2-scaling-v2")
FORMAL_Q1_INTERFACE = os.environ.get("Q1_INTERFACE_DIR")


def run():
    old_out, old_version, old_rule = base.OUT, base.VERSION, base.RULE
    try:
        base.OUT = OUT
        base.VERSION = VERSION
        base.RULE = f"{VERSION}-p0-1"
        metadata = base.run()
    finally:
        base.OUT, base.VERSION, base.RULE = old_out, old_version, old_rule
    # 旧审计函数写有 v1 固定叙述；v2 报告由实测元数据重新生成。
    inventory = pd.read_csv(OUT / "b_attachment_inventory.csv")
    inventory["actual_role"] = inventory["proposed_model_role"]
    inventory.to_csv(OUT / "b_attachment_inventory.csv", index=False)
    bad = pd.read_csv(OUT / "invalid_rows.csv")
    interface = json.loads((OUT / "q1_interface_audit.json").read_text())
    if FORMAL_Q1_INTERFACE:
        interface_dir = Path(FORMAL_Q1_INTERFACE)
        if not interface_dir.is_absolute():
            interface_dir = ROOT / interface_dir
        manifest_path = interface_dir / "manifest.json"
        formal = json.loads(manifest_path.read_text())
        interface.update(
            formal_interface_path=str(manifest_path.relative_to(ROOT)),
            formal_interface_sha256=base.sha256(manifest_path),
            formal_interface_status=formal["status"],
            model_path=formal["model"]["path"],
            model_sha256=formal["model"]["sha256"],
            input_fields=formal["input_order"],
            output_fields=formal["output_order"],
            input_count=len(formal["input_order"]),
            output_count=len(formal["output_order"]),
            ready_for_h_v=formal["status"] == "PASS" and len(formal["input_order"]) == 17 and len(formal["output_order"]) == 13,
            q_mapping_policy=formal["quality_policy"],
        )
    support_path=ROOT / "results/q1_revision_v2/mixture/support_reference.json"
    if support_path.exists():
        interface["support_rule_path"]=str(support_path.relative_to(ROOT))
        interface["support_rule_sha256"]=base.sha256(support_path)
        interface["support_rule"]=json.loads(support_path.read_text())
    (OUT / "q1_interface.json").write_text(json.dumps(interface, ensure_ascii=False, indent=2))
    pd.read_csv(OUT / "b8_groupwise_q_loss_audit.csv").to_csv(OUT / "b8_groupwise_q_loss.csv", index=False)
    loss = pd.read_csv(OUT / "loss_semantics.csv")
    loss["metric_definition"] = loss["loss_name"]
    loss["evaluation_dataset"] = loss["validation_dataset"]
    loss["tokenizer_if_known"] = loss["tokenizer"]
    loss["absolute_comparable_to_B1"] = loss["directly_comparable_to_B1"]
    loss["trend_comparable_to_B1"] = "conditional_on_loss_semantics"
    loss.to_csv(OUT / "loss_semantics.csv", index=False)
    quality=pd.read_csv(OUT/"quality_semantics.csv")
    qa_path=ROOT/"results/q1_revision_v2_1/quality/sample_quality_scores.csv.gz"
    qa=pd.read_csv(qa_path,usecols=["Q_hierarchical_balanced"])["Q_hierarchical_balanced"]
    quality.loc[len(quality)]=dict(attachment="Q_A",q_column="Q_hierarchical_balanced",min=float(qa.min()),max=float(qa.max()),mean=float(qa.mean()),std=float(qa.std()),unique_count=int(qa.nunique()),higher_means_better="yes_by_Q1_design",definition="Q1 hierarchical balanced score from 22 indicators, 0-100 design scale",generation_method="Q1 frozen scoring pipeline",same_definition_as_B6="unproven",evidence="results/q1_revision_v2_1/report/question1_v2_1_report.md; results/q1_revision_v2_1/quality/sample_quality_scores.csv.gz")
    quality.to_csv(OUT/"quality_semantics.csv",index=False)
    contract = json.loads((OUT / "field_contract.yaml").read_text())
    contract["standard_fields"]["Q_B_raw"] = contract["standard_fields"].pop("Q_raw")
    contract["standard_fields"]["Q_B_normalized"] = contract["standard_fields"].pop("Q_normalized")
    contract["standard_fields"]["Q_B_direction"] = contract["standard_fields"].pop("Q_direction")
    contract["standard_fields"]["provenance"] = "dataset_role_matrix.csv"
    (OUT / "field_contract.yaml").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    status = "PASS_WITH_WARNINGS" if interface["ready_for_h_v"] and inventory.readable.all() and metadata["source_hash_unchanged"] else "BLOCKED"
    metadata.update(stage="Q2-P0", status=status, actual_invalid_rows=len(bad), q1_quality_scores_sha256=base.sha256(qa_path), missing_requested_design_document="问题二_完整建模思路_扩展建模(1).md", available_design_document="问题二_完整建模思路_扩展建模.md")
    (OUT / "audit_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    report = ["# Q2-P0 数据合同与接口审计", "", f"版本 {VERSION}；种子 7。B 附件只读，未拟合标度律。", "", f"B1–B12 实际文件 {len(inventory)} 个；可读 {int(inventory.readable.sum())} 个。B3 为 8 个插值文件。", f"附件计数：{metadata['attachment_rows']}。非法数值行 {metadata['invalid_numeric_rows']}；重复条目 {metadata['duplicate_entries']}。B9 的非法 D 行隔离，不进入外推点。", f"B6/B7 重叠与新增：{metadata['b6_b7']}。B8 类型：{metadata['b8_data_type_counts']}。", "", "B1 的 N_params_B 与 D_tokens_B 已是十亿单位；8 个规模各 147 个 checkpoint。B12 缺精确参数量和 token，不能强行对齐。", "B1 以 scale 为组做留一规模验证；B3 只做插值轨迹核查。B2/B6–B8 是半合成；B10 Loss 是标度律估算。跨来源验证集/tokenizer 未说明，绝对 Loss 可比性未知。", "Q_A 与 Q_B 的定义、方向、数值尺度、计算口径不能证明一致，合同固定 Q_A != Q_B。B8 数值方向按原值保留。", f"第一问冻结模型当前 SHA-256={interface['model_sha256']}；17 输入、13 输出顺序核验={interface['ready_for_h_v']}；正式接口状态={interface.get('formal_interface_status', 'legacy')}。", "", "文档差异：指定的扩展建模(1).md 不存在，使用仓库内扩展建模.md；赛题正文要求更完整的质量/注意力成本，本次 P6 明确采用 C≈6ND，结论仅适用该近似。", "", f"Q2-P0 STATUS: {status}", ""]
    (OUT / "q2_p0_report.md").write_text("\n".join(report))
    (OUT / "q2_p0_audit_report.md").write_text("\n".join(report))
    return metadata
