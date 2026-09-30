"""稳定导出结构化审计结果及简报。"""
import csv
from contextlib import ExitStack
import gzip
import io
import json
from pathlib import Path


ISSUE_FIELDS = ["source_file", "source_format", "line_number", "record_id", "record_hash",
                "status", "reason_code", "details", "rule_version"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if path.suffix == ".gz":
            binary = stack.enter_context(path.open("wb"))
            compressed = stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0))
            stream = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8", newline=""))
        else:
            stream = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report(summary, output):
    lines = ["# 第一问 P0 数据审计", "", "本报告由 `PYTHONPATH=src python -m q1.audit.run` 根据本地附件生成。`valid` 仅表示通过当前规则，不证明来源真实性。", "",
             "## 文件和记录", "", "|附件|文件|读取状态|文档预期|实际行数|有效记录|隔离记录|", "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["files"]:
        lines.append(f"|{row['attachment']}|{Path(row['source_file']).name}|{row['read_status']}|{row.get('expected_rows','')}|{row.get('actual_rows','')}|{row.get('valid_rows','')}|{row.get('invalid_rows','')}|")
    lines.extend(["", "## 质量信号核查", ""])
    for item in summary["quality"]:
        counts = item["final_status_counts"]
        lines.append(f"- {item['attachment']}：实际 {item['lines_read']} 行；完整解码：{item['complete']}；有效 {counts.get('valid',0)}、共有字段完全一致的重复 {counts.get('duplicate_exact',0)}、冲突 {counts.get('duplicate_conflict',0)}、明确损坏 {counts.get('rejected_corrupt',0)}、待核实 {counts.get('suspected_unreliable',0)}。")
    lines.extend(["", "A1 解压副本与压缩流内容的 SHA-256 一致：" + str(summary["decompressed_copy"].get("same_source", "未知")) + "；副本未重复计数。",
                  "22 项指标的类型、缺失、有限值、范围与列表长度见 `quality_indicator_schema.csv`；辅助字段的结构性缺失见 `auxiliary_field_report.csv`。",
                  "", "## 配比、Loss 和映射", ""])
    for pair in summary["pairs"]:
        lines.append(f"- {Path(pair['mixture_file']).name} + {Path(pair['loss_file']).name}：{pair['matched_valid_keys']} 个有效 `index` 一一对应；配比独有 {pair['mixture_only']}，Loss 独有 {pair['loss_only']}；一一对应：{pair['one_to_one']}。")
    lines.append(f"- A12/A14 训练配方子集核查：{[item['is_subset'] for item in summary['estimate_subsets']]}；A13/A15 为已有外推估算 Loss。")
    mapping = summary["mapping"]
    lines.append(f"- A16：{mapping.get('rows',0)} 行；映射类型 {mapping.get('mapping_types',{})}；17 域完整且无重复：{mapping.get('valid',False)}。")
    lines.extend(["", "## 完整性与下一阶段输入", ""])
    for item in summary["findings"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 审计规则", "", "- JSONL 按压缩流逐行读取；解析错误逐行隔离；XZ 流错误使整个来源不可作为完整输入。",
                  "- 22 项指标按配置核查类型和有限数值；列表只核查结构，不解释为 Logits。`frac` 字段实测更像百分数尺度，但语义未证实，超过 100 暂列可疑，不判定伪造。",
                  "- 同 ID 按共有字段规范化 JSON 哈希比较。内容冲突时该 ID 全部来源隔离；相同内容保留首条作为分析视图。",
                  "- 配比和 Loss 仅按 `index` 对应。17 项配比非负，和的容差为配置中的 0.008500001，源于 17 项各保留千分位时的最大理论舍入误差。原值不改动。",
                  "- `duplicate_conflict`、`suspected_unreliable`、`unverifiable`、`rejected_corrupt` 均不得作为正式模型输入。",
                  "", "## 限制", "", "- source_manifest.json 提供文件大小和来源说明，未提供内容 SHA-256；本次 SHA-256 为本地计算，不能独立证明来源真实性。",
                  "- 三份 Question 1.x Markdown 未在仓库找到；P0 不据此推断指标方向或列表含义。",
                  "- A12–A15 是外推配方与估算 Loss，不能当成真实独立实验。", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
