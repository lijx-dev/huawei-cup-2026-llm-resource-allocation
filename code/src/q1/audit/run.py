"""P0 审计入口：PYTHONPATH=src python -m q1.audit.run。"""
import argparse
from collections import Counter
import gzip
import json
import logging
from pathlib import Path
import platform

from .duplicates import export_views, init_db
from .export import ISSUE_FIELDS, report, write_csv, write_json
from .inventory import expected_files, file_inventory, manifest_entries, sha256_file
from .quality import scan_quality, schema_rows
from .tables import audit_mapping, audit_pair, read_table, subset_of_train


def run(root):
    config_path = root / "configs/q1/audit.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = root / "results/q1/audit"
    out.mkdir(parents=True, exist_ok=True)
    files = expected_files(root)
    inventory = file_inventory(root, files, manifest_entries(root))
    expected_rows = {**config["quality_expected_rows"],
                     "A4": 512, "A5": 512, "A6": 256, "A7": 256,
                     "A8": 256, "A9": 256, "A10": 64, "A11": 64, "A16": 17}
    for item in inventory:
        item["expected_rows"] = expected_rows.get(item["attachment"], "")
    by_file = {row["source_file"]: row for row in inventory if row["source_file"]}
    issues, quality, schema, tables, pairs = [], [], [], {}, []
    db_path = out / "quality_index.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = init_db(db_path)
    for attachment in ["A1", "A2", "A3"]:
        for path in files[attachment]:
            if not path.is_file():
                continue
            logging.info("读取 %s %s", attachment, path.name)
            scan = scan_quality(path, attachment, root, config, db, issues)
            quality.append({key: value for key, value in scan.items() if key != "stats"})
            schema.extend(schema_rows(scan))
            row = by_file[path.relative_to(root).as_posix()]
            row.update(read_status="complete" if scan["complete"] else "incomplete_xz",
                       actual_rows=scan["lines_read"], valid_rows=scan["counts"].get("valid", 0),
                       invalid_rows=scan["counts"].get("rejected_corrupt", 0))
    for number in range(4, 16):
        attachment = f"A{number}"
        path = files[attachment][0]
        if not path.is_file():
            continue
        kind = "mixture" if number % 2 == 0 else "loss"
        table = read_table(path, kind, config, root, issues)
        tables[attachment] = table
        row = by_file[path.relative_to(root).as_posix()]
        row.update(read_status="complete", actual_rows=table["rows"], valid_rows=table["valid"], invalid_rows=table["invalid"])
    for number in range(4, 16, 2):
        left, right = tables.get(f"A{number}"), tables.get(f"A{number+1}")
        if left and right:
            pairs.append(audit_pair(left, right))
    subsets = [subset_of_train(tables["A4"], tables[name]) for name in ("A12", "A14") if "A4" in tables and name in tables]
    mapping_rows, mapping = [], {}
    mapping_path = files["A16"][0]
    if mapping_path.is_file():
        mapping_rows, mapping = audit_mapping(mapping_path, config["mixture_fields"], root, issues, config["rule_version"])
        by_file[mapping_path.relative_to(root).as_posix()].update(read_status="complete", actual_rows=len(mapping_rows),
                 valid_rows=sum(row["status"] == "valid" for row in mapping_rows),
                 invalid_rows=sum(row["status"] != "valid" for row in mapping_rows))
    # 与压缩文件内容哈希比较，解压副本只作同源检查，不再读成独立样本。
    copy_path = root / "data/real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl/slimpajama_quality_signal_sample.jsonl"
    copy_info = {"source_file": copy_path.relative_to(root).as_posix(), "exists": copy_path.is_file()}
    if copy_path.is_file():
        copy_info["sha256"] = sha256_file(copy_path)
        a1 = next((scan for scan in quality if scan["attachment"] == "A1"), None)
        copy_info["same_source"] = bool(a1 and a1["complete"] and copy_info["sha256"] == a1["decoded_sha256"])
    export_views(db, out, write_csv)
    statuses = dict(db.execute("SELECT status,COUNT(*) FROM records GROUP BY status").fetchall())
    for scan in quality:
        final = dict(db.execute("SELECT status,COUNT(*) FROM records WHERE source_file=? GROUP BY status",
                                (scan["source_file"],)).fetchall())
        scan["final_status_counts"] = final
        row = by_file[scan["source_file"]]
        row["valid_rows"] = final.get("valid", 0)
        row["invalid_rows"] = sum(
            count for status, count in final.items() if status not in ("valid", "duplicate_exact"))
    db.close()
    with db_path.open("rb") as raw, (out / "quality_index.sqlite.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                compressed.write(chunk)
    db_path.unlink()
    for stale in ("quality_source_view.csv", "quality_analysis_view.csv"):
        old = out / stale
        if old.exists():
            old.unlink()
    write_csv(out / "file_inventory.csv", ["attachment", "source_file", "exists", "bytes", "sha256", "manifest_bytes", "manifest_match", "expected_rows", "read_status", "actual_rows", "valid_rows", "invalid_rows"], inventory)
    write_csv(out / "quality_indicator_schema.csv", ["attachment", "source_file", "field", "types", "missing", "invalid", "list_lengths", "element_types", "minimum", "maximum", "element_minimum", "element_maximum"], schema)
    auxiliary = []
    for scan in quality:
        designed = {"id", "sub_path", "content", "_source_domain", "_source_path"} if scan["attachment"] == "A1" else {"id", "sub_path"}
        for field in ["id", "sub_path", "content", "_source_domain", "_source_path"]:
            auxiliary.append(dict(attachment=scan["attachment"], source_file=scan["source_file"], field=field,
                                  present_rows=scan["keys"].get(field, 0), status="observed" if field in designed else "structurally_absent"))
    write_csv(out / "auxiliary_field_report.csv", ["attachment", "source_file", "field", "present_rows", "status"], auxiliary)
    write_csv(out / "invalid_records.csv", ISSUE_FIELDS, (row for row in issues if row["status"] == "rejected_corrupt"))
    write_csv(out / "unverifiable_records.csv", ISSUE_FIELDS, (row for row in issues if row["status"] == "unverifiable"))
    write_csv(out / "suspected_records.csv", ISSUE_FIELDS, (row for row in issues if row["status"] == "suspected_unreliable"))
    table_fields = ["attachment", "source_file", "kind", "rows", "valid", "invalid", "header", "missing_columns", "extra_columns", "max_sum_error"]
    write_csv(out / "schema_report.csv", table_fields, (dict(attachment=name, **{k:v for k,v in table.items() if k in table_fields}) for name,table in tables.items()))
    write_csv(out / "mixture_integrity.csv", table_fields, (dict(attachment=name, **{k:v for k,v in table.items() if k in table_fields}) for name,table in tables.items() if table["kind"] == "mixture"))
    write_csv(out / "loss_integrity.csv", table_fields, (dict(attachment=name, **{k:v for k,v in table.items() if k in table_fields}) for name,table in tables.items() if table["kind"] == "loss"))
    write_csv(out / "pair_integrity.csv", ["mixture_file", "loss_file", "mixture_rows", "loss_rows", "matched_valid_keys", "mixture_only", "loss_only", "one_to_one"], pairs)
    write_csv(out / "estimate_subset_audit.csv", ["estimate_file", "train_file", "missing_train_keys", "different_mixture_keys", "is_subset"], subsets)
    write_csv(out / "domain_mapping_audit.csv", ["source_file", "line_number", "mixture_domain", "quality_domain", "mapping_type", "status", "reason_code"], mapping_rows)
    findings = [f"A1–A3 审计状态：{statuses}。", f"配比与 Loss 的六组连接：{sum(p['one_to_one'] for p in pairs)}/{len(pairs)} 组一一对应。",
                f"A16 17 域映射完整：{mapping.get('valid', False)}。", f"A1 解压副本与压缩内容相同：{copy_info.get('same_source', '未知')}。"]
    if any(not q["complete"] for q in quality):
        findings.append("至少一份 XZ 文件未完整读取；该来源已隔离，不能进入 M1。")
    if len(quality) != 3:
        findings.append("A1–A3 存在缺失附件；M1 输入不完整。")
    if statuses.get("duplicate_conflict", 0):
        findings.append("同 ID 共有字段存在冲突；这些 ID 的所有来源已隔离。")
    for scan in quality:
        expected = config["quality_expected_rows"][scan["attachment"]]
        if scan["lines_read"] != expected:
            findings.append(f"{scan['attachment']} 实际 {scan['lines_read']} 行，与说明预期 {expected} 行不一致；未补造记录。")
    if any(not item["is_subset"] for item in subsets):
        findings.append("A12/A14 与 A4 的子集关系不成立；详见 estimate_subset_audit.csv。")
    if statuses.get("valid", 0):
        findings.append("quality_analysis_view.csv.gz 提供 P1 候选来源指针；指标方向及列表语义仍需人工确认。")
    summary = {"rule_version": config["rule_version"], "config_sha256": sha256_file(config_path),
               "python_version": platform.python_version(), "files": inventory, "quality": quality,
               "record_status_counts": statuses, "issue_status_counts": dict(Counter(x["status"] for x in issues)),
               "pairs": pairs, "estimate_subsets": subsets, "mapping": mapping, "decompressed_copy": copy_info, "findings": findings,
               "m1_ready": False, "m1_blockers": ["三份 Question 1.x Markdown 未找到", "8 项列表的语义与指标方向尚未核定"]}
    usable = {"quality": {item["attachment"]: {"valid_unique_rows": item["final_status_counts"].get("valid", 0),
                                             "complete_source": item["complete"]} for item in quality},
              "mixture_loss_pairs": [{"mixture_file": item["mixture_file"], "loss_file": item["loss_file"],
                                      "matched_valid_keys": item["matched_valid_keys"],
                                      "one_to_one": item["one_to_one"],
                                      "data_role": "training" if index == 0 else "held_out" if index <= 3 else "estimated_reference"}
                                     for index, item in enumerate(pairs)],
              "mapping_complete": mapping.get("valid", False), "p1_ready": False,
              "p1_blockers": summary["m1_blockers"]}
    write_json(out / "usable_data.json", usable)
    write_json(out / "audit_summary.json", summary)
    report(summary, out / "audit_report.md")
    logging.info("审计完成：%s", out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.root.resolve())


if __name__ == "__main__":
    main()
