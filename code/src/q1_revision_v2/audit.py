"""重新逐行审计 A1–A16；只以 22 项质量字段比较同 ID。"""
import csv
import hashlib
import json
import lzma
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from q1.audit.inventory import expected_files, manifest_entries
from q1.audit.tables import audit_mapping, audit_pair, read_table, subset_of_train
from .common import paths, sha256, save_json


LEDGER_FIELDS = ["source_file", "attachment", "line_number", "record_id", "record_hash",
                 "indicator_hash", "status", "reason_code", "domain"]


def indicator_hash(record, fields):
    raw = json.dumps({field: record[field] for field in fields}, sort_keys=True,
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check_record(record, config, stats):
    if not isinstance(record, dict):
        return "not_object", ""
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        return "invalid_id", ""
    reason, detail = "", ""
    for name in config["quality_fields"]:
        value = record.get(name)
        stat = stats[name]
        if name not in record or value is None:
            stat["missing"] += 1
            if not reason:
                reason, detail = "missing_indicator", name
            continue
        stat["types"][type(value).__name__] += 1
        length = config["list_lengths"].get(name)
        if length is not None:
            if not isinstance(value, list) or len(value) != length:
                stat["invalid"] += 1
                if not reason:
                    reason, detail = "invalid_list_length_or_type", name
                continue
            values = value
            stat["lengths"][len(value)] += 1
        else:
            values = [value]
        for element in values:
            if isinstance(element, bool) or not isinstance(element, (float, int)) or not math.isfinite(element):
                stat["invalid"] += 1
                if not reason:
                    reason, detail = "invalid_numeric_element", name
                continue
            stat["minimum"] = min(stat["minimum"], element)
            stat["maximum"] = max(stat["maximum"], element)
        if name in config["log1p_fields"] and isinstance(value, (float, int)) and value < 0:
            stat["invalid"] += 1
            if not reason:
                reason, detail = "negative_log_input", name
    return reason, detail


def audit_quality(root, out, config, sources):
    db_path = out / "audit/quality_ledger.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE records (source_file TEXT,attachment TEXT,line_number INTEGER,record_id TEXT,record_hash TEXT,indicator_hash TEXT,status TEXT,reason_code TEXT,domain TEXT)")
    db.execute("CREATE INDEX records_id ON records(record_id)")
    db.execute("CREATE INDEX records_source_line ON records(source_file,line_number)")
    stats = defaultdict(lambda: {"types": Counter(), "lengths": Counter(), "minimum": float("inf"),
                                 "maximum": float("-inf"), "missing": 0, "invalid": 0})
    files_report = []
    domains = set(config["quality_domain_order"])
    for attachment, path in sources:
        source = str(path.relative_to(root))
        counts = Counter()
        complete = True
        try:
            with lzma.open(path, "rb") as stream:
                for line, raw in enumerate(stream, 1):
                    counts["rows"] += 1
                    digest = hashlib.sha256(raw).hexdigest()
                    reason, detail, record = "", "", None
                    try:
                        record = json.loads(raw.decode("utf-8"), parse_constant=float)
                        reason, detail = check_record(record, config, stats)
                    except (ValueError, UnicodeDecodeError, TypeError) as exc:
                        reason, detail = "json_parse_error", str(exc)[:120]
                    rid = record.get("id", "") if isinstance(record, dict) else ""
                    domain = record.get("_source_domain", "") if attachment == "A1" and isinstance(record, dict) else ("arxiv" if attachment == "A2" else "github")
                    if not reason and domain not in domains:
                        reason, detail = "unknown_domain", str(domain)
                    if reason:
                        status, ihash = "rejected_corrupt", ""
                    else:
                        ihash = indicator_hash(record, config["quality_fields"])
                        suspect = [name for name in config["quality_fields"] if (name.startswith("rps_") and
                                   ("frac" in name or "fraction" in name) and not isinstance(record[name], list) and record[name] > 100)]
                        if suspect:
                            status, reason, detail = "suspected_unreliable", "provisional_fraction_above_100", ",".join(suspect)
                        else:
                            status = "pending"
                    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                               (source, attachment, line, rid, digest, ihash, status, reason + (":" + detail if detail else ""), domain))
                    counts[status] += 1
                    if line % 10000 == 0:
                        db.commit()
        except (lzma.LZMAError, EOFError, OSError) as exc:
            complete = False
            db.execute("UPDATE records SET status='unverifiable',reason_code='compressed_stream_incomplete' WHERE source_file=?", (source,))
            counts["stream_error"] += 1
            files_report.append({"attachment": attachment, "source_file": source, "rows_read": counts["rows"],
                                 "complete": False, "error": f"{type(exc).__name__}: {exc}"[:180]})
        if complete:
            files_report.append({"attachment": attachment, "source_file": source, "rows_read": counts["rows"],
                                 "complete": True, "error": ""})
        db.commit()
    if not all(row["complete"] for row in files_report):
        db.close()
        raise RuntimeError("质量 XZ 来源读取不完整；已保留部分审计 SQLite，停止评分")
    conflict_ids = [row[0] for row in db.execute("SELECT record_id FROM records WHERE status IN ('pending','suspected_unreliable') GROUP BY record_id HAVING COUNT(DISTINCT indicator_hash)>1")]
    for rid in conflict_ids:
        db.execute("UPDATE records SET status='duplicate_conflict',reason_code='same_id_different_22_indicators' WHERE record_id=? AND status IN ('pending','suspected_unreliable')", (rid,))
    seen = set()
    for source, line, rid in db.execute("SELECT source_file,line_number,record_id FROM records WHERE status='pending' ORDER BY attachment,line_number"):
        status, reason = ("duplicate_exact", "same_id_same_22_indicators") if rid in seen else ("valid", "checks_passed")
        db.execute("UPDATE records SET status=?,reason_code=? WHERE source_file=? AND line_number=?", (status, reason, source, line))
        seen.add(rid)
    db.commit()
    pd.read_sql_query("SELECT * FROM records ORDER BY attachment,line_number", db).to_csv(out / "audit/source_ledger.csv.gz", index=False, compression="gzip")
    pd.read_sql_query("SELECT * FROM records WHERE status='duplicate_conflict' ORDER BY attachment,line_number", db).to_csv(out / "audit/duplicate_conflicts.csv", index=False)
    pd.read_sql_query("SELECT * FROM records WHERE status IN ('rejected_corrupt','suspected_unreliable','unverifiable') ORDER BY attachment,line_number", db).to_csv(out / "audit/exclusions.csv", index=False)
    status = dict(db.execute("SELECT status,COUNT(*) FROM records GROUP BY status").fetchall())
    attachment_counts = pd.read_sql_query("SELECT attachment,status,COUNT(*) AS n FROM records GROUP BY attachment,status", db)
    attachment_counts.to_csv(out / "audit/attachment_status.csv", index=False)
    indicator_rows = []
    for name in config["quality_fields"]:
        s = stats[name]
        indicator_rows.append({"field": name, "raw_types": json.dumps(s["types"], sort_keys=True),
                               "list_lengths": json.dumps(s["lengths"], sort_keys=True), "missing": s["missing"],
                               "invalid": s["invalid"], "element_min": s["minimum"] if s["minimum"] != float("inf") else None,
                               "element_max": s["maximum"] if s["maximum"] != float("-inf") else None})
    pd.DataFrame(indicator_rows).to_csv(out / "audit/indicator_raw_audit.csv", index=False)
    db.close()
    return {"files": files_report, "status_counts": status, "conflicting_ids": len(conflict_ids),
            "union_unique_valid": status.get("valid", 0), "ledger_path": str(db_path.relative_to(root))}


def audit_tables(root, out, config, files):
    first_mix = pd.read_csv(files["A4"][0], nrows=0)
    first_loss = pd.read_csv(files["A5"][0], nrows=0)
    mixture_fields = [c for c in first_mix if c.startswith("train_the_pile_")]
    loss_fields = [c for c in first_loss if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    if len(mixture_fields) != 17 or len(loss_fields) != 13:
        raise RuntimeError("配比或 Loss 实际字段数量不符")
    table_config = {"rule_version": config["experiment_version"], "mixture_fields": mixture_fields,
                    "loss_fields": loss_fields, "mixture_sum_tolerance": config["mixture_sum_tolerance"]}
    issues, tables, pairs = [], {}, []
    for number in range(4, 16):
        name = f"A{number}"
        tables[name] = read_table(files[name][0], "mixture" if number % 2 == 0 else "loss", table_config, root, issues)
    for number in range(4, 16, 2):
        pairs.append({"pair": f"A{number}+A{number+1}", **audit_pair(tables[f"A{number}"], tables[f"A{number+1}"])})
    mapping_rows, mapping = audit_mapping(files["A16"][0], mixture_fields, root, issues, config["experiment_version"])
    if issues or not all(p["one_to_one"] for p in pairs) or not mapping["valid"]:
        pd.DataFrame(issues).to_csv(out / "audit/table_issues.csv", index=False)
        raise RuntimeError("A4–A16 有无效行或连接/映射缺陷，停止受影响阶段")
    pd.DataFrame(pairs).to_csv(out / "audit/pair_integrity.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(out / "audit/domain_mapping_audit.csv", index=False)
    subsets = [subset_of_train(tables["A4"], tables[name]) for name in ("A12", "A14")]
    pd.DataFrame(subsets).to_csv(out / "audit/estimate_subset_audit.csv", index=False)
    return {"mixture_fields": mixture_fields, "loss_fields": loss_fields, "pairs": pairs,
            "mapping": mapping, "estimated_train_subsets": subsets}


def run(root, out, config):
    root, out = paths(root)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    files = expected_files(root)
    manifest = manifest_entries(root)
    inventory = []
    for label, paths_list in files.items():
        if len(paths_list) != 1 or not paths_list[0].is_file():
            raise RuntimeError(f"{label} 缺失或文件数异常")
        p = paths_list[0]
        key = str(p.relative_to(root / "data/real_attachments"))
        item = manifest.get(key, {})
        inventory.append({"attachment": label, "source_file": str(p.relative_to(root)), "bytes": p.stat().st_size,
                          "manifest_bytes": item.get("bytes"), "manifest_match": p.stat().st_size == item.get("bytes"),
                          "sha256": sha256(p)})
    pd.DataFrame(inventory).to_csv(out / "audit/file_inventory.csv", index=False)
    if not all(row["manifest_match"] for row in inventory):
        raise RuntimeError("文件大小与来源清单不一致；已保存清单，停止")
    sources = [(name, files[name][0]) for name in ("A1", "A2", "A3")]
    quality = audit_quality(root, out, config, sources)
    tables = audit_tables(root, out, config, files)
    summary = {"experiment_version": config["experiment_version"], "quality": quality, "tables": tables}
    save_json(out / "audit/audit_summary.json", summary)
    return summary
