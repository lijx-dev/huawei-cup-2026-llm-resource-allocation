"""流式检查质量 JSONL；不解释列表指标的语义。"""
from collections import Counter, defaultdict
import hashlib
import json
import lzma
import math

from .duplicates import classify, shared_hash


def new_stats():
    return defaultdict(lambda: {"types": Counter(), "missing": 0, "invalid": 0,
                                "lengths": Counter(), "element_types": Counter(),
                                "min": None, "max": None, "element_min": None, "element_max": None})


def _range(stat, value, prefix=""):
    lo, hi = prefix + "min", prefix + "max"
    stat[lo] = value if stat[lo] is None else min(stat[lo], value)
    stat[hi] = value if stat[hi] is None else max(stat[hi], value)


def validate(record, config, stats):
    reasons = []
    if not isinstance(record, dict):
        return ["not_object"]
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        reasons.append("missing_or_invalid_id")
    for name in config["quality_scalar_fields"] + config["quality_list_fields"]:
        stat = stats[name]
        value = record.get(name)
        if name not in record or value is None:
            stat["missing"] += 1
            reasons.append("missing_indicator:" + name)
            continue
        stat["types"][type(value).__name__] += 1
        if name in config["quality_list_fields"]:
            if not isinstance(value, list) or not value:
                stat["invalid"] += 1
                reasons.append("invalid_list:" + name)
                continue
            stat["lengths"][len(value)] += 1
            for element in value:
                stat["element_types"][type(element).__name__] += 1
                if isinstance(element, bool) or not isinstance(element, (int, float)) or not math.isfinite(element):
                    stat["invalid"] += 1
                    reasons.append("invalid_list_element:" + name)
                    break
                _range(stat, element, "element_")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            stat["invalid"] += 1
            reasons.append("invalid_scalar:" + name)
        else:
            _range(stat, value)
            if name in config["count_fields"] and value < 0:
                stat["invalid"] += 1
                reasons.append("out_of_range:" + name)
    return reasons


def scan_quality(path, attachment, root, config, db, issues):
    rel = path.relative_to(root).as_posix()
    version = config["rule_version"]
    stats = new_stats()
    fields = config["quality_scalar_fields"] + config["quality_list_fields"]
    counts = Counter()
    keys = Counter()
    decoded = hashlib.sha256()
    line_number = 0
    complete = True
    error = ""
    try:
        with lzma.open(path, "rb") as stream:
            while True:
                raw = stream.readline()
                if not raw:
                    break
                line_number += 1
                decoded.update(raw)
                record_hash = hashlib.sha256(raw).hexdigest()
                try:
                    if not raw.strip():
                        raise ValueError("empty_line")
                    # 让非标准 NaN/Infinity 进入字段级检查，以保留可用的 ID 和具体坏字段。
                    record = json.loads(raw.decode("utf-8"), parse_constant=float)
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    counts["rejected_corrupt"] += 1
                    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                               (rel, attachment, line_number, "", record_hash, "", "rejected_corrupt", "json_parse_error", version))
                    issues.append(dict(source_file=rel, source_format="jsonl.xz", line_number=line_number,
                                       record_id="", record_hash=record_hash, status="rejected_corrupt",
                                       reason_code="json_parse_error", details=str(exc)[:160], rule_version=version))
                    continue
                if isinstance(record, dict):
                    keys.update(record.keys())
                reasons = validate(record, config, stats)
                if reasons:
                    counts["rejected_corrupt"] += 1
                    record_id = record.get("id", "") if isinstance(record, dict) else ""
                    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                               (rel, attachment, line_number, record_id, record_hash, "", "rejected_corrupt", reasons[0], version))
                    issues.append(dict(source_file=rel, source_format="jsonl.xz", line_number=line_number,
                                       record_id=record_id,
                                       record_hash=record_hash, status="rejected_corrupt",
                                       reason_code=reasons[0], details=";".join(reasons)[:500], rule_version=version))
                    continue
                # 文档未给 frac 字段的法定范围；>100 仅按“疑似百分数”假设标记。
                suspect_fields = [name for name in config["fraction_fields"]
                                  if record[name] > config["provisional_percentage_upper_bound"]]
                status = classify(db, rel, attachment, line_number, record["id"],
                                  record_hash, shared_hash(record, fields), version)
                if suspect_fields and status == "valid":
                    status = "suspected_unreliable"
                    db.execute("UPDATE records SET status=?, reason_code=? WHERE source_file=? AND line_number=?",
                               (status, "provisional_percentage_above_100", rel, line_number))
                    issues.append(dict(source_file=rel, source_format="jsonl.xz", line_number=line_number,
                                       record_id=record["id"], record_hash=record_hash, status=status,
                                       reason_code="provisional_percentage_above_100",
                                       details=";".join(suspect_fields), rule_version=version))
                counts[status] += 1
                if line_number % 10000 == 0:
                    db.commit()
    except (lzma.LZMAError, EOFError, OSError) as exc:
        complete = False
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
        issues.append(dict(source_file=rel, source_format="jsonl.xz", line_number=line_number + 1,
                           record_id="", record_hash="", status="unverifiable",
                           reason_code="compressed_stream_incomplete", details=error, rule_version=version))
        db.execute("UPDATE records SET status='unverifiable', reason_code='compressed_stream_incomplete' WHERE source_file=?", (rel,))
    db.commit()
    return {"attachment": attachment, "source_file": rel, "lines_read": line_number,
            "complete": complete, "error": error, "decoded_sha256": decoded.hexdigest() if complete else "",
            "counts": dict(counts), "keys": dict(keys), "stats": stats}


def schema_rows(scan):
    rows = []
    for field, stat in scan["stats"].items():
        rows.append(dict(attachment=scan["attachment"], source_file=scan["source_file"],
                         field=field, types=json.dumps(stat["types"], sort_keys=True),
                         missing=stat["missing"], invalid=stat["invalid"],
                         list_lengths=json.dumps(stat["lengths"], sort_keys=True),
                         element_types=json.dumps(stat["element_types"], sort_keys=True),
                         minimum=stat["min"], maximum=stat["max"],
                         element_minimum=stat["element_min"], element_maximum=stat["element_max"]))
    return rows
