"""按 ID 和共有字段判断重复，保留完整来源定位。"""
import hashlib
import json
import sqlite3


def shared_hash(record, fields):
    shared = {name: record[name] for name in ["id", "sub_path", *fields] if name in record}
    payload = json.dumps(shared, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_db(path):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS records (source_file TEXT, attachment TEXT, line_number INTEGER, record_id TEXT, record_hash TEXT, shared_hash TEXT, status TEXT, reason_code TEXT, rule_version TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS records_id ON records(record_id)")
    db.execute("CREATE INDEX IF NOT EXISTS records_source ON records(source_file)")
    db.commit()
    return db


def classify(db, source, attachment, line, record_id, record_hash, comparable_hash, version):
    prior = db.execute("SELECT shared_hash, status FROM records WHERE record_id=? AND status!='rejected_corrupt' LIMIT 1", (record_id,)).fetchone()
    if prior is None:
        status, reason = "valid", "checks_passed"
    elif prior[0] == comparable_hash and prior[1] != "duplicate_conflict":
        status, reason = "duplicate_exact", "same_id_shared_fields"
    else:
        status, reason = "duplicate_conflict", "same_id_conflicting_shared_fields"
        db.execute("UPDATE records SET status='duplicate_conflict', reason_code=? WHERE record_id=? AND status!='rejected_corrupt'",
                   (reason, record_id))
    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
               (source, attachment, line, record_id, record_hash, comparable_hash, status, reason, version))
    return status


def export_views(db, out, write_csv):
    columns = ["source_file", "attachment", "line_number", "record_id", "record_hash",
               "shared_hash", "status", "reason_code", "rule_version"]
    for name, where in [("quality_source_view.csv.gz", "1=1"),
                        ("quality_analysis_view.csv.gz", "status='valid'"),
                        ("duplicate_records.csv", "status='duplicate_exact'"),
                        ("duplicate_conflicts.csv", "status='duplicate_conflict'")]:
        rows = db.execute(f"SELECT * FROM records WHERE {where} ORDER BY source_file,line_number")
        write_csv(out / name, columns, (dict(zip(columns, row)) for row in rows))
