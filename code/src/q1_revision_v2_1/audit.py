"""先于 P1 逐行核清 v2 字段 invalid 的记录级去向。"""
import hashlib
import json
import lzma
import math
from collections import defaultdict

import pandas as pd

from q1.audit.inventory import expected_files
from .common import paths, save_json, sha256, file_manifest


ACCEPTED = {"valid", "duplicate_exact"}


def invalid_count(record, field, length):
    """返回单条记录的非法元素数；缺失/列表形状异常计为一项。"""
    if not isinstance(record, dict) or field not in record or record[field] is None:
        return 1
    value = record[field]
    if length is None:
        values = [value]
    elif not isinstance(value, list) or len(value) != length:
        return 1
    else:
        values = value
    return sum(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values)


def summarize_field(field, found, expected):
    """将元素级 invalid 与行级、ID 级隔离关系分开统计。"""
    raw_elements = sum(x[2] for x in found)
    raw_rows = len(found)
    ids = {x[0] for x in found}
    accepted = [x for x in found if x[1] in ACCEPTED]
    if raw_elements != expected:
        raise RuntimeError(f"{field} 的 v2 字段 invalid 次数无法复算: {raw_elements} != {expected}")
    return {"field": field, "raw_invalid_rows": raw_rows,
            "raw_invalid_elements": raw_elements, "unique_invalid_ids": len(ids),
            "duplicate_invalid_rows": raw_rows-len(ids),
            "already_excluded_structural_invalid": sum(x[1]=="rejected_corrupt" for x in found),
            "already_excluded_suspected": sum(x[1]=="suspected_unreliable" for x in found),
            "remaining_invalid_after_dedup": len({x[0] for x in accepted}),
            "used_in_scoring": bool(accepted),
            "handling_rule": "stop_if_invalid_accepted; otherwise keep_v2_quarantine"}


def run(root, config, v2_config):
    root, out, old = paths(root)
    dest = out / "audit"
    dest.mkdir(parents=True, exist_ok=True)
    old_meta = json.loads((old / "metadata.json").read_text())
    if old_meta["experiment_version"] != "q1-revision-v2" or old_meta["random_seed"] != config["seed"]:
        raise RuntimeError("v2 版本或随机种子不一致")
    if old_meta["config_sha256"] != sha256(root / "configs/q1_revision_v2/config.json"):
        raise RuntimeError("v2 配置已变化，停止复用历史数据")
    for rel, expected in old_meta["code_sha256"].items():
        if sha256(root / rel) != expected:
            raise RuntimeError(f"v2 源代码已变化: {rel}")
    inventory = pd.read_csv(old / "audit/file_inventory.csv")
    for row in inventory.itertuples():
        path = root / row.source_file
        if sha256(path) != row.sha256 or old_meta["input_file_sha256"].get(row.source_file) != row.sha256:
            raise RuntimeError(f"原始附件与 v2 记录不一致: {row.source_file}")
    old_hashes = file_manifest(old)
    save_json(dest / "v2_result_hashes_before.json", old_hashes)
    ledger = pd.read_csv(old / "audit/source_ledger.csv.gz", keep_default_na=False)
    ledger = ledger[ledger.attachment.isin(("A1", "A2", "A3"))]
    lookup = {(r.attachment, int(r.line_number)): r for r in ledger.itertuples()}
    fields = v2_config["quality_fields"]
    if len(fields) != 22 or len(set(fields)) != 22:
        raise RuntimeError("v2 质量字段不是唯一 22 项")
    field_rows = defaultdict(list)
    invalid_details = []
    files = expected_files(root)
    scanned = 0
    for attachment in ("A1", "A2", "A3"):
        with lzma.open(files[attachment][0], "rb") as stream:
            for line, raw in enumerate(stream, 1):
                scanned += 1
                entry = lookup.get((attachment, line))
                if entry is None or hashlib.sha256(raw).hexdigest() != entry.record_hash:
                    raise RuntimeError(f"v2 审计账本与原始行不一致: {attachment}:{line}")
                try:
                    record = json.loads(raw.decode("utf-8"), parse_constant=float)
                except (ValueError, UnicodeDecodeError, TypeError):
                    continue
                for field in fields:
                    count = invalid_count(record, field, v2_config["list_lengths"].get(field))
                    if field in v2_config["log1p_fields"] and count == 0 and record[field] < 0:
                        count = 1
                    if count:
                        rid = str(record.get("id") or f"@{attachment}:{line}")
                        field_rows[field].append((rid, entry.status, count))
                        invalid_details.append({"field": field, "attachment": attachment,
                                                "line_number": line, "record_id": rid,
                                                "record_hash": entry.record_hash,
                                                "status": entry.status, "invalid_elements": count})
    if scanned != len(ledger):
        raise RuntimeError("原始质量行数与 v2 账本不一致")
    old_audit = pd.read_csv(old / "audit/indicator_raw_audit.csv").set_index("field")
    rows = []
    for field in fields:
        found = field_rows[field]
        expected = int(old_audit.loc[field, "invalid"]) + int(old_audit.loc[field, "missing"])
        rows.append(summarize_field(field, found, expected))
    resolution = pd.DataFrame(rows)
    resolution.to_csv(dest / "field_invalid_resolution.csv", index=False)
    pd.DataFrame(invalid_details).to_csv(dest / "field_invalid_rows.csv", index=False)
    if resolution.remaining_invalid_after_dedup.sum() or resolution.used_in_scoring.any():
        raise RuntimeError("非法质量字段仍进入 v2 完整评分，停止 P1")
    scored = pd.read_csv(old / "quality/sample_scores.csv.gz", usecols=["attachment", "line_number", "record_id", "status"])
    if not set(scored.status).issubset(ACCEPTED) or len(scored) != (ledger.status.isin(ACCEPTED)).sum():
        raise RuntimeError("v2 样本评分视图包含隔离行或数量不符")
    summary = {"raw_quality_rows": scanned, "accepted_source_rows": len(scored),
               "valid_unique": int((ledger.status == "valid").sum()),
               "status_counts": ledger.status.value_counts().to_dict(),
               "v2_model_sha256_in_metadata": any("model" in k.lower() and "sha" in k.lower() for k in old_meta),
               "v2_model_file_sha256": old_hashes.get("mixture/fitted_models.joblib")}
    save_json(dest / "audit_summary.json", summary)
    return summary
