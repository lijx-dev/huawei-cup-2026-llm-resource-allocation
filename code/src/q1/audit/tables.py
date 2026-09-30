"""配比、Loss 与领域映射的 CSV 结构审计。"""
import csv
from collections import Counter
import hashlib
import json
import math


def _number(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_table(path, kind, config, root, issues):
    rel = path.relative_to(root).as_posix()
    version = config["rule_version"]
    expected = config["mixture_fields" if kind == "mixture" else "loss_fields"]
    rows, valid_keys, seen_keys, values_by_key, counts = [], set(), set(), {}, Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = sorted(set(["index", *expected]) - set(header))
        extra = sorted(set(header) - set(["index", *expected]))
        for line, row in enumerate(reader, 2):
            counts["rows"] += 1
            key = row.get("index", "")
            reasons = []
            if not key or not key.strip():
                reasons.append("missing_index")
            if key in seen_keys:
                reasons.append("duplicate_index")
            seen_keys.add(key)
            if missing or None in row:
                reasons.append("schema_mismatch")
            values = [_number(row.get(field)) for field in expected]
            if any(value is None for value in values):
                reasons.append("nonfinite_or_missing_number")
            elif kind == "mixture":
                if any(value < 0 for value in values):
                    reasons.append("negative_mixture")
                total = sum(values)
                if abs(total - 1) > config["mixture_sum_tolerance"]:
                    reasons.append("mixture_sum_outside_rounding_tolerance")
                counts["max_sum_error"] = max(counts["max_sum_error"], abs(total - 1))
            elif any(value < 0 for value in values):
                reasons.append("negative_loss")
            if reasons:
                counts["invalid"] += 1
                issues.append(dict(source_file=rel, source_format="csv", line_number=line, record_id=key,
                                   record_hash=_row_hash(row), status="rejected_corrupt", reason_code=reasons[0],
                                   details=";".join(reasons), rule_version=version))
            else:
                counts["valid"] += 1
                valid_keys.add(key)
                values_by_key[key] = values
            rows.append((key, line, not reasons))
    return dict(source_file=rel, kind=kind, rows=counts["rows"], valid=counts["valid"],
                invalid=counts["invalid"], header=header, missing_columns=missing,
                extra_columns=extra, max_sum_error=counts["max_sum_error"] if kind == "mixture" else "",
                keys=valid_keys, values_by_key=values_by_key, all_rows=rows)


def subset_of_train(train, estimate):
    missing = sorted(estimate["keys"] - train["keys"])
    different = sorted(key for key in estimate["keys"] & train["keys"]
                       if estimate["values_by_key"][key] != train["values_by_key"][key])
    return {"estimate_file": estimate["source_file"], "train_file": train["source_file"],
            "missing_train_keys": missing, "different_mixture_keys": different,
            "is_subset": not missing and not different}


def audit_pair(left, right):
    a, b = left["keys"], right["keys"]
    return dict(mixture_file=left["source_file"], loss_file=right["source_file"],
                mixture_rows=left["rows"], loss_rows=right["rows"],
                matched_valid_keys=len(a & b), mixture_only=len(a - b), loss_only=len(b - a),
                one_to_one=(left["invalid"] == 0 and right["invalid"] == 0 and a == b))


def audit_mapping(path, mixture_fields, root, issues, version):
    rel = path.relative_to(root).as_posix()
    expected = {name.removeprefix("train_the_pile_") for name in mixture_fields}
    seen = set()
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing_fields = sorted({"mixture_domain", "quality_domain", "mapping_type"} - set(header))
        for line, row in enumerate(reader, 2):
            domain = row.get("mixture_domain", "")
            reason = ""
            if missing_fields or None in row:
                reason = "schema_mismatch"
            elif not domain or not row.get("quality_domain"):
                reason = "missing_mapping_value"
            elif domain in seen:
                reason = "duplicate_mapping"
            elif row.get("mapping_type") not in {"direct", "near_direct", "inferred"}:
                reason = "unknown_mapping_type"
            seen.add(domain)
            rows.append(dict(source_file=rel, line_number=line, mixture_domain=domain,
                             quality_domain=row.get("quality_domain", ""), mapping_type=row.get("mapping_type", ""),
                             status="valid" if not reason else "rejected_corrupt", reason_code=reason))
            if reason:
                issues.append(dict(source_file=rel, source_format="csv", line_number=line, record_id=domain,
                                   record_hash=_row_hash(row), status="rejected_corrupt", reason_code=reason,
                                   details="", rule_version=version))
    return rows, dict(source_file=rel, rows=len(rows), missing_columns=missing_fields,
                      missing_domains=sorted(expected - seen), unexpected_domains=sorted(seen - expected),
                      mapping_types=dict(Counter(row["mapping_type"] for row in rows)),
                      valid=(not missing_fields and not expected - seen and not seen - expected and
                             all(row["status"] == "valid" for row in rows)))
