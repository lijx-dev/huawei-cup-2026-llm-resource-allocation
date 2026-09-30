"""P1 正式入口：仅复用 P0 记录状态，按 A1 拟合并对 A2/A3 固定应用。"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import logging
import lzma
import math
from pathlib import Path
import platform

from q1.audit.export import write_csv, write_json
from q1.audit.inventory import sha256_file
from .domain_quality import summarize_domain
from .entropy import entropy_spearman_weights
from .normalization import fit_minmax, transform_minmax
from .preprocessing import load_schema, require_schema_ready, scalarize
from .scoring import score_row, topsis_closeness
from .validation import compare_scores


def load_references(path):
    selected = defaultdict(dict)
    counts = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] not in {"valid", "duplicate_exact"}:
                continue
            if row["attachment"] == "A1" and row["status"] != "valid":
                continue
            selected[row["source_file"]][int(row["line_number"])] = row
            counts[(row["attachment"], row["status"])] += 1
    return selected, counts


def audit_a1_domains(path, root, source_view):
    """按 P0 行状态统计 A1 领域；无法解析的行单列为 unknown。"""
    relative = path.relative_to(root).as_posix()
    statuses = {}
    with gzip.open(source_view, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source_file"] == relative and row["attachment"] == "A1":
                statuses[int(row["line_number"])] = row["status"]
    counts = Counter()
    with lzma.open(path, "rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if line_number not in statuses:
                raise RuntimeError(f"A1 行缺少 P0 状态：{relative}:{line_number}")
            try:
                domain = json.loads(raw.decode("utf-8")).get("_source_domain") or "unknown"
            except (UnicodeError, ValueError, TypeError):
                domain = "unknown"
            counts[(domain, statuses[line_number])] += 1
    if sum(counts.values()) != len(statuses):
        raise RuntimeError(f"A1 P0 状态行数与原始文件不符：{relative}")
    return counts


def iter_audited_records(path, root, references):
    relative = path.relative_to(root).as_posix()
    selected = references.get(relative, {})
    found = 0
    with lzma.open(path, "rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            ref = selected.get(line_number)
            if ref is None:
                continue
            if hashlib.sha256(raw).hexdigest() != ref["record_hash"]:
                raise RuntimeError(f"P0 后原始记录已变化：{relative}:{line_number}")
            record = json.loads(raw.decode("utf-8"), parse_constant=float)
            if record["id"] != ref["record_id"]:
                raise RuntimeError(f"P0 记录 ID 不一致：{relative}:{line_number}")
            found += 1
            yield record, ref
    if found != len(selected):
        raise RuntimeError(f"P0 记录指针未全部定位：{relative}，{found}/{len(selected)}")


def prepare(record, specs):
    values = []
    for spec in specs:
        raw = record[spec["name"]]
        bounds = spec["allowed_raw_range"]
        if bounds is not None and not isinstance(raw, list):
            low, high = bounds
            if low is not None and raw < low or high is not None and raw > high:
                raise ValueError(f"{spec['name']}: 超出配置合法范围")
        values.append(scalarize(raw, spec))
    return values


def _stats_bucket():
    return {"scores": [], "equal": [], "in_reference": [], "indicator_sum": None,
            "indicator_sumsq": None, "hist": [0]*20}


def _add_stats(bucket, normalized, score, inside):
    bucket["scores"].append(score["Q"])
    bucket["equal"].append(score["Q_equal"])
    if inside:
        bucket["in_reference"].append(score["Q"])
    if bucket["indicator_sum"] is None:
        bucket["indicator_sum"] = [0.0]*len(normalized)
        bucket["indicator_sumsq"] = [0.0]*len(normalized)
    for j,value in enumerate(normalized):
        bucket["indicator_sum"][j] += value
        bucket["indicator_sumsq"][j] += value*value
    bucket["hist"][min(19,int(score["Q"]//5))] += 1


def write_clipping_summary(output, metadata, fields):
    rows = []
    for source in ("A2", "A3"):
        n = sum(count for key,count in metadata["scored_counts"].items() if key.startswith(source+":"))
        if n == 0:
            raise RuntimeError(f"{source} 没有成功评分的记录")
        for name in fields:
            count = metadata["clipped_by_source"].get(source, {}).get(name, 0)
            rows.append({"attachment":source,"name":name,"scored_n":n,
                         "clipped_n":count,"clipped_rate":count/n})
    write_csv(output / "boundary_clipping_by_indicator.csv",
              ["attachment","name","scored_n","clipped_n","clipped_rate"],rows)


def run(root):
    audit_dir = root / "results/q1/audit"
    output = root / "results/q1/quality"
    figures = root / "results/q1/figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    schema_path = root / "configs/q1/indicator_schema.json"
    config_path = root / "configs/q1/quality.json"
    audit_config_path = root / "configs/q1/audit.json"
    design_path = root / "question1-1(1).md"
    schema = load_schema(schema_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit_config = json.loads(audit_config_path.read_text(encoding="utf-8"))
    require_schema_ready(schema, audit_config["quality_scalar_fields"], audit_config["quality_list_fields"])
    if not design_path.is_file():
        raise RuntimeError("缺少 question1-1(1).md")
    audit = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    sources = [entry for entry in audit["files"] if entry["attachment"] in {"A1", "A2", "A3"}]
    if len(sources) != 3 or any(entry["read_status"] != "complete" for entry in sources):
        raise RuntimeError("P0 质量来源不完整")
    for entry in sources:
        path = root / entry["source_file"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"输入文件 SHA-256 与 P0 不符：{path}")
    refs, expected = load_references(audit_dir / "quality_source_view.csv.gz")
    specs = schema["indicators"]
    fields = [s["name"] for s in specs]
    directions = [s["direction"] for s in specs]
    a1 = []
    exclusions = []
    for entry in sources:
        if entry["attachment"] != "A1":
            continue
        for record, ref in iter_audited_records(root / entry["source_file"], root, refs):
            try:
                values = prepare(record, specs)
            except (KeyError, ValueError) as exc:
                exclusions.append({"source_file":ref["source_file"],"line_number":ref["line_number"],
                                   "record_id":ref["record_id"],"reason":str(exc)[:160]})
                continue
            domain = record.get("_source_domain")
            if domain not in {"arxiv","book","c4","commoncrawl","github","stackexchange","wikipedia"}:
                raise RuntimeError(f"A1 领域未确认：{ref['source_file']}:{ref['line_number']}")
            a1.append((values, record["qurater"], domain, ref))
    if len(a1) < 2:
        raise RuntimeError("A1 可拟合样本不足")
    parameters = fit_minmax([item[0] for item in a1], directions)
    normalized_a1 = [transform_minmax(item[0], parameters, "reject_out_of_reference")[0] for item in a1]
    model = entropy_spearman_weights(normalized_a1)
    weights = model["weights"]
    logging.info("A1 完整案例 %s；权重和 %.12f", len(a1), math.fsum(weights))
    normalization_records = [dict(name=spec["name"], scalarization=spec["scalarization"],
                                  fit_count=len(a1), **param)
                             for spec,param in zip(specs,parameters)]
    write_json(output / "normalization_parameters.json", normalization_records)
    write_csv(output / "normalization_parameters.csv", ["name","scalarization","fit_count","minimum","maximum","direction"],
              normalization_records)
    write_csv(output / "scalarization_report.csv",
              ["name","raw_type","expected_length","scalarization","direction","transformed_min_A1","transformed_max_A1","fit_count","model_assumption"],
              ({"name":spec["name"],"raw_type":spec["raw_type"],"expected_length":spec["expected_length"],
                "scalarization":spec["scalarization"],"direction":spec["direction"],
                "transformed_min_A1":param["minimum"],"transformed_max_A1":param["maximum"],
                "fit_count":len(a1),"model_assumption":spec.get("model_assumption","")}
               for spec,param in zip(specs,parameters)))
    order = sorted(range(22), key=lambda j:weights[j], reverse=True)
    rank = {j:i+1 for i,j in enumerate(order)}
    write_csv(output / "entropy_weights.csv", ["name","entropy","difference","independence","information","weight","rank"],
              ({"name":fields[j],"entropy":model["entropy"][j],"difference":model["difference"][j],
                "independence":model["independence"][j],"information":model["information"][j],
                "weight":weights[j],"rank":rank[j]} for j in range(22)))
    write_csv(output / "spearman_correlation.csv", ["name",*fields],
              (dict(name=name, **dict(zip(fields,row))) for name,row in zip(fields,model["spearman"])))
    write_csv(output / "indicator_metadata.csv", ["name","raw_type","expected_length","scalarization","direction","semantic","model_assumption"], specs)
    score_fields = ["record_id","source_file","line_number","record_hash","attachment","domain","audit_status","model_version",
                    "union_membership","q","Q","Q_equal", "C_topsis",*['z_'+name for name in fields]]
    buckets = defaultdict(_stats_bucket)
    clipped = defaultdict(Counter)
    scored = Counter()
    comparison_main, comparison_equal, comparison_topsis = [], [], []
    def emit(values, domain, ref, attachment):
        z, flags = transform_minmax(values, parameters, config["out_of_reference"])
        score = score_row(z,weights)
        for name,flag in zip(fields,flags):
            if flag: clipped[attachment][name] += 1
        inside = not any(flags)
        for view in [attachment, "union" if ref["status"] == "valid" else None,
                     attachment+"_nonoverlap" if attachment in {"A2", "A3"} and ref["status"] == "valid" else None]:
            if view:
                _add_stats(buckets[(view,domain)], z, score, inside)
        scored[(attachment,ref["status"])] += 1
        if attachment == "A1":
            comparison_main.append(score["Q"])
            comparison_equal.append(score["Q_equal"])
            comparison_topsis.append(100*topsis_closeness(z,weights))
        return {"record_id":ref["record_id"],"source_file":ref["source_file"],"line_number":ref["line_number"],
                "record_hash":ref["record_hash"],"attachment":attachment,"domain":domain,"audit_status":ref["status"],
                "model_version":config["model_version"],
                "union_membership":ref["status"] == "valid",**score,"C_topsis":topsis_closeness(z,weights),
                **{'z_'+name:value for name,value in zip(fields,z)}}
    def a1_rows():
        for values,_,domain,ref in a1:
            yield emit(values,domain,ref,"A1")
    write_csv(output / "sample_quality_scores.csv.gz", score_fields, a1_rows())
    def extended_rows():
        for entry in sources:
            attachment = entry["attachment"]
            if attachment == "A1":
                continue
            for record,ref in iter_audited_records(root / entry["source_file"],root,refs):
                try:
                    values = prepare(record,specs)
                except (KeyError,ValueError) as exc:
                    exclusions.append({"source_file":ref["source_file"],"line_number":ref["line_number"],
                                       "record_id":ref["record_id"],"reason":str(exc)[:160]})
                    continue
                yield emit(values,"arxiv" if attachment == "A2" else "github",ref,attachment)
    write_csv(output / "extended_quality_scores.csv.gz", score_fields, extended_rows())
    write_csv(output / "scoring_exclusions.csv", ["source_file","line_number","record_id","reason"], exclusions)
    a1_source = next(s for s in sources if s["attachment"] == "A1")
    a1_domain_counts = audit_a1_domains(root / a1_source["source_file"], root, audit_dir / "quality_source_view.csv.gz")
    domain_exclusions = []
    for domain in sorted({d for d,_ in a1_domain_counts}):
        raw_n = sum(n for (d,_),n in a1_domain_counts.items() if d == domain)
        valid_n = a1_domain_counts[(domain,"valid")]
        domain_exclusions.append({"domain":domain,"raw_n":raw_n,"p0_valid_n":valid_n,
                                  "rejected_corrupt_n":a1_domain_counts[(domain,"rejected_corrupt")],
                                  "suspected_unreliable_n":a1_domain_counts[(domain,"suspected_unreliable")],
                                  "unverifiable_n":a1_domain_counts[(domain,"unverifiable")],
                                  "p1_scored_n":len(buckets[("A1",domain)]["scores"]) if ("A1",domain) in buckets else 0,
                                  "excluded_n":raw_n-(len(buckets[("A1",domain)]["scores"]) if ("A1",domain) in buckets else 0),
                                  "excluded_rate":(raw_n-(len(buckets[("A1",domain)]["scores"]) if ("A1",domain) in buckets else 0))/raw_n})
    write_csv(output / "domain_exclusions.csv",list(domain_exclusions[0]),domain_exclusions)
    rows = []
    for (view,domain),bucket in sorted(buckets.items()):
        info = summarize_domain(bucket["scores"])
        info.update(view=view,domain=domain,in_reference_n=len(bucket["in_reference"]),
                    in_reference_mean_Q=(math.fsum(bucket["in_reference"])/len(bucket["in_reference"])) if bucket["in_reference"] else "",
                    mean_equal_Q=math.fsum(bucket["equal"])/len(bucket["equal"]))
        rows.append(info)
    domain_fields = ["view","domain","n","mean_Q","median_Q","std_Q","q25_Q","q75_Q","ci95_low_Q","ci95_high_Q",
                     "ci_method","in_reference_n","in_reference_mean_Q","mean_equal_Q"]
    write_csv(output / "domain_quality_summary.csv",domain_fields,rows)
    by_key = {(r["view"],r["domain"]):r for r in rows}
    comparisons=[]
    for source,domain in [("A2","arxiv"),("A3","github")]:
        sample=by_key[("A1",domain)]; extension=by_key[(source,domain)]; nonoverlap=by_key[(source+"_nonoverlap",domain)]
        overlap=expected[(source,"duplicate_exact")]
        comparisons.append({"domain":domain,"a1_n":sample["n"],"extension_n":extension["n"],
                            "extension_nonoverlap_n":nonoverlap["n"],
                            "overlap_n":overlap,"overlap_fraction":overlap/extension["n"],
                            "a1_mean_Q":sample["mean_Q"],"extension_mean_Q":extension["mean_Q"],
                            "extension_nonoverlap_mean_Q":nonoverlap["mean_Q"],
                            "extension_median_Q":extension["median_Q"],"extension_std_Q":extension["std_Q"],
                            "mean_difference_Q":extension["mean_Q"]-sample["mean_Q"],
                            "in_reference_n":extension["in_reference_n"],
                            "in_reference_mean_Q":extension["in_reference_mean_Q"]})
    write_csv(output / "extended_domain_comparison.csv", list(comparisons[0]), comparisons)
    write_csv(output / "boundary_sensitivity.csv",["attachment","domain","n","in_reference_n","clipped_record_n","main_mean_Q","in_reference_mean_Q","difference_Q"],
              ({"attachment":source,"domain":domain,"n":by_key[(source,domain)]["n"],
                "in_reference_n":by_key[(source,domain)]["in_reference_n"],
                "clipped_record_n":by_key[(source,domain)]["n"]-by_key[(source,domain)]["in_reference_n"],
                "main_mean_Q":by_key[(source,domain)]["mean_Q"],
                "in_reference_mean_Q":by_key[(source,domain)]["in_reference_mean_Q"],
                "difference_Q":by_key[(source,domain)]["mean_Q"]-by_key[(source,domain)]["in_reference_mean_Q"]}
               for source,domain in [("A2","arxiv"),("A3","github")]))
    indicator_comparisons=[]
    for source,domain in [("A2","arxiv"),("A3","github")]:
        for j,name in enumerate(fields):
            a=buckets[("A1",domain)]; b=buckets[(source,domain)]
            for bucket,view in [(a,"A1"),(b,source)]:
                n=len(bucket["scores"]); mean=bucket["indicator_sum"][j]/n
                variance=max(0.0,bucket["indicator_sumsq"][j]/n-mean*mean)
                indicator_comparisons.append({"domain":domain,"view":view,"name":name,"n":n,
                                              "normalized_mean":mean,"normalized_std":math.sqrt(variance)})
    write_csv(output / "indicator_domain_comparison.csv",["domain","view","name","n","normalized_mean","normalized_std"],indicator_comparisons)
    robustness=[{"comparison":"equal_weight_A1",**compare_scores(comparison_main,comparison_equal)},
                {"comparison":"weighted_topsis_A1",**compare_scores(comparison_main,comparison_topsis)}]
    # QuRater 四个独立评分并非有序等级；等权均值仅作预先指定的替代口径。
    q_index=fields.index("qurater")
    alternative=[]
    for values,raw_q,_,_ in a1:
        row=values.copy();row[q_index]=math.fsum(raw_q)/4
        alternative.append(row)
    alt_params=fit_minmax(alternative,directions)
    alt_z=[transform_minmax(row,alt_params,"reject_out_of_reference")[0] for row in alternative]
    alt_weights=entropy_spearman_weights(alt_z)["weights"]
    alt_scores=[score_row(z,alt_weights)["Q"] for z in alt_z]
    robustness.append({"comparison":"qurater_four_dimension_mean_A1",**compare_scores(comparison_main,alt_scores)})
    write_csv(output / "quality_robustness.csv",["comparison","n","pearson","spearman","mean_difference"],robustness)
    metadata={"model_version":config["model_version"],"schema_version":schema["schema_version"],
              "input_sha256":{s["attachment"]:s["sha256"] for s in sources},
              "code_sha256":{path.relative_to(root).as_posix():sha256_file(path)
                             for path in sorted((root / "src/q1/quality").glob("*.py"))},
              "audit_summary_sha256":sha256_file(audit_dir / "audit_summary.json"),
              "design_sha256":sha256_file(design_path),"schema_sha256":sha256_file(schema_path),
              "config_sha256":sha256_file(config_path),"python_version":platform.python_version(),
              "fit_count_A1":len(a1),"scored_counts":{f"{a}:{status}":n for (a,status),n in scored.items()},
              "p0_status_counts":audit["record_status_counts"],"p1_exclusions":len(exclusions),
              "clipped_by_source":{a:dict(c) for a,c in clipped.items()},
              "normalization_policy":config["out_of_reference"],
              "uncertainty_method":config["domain_ci_method"],"quality_weight_note":"统计差异与冗余权重，非训练效果的因果重要性"}
    write_json(output / "quality_model_metadata.json",metadata)
    write_clipping_summary(output,metadata,fields)
    from .visualization import make_all_figures
    make_all_figures(figures,fields,weights,model["spearman"],buckets,rows,comparisons,comparison_main,comparison_equal)
    from .report import write_report
    write_report(output / "quality_report.md",metadata,rows,comparisons,robustness,config,domain_exclusions)
    logging.info("P1 完成：%s", output)
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run(args.root.resolve())


if __name__ == "__main__":
    main()
