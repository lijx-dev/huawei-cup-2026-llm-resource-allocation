"""对 P2 落盘结果进行独立的记录数、模型继承与公式复核。"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from q1.audit.export import write_json
from q1.audit.inventory import sha256_file


def verify(root):
    quality=root/"results/q1/quality";conflict=root/"results/q1/conflict";figures=root/"results/q1/figures"
    metadata=json.loads((conflict/"conflict_model_metadata.json").read_text(encoding="utf-8"))
    for relative,digest in metadata["code_sha256"].items():
        if sha256_file(root/relative)!=digest:raise RuntimeError(f"P2 代码校验不一致：{relative}")
    for name,digest in metadata["p1_input_sha256"].items():
        if sha256_file(quality/name)!=digest:raise RuntimeError(f"P1 输入变化：{name}")
    if sha256_file(root/"configs/q1/conflict.json")!=metadata["config_sha256"]:
        raise RuntimeError("P2 配置变化")
    with (quality/"entropy_weights.csv").open(encoding="utf-8",newline="") as stream:
        weights=np.array([float(row["weight"]) for row in csv.DictReader(stream)])
    names=metadata["indicator_names"]
    first,second=np.triu_indices(len(names),1)
    pair_weights=weights[first]*weights[second]
    pair_weights/=pair_weights.sum()
    counts={};duplicates={};checked=0;high=[]
    for quality_file,conflict_file in [("sample_quality_scores.csv.gz","sample_conflict_scores.csv.gz"),
                                       ("extended_quality_scores.csv.gz","extended_conflict_scores.csv.gz")]:
        with gzip.open(quality/quality_file,"rt",encoding="utf-8",newline="") as qstream,gzip.open(conflict/conflict_file,"rt",encoding="utf-8",newline="") as cstream:
            qreader=csv.DictReader(qstream);creader=csv.DictReader(cstream)
            n=0
            for qrow,crow in zip(qreader,creader,strict=True):
                n+=1
                for key in ("record_id","source_file","line_number","record_hash","attachment","domain","audit_status","union_membership"):
                    if qrow[key]!=crow[key]:raise RuntimeError(f"P1/P2 记录指针不一致：{conflict_file}:{n}")
                if crow["quality_model_version"]!=metadata["quality_model_version"] or crow["model_version"]!=metadata["model_version"]:
                    raise RuntimeError("样本级模型版本不一致")
                c=float(crow["C"])
                if not 0<=c<=1 or not math.isclose(float(qrow["Q"]),float(crow["Q"]),abs_tol=1e-10):
                    raise RuntimeError("P2 分数越界或 P1 质量分变化")
                for quantile,tau in zip((85,90,95),metadata["thresholds"]):
                    if (crow[f"high_conflict_{quantile}"]=="True")!=(c>tau):
                        raise RuntimeError("高冲突标记未使用 A1 固定阈值")
                if n%997==0:
                    z=np.array([float(qrow["z_"+name]) for name in names])
                    if not math.isclose(c,float(np.sum(np.abs(z[first]-z[second])*pair_weights)),abs_tol=1e-12):
                        raise RuntimeError("抽检样本的连续冲突公式不一致")
                    checked+=1
                if crow["attachment"]=="A1":duplicates[crow["record_id"]]=c
                elif crow["audit_status"]=="duplicate_exact" and not math.isclose(c,duplicates[crow["record_id"]],abs_tol=1e-12):
                    raise RuntimeError("与 A1 精确重叠的样本 C 不一致")
                if crow["attachment"]=="A1" and c>metadata["thresholds"][1]:high.append(c)
            counts[conflict_file]=n
    if counts!={"sample_conflict_scores.csv.gz":metadata["scored_counts"]["A1:valid"],
                "extended_conflict_scores.csv.gz":sum(n for key,n in metadata["scored_counts"].items() if key.startswith(("A2:","A3:")))}:
        raise RuntimeError("P2 输出行数与输入元数据不一致")
    with gzip.open(conflict/"indicator_pair_contributions.csv.gz","rt",encoding="utf-8",newline="") as stream:
        pairs=[row for row in csv.DictReader(stream) if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]=="0.9"]
    with (conflict/"dimension_weighted_contributions.csv").open(encoding="utf-8",newline="") as stream:
        dimensions=[row for row in csv.DictReader(stream) if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]=="0.9"]
    high_mean=float(np.mean(high))
    if len(pairs)!=231 or not math.isclose(sum(float(row["H_weighted"]) for row in pairs),high_mean,abs_tol=1e-10):
        raise RuntimeError("231 对平均贡献不能重构高冲突样本平均 C")
    if not math.isclose(sum(float(row["mean_weighted_contribution"]) for row in dimensions),high_mean,abs_tol=1e-10):
        raise RuntimeError("维度加权贡献不能重构平均 C")
    with (figures/"p2_figure_manifest.csv").open(encoding="utf-8",newline="") as stream:
        figure_rows=list(csv.DictReader(stream))
    for row in figure_rows:
        path=figures/row["file"]
        if path.suffix==".svg":ET.parse(path)
        elif path.suffix==".png":
            with Image.open(path) as image:image.verify()
    result={"model_version":metadata["model_version"],"scored_rows":counts,"A1_high_conflict_90":len(high),
            "recomputed_record_count":checked,"exact_duplicate_C_consistent":sum(n for key,n in metadata["scored_counts"].items() if key.endswith("duplicate_exact")),
            "pair_contribution_sum":high_mean,"verified_figures":len(figure_rows),"status":"passed"}
    write_json(conflict/"verification_report.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    print(json.dumps(verify(args.root.resolve()),ensure_ascii=False,sort_keys=True))


if __name__=="__main__":main()
