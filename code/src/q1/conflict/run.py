"""P2 可复现实验入口：P1 固定模型 → 聚类 → 冲突 → 扩展集验证。"""
import argparse
from collections import Counter
import csv
import json
import logging
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from q1.audit.export import write_csv, write_json
from q1.audit.inventory import sha256_file
from .analysis import Summary, compare_extensions, domain_rows, relation_rows
from .clustering import adjusted_rand, checked_dimensions, distance_matrix, evaluate
from .input import iter_p1_batches, load_p1_model
from .scoring import prepare_pairs, score_batch


def fixed_reference_thresholds(path,names,weights,pairs,version,chunk_size,quantiles):
    first,second,pair_weights,_=pairs
    conflicts=[];quality=[];count=0
    for rows,z,q in iter_p1_batches(path,names,weights,version,chunk_size,{"A1"}):
        conflicts.append(np.sum(np.abs(z[:,first]-z[:,second])*pair_weights,axis=1))
        quality.append(q)
        count+=len(rows)
    if count<100:
        raise RuntimeError("A1 合格样本不足，不能确定分位数阈值")
    c=np.concatenate(conflicts)
    return np.quantile(c,quantiles),float(np.median(np.concatenate(quality))),count


def cluster_sensitivity(label_sets,primary_labels,high_z,names,weights):
    rows=[]
    for (method,k),labels in sorted(label_sets.items(),key=lambda item:(item[0][0],item[0][1])):
        groups=[np.flatnonzero(labels==label) for label in sorted(set(labels))]
        scores=np.column_stack([high_z[:,indices]@(weights[indices]/np.sum(weights[indices])) for indices in groups])
        first,second=np.triu_indices(len(groups),1)
        top=np.argmax(np.abs(scores[:,first]-scores[:,second]),axis=1)
        counts=np.bincount(top,minlength=len(first))
        winner=int(np.argmax(counts))
        left="|".join(names[i] for i in groups[first[winner]])
        right="|".join(names[i] for i in groups[second[winner]])
        rows.append({"linkage_distance":method,"k":k,"ARI_vs_primary":adjusted_rand(labels,primary_labels),
                     "singleton_count":sum(len(g)==1 for g in groups),
                     "top_dimension_1_members":left,"top_dimension_2_members":right,
                     "top_dimension_pair_share":float(counts[winner]/len(high_z))})
    return rows


def representative_cases(candidates,median_quality,threshold,count):
    groups={"high_Q_high_C":[],"high_Q_low_C":[],"low_Q_high_C":[],"low_Q_low_C":[]}
    for item in candidates:
        label=("high_Q" if item["Q"]>=median_quality else "low_Q")+("_high_C" if item["C"]>threshold else "_low_C")
        groups[label].append(item)
    selected=[]
    for label,items in groups.items():
        is_high=label.endswith("high_C")
        for rank,item in enumerate(sorted(items,key=lambda r:((-r["C"] if is_high else r["C"]),r["record_id"]))[:count],1):
            selected.append({"quadrant":label,"rank":rank,**item})
    return selected


def run(root):
    config_path=root/"configs/q1/conflict.json"
    design_path=root/"question1-2(1).md"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if not design_path.is_file():raise RuntimeError("缺少 question1-2(1).md")
    model=load_p1_model(root,config["required_quality_model_version"])
    names=model["names"];weights=model["weights"];corr=model["correlation"]
    eval_rows,label_sets=evaluate(corr,config["candidate_k"])
    labels=label_sets[(config["linkage"],config["selected_k"])]
    groups=checked_dimensions(names,labels,config["expected_clusters"],weights)
    d=distance_matrix(corr)
    tree=linkage(squareform(d),method=config["linkage"])
    pairs=prepare_pairs(weights)
    if len(pairs[0])!=231:raise RuntimeError("22 项指标应有 231 个指标对")
    dim_first,dim_second=np.triu_indices(len(groups),1)
    quantiles=config["threshold_quantiles"]
    if sorted(quantiles)!=quantiles or config["primary_threshold_quantile"] not in quantiles:
        raise ValueError("阈值配置非法")
    thresholds,median_quality,fit_count=fixed_reference_thresholds(model["paths"]["sample_quality_scores.csv.gz"],names,weights,pairs,
                                                                     model["metadata"]["model_version"],config["chunk_size"],quantiles)
    if fit_count!=model["metadata"]["fit_count_A1"]:raise RuntimeError("A1 拟合样本量与 P1 元数据不一致")
    logging.info("A1 固定阈值：%s；K=%s",thresholds,config["selected_k"])
    out=root/"results/q1/conflict";out.mkdir(parents=True,exist_ok=True)
    figures=root/"results/q1/figures";figures.mkdir(parents=True,exist_ok=True)
    write_json(out/"conflict_thresholds.json",{"quantiles":quantiles,"thresholds":[float(x) for x in thresholds],
                                                "reference":"A1 audited valid, fixed P1 model","rule":"C > tau",
                                                "sample_count":fit_count,"quality_median_A1":median_quality})
    write_csv(out/"cluster_evaluation.csv",list(eval_rows[0]),eval_rows)
    cluster_rows=[]
    for group in groups:
        for index in group["indices"]:
            cluster_rows.append({"dimension":group["dimension"],"label":group["label"],"indicator":names[index],
                                 "indicator_weight":float(weights[index]),"dimension_weight_sum":group["weight_sum"]})
    write_csv(out/"indicator_clusters.csv",list(cluster_rows[0]),cluster_rows)
    summary=Summary(len(pairs[0]),len(dim_first),thresholds)
    scored=Counter();case_candidates=[];high_z=[]
    fields=["record_id","source_file","line_number","record_hash","attachment","domain","audit_status","union_membership",
            "quality_model_version","model_version","Q","C",*['S_'+g["dimension"] for g in groups],
            "top_dimension_1","top_dimension_2","max_dimension_gap","top_indicator_1","top_indicator_2",
            *[f"high_conflict_{int(q*100)}" for q in quantiles]]
    def output_rows(path,attachments):
        for source_rows,z,q in iter_p1_batches(path,names,weights,model["metadata"]["model_version"],config["chunk_size"],attachments):
            score=score_batch(z,groups,pairs)
            source=np.array([r["attachment"] for r in source_rows]);domain=np.array([r["domain"] for r in source_rows])
            status=np.array([r["audit_status"] for r in source_rows])
            for attachment in sorted(set(source)):
                mask_source=source==attachment
                for current_domain in sorted(set(domain[mask_source])):
                    mask=mask_source&(domain==current_domain)
                    summary.add(attachment,current_domain,q,score,mask)
                    if attachment in {"A2","A3"}:
                        summary.add(attachment+"_nonoverlap",current_domain,q,score,mask&(status=="valid"))
                    summary.add("union",current_domain,q,score,mask&(status=="valid"))
            if attachments=={"A1"}:
                high_z.append(z[score["conflict"]>thresholds[quantiles.index(config["primary_threshold_quantile"])]])
            for i,row in enumerate(source_rows):
                attachment=row["attachment"]
                scored[(attachment,row["audit_status"])]+=1
                dp=int(score["top_dimension_pair"][i]);ip=int(score["top_indicator_pair"][i])
                output={key:row[key] for key in ("record_id","source_file","line_number","record_hash","attachment","domain","audit_status","union_membership")}
                output.update({"quality_model_version":row["model_version"],"model_version":config["model_version"],
                               "Q":float(q[i]),"C":float(score["conflict"][i]),
                               "top_dimension_1":groups[dim_first[dp]]["dimension"],"top_dimension_2":groups[dim_second[dp]]["dimension"],
                               "max_dimension_gap":float(score["top_dimension_gap"][i]),
                               "top_indicator_1":names[pairs[0][ip]],"top_indicator_2":names[pairs[1][ip]]})
                output.update({"S_"+g["dimension"]:float(score["dimension"][i,j]) for j,g in enumerate(groups)})
                output.update({f"high_conflict_{int(quantile*100)}":bool(score["conflict"][i]>thresholds[t])
                               for t,quantile in enumerate(quantiles)})
                if attachment=="A1":
                    case_candidates.append({key:output[key] for key in ("record_id","source_file","line_number","record_hash","domain","audit_status","Q","C","top_dimension_1","top_dimension_2","top_indicator_1","top_indicator_2",*['S_'+g["dimension"] for g in groups])})
                yield output
    write_csv(out/"sample_conflict_scores.csv.gz",fields,output_rows(model["paths"]["sample_quality_scores.csv.gz"],{"A1"}))
    write_csv(out/"extended_conflict_scores.csv.gz",fields,output_rows(model["paths"]["extended_quality_scores.csv.gz"],{"A2","A3"}))
    expected=model["metadata"]["scored_counts"]
    if {f"{attachment}:{status}":count for (attachment,status),count in scored.items()}!=expected:
        raise RuntimeError("P2 评分数量与 P1 元数据不一致")
    if summary.buckets[("union","all")].n!=sum(count for (attachment,status),count in scored.items() if status=="valid"):
        raise RuntimeError("联合去重视图数量与 P1 有效记录不一致")
    from .analysis import domain_rows as summarize_domains
    domain_stats=summarize_domains(summary,quantiles,config["bootstrap_repetitions"],config["bootstrap_seed"])
    relations,bins=relation_rows(summary)
    comparisons=compare_extensions(summary,quantiles.index(config["primary_threshold_quantile"]))
    from .reporting import write_tables
    pair_rows,dim_rows,weighted_dim_rows,sensitivity=write_tables(out,names,groups,pairs,(dim_first,dim_second),summary,quantiles,domain_stats,relations,bins,comparisons)
    high_matrix=np.concatenate(high_z)
    clustering_rows=cluster_sensitivity(label_sets,labels,high_matrix,names,weights)
    write_csv(out/"clustering_sensitivity.csv",list(clustering_rows[0]),clustering_rows)
    cases=representative_cases(case_candidates,median_quality,float(thresholds[quantiles.index(config["primary_threshold_quantile"])]),config["example_count_per_quadrant"])
    write_csv(out/"representative_cases.csv",list(cases[0]),cases)
    metadata={"model_version":config["model_version"],"quality_model_version":model["metadata"]["model_version"],
              "indicator_names":names,"p1_input_sha256":model["input_sha256"],"raw_input_sha256_from_P1":model["metadata"]["input_sha256"],
              "config_sha256":sha256_file(config_path),"design_sha256":sha256_file(design_path),
              "code_sha256":{p.relative_to(root).as_posix():sha256_file(p) for p in sorted((root/"src/q1/conflict").glob("*.py"))},
              "scored_counts":{f"{a}:{s}":n for (a,s),n in scored.items()},"p0_status_counts":model["metadata"]["p0_status_counts"],
              "p2_exclusions":0,"union_valid_count":summary.buckets[("union","all")].n,
              "quality_median_A1":median_quality,
              "thresholds":[float(x) for x in thresholds],"threshold_quantiles":quantiles,
              "selected_k":config["selected_k"],"linkage":config["linkage"],"distance":config["distance"],
              "python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,
              "matplotlib_version":matplotlib.__version__,"bootstrap_seed":config["bootstrap_seed"],
              "bootstrap_repetitions":config["bootstrap_repetitions"]}
    write_json(out/"conflict_model_metadata.json",metadata)
    from .figures import make_figures
    make_figures(figures,names,tree,groups,summary,domain_stats,quantiles,pair_rows,dim_rows,pairs[0],pairs[1],dim_first,dim_second)
    from .reporting import write_report
    write_report(out/"conflict_report.md",metadata,groups,eval_rows,domain_stats,relations,comparisons,pair_rows,dim_rows,weighted_dim_rows,clustering_rows,sensitivity,cases,config)
    logging.info("P2 完成：%s",out)
    return metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run(args.root.resolve())


if __name__=="__main__":
    main()
