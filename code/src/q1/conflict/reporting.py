"""P2 表格汇总与中文实验报告。"""
import numpy as np

from q1.audit.export import write_csv


def aggregate_dimension_contributions(values, first, second, group_of):
    result={}
    for p,value in enumerate(values):
        key=tuple(sorted((group_of[int(first[p])],group_of[int(second[p])])))
        result[key]=result.get(key,0.)+float(value)
    return result


def write_tables(output, names, groups, pairs, dim_pairs, summary, quantiles, domain_rows, relations, bins, comparisons):
    first,second,_,denominator=pairs
    dim_first,dim_second=dim_pairs
    pair_rows=[];dim_rows=[];weighted_dim_rows=[];sensitivity=[]
    group_of={int(index):group["dimension"] for group in groups for index in group["indices"]}
    for (view,domain),bucket in sorted(summary.buckets.items()):
        for t,q in enumerate(quantiles):
            if bucket.high_n[t]==0:
                sensitivity.append({"view":view,"domain":domain,"quantile":q,"threshold":float(summary.thresholds[t]),
                                    "n":bucket.n,"high_n":0,"rate":0.,"top_indicator_1":"","top_indicator_2":"",
                                    "top_dimension_1":"","top_dimension_2":""})
                continue
            values=bucket.pair_sum[t]/bucket.high_n[t]
            unweighted=bucket.gap_sum[t]/bucket.high_n[t]
            order=sorted(range(len(values)),key=lambda j:(-values[j],j))
            for rank,p in enumerate(order,1):
                pair_rows.append({"view":view,"domain":domain,"quantile":q,"high_n":int(bucket.high_n[t]),
                                  "rank":rank,"indicator_1":names[first[p]],"indicator_2":names[second[p]],
                                  "H_weighted":float(values[p]),"U_unweighted":float(unweighted[p]),
                                  "weight_normalizer":denominator})
            weighted_groups=aggregate_dimension_contributions(values,first,second,group_of)
            for rank,(group_pair,value) in enumerate(sorted(weighted_groups.items(),key=lambda item:(-item[1],item[0])),1):
                weighted_dim_rows.append({"view":view,"domain":domain,"quantile":q,"high_n":int(bucket.high_n[t]),
                                          "rank":rank,"dimension_1":group_pair[0],"dimension_2":group_pair[1],
                                          "mean_weighted_contribution":value,
                                          "share_of_mean_C":value/float(np.sum(values)) if np.sum(values)>0 else ""})
            dim_values=bucket.dim_counts[t]/bucket.high_n[t]
            dim_order=sorted(range(len(dim_values)),key=lambda p:(-dim_values[p],p))
            for rank,p in enumerate(dim_order,1):
                dim_rows.append({"view":view,"domain":domain,"quantile":q,"high_n":int(bucket.high_n[t]),
                                 "rank":rank,"dimension_1":groups[dim_first[p]]["dimension"],
                                 "dimension_2":groups[dim_second[p]]["dimension"],
                                 "count":int(bucket.dim_counts[t,p]),"share":float(dim_values[p])})
            sensitivity.append({"view":view,"domain":domain,"quantile":q,"threshold":float(summary.thresholds[t]),
                                "n":bucket.n,"high_n":int(bucket.high_n[t]),
                                "rate":float(bucket.high_n[t]/bucket.n),
                                "top_indicator_1":names[first[order[0]]],"top_indicator_2":names[second[order[0]]],
                                "top_dimension_1":groups[dim_first[dim_order[0]]]["dimension"],
                                "top_dimension_2":groups[dim_second[dim_order[0]]]["dimension"]})
    write_csv(output/"indicator_pair_contributions.csv.gz",list(pair_rows[0]),pair_rows)
    write_csv(output/"top_indicator_conflicts.csv",list(pair_rows[0]),[r for r in pair_rows if r["rank"]<=20])
    write_csv(output/"dimension_pair_contributions.csv",list(dim_rows[0]),dim_rows)
    write_csv(output/"dimension_weighted_contributions.csv",list(weighted_dim_rows[0]),weighted_dim_rows)
    write_csv(output/"top_dimension_conflicts.csv",list(dim_rows[0]),[r for r in dim_rows if r["rank"]<=10])
    write_csv(output/"threshold_sensitivity.csv",list(sensitivity[0]),sensitivity)
    primary=quantiles.index(.9)
    global_bucket=summary.buckets[("union","all")]
    weighted=global_bucket.pair_sum[primary]/global_bucket.high_n[primary]
    matrix_rows=[]
    for j,name in enumerate(names):
        row={"name":name}
        for k,other in enumerate(names):
            if j==k: row[other]=0.
            else:
                p=next(p for p,(a,b) in enumerate(zip(first,second)) if (a==j and b==k) or (a==k and b==j))
                row[other]=float(weighted[p])
        matrix_rows.append(row)
    write_csv(output/"indicator_conflict_matrix.csv",["name",*names],matrix_rows)
    dim_values=global_bucket.dim_counts[primary]/global_bucket.high_n[primary]
    dim_matrix=[]
    for j,group in enumerate(groups):
        row={"dimension":group["dimension"]}
        for k,other in enumerate(groups):
            if j==k: row[other["dimension"]]=0.
            else:
                p=next(p for p,(a,b) in enumerate(zip(dim_first,dim_second)) if (a==j and b==k) or (a==k and b==j))
                row[other["dimension"]]=float(dim_values[p])
        dim_matrix.append(row)
    write_csv(output/"dimension_conflict_matrix.csv",["dimension",*[g["dimension"] for g in groups]],dim_matrix)
    write_csv(output/"domain_conflict_summary.csv",list(domain_rows[0]),domain_rows)
    write_csv(output/"quality_conflict_relation.csv",list(relations[0]),relations)
    write_csv(output/"quality_conflict_bins.csv",list(bins[0]),bins)
    write_csv(output/"extended_conflict_comparison.csv",list(comparisons[0]),comparisons)
    return pair_rows,dim_rows,weighted_dim_rows,sensitivity


def write_report(path, metadata, groups, evaluations, domain_rows, relation_rows, comparisons, pair_rows, dim_rows, weighted_dim_rows, clustering_sensitivity, sensitivity, examples, config):
    primary=config["primary_threshold_quantile"]
    sample=next(row for row in domain_rows if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]==primary)
    a1_domains=[row for row in domain_rows if row["view"]=="A1" and row["domain"]!="all" and row["quantile"]==primary]
    a1_relation=next(row for row in relation_rows if row["view"]=="A1" and row["domain"]=="all")
    top_ind=[row for row in pair_rows if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]==primary and row["rank"]<=10]
    top_dim=[row for row in dim_rows if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]==primary and row["rank"]<=5]
    top_weighted_dim=[row for row in weighted_dim_rows if row["view"]=="A1" and row["domain"]=="all" and row["quantile"]==primary and row["rank"]<=5]
    a1_sensitivity=[row for row in sensitivity if row["view"]=="A1" and row["domain"]=="all"]
    a1_indicator_stable=len({(row["top_indicator_1"],row["top_indicator_2"]) for row in a1_sensitivity})==1
    a1_dimension_stable=len({(row["top_dimension_1"],row["top_dimension_2"]) for row in a1_sensitivity})==1
    complete_ari=next(row["ARI_vs_primary"] for row in clustering_sensitivity if row["linkage_distance"]=="complete" and row["k"]==config["selected_k"])
    absolute_ari=next(row["ARI_vs_primary"] for row in clustering_sensitivity if row["linkage_distance"]=="average_abs" and row["k"]==config["selected_k"])
    lines=["# 第一问 P2 质量指标冲突分析", "",
           "由 `.venv/bin/python -m q1.conflict.run` 根据 P1 落盘的标准化矩阵、权重和 Spearman 矩阵生成。冲突表示指标评价不一致，不等于坏数据、幻觉或质量低；所有高冲突合法样本均保留。", "",
           "## 固定模型与输入", "",
           f"- P1 模型：`{metadata['quality_model_version']}`；P2 模型：`{metadata['model_version']}`；22 项权重及 22×22 Spearman 矩阵直接读取 P1，未重新拟合。",
           f"- 逐来源处理：{metadata['scored_counts']}；P0 状态见元数据；P2 额外排除：{metadata['p2_exclusions']}。",
           "- A2/A3 来源视图含与 A1 精确重叠的记录；独立迁移比较使用 `_nonoverlap` 视图，联合视图只含 P0 `valid`。",
           "- 输入及代码 SHA-256、配置、依赖版本和随机种子见 `conflict_model_metadata.json`。", "",
           "## 指标聚类", "",
           f"主方案 d(j,k)=1−ρ(j,k)、average linkage，K={config['selected_k']}。{config['selection_reason']}", "",
           "|维度|解释性名称|指标|组权重和|", "|---|---|---|---:|"]
    for group in groups:
        lines.append(f"|{group['dimension']}|{group['label']}|{', '.join(metadata['indicator_names'][int(i)] for i in group['indices'])}|{group['weight_sum']:.4f}|")
    lines.extend(["", "候选 K=2–8 的轮廓系数、cophenetic 相关和单指标簇数量见 `cluster_evaluation.csv`；不同 K、complete linkage 与 1−|ρ| 对照见 `clustering_sensitivity.csv`。簇名是对统计分组的解释，不是已证实的潜在因子。", "",
                  "## A1 冲突与领域差异", "",
                  f"A1 n={sample['n']}，C 均值 {sample['mean_C']:.4f}、中位数 {sample['median_C']:.4f}、标准差 {sample['std_C']:.4f}；A1 固定阈值："+
                  ", ".join(f"τ{int(q*100)}={value:.6f}" for q,value in zip(config['threshold_quantiles'],metadata['thresholds']))+"。判定使用严格大于阈值。", "",
                  "|领域|n|平均 C|高冲突 n|90% 阈值冲突率|95% Bootstrap 区间|95% Wilson 区间|", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in a1_domains:
        lines.append(f"|{row['domain']}|{row['n']}|{row['mean_C']:.4f}|{row['high_n']}|{row['conflict_rate']:.2%}|[{row['rate_ci95_low']:.2%}, {row['rate_ci95_high']:.2%}]|[{row['wilson_ci95_low']:.2%}, {row['wilson_ci95_high']:.2%}]|")
    lines.extend(["", "置信区间把当前记录视作独立同分布的经验样本。二元标记的有放回 Bootstrap 用等价的 Binomial(n, p̂) 抽样实现；零事件时经验 Bootstrap 会退化为 [0,0]，故同时报告 Wilson 区间。区间不能证明来源真实性或总体代表性。", "",
                  "## 冲突来源（A1，τ90）", "",
                  "每条样本的 C 为 231 个加权指标对贡献之和。`H_weighted` 是高冲突样本的平均贡献；`U_unweighted` 是同一指标对的原始归一化差距均值，二者不能混为一谈。", "",
                  "|排名|指标 1|指标 2|H|U|", "|---:|---|---|---:|---:|"])
    for row in top_ind:
        lines.append(f"|{row['rank']}|{row['indicator_1']}|{row['indicator_2']}|{row['H_weighted']:.5f}|{row['U_unweighted']:.3f}|")
    lines.extend(["", "|排名|维度 1|维度 2|高冲突样本数|比例|", "|---:|---|---|---:|---:|"])
    for row in top_dim:
        lines.append(f"|{row['rank']}|{row['dimension_1']}|{row['dimension_2']}|{row['count']}|{row['share']:.2%}|")
    lines.extend(["", "最大维度分差对每个维度先除以组权重和，故低权重的 G2/G3 仍可能常成为最大分差。下表另外把 231 对加权指标贡献按维度归属相加；这才与连续冲突分数 C 的加权口径一致。同维度内部也可产生贡献。", "",
                  "|排名|维度 1|维度 2|平均加权贡献|占高冲突样本平均 C|", "|---:|---|---|---:|---:|"])
    for row in top_weighted_dim:
        lines.append(f"|{row['rank']}|{row['dimension_1']}|{row['dimension_2']}|{row['mean_weighted_contribution']:.5f}|{row['share_of_mean_C']:.2%}|")
    lines.extend(["", "## Q 与 C 的关系", "",
                  f"A1 Pearson={a1_relation['pearson_Q_C']:.4f}，Spearman={a1_relation['spearman_Q_C']:.4f}。分箱均值和二维密度见 `quality_conflict_bins.csv` 与图表；相关关系不是因果效应。", "",
                  f"四类 A1 典型样本按 A1 质量分中位数 Q={metadata['quality_median_A1']:.3f} 和固定 τ90 分组；“高/低质量”仅指这个相对分界。指针、分数、最大冲突维度和指标对见 `representative_cases.csv`；未复制原始长文本。", "",
                  "## 扩展集迁移（排除 A1 重叠）", "",
                  "|领域|A1 n|扩展新增 n|A1 平均 C|扩展平均 C|均值差|A1 高冲突率|扩展高冲突率|KS D|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        lines.append(f"|{row['domain']}|{row['a1_n']}|{row['extension_nonoverlap_n']}|{row['a1_mean_C']:.4f}|{row['extension_mean_C']:.4f}|{row['mean_difference_extension_minus_A1']:.4f}|{row['a1_rate_90']:.2%}|{row['extension_rate_90']:.2%}|{row['ks_D']:.4f}|")
    lines.extend(["", "扩展集使用 A1 的三个固定阈值。KS 的 p 值与效应量均在 `extended_conflict_comparison.csv`；大样本可使很小的差异显著，因此重点看均值差、冲突率差及 KS D。", "",
                  f"arxiv 扩展新增记录在 τ90 上仅有 {round(comparisons[0]['extension_rate_90']*comparisons[0]['extension_nonoverlap_n'])} 条高冲突，相关冲突来源排名仅供描述，不宜据此宣称稳定迁移。", "",
                  "## 敏感性与处理规则", "",
                  "`threshold_sensitivity.csv` 对 τ85/τ90/τ95 分别重算各视图领域率、首要维度对及指标对；`clustering_sensitivity.csv` 比较 K、linkage 和距离定义。若某项排名改变，按表格报告，不能按期待选择阈值。", "",
                  f"A1 全样本在三档阈值下首位指标对{'保持一致' if a1_indicator_stable else '发生变化'}、首位最大分差维度对{'保持一致' if a1_dimension_stable else '发生变化'}。固定 K={config['selected_k']} 时 complete 与主簇 ARI={complete_ari:.4f}，1−|ρ| 对照 ARI={absolute_ari:.4f}；后者表明距离定义对维度解释有影响。", "",
                  "P0 已确认损坏和未核实记录继续隔离；合法高冲突记录保留其原指标、P1 质量分和 P2 冲突标记。代码、数学文本等领域可能受自然语言指标适用性影响，仅做领域标记与敏感性分析，不自动删除或修改指标。", "",
                  "## 复现与限制", "",
                  "- 逐样本 CSV.gz 包含原始指针、P0 状态、P1/P2 模型版本、Q、C、5 个维度得分、主要冲突指标对及三档阈值标记。",
                  "- 方向及标量化均继承 P1 的建模假设；P2 不能独立证明这些假设正确。",
                  "- 维度对按 |Sg−Sh| 最大值归属；此归属与所有 231 对的加权贡献分解是两种不同统计口径。",
                  "- A1 与扩展集存在精确重叠，因此只把排除重叠后的比较称作迁移检验；仍不能证明样本独立同分布。", "",
                  "## 后续阶段接口", "",
                  "后续配比研究可复用 P1 的 `Q` 与本阶段的 `C`、三档固定阈值、五维定义及领域汇总，字段和来源指针见两份逐样本冲突文件。质量信号和配方实验没有逐记录真实配对；不能把这些样本与配方表直接拼成联合观测，也不能把 `C` 当作已证实的训练收益因子。", ""])
    path.write_text("\n".join(lines),encoding="utf-8")
