"""适用于论文插图的 P2 图表；每幅图标注固定模型和输入视图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from q1.audit.export import write_csv


def _save(fig, directory, stem, source, manifest):
    fig.text(.01,.005,f"Source: {source} | P1 q1-p1-v1 | P2 q1-p2-v1",fontsize=7,color="#555")
    for suffix in ("svg","png"):
        path=directory/f"p2_{stem}.{suffix}"
        fig.savefig(path,dpi=180,bbox_inches="tight",metadata={"Date":None} if suffix=="svg" else None)
        manifest.append({"file":path.name,"source_result":source,"model_version":"q1-p2-v1"})
    plt.close(fig)


def make_figures(directory,names,tree,groups,summary,domain_rows,quantiles,pair_rows,dim_rows,first,second,dim_first,dim_second):
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","svg.hashsalt":"q1-p2-v1"})
    manifest=[]
    fig,ax=plt.subplots(figsize=(12,7))
    dendrogram(tree,labels=names,leaf_rotation=90,leaf_font_size=7,color_threshold=0,ax=ax)
    ax.set(title="Indicator hierarchy: average linkage, distance = 1 - Spearman rho",ylabel="Correlation distance")
    fig.subplots_adjust(bottom=.38)
    _save(fig,directory,"dendrogram","P1 Spearman matrix; cluster_evaluation.csv",manifest)

    fig,ax=plt.subplots(figsize=(10,4))
    labels=[g["dimension"] for g in groups]
    x=np.arange(len(groups));weights=[g["weight_sum"] for g in groups];counts=[len(g["indices"]) for g in groups]
    ax.bar(x,weights,color="#3977a8")
    ax.set_xticks(x,labels);ax.set(ylabel="Sum of P1 indicator weights",title="Five fixed quality dimensions (definitions in indicator_clusters.csv)")
    for i,(w,n) in enumerate(zip(weights,counts)):
        ax.text(i,w+.005,f"{n} indicator{'s' if n!=1 else ''}",ha="center",fontsize=8)
    _save(fig,directory,"dimension_structure","indicator_clusters.csv",manifest)

    _,union_c=summary.buckets[("union","all")].arrays()
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(union_c,bins=50,color="#3977a8",alpha=.85)
    ax.axvline(summary.thresholds[1],color="#d85a45",ls="--",label="A1 90th percentile")
    ax.set(xlabel="Continuous conflict index C (0-1)",ylabel="Unique records",title="Conflict distribution of A1 ∪ A2 ∪ A3")
    ax.legend();_save(fig,directory,"conflict_distribution","union deduplicated conflict scores",manifest)

    domains=sorted(domain for view,domain in summary.buckets if view=="A1" and domain!="all")
    fig,ax=plt.subplots(figsize=(10,4))
    data=[summary.buckets[("A1",d)].arrays()[1] for d in domains]
    ax.boxplot(data,tick_labels=domains,showfliers=False)
    ax.set(ylabel="Conflict index C",title="A1 conflict distribution by domain")
    ax.tick_params(axis="x",rotation=25)
    _save(fig,directory,"domain_distributions","A1 audited valid conflict scores",manifest)

    rates=[next(r["conflict_rate"] for r in domain_rows if r["view"]=="A1" and r["domain"]==d and r["quantile"]==.9) for d in domains]
    lows=[next(r["wilson_ci95_low"] for r in domain_rows if r["view"]=="A1" and r["domain"]==d and r["quantile"]==.9) for d in domains]
    highs=[next(r["wilson_ci95_high"] for r in domain_rows if r["view"]=="A1" and r["domain"]==d and r["quantile"]==.9) for d in domains]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(domains,rates,color="#3977a8")
    ax.errorbar(domains,rates,yerr=np.maximum(0,np.array([np.array(rates)-lows,np.array(highs)-rates])),fmt="none",ecolor="#222",capsize=3)
    ax.set(ylabel="Share with C > fixed A1 tau90",title="A1 high conflict rate by domain; 95% Wilson CI")
    ax.tick_params(axis="x",rotation=25)
    _save(fig,directory,"domain_conflict_rates","domain_conflict_summary.csv; A1 fixed tau90",manifest)

    q,c=summary.buckets[("A1","all")].arrays()
    fig,ax=plt.subplots(figsize=(7,5))
    image=ax.hexbin(q,c,gridsize=50,mincnt=1,cmap="viridis",bins="log")
    ax.axhline(summary.thresholds[1],ls="--",color="white",lw=1)
    ax.set(xlabel="P1 quality score Q (0-100)",ylabel="Conflict index C (0-1)",title="A1 quality versus conflict density")
    fig.colorbar(image,ax=ax,label="log count")
    _save(fig,directory,"quality_vs_conflict","A1 Q and C; fixed P1 scoring",manifest)

    primary=[r for r in pair_rows if r["view"]=="union" and r["domain"]=="all" and r["quantile"]==.9]
    lookup={(names.index(r["indicator_1"]),names.index(r["indicator_2"])):r["H_weighted"] for r in primary}
    matrix=np.zeros((len(names),len(names)))
    for (i,j),value in lookup.items():matrix[i,j]=matrix[j,i]=value
    fig,ax=plt.subplots(figsize=(11,9))
    image=ax.imshow(matrix,cmap="YlOrRd")
    ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7)
    ax.set_yticks(range(len(names)),names,fontsize=7)
    ax.set(title="Weighted indicator pair contribution H; unique high conflict records")
    fig.colorbar(image,ax=ax,label="Mean weighted contribution")
    fig.subplots_adjust(left=.33,bottom=.36,right=.89)
    _save(fig,directory,"indicator_contributions","indicator_conflict_matrix.csv; union valid at tau90",manifest)

    primary_dim=[r for r in dim_rows if r["view"]=="union" and r["domain"]=="all" and r["quantile"]==.9]
    dlookup={(int(r["dimension_1"][1:])-1,int(r["dimension_2"][1:])-1):r["share"] for r in primary_dim}
    dmatrix=np.zeros((len(groups),len(groups)))
    for (i,j),value in dlookup.items():dmatrix[i,j]=dmatrix[j,i]=value
    fig,ax=plt.subplots(figsize=(6,5))
    image=ax.imshow(dmatrix,cmap="YlGnBu",vmin=0,vmax=max(.01,float(dmatrix.max())))
    ax.set_xticks(range(len(groups)),[g["dimension"] for g in groups]);ax.set_yticks(range(len(groups)),[g["dimension"] for g in groups])
    ax.set(title="Most divergent dimension pair among high conflict records")
    fig.colorbar(image,ax=ax,label="Share of high conflict records")
    _save(fig,directory,"dimension_contributions","dimension_conflict_matrix.csv; union valid at tau90",manifest)

    for source,domain in [("A2_nonoverlap","arxiv"),("A3_nonoverlap","github")]:
        a=summary.buckets[("A1",domain)].arrays()[1];b=summary.buckets[(source,domain)].arrays()[1]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.hist(a,bins=45,density=True,histtype="step",lw=1.8,label=f"A1 n={len(a)}")
        ax.hist(b,bins=45,density=True,histtype="step",lw=1.8,label=f"{source} n={len(b)}")
        ax.axvline(summary.thresholds[1],color="#d85a45",ls="--",label="fixed A1 tau90")
        ax.set(xlabel="Conflict index C",ylabel="Density",title=f"{domain}: A1 versus nonoverlapping extension")
        ax.legend()
        _save(fig,directory,f"extension_{domain}",f"A1 and {source} P1 scored records",manifest)

    fig,ax=plt.subplots(figsize=(9,4))
    for view,domain,label in [("A1","arxiv","A1 arxiv"),("A2_nonoverlap","arxiv","A2 new arxiv"),
                              ("A1","github","A1 github"),("A3_nonoverlap","github","A3 new github")]:
        points=[next(r["conflict_rate"] for r in domain_rows if r["view"]==view and r["domain"]==domain and r["quantile"]==q) for q in quantiles]
        ax.plot([int(q*100) for q in quantiles],points,marker="o",label=label)
    ax.set(xlabel="A1 reference quantile (%)",ylabel="Share above fixed threshold",title="Threshold sensitivity with A1 fixed thresholds")
    ax.legend(ncol=2)
    _save(fig,directory,"threshold_sensitivity","threshold_sensitivity.csv",manifest)
    write_csv(directory/"p2_figure_manifest.csv",list(manifest[0]),manifest)
    return manifest
