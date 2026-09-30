"""v2.1 的统计图均从已保存 CSV/NPY 生成并附输入哈希。"""
import json
import os
from pathlib import Path

from .common import paths, save_json, sha256

_cache = Path(__file__).resolve().parents[2] / "results/q1_revision_v2_1/.matplotlib_cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def plot_recipe_distance_ecdf(distances, output_path):
    """绘制最近邻距离 ECDF，并显式标注完全重合的 1M/60M 曲线。"""
    fig,ax=plt.subplots(figsize=(8,5))
    values_1m=np.sort(distances.loc[distances.split=="test_1m","nearest_distance"].to_numpy())
    values_60m=np.sort(distances.loc[distances.split=="test_60m","nearest_distance"].to_numpy())
    same_1m_60m=(len(values_1m)==len(values_60m) and np.array_equal(values_1m,values_60m))
    series=[("train_1m","train_1m")]
    if same_1m_60m:
        series.append(("test_1m",f"test_1m / test_60m (identical, n={len(values_1m)} each)"))
        series.append(("test_1b","test_1b"))
    else:
        series.extend((name,name) for name in ("test_1m","test_60m","test_1b"))
    for name,label in series:
        values=np.sort(distances.loc[distances.split==name,"nearest_distance"].to_numpy())
        ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",label=label)
    ax.set_xlabel("Euclidean distance to nearest training recipe")
    ax.set_ylabel("ECDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path,dpi=180)
    plt.close(fig)


def run(root, config):
    _, out, _ = paths(root)
    dest = out / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    source = []
    weights_file = out / "quality/hierarchical_weights.csv"
    weights = pd.read_csv(weights_file).sort_values("global_weight")
    fig,ax=plt.subplots(figsize=(9,8))
    y=np.arange(len(weights))
    ax.barh(y-.2,weights.old_v2_global_weight,height=.4,label="v2 global")
    ax.barh(y+.2,weights.global_weight,height=.4,label="v2.1 hierarchical")
    ax.set_yticks(y,weights.field,fontsize=7)
    ax.set_xlabel("Global weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "weight_comparison.png",dpi=180)
    plt.close(fig)
    source.append(weights_file)
    linkage_file=out / "conflict/primary_linkage.npy"
    fields_file=out / "quality/normalization_parameters.csv"
    tree=np.load(linkage_file)
    names=pd.read_csv(fields_file).field.tolist()
    fig,ax=plt.subplots(figsize=(11,6))
    dendrogram(tree,labels=names,leaf_rotation=90,leaf_font_size=7,ax=ax)
    ax.set_title("Exploratory indicator hierarchy: signed Spearman distance")
    fig.tight_layout()
    fig.savefig(dest / "indicator_dendrogram.png",dpi=180)
    plt.close(fig)
    source.extend([linkage_file,fields_file])
    distance_file=out / "mixture/nearest_train_distance.csv"
    distances=pd.read_csv(distance_file)
    plot_recipe_distance_ecdf(distances,dest / "recipe_distance_ecdf.png")
    source.append(distance_file)
    coverage_file=out / "q_mapping/qmapped_coverage.csv"
    coverage=pd.read_csv(coverage_file)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in ("train_1m","test_1m","test_60m","test_1b"):
        values=coverage.loc[coverage.split==name,"coverage"]
        ax.hist(values,bins=np.linspace(0,1,31),density=True,histtype="step",label=name)
    ax.set_xlabel("Six-domain mapped mixture coverage")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "qmapped_coverage_distribution.png",dpi=180)
    plt.close(fig)
    source.append(coverage_file)
    placebo_file=out / "q_mapping/qmapped_placebo.csv"
    summary_file=out / "q_mapping/qmapped_placebo_summary.csv"
    placebo=pd.read_csv(placebo_file)
    summary=pd.read_csv(summary_file)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,method in zip(axes,summary.quality_method):
        values=placebo.loc[placebo.quality_method==method,"delta_pooled_r2_cv"]
        real=summary.loc[summary.quality_method==method,"real_delta_pooled_r2_cv"].iloc[0]
        ax.hist(values,bins=35,alpha=.8)
        ax.axvline(real,color="red",label="real mapping")
        ax.set_title(method)
        ax.set_xlabel("CV pooled R2 gain")
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "qmapped_placebo.png",dpi=180)
    plt.close(fig)
    source.extend([placebo_file,summary_file])
    wfile=out / "conflict/weighted_pair_contributions.csv"
    efile=out / "conflict/equal_pair_disagreement.csv"
    weighted=pd.read_csv(wfile)
    equal=pd.read_csv(efile)
    selected=((weighted.attachment=="A1") & (weighted.high_set_method=="C_weighted_hierarchical"))
    selected_e=((equal.attachment=="A1") & (equal.high_set_method=="C_weighted_hierarchical"))
    matrices=[]
    for frame,col,mask in ((weighted,"H_weighted",selected),(equal,"H_equal_raw",selected_e)):
        matrix=np.zeros((len(names),len(names)))
        index={name:i for i,name in enumerate(names)}
        for row in frame[mask].itertuples():
            a,b=index[row.field_j],index[row.field_k]
            matrix[a,b]=matrix[b,a]=getattr(row,col)
        matrices.append(matrix)
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    for ax,matrix,title in zip(axes,matrices,("Weighted contribution","Raw equal-pair disagreement")):
        image=ax.imshow(matrix,cmap="magma")
        ax.set_title(title)
        ax.set_xticks(range(len(names)),names,rotation=90,fontsize=5)
        ax.set_yticks(range(len(names)),names,fontsize=5)
        fig.colorbar(image,ax=ax,shrink=.65)
    fig.tight_layout()
    fig.savefig(dest / "pair_disagreement_comparison.png",dpi=180)
    plt.close(fig)
    source.extend([wfile,efile])
    save_json(dest / "figure_sources.json", {"experiment_version":config["experiment_version"],
              "seed":config["seed"],"input_sha256":{str(p.relative_to(out)):sha256(p) for p in source}})
