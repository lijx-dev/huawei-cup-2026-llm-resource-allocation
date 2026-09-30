"""基于 P1 Spearman 矩阵的平均连接层次聚类及稳定性分析。"""
from math import comb

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, cophenet
from scipy.spatial.distance import squareform


def distance_matrix(correlation, absolute=False):
    r = np.asarray(correlation, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.all(np.isfinite(r)) or not np.allclose(r,r.T,atol=1e-9) or not np.allclose(np.diag(r),1,atol=1e-9) or np.any(np.abs(r)>1+1e-9):
        raise ValueError("Spearman 矩阵非法")
    d = 1 - (np.abs(r) if absolute else r)
    np.fill_diagonal(d,0)
    if np.any(d < -1e-9):
        raise ValueError("相关距离为负")
    return np.maximum(d,0)


def cluster_labels(distance, method, k):
    if method not in {"average","complete"}:
        raise ValueError("预计算相关距离仅使用 average/complete；Ward 不适用")
    tree = linkage(squareform(distance,checks=True),method=method)
    labels = fcluster(tree,k,criterion="maxclust")
    if len(set(labels)) != k:
        raise ValueError("关联距离的并列值使实际簇数与目标 K 不一致")
    return tree,labels


def silhouette_from_distance(distance, labels):
    labels = np.asarray(labels)
    values = []
    groups = [np.flatnonzero(labels==label) for label in np.unique(labels)]
    for i in range(len(labels)):
        own = next(group for group in groups if i in group)
        if len(own) == 1:
            values.append(0.0)
            continue
        a = float(np.mean(distance[i,own[own!=i]]))
        b = min(float(np.mean(distance[i,group])) for group in groups if i not in group)
        values.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return float(np.mean(values))


def adjusted_rand(left,right):
    """按指标对列联计数计算 ARI，不依赖 scikit-learn。"""
    if len(left) != len(right) or len(left)<2:
        raise ValueError("聚类标签长度非法")
    left=np.asarray(left);right=np.asarray(right);n=len(left)
    a_labels,a_counts=np.unique(left,return_counts=True)
    b_labels,b_counts=np.unique(right,return_counts=True)
    same=sum(comb(int(np.sum((left==a)&(right==b))),2) for a in a_labels for b in b_labels)
    row=sum(comb(int(x),2) for x in a_counts)
    col=sum(comb(int(x),2) for x in b_counts)
    total=comb(n,2)
    expected=row*col/total
    denominator=(row+col)/2-expected
    return float((same-expected)/denominator) if denominator else 1.0


def evaluate(correlation, candidates):
    d=distance_matrix(correlation)
    condensed=squareform(d)
    result=[]
    label_sets={}
    for method in ("average","complete"):
        tree=linkage(condensed,method=method)
        coph=float(cophenet(tree,condensed)[0])
        for k in candidates:
            labels=fcluster(tree,k,criterion="maxclust")
            if len(set(labels)) != k:
                raise ValueError(f"{method} K={k} 得到的簇数不符")
            sizes=np.bincount(labels)[1:]
            result.append({"distance":"1-rho","linkage":method,"k":k,
                           "silhouette":silhouette_from_distance(d,labels),
                           "cophenetic_correlation":coph,"singleton_count":int(np.sum(sizes==1)),
                           "min_cluster_size":int(np.min(sizes)),"max_cluster_size":int(np.max(sizes))})
            label_sets[(method,k)]=labels
    absolute=distance_matrix(correlation,absolute=True)
    for k in candidates:
        _,labels=cluster_labels(absolute,"average",k)
        label_sets[("average_abs",k)]=labels
    return result,label_sets


def checked_dimensions(names, labels, specifications, weights):
    if len(specifications)!=len(set(labels)):
        raise RuntimeError("P2 配置的维度数与聚类结果不一致")
    groups=[]
    for label,spec in zip(sorted(set(labels)),specifications):
        indices=np.flatnonzero(labels==label)
        actual={names[i] for i in indices}
        if actual!=set(spec["indicators"]):
            raise RuntimeError(f"P1 相关结构已变化，{spec['dimension']} 语义标签需要重新审核")
        total=float(np.sum(weights[indices]))
        if total<=0:
            raise RuntimeError("维度簇权重为零")
        groups.append({"dimension":spec["dimension"],"label":spec["label"],
                       "indices":indices,"weight_sum":total,
                       "indicator_weights":weights[indices]/total})
    return groups
