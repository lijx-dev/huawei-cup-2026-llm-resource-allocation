"""连续冲突指数、231 对贡献与维度得分的向量化计算。"""
import numpy as np


def prepare_pairs(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=1 or len(w)<2 or np.any(~np.isfinite(w)) or np.any(w<0) or not np.isclose(w.sum(),1,atol=1e-10):
        raise ValueError("权重非法")
    first,second=np.triu_indices(len(w),1)
    products=w[first]*w[second]
    denominator=float(products.sum())
    if denominator<=0:
        raise ValueError("冲突指数分母为零")
    return first,second,products/denominator,denominator


def score_batch(z,groups,pairs):
    values=np.asarray(z,dtype=np.float64)
    if values.ndim!=2 or np.any(~np.isfinite(values)) or np.any(values<0) or np.any(values>1):
        raise ValueError("标准化矩阵非法")
    first,second,pair_weights,_=pairs
    if values.shape[1] != len(set(np.r_[first,second])):
        raise ValueError("指标列数不一致")
    gaps=np.abs(values[:,first]-values[:,second])
    contributions=gaps*pair_weights
    conflict=contributions.sum(axis=1)
    if np.any(conflict < -1e-12) or np.any(conflict > 1+1e-12):
        raise ValueError("冲突指数超出 [0,1]")
    dimension=np.column_stack([values[:,g["indices"]] @ g["indicator_weights"] for g in groups])
    # 上式中的组内权重已经除以组权重和，直接得到 [0,1] 维度得分。
    if np.any(dimension< -1e-12) or np.any(dimension>1+1e-12):
        raise ValueError("维度得分超出 [0,1]")
    dim_first,dim_second=np.triu_indices(len(groups),1)
    dim_gap=np.abs(dimension[:,dim_first]-dimension[:,dim_second])
    return {"conflict":conflict,"dimension":dimension,"pair_gaps":gaps,
            "pair_contributions":contributions,"top_indicator_pair":np.argmax(contributions,axis=1),
            "top_dimension_pair":np.argmax(dim_gap,axis=1),"top_dimension_gap":np.max(dim_gap,axis=1),
            "dim_first":dim_first,"dim_second":dim_second}
