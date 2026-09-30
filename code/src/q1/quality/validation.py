"""主评分与等权基准的相关和排序对照。"""
from .entropy import average_ranks, pearson


def compare_scores(main, reference):
    if len(main) != len(reference) or len(main) < 2:
        raise ValueError("对照评分长度非法")
    return {"n":len(main),"pearson":pearson(main,reference),
            "spearman":pearson(average_ranks(main),average_ranks(reference)),
            "mean_difference":sum(a-b for a,b in zip(main,reference))/len(main)}
