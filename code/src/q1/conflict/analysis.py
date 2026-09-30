"""领域、阈值和冲突来源的流式汇总；不改变 P1 单条记录。"""
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp


class Bucket:
    def __init__(self, pair_count, dimension_pair_count, threshold_count):
        self.q_parts=[]
        self.c_parts=[]
        self.n=0
        self.high_n=np.zeros(threshold_count,dtype=np.int64)
        self.pair_sum=np.zeros((threshold_count,pair_count),dtype=np.float64)
        self.gap_sum=np.zeros((threshold_count,pair_count),dtype=np.float64)
        self.dim_counts=np.zeros((threshold_count,dimension_pair_count),dtype=np.int64)

    def add(self,q,c,gaps,contributions,top_dimension,thresholds):
        if not len(c):return
        self.n+=len(c)
        self.q_parts.append(np.asarray(q,dtype=np.float64).copy())
        self.c_parts.append(np.asarray(c,dtype=np.float64).copy())
        for t,tau in enumerate(thresholds):
            high=c>tau
            self.high_n[t]+=int(np.count_nonzero(high))
            if np.any(high):
                self.pair_sum[t]+=np.sum(contributions[high],axis=0)
                self.gap_sum[t]+=np.sum(gaps[high],axis=0)
                self.dim_counts[t]+=np.bincount(top_dimension[high],minlength=self.dim_counts.shape[1])

    def arrays(self):
        return np.concatenate(self.q_parts),np.concatenate(self.c_parts)


class Summary:
    def __init__(self,pair_count,dimension_pair_count,thresholds):
        self.pair_count=pair_count
        self.dimension_pair_count=dimension_pair_count
        self.thresholds=np.asarray(thresholds,dtype=np.float64)
        self.buckets=defaultdict(lambda:Bucket(pair_count,dimension_pair_count,len(thresholds)))

    def add(self,view,domain,q,score,selection=None):
        mask=np.ones(len(q),dtype=bool) if selection is None else np.asarray(selection,dtype=bool)
        if not np.any(mask):return
        for key in [(view,domain),(view,"all")]:
            self.buckets[key].add(q[mask],score["conflict"][mask],score["pair_gaps"][mask],
                                  score["pair_contributions"][mask],score["top_dimension_pair"][mask],self.thresholds)


def bootstrap_rate_interval(n, high_n, repetitions, rng):
    """二元标记的有放回非参数 Bootstrap 等价于 Binomial(n, p̂)。"""
    if n<=0 or repetitions<2:raise ValueError("Bootstrap 样本数或重复次数非法")
    draws=rng.binomial(n,high_n/n,size=repetitions)/n
    return tuple(float(x) for x in np.quantile(draws,[0.025,0.975]))


def wilson_interval(n, high_n, z=1.96):
    """零事件时用于补充展示的不退化二项比例区间。"""
    if n<=0 or not 0<=high_n<=n:raise ValueError("二项比例样本量非法")
    p=high_n/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    low=max(0.,center-half)
    high=min(1.,center+half)
    if high_n==0:low=0.
    if high_n==n:high=1.
    return low,high


def domain_rows(summary,quantiles,repetitions,seed):
    rng=np.random.default_rng(seed)
    rows=[]
    for (view,domain),bucket in sorted(summary.buckets.items()):
        q,c=bucket.arrays()
        for t,quantile in enumerate(quantiles):
            high_n=int(bucket.high_n[t])
            low,high=bootstrap_rate_interval(bucket.n,high_n,repetitions,rng)
            wilson_low,wilson_high=wilson_interval(bucket.n,high_n)
            rows.append({"view":view,"domain":domain,"n":bucket.n,"quantile":quantile,
                         "threshold":float(summary.thresholds[t]),"mean_C":float(np.mean(c)),
                         "median_C":float(np.median(c)),"std_C":float(np.std(c,ddof=1)) if len(c)>1 else 0.,
                         "q25_C":float(np.quantile(c,.25)),"q75_C":float(np.quantile(c,.75)),
                         "high_n":high_n,"conflict_rate":high_n/bucket.n,
                         "rate_ci95_low":low,"rate_ci95_high":high,
                         "wilson_ci95_low":wilson_low,"wilson_ci95_high":wilson_high,
                         "ci_method":"IID binary bootstrap, binomial equivalent"})
    return rows


def relation_rows(summary):
    rows=[];bins=[]
    for (view,domain),bucket in sorted(summary.buckets.items()):
        q,c=bucket.arrays()
        pearson=float(pearsonr(q,c).statistic) if np.std(q)>0 and np.std(c)>0 else float('nan')
        spearman=float(spearmanr(q,c).statistic) if np.std(q)>0 and np.std(c)>0 else float('nan')
        rows.append({"view":view,"domain":domain,"n":bucket.n,"pearson_Q_C":pearson,"spearman_Q_C":spearman})
        for lower in range(0,100,10):
            mask=(q>=lower)&(q<(lower+10) if lower<90 else q<=100)
            bins.append({"view":view,"domain":domain,"Q_bin_low":lower,"Q_bin_high":lower+10,
                         "n":int(np.count_nonzero(mask)),
                         "mean_C":float(np.mean(c[mask])) if np.any(mask) else ""})
    return rows,bins


def compare_extensions(summary,primary_index):
    rows=[]
    for sample_view,extension_view,domain in [("A1","A2_nonoverlap","arxiv"),("A1","A3_nonoverlap","github")]:
        left=summary.buckets[(sample_view,domain)]
        right=summary.buckets[(extension_view,domain)]
        _,a=left.arrays();_,b=right.arrays()
        ks=ks_2samp(a,b,method="asymp")
        rows.append({"domain":domain,"a1_n":len(a),"extension_nonoverlap_n":len(b),
                     "a1_mean_C":float(np.mean(a)),"extension_mean_C":float(np.mean(b)),
                     "mean_difference_extension_minus_A1":float(np.mean(b)-np.mean(a)),
                     "median_difference_extension_minus_A1":float(np.median(b)-np.median(a)),
                     "ks_D":float(ks.statistic),"ks_pvalue":float(ks.pvalue),
                     "a1_rate_90":float(left.high_n[primary_index]/left.n),
                     "extension_rate_90":float(right.high_n[primary_index]/right.n),
                     "rate_difference_extension_minus_A1":float(right.high_n[primary_index]/right.n-left.high_n[primary_index]/left.n),
                     "overlap_excluded":True})
    return rows
