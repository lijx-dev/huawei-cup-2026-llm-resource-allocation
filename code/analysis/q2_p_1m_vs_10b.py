# -*- coding: utf-8 -*-
"""1M(训练) vs 10B(估计) vs 70B(估计)：同配方跨规模对照（63 个共享配方）"""
import numpy as np
import pandas as pd
import os
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'


def rd(f):
    return pd.read_csv(os.path.join(A, f)).iloc[:, 1:].values.astype(float)


def cols(f):
    return [c.replace('metric/the_pile_', '').replace('_val_loss', '')
            for c in pd.read_csv(os.path.join(A, f)).columns[1:]]


Xtr, Ytr, ttr = rd('train_mixture_1m.csv'), rd('train_pile_loss_1m.csv'), cols('train_pile_loss_1m.csv')
X10, Y10, t10 = rd('est_mixture_10b.csv'), rd('est_pile_loss_10b.csv'), cols('est_pile_loss_10b.csv')
X70, Y70, t70 = rd('est_mixture_70b.csv'), rd('est_pile_loss_70b.csv'), cols('est_pile_loss_70b.csv')
ktr = {tuple(np.round(r, 6)): i for i, r in enumerate(Xtr)}
k10 = {tuple(np.round(r, 6)): i for i, r in enumerate(X10)}
k70 = {tuple(np.round(r, 6)): i for i, r in enumerate(X70)}
common = sorted(set(ktr) & set(k10))
i_tr = [ktr[c] for c in common]
i10 = [k10[c] for c in common]
i70 = [k70[c] for c in common]
print(f'共享配方 {len(common)} 个：1M(训练) vs 10B(估计) vs 70B(估计)')
print()
print('逐目标：同配方跨规模归一化剖面一致性 + 比值恒定性')
print(f"{'target':<20}{'r(1M,10B)':>11}{'R2':>8}{'斜率':>8}{'截距':>9}{'比值CV':>9}{'比值corr':>10}")
for j, t in enumerate(ttr):
    if t not in t10:
        continue
    j10 = t10.index(t)

    def fit(X, Y, jj):
        Xc = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(Xc, Y[:, jj], rcond=None)
        return Xc @ b

    p_tr = fit(Xtr, Ytr, j)[i_tr]
    p10 = fit(X10, Y10, j10)[i10]
    v1 = p_tr / p_tr.mean()
    v2 = p10 / p10.mean()
    r = np.corrcoef(v1, v2)[0, 1]
    sl, ic, rr, pv, se = stats.linregress(v2, v1)
    ratio = p_tr / p10
    print(f"{t:<20}{r:>11.4f}{rr**2:>8.4f}{sl:>8.4f}{ic:>+9.4f}"
          f"{ratio.std()/ratio.mean():>9.4f}{stats.pearsonr(ratio, p10)[0]:>10.4f}")

print()
print('汇总：')
rs = []
for j, t in enumerate(ttr):
    if t not in t10:
        continue
    j10 = t10.index(t)

    def fit(X, Y, jj):
        Xc = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(Xc, Y[:, jj], rcond=None)
        return Xc @ b

    p_tr = fit(Xtr, Ytr, j)[i_tr]
    p10 = fit(X10, Y10, j10)[i10]
    rs.append(np.corrcoef(p_tr / p_tr.mean(), p10 / p10.mean())[0, 1])
print(f'  剖面相关 r：中位={np.median(rs):.4f}  最小={min(rs):.4f}  最大={max(rs):.4f}')
