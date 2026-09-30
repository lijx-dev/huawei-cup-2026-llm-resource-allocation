# -*- coding: utf-8 -*-
"""1m vs 60m 真实共同配方：斜率 s 与截距 a 的逐目标推断

Form A（p 只挂 token 项，且两切片 D 相同）: v1=v2 -> s=1，a=(u1-u2)>0
Form B（p 挂全部可约项）:                     s=R1/R2>1，a=E(1-s)<0
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'
X1 = pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
X2 = pd.read_csv(os.path.join(A, 'test_mixture_60m.csv')).iloc[:, 1:].values.astype(float)
Y1df = pd.read_csv(os.path.join(A, 'test_pile_loss_1m.csv'))
Y2df = pd.read_csv(os.path.join(A, 'test_pile_loss_60m.csv'))
tgt = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in Y1df.columns[1:]]
Y1 = Y1df.iloc[:, 1:].values.astype(float)
Y2 = Y2df.iloc[:, 1:].values.astype(float)
k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(X1)}
k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(X2)}
common = sorted(set(k1) & set(k2))
i1 = [k1[c] for c in common]
i2 = [k2[c] for c in common]

print(f'共同配方 n = {len(common)}')
print(f"{'target':<18}{'s':>8}{'se_s':>8}{'s-1':>8}{'p(s=1)':>10}{'a':>9}{'p(a=0)':>10}")
sl, ic, p_s, p_a = [], [], [], []
for j, t in enumerate(tgt):
    l1, l2 = Y1[i1, j], Y2[i2, j]
    s, a, r, pv, se = stats.linregress(l2, l1)
    # 斜率与 1 的差异检验
    ts = (s - 1) / se
    ps = 2 * stats.t.sf(abs(ts), len(l1) - 2)
    # 截距与 0 的差异检验
    n = len(l1)
    xm = l2.mean()
    se_a = np.sqrt(np.sum((l1 - (a + s * l2)) ** 2) / (n - 2)) * np.sqrt(1 / n + xm ** 2 / np.sum((l2 - xm) ** 2))
    ta = a / se_a
    pa = 2 * stats.t.sf(abs(ta), n - 2)
    sl.append(s); ic.append(a); p_s.append(ps); p_a.append(pa)
    print(f'{t:<18}{s:>8.4f}{se:>8.4f}{s-1:>+8.4f}{ps:>10.2e}{a:>+9.4f}{pa:>10.2e}')

sl = np.array(sl); ic = np.array(ic)
print(f'\n  s 中位={np.median(sl):.4f}，s>1 的比例={np.mean(sl>1):.0%}，'
      f's 与 1 有显著差异的目标={int(np.sum(np.array(p_s)<0.05))}/{len(sl)}')
print(f'  a 中位={np.median(ic):+.4f}，a>0 的比例={np.mean(ic>0):.0%}，'
      f'a 与 0 有显著差异的目标={int(np.sum(np.array(p_a)<0.05))}/{len(ic)}')
# 组合检验（各目标 z 值 Stouffer）
z_s = (sl - 1) / np.array([stats.linregress(Y2[i2, j], Y1[i1, j])[4] for j in range(len(tgt))])
z_a = ic / np.array([
    np.sqrt(np.sum((Y1[i1, j] - (np.polyfit(Y2[i2, j], Y1[i1, j], 1)[1] +
     np.polyfit(Y2[i2, j], Y1[i1, j], 1)[0] * Y2[i2, j])) ** 2) / (len(common) - 2)) *
    np.sqrt(1 / len(common) + Y2[i2, j].mean() ** 2 / np.sum((Y2[i2, j] - Y2[i2, j].mean()) ** 2))
    for j in range(len(tgt))])
print(f'\n  Stouffer 组合（s>1）: Z = {z_s.sum()/np.sqrt(len(z_s)):.3f}, '
      f'p = {stats.norm.sf(z_s.sum()/np.sqrt(len(z_s))):.3e}')
print(f'  Stouffer 组合（a>0）: Z = {z_a.sum()/np.sqrt(len(z_a)):.3f}, '
      f'p = {stats.norm.sf(z_a.sum()/np.sqrt(len(z_a))):.3e}')
