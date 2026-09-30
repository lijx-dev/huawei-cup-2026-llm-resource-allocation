# -*- coding: utf-8 -*-
"""量化"设计空间偏移量"：p0_eq、团队 §5.6 的 272 个转移点、A6–A7 检验配方
各自到 A4 训练配方凸包的 L1 距离，并给出分位对照。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

A = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
X = pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
T = pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
COLS = [c.replace('train_the_pile_', '')
        for c in pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).columns[1:]]


def l1_inf(x, X):
    n, d = X.shape
    c = np.concatenate([np.zeros(n), np.ones(2 * d)])
    A_eq = np.zeros((d + 1, n + 2 * d))
    A_eq[:d, :n] = X.T
    A_eq[:d, n:n + d] = -np.eye(d)
    A_eq[:d, n + d:] = np.eye(d)
    A_eq[d, :n] = 1.0
    r = linprog(c, A_eq=A_eq, b_eq=np.concatenate([x, [1.0]]),
                bounds=[(0, None)] * (n + 2 * d), method='highs')
    return float(r.fun)


p0_mean, p0_eq = X.mean(0), np.full(17, 1.0 / 17)
d_te = np.array([l1_inf(T[i], X) for i in range(len(T))])
print('A6–A7 检验配方到训练凸包的 L1 距离：'
      f'中位 {np.median(d_te):.4f}  p95 {np.percentile(d_te,95):.4f}  max {d_te.max():.4f}')

for nm, v in [('p0_A4mean', p0_mean), ('p0_eq', p0_eq)]:
    v = np.asarray(v, float)
    f = l1_inf(v, X)
    pct = float((d_te < f).mean() * 100)
    nn = float(np.min(np.sqrt(((X - v) ** 2).sum(1))))
    print(f'  {nm:<10} L1 = {f:.4f}  （超过 {pct:.1f}% 的 A6–A7 检验配方）  最近邻距离 = {nn:.4f}')

DELTA = 0.02
ds = []
for j in range(17):
    for k in range(17):
        if j == k:
            continue
        p = p0_eq.copy()
        p[j] += DELTA
        p[k] -= DELTA
        if p.min() < 0:
            continue
        ds.append(l1_inf(p, X))
ds = np.array(ds)
print(f'\n团队 §5.6 的 {len(ds)} 个转移点（δ=0.02，以 p0_eq 为基准）：')
print(f'  L1 距离 中位 {np.median(ds):.4f}  min {ds.min():.4f}  max {ds.max():.4f}')
print(f'  超过 A6–A7 检验配方 p95（{np.percentile(d_te,95):.4f}）的点数：'
      f'{int((ds > np.percentile(d_te,95)).sum())}/{len(ds)}')
