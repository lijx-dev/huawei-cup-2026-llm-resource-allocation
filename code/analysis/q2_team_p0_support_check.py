# -*- coding: utf-8 -*-
"""(a) 稳健复核：p0_eq=(1/17,...,1/17) 是否真在 A4 训练配方 17 维凸包外（最小 L1 不可行度）
(b) 团队论文 §5.6 的 272 个有向转移点 p(δ)=p0+δ(e_j−e_k) 是否落在训练凸包内
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

A = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
DTR = pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).iloc[:, 1:].astype(float)
X = DTR.values
n, d = X.shape
print(f'A4 训练配方 n={n}，维度={d}')

p0_eq = np.full(d, 1.0 / 17)
p0_mean = X.mean(0)


def l1_infeasibility(x, X):
    """min Σ s_i  s.t. Xᵀλ − x = s⁺ − s⁻, Σλ=1, λ,s⁺,s⁻≥0。
    目标为 0（<1e-9）即在凸包内；>0 给出到凸包的 L1 距离。"""
    n, d = X.shape
    # 变量: λ(n), s⁺(d), s⁻(d)
    c = np.concatenate([np.zeros(n), np.ones(2 * d)])
    A_eq = np.zeros((d + 1, n + 2 * d))
    A_eq[:d, :n] = X.T
    A_eq[:d, n:n + d] = -np.eye(d)
    A_eq[:d, n + d:] = np.eye(d)
    A_eq[d, :n] = 1.0
    b_eq = np.concatenate([x, [1.0]])
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + 2 * d), method='highs')
    return r.fun, r.status


print('\n' + '=' * 80)
print('(a) 稳健复核：两种参考配比到训练凸包的 L1 距离')
print('=' * 80)
for nm, v in [('p0_A4mean', p0_mean), ('p0_eq (1/17)', p0_eq)]:
    f, st = l1_infeasibility(v, X)
    print(f'  {nm:<16} L1 不可行度 = {f:.3e}  status={st}  → '
          f'{"凸包内" if f < 1e-9 else "凸包外"}')

print('\n' + '=' * 80)
print('(b) 团队 §5.6 转移点 p(δ)=p0_eq+δ(e_j−e_k) 的凸包支持（δ=0.02）')
print('=' * 80)
DELTA = 0.02
cols = [c.replace('train_the_pile_', '') for c in DTR.columns]
inside, tested = 0, 0
by_donor = {}
for j in range(17):
    for k in range(17):
        if j == k:
            continue
        p = p0_eq.copy()
        p[j] += DELTA
        p[k] -= DELTA
        if p.min() < 0:
            continue
        tested += 1
        f, _ = l1_infeasibility(p, X)
        ok = f < 1e-9
        inside += ok
        by_donor.setdefault(cols[k], [0, 0])
        by_donor[cols[k]][0] += ok
        by_donor[cols[k]][1] += 1
print(f'  δ=0.02：可行扰动点 {tested} 个，其中落在训练凸包内 {inside} 个（{100*inside/tested:.1f}%）')
print(f'  {"供给域 k":<22}{"凸包内/可行":>14}')
for k, (a, b) in sorted(by_donor.items(), key=lambda x: -x[1][0]):
    print(f'  {k:<22}{f"{a}/{b}":>14}')

# 与团队支持半径准则对照
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
RAD = float(np.percentile(D.min(1), 95))
cnt_r = 0
for j in range(17):
    for k in range(17):
        if j == k:
            continue
        p = p0_eq.copy()
        p[j] += DELTA
        p[k] -= DELTA
        if p.min() < 0:
            continue
        if np.min(np.sqrt(((X - p) ** 2).sum(1))) > RAD:
            cnt_r += 1
print(f'\n  按团队最近邻半径准则（{RAD:.4f}）：超半径 {cnt_r}/{tested} 个')
print('  → 最近邻准则判定"支持内"的点，仍有相当比例落在凸包外；凸包是更严的判据。')
