# -*- coding: utf-8 -*-
"""诊断：为什么 A6–A7/A8–A9 的 256 条检验配方全部落在 A4 训练配方凸包之外。
检查 (i) LP 自检（训练点的凸组合是否被判为凸包内）(ii) 训练/检验配比的极端程度分布差异
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

A = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'


def raw(stem):
    df = pd.read_csv(os.path.join(A, stem + '.csv')).iloc[:, 1:].astype(float)
    return df.values


Xtr, Xte = raw('train_mixture_1m'), raw('test_mixture_1m')


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
    return r.fun


print('=' * 76)
print('(i) LP 自检')
print('=' * 76)
print(f'  训练配方自身（应=0）            L1 = {l1_inf(Xtr[0], Xtr):.3e}')
c2 = 0.5 * Xtr[0] + 0.5 * Xtr[1]
print(f'  两条训练配方的中点（应=0）      L1 = {l1_inf(c2, Xtr):.3e}')
print(f'  512 条训练配方均值（应=0）      L1 = {l1_inf(Xtr.mean(0), Xtr):.3e}')
ext = np.zeros(17)
ext[0] = 0.9
ext[1:] = 0.1 / 16
print(f'  极端点 (0.9, 0.1/16×16)（应>0） L1 = {l1_inf(ext, Xtr):.3e}')

print('\n' + '=' * 76)
print('(ii) 训练 vs 检验配比的极端程度')
print('=' * 76)


def desc(name, X):
    mx = X.max(1)
    ent = -(np.where(X > 0, X * np.log(np.maximum(X, 1e-300)), 0)).sum(1)
    nz = (X > 1e-9).sum(1)
    print(f'  {name:<12}n={len(X):<5} 最大单域占比 中位={np.median(mx):.3f} p95={np.percentile(mx,95):.3f} '
          f'max={mx.max():.3f} | 有效域数 中位={np.median(nz):.0f} | 熵 中位={np.median(ent):.3f} '
          f'p95={np.percentile(ent,95):.3f}')
    return mx, ent, nz


desc('A4 训练', Xtr)
desc('A6–A7 检验', Xte)
print('\n  配比取值是否量化到 0.01 网格：')
for nm, X in [('A4 训练', Xtr), ('A6–A7 检验', Xte)]:
    v = X.ravel()
    v = v[v > 1e-9]
    q = np.abs(v * 100 - np.round(v * 100))
    print(f'    {nm}: |100p − round(100p)| 的 p95 = {np.percentile(q,95):.2e}  '
          f'max = {q.max():.2e}  → {"0.01 网格" if q.max() < 1e-6 else "非网格"}')

print('\n  检验配方到训练凸包的 L1 距离分布（A6–A7，256 条）：')
d = np.array([l1_inf(Xte[i], Xtr) for i in range(len(Xte))])
print(f'    min={d.min():.4f}  中位={np.median(d):.4f}  p95={np.percentile(d,95):.4f}  max={d.max():.4f}')
print(f'    全部 >0 的比例 = {(d>1e-9).mean()*100:.1f}%')
