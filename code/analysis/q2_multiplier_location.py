# -*- coding: utf-8 -*-
"""乘子位置识别：质量效应挂在 N 项还是 D 项？

M0 经典   : L = E + A N^-a + B D^-b
M1 挂N项  : L = E + A N^-a Q^-d + B D^-b
M2 挂D项  : L = E + A N^-a + B D^-b Q^-d
M3 双挂   : L = E + A N^-a Q^-d1 + B D^-b Q^-d2
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))

N = b7.N_params_B.values
D = b7.D_tokens_B.values
Q = b7.Q_score.values
L = b7.val_loss.values

print('=== 1) 振幅回归（45 个 (N,D) 组）===')
rows = []
for (n, d), sub in b7.groupby(['N_params_B', 'D_tokens_B']):
    sub = sub.sort_values('Q_score')
    amp = sub.val_loss.max() - sub.val_loss.min()
    sl = np.polyfit(sub.Q_score, sub.val_loss, 1)[0]
    rows.append((n, d, amp, sl))
A = pd.DataFrame(rows, columns=['N', 'D', 'amp', 'slope'])

lN, lD = np.log(A.N.values), np.log(A.D.values)
print(f'  设计矩阵共线检查 corr(lnN, lnD) = {np.corrcoef(lN, lD)[0,1]:+.4f}')

y = np.log(A.amp.values)
X = np.column_stack([np.ones(len(A)), lN, lD])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f'  ln(amp) = {coef[0]:.4f} + {coef[1]:.4f}*lnN + {coef[2]:.4f}*lnD   R2={r2:.4f}')
print(f'  → 若质量挂 N 项，期望 lnN 系数≈-0.34、lnD 系数≈0')
print(f'  → 若质量挂 D 项，期望 lnN 系数≈0、lnD 系数≈-0.28')

Xs = np.column_stack([(lN - lN.mean()) / lN.std(), (lD - lD.mean()) / lD.std()])
ys = (y - y.mean()) / y.std()
cs, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A)), Xs]), ys, rcond=None)
print(f'  标准化系数: lnN={cs[1]:+.3f}, lnD={cs[2]:+.3f}  (绝对值大者为主导)')

print()
print('=== 2) 四种模型非线性最小二乘对比（B7 全量 450 点）===')
n_obs = len(L)


def fit(kind):
    def unpack(p):
        E = p[0]
        a = np.exp(p[1])
        al = np.exp(p[2])
        b = np.exp(p[3])
        be = np.exp(p[4])
        if kind in ('M1', 'M2', 'M3'):
            d = np.exp(p[5])
        if kind == 'M3':
            d2 = np.exp(p[6])
        if kind == 'M0':
            return E, a, al, b, be
        if kind in ('M1', 'M2'):
            return E, a, al, b, be, d
        return E, a, al, b, be, d, d2

    def model(p):
        if kind == 'M0':
            E, a, al, b, be = unpack(p)
            return E + a * N ** (-al) + b * D ** (-be)
        if kind == 'M1':
            E, a, al, b, be, d = unpack(p)
            return E + a * N ** (-al) * Q ** (-d) + b * D ** (-be)
        if kind == 'M2':
            E, a, al, b, be, d = unpack(p)
            return E + a * N ** (-al) + b * D ** (-be) * Q ** (-d)
        E, a, al, b, be, d, d2 = unpack(p)
        return E + a * N ** (-al) * Q ** (-d) + b * D ** (-be) * Q ** (-d2)

    p0 = [1.5, 0.0, np.log(0.34), 0.0, np.log(0.28), np.log(0.2)]
    if kind == 'M3':
        p0 = p0 + [np.log(0.2)]
    if kind == 'M0':
        p0 = p0[:5]
    res = least_squares(lambda p: model(p) - L, p0, max_nfev=20000)
    rss = np.sum(res.fun ** 2)
    k = len(p0)
    aic = n_obs * np.log(rss / n_obs) + 2 * k
    rmse = np.sqrt(rss / n_obs)
    return unpack(res.x), rmse, aic, k


print(f'{"模型":<22}{"RMSE":>10}{"AIC":>12}{"参数":>6}   参数估计')
best = None
for kind, name in [('M0', 'M0 经典(无Q)'), ('M1', 'M1 质量挂N项'), ('M2', 'M2 质量挂D项'), ('M3', 'M3 双挂')]:
    pars, rmse, aic, k = fit(kind)
    ps = ', '.join(f'{v:.4f}' for v in pars)
    print(f'{name:<22}{rmse:>10.5f}{aic:>12.2f}{k:>6}   {ps}')
    if best is None or aic < best[1]:
        best = (name, aic)
print(f'→ AIC 最小: {best[0]}')