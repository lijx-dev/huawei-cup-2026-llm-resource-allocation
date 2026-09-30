# -*- coding: utf-8 -*-
"""乘子位置：显著性诊断 + bootstrap 置信区间 + B8 反向验证"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))

print('=== 1) 振幅回归标准误 ===')
rows = []
for (n, d), sub in b7.groupby(['N_params_B', 'D_tokens_B']):
    sub = sub.sort_values('Q_score')
    rows.append((n, d, sub.val_loss.max() - sub.val_loss.min()))
A = pd.DataFrame(rows, columns=['N', 'D', 'amp'])
lN, lD = np.log(A.N.values), np.log(A.D.values)
y = np.log(A.amp.values)
X = np.column_stack([np.ones(len(A)), lN, lD])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
rss = np.sum((y - X @ coef) ** 2)
n, k = len(A), 3
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * rss / (n - k))
tval = coef / se
for i, nm in enumerate(['const', 'lnN', 'lnD']):
    print(f'  {nm:>6}: 系数={coef[i]:+.4f}  se={se[i]:.4f}  t={tval[i]:+.2f}')
print(f'  (t 临界值 |t|>2.02 为 5% 显著, n-k={n-k})')

print()
print('=== 2) 分层诊断：固定一维看另一维 ===')
print('  [固定 D，振幅随 N 的斜率]')
for d, sub in A.groupby('D'):
    c = np.polyfit(np.log(sub.N), np.log(sub.amp), 1)
    print(f'    D={d:>6}: d ln(amp)/d lnN = {c[0]:+.4f}')
print('  [固定 N，振幅随 D 的斜率]')
for nn, sub in A.groupby('N'):
    c = np.polyfit(np.log(sub.D), np.log(sub.amp), 1)
    print(f'    N={nn:>6}: d ln(amp)/d lnD = {c[0]:+.4f}')

print()
print('=== 3) M3 双挂模型的 bootstrap 置信区间（按 (N,D) 组重抽样）===')
N_, D_, Q_, L_ = b7.N_params_B.values, b7.D_tokens_B.values, b7.Q_score.values, b7.val_loss.values
groups = list(b7.groupby(['N_params_B', 'D_tokens_B']))
rng = np.random.default_rng(20260924)


def fit_m3(idx_groups):
    sub = pd.concat([groups[i][1] for i in idx_groups])
    n_, d_, q_, l_ = sub.N_params_B.values, sub.D_tokens_B.values, sub.Q_score.values, sub.val_loss.values

    def model(p):
        E, a, al, b, be, d1, d2 = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5]), np.exp(p[6])
        return E + a * n_ ** (-al) * q_ ** (-d1) + b * d_ ** (-be) * q_ ** (-d2)

    p0 = [1.56, np.log(0.51), np.log(0.26), np.log(1.21), np.log(0.25), np.log(0.14), np.log(0.10)]
    res = least_squares(lambda p: model(p) - l_, p0, max_nfev=20000)
    return np.exp(res.x[[2, 4, 5, 6]])


base = fit_m3(range(len(groups)))
print(f'  全样本估计: alpha={base[0]:.4f}, beta={base[1]:.4f}, delta_N={base[2]:.4f}, delta_D={base[3]:.4f}')
boots = []
for _ in range(120):
    idx = rng.integers(0, len(groups), len(groups))
    try:
        boots.append(fit_m3(idx))
    except Exception:
        pass
boots = np.array(boots)
for i, nm in enumerate(['alpha', 'beta', 'delta_N', 'delta_D']):
    lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
    print(f'  {nm:>8}: 95%CI = [{lo:.4f}, {hi:.4f}]')
print(f'  delta_N/delta_D 比值中位数 = {np.median(boots[:,2]/boots[:,3]):.3f}')

print()
print('=== 4) B8 方向验证：Q 替换为 1-Q 后与 B6/B7 规律是否一致 ===')
b8c = b8[b8.data_type == 'calibrated'].copy()
for tag, sub in [('原始 Q', b8c), ('Q -> 1-Q', b8c.assign(Q_score=1 - b8c.Q_score))]:
    cors = []
    for (n, d), g in sub.groupby(['N_params_B', 'D_tokens_B']):
        if g.Q_score.nunique() >= 4:
            cors.append(np.corrcoef(g.Q_score, g.val_loss)[0, 1])
    print(f'  [{tag}] 组内 corr(Q,L) 中位数 = {np.median(cors):+.3f}  (B6/B7 为 -0.93)')
print()
print('  B8 calibrated 中 N=0.07, D=5 的原始记录：')
print(b8c[(b8c.N_params_B == 0.07) & (b8c.D_tokens_B == 5)][['Q_score', 'val_loss']].to_string(index=False))
print('  B6 中 N=0.07, D=10 的原始记录（对照）：')
print(b7[(b7.N_params_B == 0.07) & (b7.D_tokens_B == 10)][['Q_score', 'val_loss']].to_string(index=False))