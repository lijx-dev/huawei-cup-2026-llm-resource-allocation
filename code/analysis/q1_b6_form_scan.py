# -*- coding: utf-8 -*-
"""审阅用：扫描 B6 质量项的不同挂载形式，找出哪一形式能复现论文 5.4.2 的
   M_0 RMSE=0.128762、M_Q RMSE=0.076087、gamma_Q=0.317893。
   同时核对 Q 与 Loss 的实测方向。
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

b6 = pd.read_csv(r'd:\F题\F题\real_attachments\B_scaling_laws\supplementary_NQ_experiment.csv')
N, D, Q, L = (b6[c].values for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
Q0 = 0.5

print('=' * 88)
print('[0] B6 中 Q 与 Loss 的实测方向（组内，固定 N,D）')
print('=' * 88)
g = b6.groupby(['N_params_B', 'D_tokens_B'])
pos = neg = 0
for _, sub in g:
    s = sub.sort_values('Q_score')
    r = np.corrcoef(s.Q_score, s.val_loss)[0, 1]
    if r > 0:
        pos += 1
    else:
        neg += 1
print(f'  45 个 (N,D) 组中，Q 与 val_loss 组内相关的符号：正 {pos} 组，负 {neg} 组')
print(f'  全体 Spearman(Q, val_loss) = {pd.Series(Q).corr(pd.Series(L), method="spearman"):+.4f}')
print('  => 若多数组为负，说明 Q 越大 Loss 越小（质量正向）。')


def fit(form, npar, inits, nstart=40):
    best = None
    for s in range(nstart):
        rng = np.random.default_rng(s)
        th0 = np.array(inits) + rng.normal(0, 0.4, npar)
        sol = least_squares(lambda th: form(th) - L, th0, method='lm', max_nfev=200000)
        if best is None or sol.cost < best.cost:
            best = sol
    return np.sqrt(2 * best.cost / len(L)), best.x


forms = {
    'M0 无质量项  E+A N^-a+B D^-b':
        (lambda th: np.exp(th[0]) + np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4]),
         5, np.log([1.7, 0.35, 0.34, 1.24, 0.28])),
    '(i) 挂D项 exp{+g(Q-Q0)}':
        (lambda th: np.exp(th[0]) + np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4]) * np.exp(th[5] * (Q - Q0)),
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]) .tolist() + [0.3]),
    '(ii) 双挂 exp{+g(Q-Q0)}':
        (lambda th: np.exp(th[0]) + (np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4])) * np.exp(th[5] * (Q - Q0)),
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]).tolist() + [0.3]),
    '(iii) 挂N项 exp{+g(Q-Q0)}':
        (lambda th: np.exp(th[0]) + np.exp(th[1]) * N ** -np.exp(th[2]) * np.exp(th[5] * (Q - Q0)) + np.exp(th[3]) * D ** -np.exp(th[4]),
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]).tolist() + [0.3]),
    '(iv) 挂D项 exp{-g(Q-Q0)}':
        (lambda th: np.exp(th[0]) + np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4]) * np.exp(-th[5] * (Q - Q0)),
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]).tolist() + [0.3]),
    '(v) 挂D项 (Q/Q0)^g':
        (lambda th: np.exp(th[0]) + np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4]) * (Q / Q0) ** th[5],
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]).tolist() + [0.3]),
    '(vi) 双挂 exp{-g(Q-Q0)}':
        (lambda th: np.exp(th[0]) + (np.exp(th[1]) * N ** -np.exp(th[2]) + np.exp(th[3]) * D ** -np.exp(th[4])) * np.exp(-th[5] * (Q - Q0)),
         6, np.log([1.7, 0.35, 0.34, 1.24, 0.28]).tolist() + [0.3]),
}

print('\n' + '=' * 88)
print('[1] 各形式的自由拟合结果（对照论文 M_0 RMSE=0.128762, M_Q RMSE=0.076087, gamma=0.317893）')
print('=' * 88)
print(f"{'形式':<32}{'RMSE':>10}{'E':>9}{'A':>8}{'alpha':>8}{'B':>8}{'beta':>8}{'gamma':>10}")
for k, (f, npar, ini) in forms.items():
    r, th = fit(f, npar, ini)
    p = np.exp(th[:5])
    gg = th[5] if npar == 6 else np.nan
    print(f'{k:<32}{r:>10.6f}{p[0]:>9.4f}{p[1]:>8.4f}{p[2]:>8.5f}{p[3]:>8.4f}{p[4]:>8.5f}{gg:>10.6f}')
