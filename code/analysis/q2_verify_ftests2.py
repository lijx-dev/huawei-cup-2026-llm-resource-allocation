# -*- coding: utf-8 -*-
"""核对：文档中嵌套 F 检验应基于"指数形式"还是"幂律形式"（B6∪B7, n=450）"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import stats

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment_expanded.csv'))
d = pd.concat([b6, b7], ignore_index=True).drop_duplicates(
    subset=['N_params_B', 'D_tokens_B', 'Q_score'])
N = d.N_params_B.values.astype(float); D = d.D_tokens_B.values.astype(float)
Q = d.Q_score.values.astype(float); L = d.val_loss.values.astype(float)
n = len(L)
LO = [0.2, 1e-8, 1e-3, 1e-8, 1e-3, 1e-3, 1e-3, 1e-3]
HI = [4.0, 1e8, 3.0, 1e8, 3.0, 10.0, 10.0, 10.0]


def fit(f, p0):
    k = len(p0)
    r = least_squares(lambda p: f(p) - L, p0, bounds=(LO[:k], HI[:k]), max_nfev=200000)
    return np.sum(r.fun ** 2), k


def Ft(a, ka, b_, kb):
    f = ((a - b_) / (kb - ka)) / (b_ / (n - kb))
    return f, kb - ka, n - kb, 1 - stats.f.cdf(f, kb - ka, n - kb)


print('=' * 76)
print('【指数形式】')
print('=' * 76)
g0 = lambda p: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4])
gE = lambda p: p[0] + p[1] * (1 - Q) + p[2] * N ** (-p[3]) + p[4] * D ** (-p[5])
gD = lambda p: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]) * np.exp(-p[5] * Q)
gN = lambda p: p[0] + p[1] * N ** (-p[2]) * np.exp(-p[5] * Q) + p[3] * D ** (-p[4])
g2 = lambda p: (p[0] + p[1] * N ** (-p[2]) * np.exp(-p[5] * Q)
                + p[3] * D ** (-p[4]) * np.exp(-p[6] * Q))
g3 = lambda p: (p[0] + p[1] * (1 - Q) + p[2] * N ** (-p[3]) * np.exp(-p[6] * Q)
                + p[4] * D ** (-p[5]) * np.exp(-p[7] * Q))

r0 = fit(g0, [1.7, .35, .34, 1.24, .28])
rE = fit(gE, [1.52, .36, .53, .2832, 1.33, .2996])
rD = fit(gD, [.68, .53, .2832, 1.85, .0831, .098])
rN = fit(gN, [1.53, .60, .2237, 1.33, .2996, .1722])
r2 = fit(g2, [1.56, .51, .2643, 1.21, .2489, .4152, .2469])
r3 = fit(g3, [1.5, .36, .5, .26, 1.2, .25, .35, .13])
print(f'RSS: 经典={r0[0]:.5f} 改E={rE[0]:.5f} 挂D={rD[0]:.5f} 挂N={rN[0]:.5f} 双挂={r2[0]:.5f} 三项={r3[0]:.5f}')
for tag, a, ka, b_, kb in [('经典→改E', *r0, *rE), ('经典→挂D', *r0, *rD), ('经典→挂N', *r0, *rN),
                           ('挂D→双挂', *rD, *r2), ('挂N→双挂', *rN, *r2), ('双挂→三项', *r2, *r3)]:
    f, d1, d2, p = Ft(a, ka, b_, kb)
    print(f'  {tag:<10} F({d1},{d2}) = {f:>8.1f}   p = {p:.2e}')

print()
print('=' * 76)
print('【幂律形式】')
print('=' * 76)
hD = lambda p: p[0] + p[1] * N ** (-p[2]) + p[3] * D ** (-p[4]) * Q ** (-p[5])
hN = lambda p: p[0] + p[1] * N ** (-p[2]) * Q ** (-p[5]) + p[3] * D ** (-p[4])
h2 = lambda p: p[0] + p[1] * N ** (-p[2]) * Q ** (-p[5]) + p[3] * D ** (-p[4]) * Q ** (-p[6])
sD = fit(hD, [.68, .53, .2832, 1.85, .0831, .098])
sN = fit(hN, [1.53, .60, .2237, 1.33, .2996, .1722])
s2 = fit(h2, [1.56, .51, .2643, 1.21, .2489, .1418, .0999])
for tag, a, ka, b_, kb in [('挂D→双挂', *sD, *s2), ('挂N→双挂', *sN, *s2)]:
    f, d1, d2, p = Ft(a, ka, b_, kb)
    print(f'  {tag:<10} F({d1},{d2}) = {f:>8.1f}   p = {p:.2e}')
