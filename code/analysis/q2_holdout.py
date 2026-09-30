# -*- coding: utf-8 -*-
"""五种结构的留出验证：质量改E(M4a) vs 挂N(M1) vs 挂D(M2) vs 双挂(M3) vs 三项全挂(M5)"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))

LOGE = np.log


def unpack(p):
    return p[0], p[1], np.exp(p[2]), np.exp(p[3]), p[4], np.exp(p[5]), np.exp(p[6]), np.exp(p[7])


def M1(p, N, D, Q):   # 质量挂 N 项
    E, A, a, B, b, d = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5])
    return E + A * N ** (-a) * Q ** (-d) + B * D ** (-b)


def M2(p, N, D, Q):   # 质量挂 D 项（思路 2/3/4 主方案）
    E, A, a, B, b, d = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5])
    return E + A * N ** (-a) + B * D ** (-b) * Q ** (-d)


def M3(p, N, D, Q):   # 双挂
    E, A, a, B, b, d1, d2 = (p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]),
                             np.exp(p[4]), np.exp(p[5]), np.exp(p[6]))
    return E + A * N ** (-a) * Q ** (-d1) + B * D ** (-b) * Q ** (-d2)


def M4a(p, N, D, Q):  # 质量改损失下限（思路四备选）
    E0, E1, A, a, B, b = p[0], p[1], np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5])
    return E0 + E1 * (1 - Q) + A * N ** (-a) + B * D ** (-b)


def M5(p, N, D, Q):   # 三项全挂：E + N项 + D项
    E0, E1, A, a, B, b, d1, d2 = (p[0], p[1], np.exp(p[2]), np.exp(p[3]),
                                  np.exp(p[4]), np.exp(p[5]), np.exp(p[6]), np.exp(p[7]))
    return E0 + E1 * (1 - Q) + A * N ** (-a) * Q ** (-d1) + B * D ** (-b) * Q ** (-d2)


MODELS = {
    'M1 挂N项': (M1, 6, [1.53, LOGE(0.60), LOGE(0.2237), LOGE(1.33), LOGE(0.2996), LOGE(0.1722)]),
    'M2 挂D项': (M2, 6, [0.68, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), LOGE(0.0980)]),
    'M3 双挂': (M3, 7, [1.56, LOGE(0.51), LOGE(0.2643), LOGE(1.21), LOGE(0.2489), LOGE(0.1418), LOGE(0.0999)]),
    'M4a 改E': (M4a, 6, [1.52, 0.36, LOGE(0.53), LOGE(0.2832), LOGE(1.33), LOGE(0.2996)]),
    'M5 三项全挂': (M5, 8, [1.5, 0.36, LOGE(0.5), LOGE(0.26), LOGE(1.2), LOGE(0.25), LOGE(0.14), LOGE(0.10)]),
}

N_, D_, Q_, L_ = b7.N_params_B.values, b7.D_tokens_B.values, b7.Q_score.values, b7.val_loss.values
n = len(L_)
lo = [0.2, -2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), LOGE(1e-3)]
hi = [4, 6, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), LOGE(10)]

print('=' * 78)
print('1) 全样本拟合（B7, 450 点）')
print('=' * 78)
print(f'{"模型":<14}{"k":>3}{"RMSE":>10}{"AIC":>11}{"BIC":>11}')
full = {}
for name, (f, k, p0) in MODELS.items():
    res = least_squares(lambda p: f(p, N_, D_, Q_) - L_, p0,
                        bounds=(lo[:k], hi[:k]), max_nfev=40000)
    rss = np.sum(res.fun ** 2)
    rmse = np.sqrt(rss / n)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    full[name] = (res.x, rmse, aic)
    print(f'{name:<14}{k:>3}{rmse:>10.5f}{aic:>11.2f}{bic:>11.2f}')

print()
print('  M4a / M5 的参数（看 E1 是否显著非零）:')
for name in ['M4a 改E', 'M5 三项全挂']:
    x = full[name][0]
    print(f'    {name}: E0={x[0]:.4f}, E1={x[1]:.4f}, A={np.exp(x[2]):.4f}, a={np.exp(x[3]):.4f}, '
          f'B={np.exp(x[4]):.4f}, b={np.exp(x[5]):.4f}'
          + (f', d1={np.exp(x[6]):.4f}, d2={np.exp(x[7]):.4f}' if len(x) > 6 else ''))

print()
print('=' * 78)
print('2) 留出验证：留一 (N,D) 组（共 45 组），预测该组 10 个点')
print('=' * 78)
groups = list(b7.groupby(['N_params_B', 'D_tokens_B']))
loo = {m: [] for m in MODELS}
for i in range(len(groups)):
    tr = pd.concat([g for j, (_, g) in enumerate(groups) if j != i])
    te = groups[i][1]
    for name, (f, k, p0) in MODELS.items():
        try:
            res = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values,
                                             tr.Q_score.values) - tr.val_loss.values,
                                p0, bounds=(lo[:k], hi[:k]), max_nfev=4000)
            pr = f(res.x, te.N_params_B.values, te.D_tokens_B.values, te.Q_score.values)
            loo[name].extend(te.val_loss.values - pr)
        except Exception:
            loo[name].extend([np.nan] * len(te))
print(f'{"模型":<14}{"留出 RMSE":>12}{"最大绝对误差":>14}')
for name in MODELS:
    e = np.array(loo[name], dtype=float)
    print(f'{name:<14}{np.sqrt(np.nanmean(e**2)):>12.5f}{np.nanmax(np.abs(e)):>14.4f}')

print()
print('=' * 78)
print('3) 更强留出：整条 N 留出（9 折，每次留出某 N 下全部 5 个 D）')
print('=' * 78)
looN = {m: [] for m in MODELS}
for nv in sorted(b7.N_params_B.unique()):
    tr = b7[b7.N_params_B != nv]
    te = b7[b7.N_params_B == nv]
    for name, (f, k, p0) in MODELS.items():
        try:
            res = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values,
                                             tr.Q_score.values) - tr.val_loss.values,
                                p0, bounds=(lo[:k], hi[:k]), max_nfev=4000)
            pr = f(res.x, te.N_params_B.values, te.D_tokens_B.values, te.Q_score.values)
            looN[name].extend(te.val_loss.values - pr)
        except Exception:
            looN[name].extend([np.nan] * len(te))
print(f'{"模型":<14}{"留出 RMSE":>12}{"最大绝对误差":>14}')
for name in MODELS:
    e = np.array(looN[name], dtype=float)
    print(f'{name:<14}{np.sqrt(np.nanmean(e**2)):>12.5f}{np.nanmax(np.abs(e)):>14.4f}')