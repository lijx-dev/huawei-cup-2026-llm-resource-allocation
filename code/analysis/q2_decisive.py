# -*- coding: utf-8 -*-
"""三种决定性检验
A. 留出泛化：留一(N,D)组 / 留一N —— 全样本 AIC 高不代表泛化好
B. 锚定 B1 的 beta=0.2799（真实数据独立估出）后，各思路还剩多少自由度、拟合变差多少
C. 振幅 A(N,D)=max_Q L - min_Q L 的 N/D 依赖 —— 判定“质量只能挂 D 项”是否成立
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
LOGE = np.log

# B1 独立估出的真实标度律参数（真实数据、非半合成）
N1, D1, L1 = (b1.N_params_B.values.astype(float), b1.D_tokens_B.values.astype(float),
              b1.val_loss.values.astype(float))
r1 = least_squares(lambda p: p[0] + np.exp(p[1]) * N1 ** (-np.exp(p[2])) + np.exp(p[3]) * D1 ** (-np.exp(p[4])) - L1,
                   [1.5, LOGE(0.35), LOGE(0.34), LOGE(1.24), LOGE(0.28)],
                   bounds=([0, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                           [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]), max_nfev=80000)
E1v, A1v, al1v, B1v, be1v = r1.x[0], np.exp(r1.x[1]), np.exp(r1.x[2]), np.exp(r1.x[3]), np.exp(r1.x[4])
print(f'[B1 真实数据锚点] E={E1v:.4f} A={A1v:.4f} alpha={al1v:.4f} B={B1v:.4f} beta={be1v:.4f}')

# ---------- 模型（N,D,Q 显式传参，便于留出） ----------
def M_Dpow(p, N, D, Q):
    E, A, a, B, b, d = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5])
    return E + A * N ** (-a) + B * D ** (-b) * Q ** (-d)

def M_NDpow(p, N, D, Q):
    E, A, a, B, b, d1, d2 = (p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]),
                             np.exp(p[5]), np.exp(p[6]))
    return E + A * N ** (-a) * Q ** (-d1) + B * D ** (-b) * Q ** (-d2)

def M_Dexp(p, N, D, Q):
    E, A, a, B, b, r = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), p[5]
    return E + A * N ** (-a) + B * D ** (-b) * np.exp(-r * Q)

def M_DexpG(p, N, D, Q):
    E, A, a, B, b, r, G = (p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), p[5], p[6])
    return E + A * N ** (-a) + B * D ** (-b) * np.exp(-r * Q) + G * (1 - Q)

def M_Elin(p, N, D, Q):
    E0, E1, A, a, B, b = p[0], p[1], np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), np.exp(p[5])
    return E0 + E1 * (1 - Q) + A * N ** (-a) + B * D ** (-b)

def M_NDexp(p, N, D, Q):
    E, A, a, B, b, r1, r2 = (p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), p[5], p[6])
    return E + A * N ** (-a) * np.exp(-r1 * Q) + B * D ** (-b) * np.exp(-r2 * Q)

def M_NDexpE(p, N, D, Q):
    E0, E1, A, a, B, b, r1, r2 = (p[0], p[1], np.exp(p[2]), np.exp(p[3]), np.exp(p[4]),
                                  np.exp(p[5]), p[6], p[7])
    return E0 + E1 * (1 - Q) + A * N ** (-a) * np.exp(-r1 * Q) + B * D ** (-b) * np.exp(-r2 * Q)

MODELS = {
    'P1 挂D·幂律(思路一)':   (M_Dpow,   6, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), LOGE(0.0980)],
                              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10)]),
    'P2 双挂·幂律(思路二)':   (M_NDpow,  7, [1.56, LOGE(0.51), LOGE(0.2643), LOGE(1.21), LOGE(0.2489), LOGE(0.1418), LOGE(0.0999)],
                              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), LOGE(1e-3)],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), LOGE(10)]),
    'P3 挂D·指数(思路三/四)': (M_Dexp,   6, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), 1.0],
                              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
    'P3G 挂D·指数+附加G':     (M_DexpG,  7, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), 1.0, 0.0],
                              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -5],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 5]),
    'P4 改E·线性':            (M_Elin,   6, [1.52, 0.36, LOGE(0.53), LOGE(0.2832), LOGE(1.33), LOGE(0.2996)],
                              [0.2, -5, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                              [4, 5, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]),
    'P5 双挂·指数':           (M_NDexp,  7, [1.56, LOGE(0.51), LOGE(0.2643), LOGE(1.21), LOGE(0.2489), 1.5, 1.0],
                              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
    'P6 三项全挂·指数':       (M_NDexpE, 8, [1.5, 0.36, LOGE(0.5), LOGE(0.26), LOGE(1.2), LOGE(0.25), 1.5, 1.0],
                              [0.2, -5, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                              [4, 5, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
}

N_, D_, Q_, L_ = (b7.N_params_B.values.astype(float), b7.D_tokens_B.values.astype(float),
                  b7.Q_score.values.astype(float), b7.val_loss.values.astype(float))
n = len(L_)

print()
print('=' * 100)
print('A) 留出泛化')
print('=' * 100)
groups = list(b7.groupby(['N_params_B', 'D_tokens_B']))
resA = {m: [] for m in MODELS}
for i in range(len(groups)):
    tr = pd.concat([g for j, (_, g) in enumerate(groups) if j != i])
    te = groups[i][1]
    for name, (f, k, p0, lo, hi) in MODELS.items():
        try:
            r = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values,
                                           tr.Q_score.values) - tr.val_loss.values,
                              p0, bounds=(lo, hi), max_nfev=3000)
            resA[name].extend(te.val_loss.values - f(r.x, te.N_params_B.values, te.D_tokens_B.values,
                                                     te.Q_score.values))
        except Exception:
            resA[name].extend([np.nan] * len(te))
resB = {m: [] for m in MODELS}
for nv in sorted(b7.N_params_B.unique()):
    tr, te = b7[b7.N_params_B != nv], b7[b7.N_params_B == nv]
    for name, (f, k, p0, lo, hi) in MODELS.items():
        try:
            r = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values,
                                           tr.Q_score.values) - tr.val_loss.values,
                              p0, bounds=(lo, hi), max_nfev=3000)
            resB[name].extend(te.val_loss.values - f(r.x, te.N_params_B.values, te.D_tokens_B.values,
                                                     te.Q_score.values))
        except Exception:
            resB[name].extend([np.nan] * len(te))
print(f'{"模型":<24}{"全样本RMSE":>11}{"留一(N,D)组RMSE":>16}{"留一N RMSE":>13}{"留一N最大误差":>14}')
full_rmse = {}
for name, (f, k, p0, lo, hi) in MODELS.items():
    r = least_squares(lambda p: f(p, N_, D_, Q_) - L_, p0, bounds=(lo, hi), max_nfev=40000)
    full_rmse[name] = np.sqrt(np.mean(r.fun ** 2))
for name in MODELS:
    a = np.array(resA[name], float); b = np.array(resB[name], float)
    print(f'{name:<24}{full_rmse[name]:>11.5f}{np.sqrt(np.nanmean(a**2)):>16.5f}'
          f'{np.sqrt(np.nanmean(b**2)):>13.5f}{np.nanmax(np.abs(b)):>14.4f}')

print()
print('=' * 100)
print('B) 用真实数据锚定 alpha,beta 后，各思路的拟合与残差自由度')
print('=' * 100)

def anchored(kind):
    def model(p, N, D, Q):
        if kind == 'Dpow':
            A, B, d = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
            return E1v + A * N ** (-al1v) + B * D ** (-be1v) * Q ** (-d)
        if kind == 'NDpow':
            A, B, d1, d2 = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[3])
            return E1v + A * N ** (-al1v) * Q ** (-d1) + B * D ** (-be1v) * Q ** (-d2)
        if kind == 'Dexp':
            A, B, r = np.exp(p[0]), np.exp(p[1]), p[2]
            return E1v + A * N ** (-al1v) + B * D ** (-be1v) * np.exp(-r * Q)
        if kind == 'Elin':
            A, B, E1 = np.exp(p[0]), np.exp(p[1]), p[2]
            return E1v + E1 * (1 - Q) + A * N ** (-al1v) + B * D ** (-be1v)
        A, B, r1, r2 = np.exp(p[0]), np.exp(p[1]), p[2], p[3]
        return E1v + A * N ** (-al1v) * np.exp(-r1 * Q) + B * D ** (-be1v) * np.exp(-r2 * Q)
    return model

p0s = {'Dpow': [LOGE(1.2), LOGE(1.2), LOGE(0.2)], 'NDpow': [LOGE(1.2), LOGE(1.2), LOGE(0.14), LOGE(0.10)],
       'Dexp': [LOGE(1.2), LOGE(1.2), 1.0], 'Elin': [LOGE(1.2), LOGE(1.2), 0.3],
       'NDexp': [LOGE(1.2), LOGE(1.2), 1.5, 1.0]}
los = {'Dpow': [LOGE(1e-8), LOGE(1e-8), LOGE(1e-3)], 'NDpow': [LOGE(1e-8), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
       'Dexp': [LOGE(1e-8), LOGE(1e-8), -20], 'Elin': [LOGE(1e-8), LOGE(1e-8), -5],
       'NDexp': [LOGE(1e-8), LOGE(1e-8), -20, -20]}
his = {'Dpow': [LOGE(1e8), LOGE(1e8), LOGE(10)], 'NDpow': [LOGE(1e8), LOGE(1e8), LOGE(10), LOGE(10)],
       'Dexp': [LOGE(1e8), LOGE(1e8), 20], 'Elin': [LOGE(1e8), LOGE(1e8), 5],
       'NDexp': [LOGE(1e8), LOGE(1e8), 20, 20]}
labels = {'Dpow': 'P1 挂D·幂律', 'NDpow': 'P2 双挂·幂律', 'Dexp': 'P3 挂D·指数',
          'Elin': 'P4 改E·线性', 'NDexp': 'P5 双挂·指数'}
for kind in ['Dpow', 'Dexp', 'Elin', 'NDpow', 'NDexp']:
    m = anchored(kind)
    r = least_squares(lambda p: m(p, N_, D_, Q_) - L_, p0s[kind], bounds=(los[kind], his[kind]), max_nfev=40000)
    rss = np.sum(r.fun ** 2); k = len(p0s[kind])
    print(f'{labels[kind]:<16} k={k}  RMSE={np.sqrt(rss/n):.5f}  AIC={n*np.log(rss/n)+2*k:.2f}  '
          f'参数={np.round(np.concatenate([np.exp(r.x[:2]), r.x[2:] if kind in ("Dexp","Elin","NDexp") else np.exp(r.x[2:])]), 4)}')
print(f'  （锚定 E={E1v:.4f}, alpha={al1v:.4f}, beta={be1v:.4f} 后，只有质量部分参数自由）')

print()
print('=' * 100)
print('C) 振幅 A(N,D) = max_Q L - min_Q L 的 N / D 依赖（判定质量挂哪个通道）')
print('=' * 100)
piv = b7.groupby(['N_params_B', 'D_tokens_B']).val_loss.agg(lambda s: s.max() - s.min()).unstack()
print('振幅矩阵（行=N, 列=D）:')
print(piv.round(4).to_string())
Ns, Ds = np.array(piv.index, float), np.array(piv.columns, float)
print()
print('固定 D，振幅随 N 的 log-log 局部斜率（相邻 N 之间）:')
for j, d in enumerate(Ds):
    col = piv.iloc[:, j].values.astype(float)
    sl = np.diff(np.log(col)) / np.diff(np.log(Ns))
    print(f'  D={d:>7.0f}: ' + '  '.join(f'{s:+.3f}' for s in sl))
print('固定 N，振幅随 D 的 log-log 局部斜率:')
for i, nv in enumerate(Ns):
    row = piv.iloc[i, :].values.astype(float)
    sl = np.diff(np.log(row)) / np.diff(np.log(Ds))
    print(f'  N={nv:>9.3f}: ' + '  '.join(f'{s:+.3f}' for s in sl))
# 全局回归
X = []
y = []
for i, nv in enumerate(Ns):
    for j, d in enumerate(Ds):
        y.append(np.log(piv.iloc[i, j])); X.append([1.0, np.log(nv), np.log(d)])
X, y = np.array(X), np.array(y)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ coef
r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
print(f'\n全局 log A = {coef[0]:.4f} {coef[1]:+.4f} logN {coef[2]:+.4f} logD   (R²={r2:.4f})')
print(f'  实测：振幅随 N 衰减指数 = {coef[1]:+.4f}；随 D 衰减指数 = {coef[2]:+.4f}')
print(f'  纯挂D项模型预测：N 指数 = 0.0000, D 指数 = -beta（B1 锚点 {-be1v:.4f}）')
print(f'  双挂模型预测：N 指数介于 0 与 -alpha 之间（B1 锚点 -{al1v:.4f}），D 指数介于 0 与 -beta 之间')

print()
print('=' * 100)
print('D) 固定 (N,D) 组内，质量效应沿 N 的衰减 vs 标度指数 alpha')
print('=' * 100)
semi = []
for (nv, d), sub in b7.groupby(['N_params_B', 'D_tokens_B']):
    s = sub.sort_values('Q_score')
    q, l = s.Q_score.values, s.val_loss.values
    slope = np.polyfit(np.log(q), l, 1)[0]      # dL/dlnQ
    semi.append((nv, d, slope, l.max() - l.min()))
sd = pd.DataFrame(semi, columns=['N', 'D', 'dL_dlnQ', 'amp'])
# 在每个 D 上回归 log|dL/dlnQ| ~ logN
print(f'{"D":>8}{"corr(logN, log|dL/dlnQ|)":>28}{"斜率":>10}')
for d, sub in sd.groupby('D'):
    x = np.log(sub.N.values); yv = np.log(np.abs(sub.dL_dlnQ.values))
    c = np.corrcoef(x, yv)[0, 1]
    print(f'{d:>8.0f}{c:>28.3f}{np.polyfit(x, yv, 1)[0]:>10.3f}')
x = np.log(sd.N.values); yv = np.log(np.abs(sd.dL_dlnQ.values))
m = np.polyfit(x, yv, 1)
print(f'\n全局：dL/dlnQ ∝ N^{m[0]:.4f}  (R²={np.corrcoef(x, yv)[0,1]**2:.4f})')
print(f'  纯挂D项模型预测该指数 = 0；双挂·幂律预测 = -alpha = -{al1v:.4f}（B1锚点）')
sd.to_csv(r'd:\F题\q2_amp_by_group.csv', index=False)