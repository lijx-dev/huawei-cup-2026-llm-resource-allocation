# -*- coding: utf-8 -*-
"""决定性检验：乘子形式要求 alpha(主效应) == gamma(质量效应的N依赖)
若不等，则单一alpha的乘子形式不自洽
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
N7 = b7['N_params_B'].values.astype(float)
D7 = b7['D_tokens_B'].values.astype(float)
Q7 = b7['Q_score'].values.astype(float)
L7 = b7['val_loss'].values.astype(float)

def fit_and_report(name, func, x0, bounds, npar):
    try:
        r = least_squares(lambda p: func(p, N7, D7, Q7) - L7, x0=x0, bounds=bounds, max_nfev=100000)
        pred = func(r.x, N7, D7, Q7)
        r2 = 1 - np.sum((L7-pred)**2)/np.sum((L7-L7.mean())**2)
        rmse = np.sqrt(np.mean((L7-pred)**2))
        n = len(L7)
        aic = n*np.log(np.sum((L7-pred)**2)/n) + 2*npar
        print(f'  {name}')
        print(f'    参数: {np.round(r.x, 4)}')
        print(f'    R²={r2:.5f}, RMSE={rmse:.5f}, AIC={aic:.2f}')
        return r, r2, rmse, aic
    except Exception as e:
        print(f'  {name}: FAIL {e}')
        return None, -9, -9, 9e9

print('=' * 74)
print('检验1: 乘子形式（alpha == gamma 约束）')
print('=' * 74)
# L = E + A*N^-a*Q^-d + B*D^-b   （乘子：质量项与N项共用指数a）
r_mul, r2_mul, rm_mul, aic_mul = fit_and_report(
    '乘子: L=E+A*N^-a*Q^-δ+B*D^-b',
    lambda p, N, D, Q: p[0] + p[1]*N**(-p[2])*Q**(-p[5]) + p[3]*D**(-p[4]),
    [1.5, 0.6, 0.22, 1.33, 0.30, 0.17],
    ([0.1,1e-8,1e-3,1e-8,1e-3,1e-3], [4,1e8,3,1e8,3,10]), 6)

print('\n' + '=' * 74)
print('检验2: 广义乘子（alpha 与 gamma 独立）')
print('=' * 74)
# L = E + A*N^-a + B*D^-b + c*N^-g*Q^-d  （质量项N指数独立）
r_gen, r2_gen, rm_gen, aic_gen = fit_and_report(
    '广义: L=E+A*N^-a+B*D^-b+c*N^-g*Q^-δ',
    lambda p, N, D, Q: p[0] + p[1]*N**(-p[2]) + p[3]*D**(-p[4]) + p[5]*N**(-p[6])*Q**(-p[7]),
    [1.5, 0.5, 0.29, 1.33, 0.30, 0.1, 0.16, 0.17],
    ([0.1,1e-8,1e-3,1e-8,1e-3,-1e8,1e-3,1e-3], [4,1e8,3,1e8,3,1e8,3,10]), 8)

print('\n' + '=' * 74)
print('检验3: 质量项同时乘 N 项与 D 项')
print('=' * 74)
r_both, r2_both, rm_both, aic_both = fit_and_report(
    '双乘子: L=E+A*N^-a*Q^-δ+B*D^-b*Q^-ε',
    lambda p, N, D, Q: p[0] + p[1]*N**(-p[2])*Q**(-p[5]) + p[3]*D**(-p[4])*Q**(-p[6]),
    [1.5, 0.6, 0.22, 1.33, 0.30, 0.17, 0.01],
    ([0.1,1e-8,1e-3,1e-8,1e-3,1e-3,1e-3], [4,1e8,3,1e8,3,10,10]), 7)

print('\n' + '=' * 74)
print('核心结论对比')
print('=' * 74)
print(f'  实测 dL/dQ ∝ N^(-0.162)   [独立于拟合的测量]')
print(f'  B1 经典标度律 alpha = 0.3400')
print(f'  固定Q逐组拟合 alpha 均值 ≈ 0.29')
print()
if r_mul is not None:
    print(f'  乘子形式:     alpha(=gamma) = {r_mul.x[2]:.4f}  ← 被迫折中')
if r_gen is not None:
    print(f'  广义形式:     alpha(主效应) = {r_gen.x[2]:.4f}, gamma(质量项N指数) = {r_gen.x[6]:.4f}')
    print(f'    → alpha 与 gamma 差异 = {abs(r_gen.x[2]-r_gen.x[6]):.4f}')
if r_both is not None:
    print(f'  双乘子形式:   alpha = {r_both.x[2]:.4f}, delta = {r_both.x[5]:.4f}, epsilon = {r_both.x[6]:.4f}')

print('\n  AIC 比较（越小越好）:')
print(f'    乘子(6参):    AIC={aic_mul:.2f}')
print(f'    广义(8参):    AIC={aic_gen:.2f}')
print(f'    双乘子(7参):  AIC={aic_both:.2f}')

# ---------- 稳定性检验：B6 上重复广义拟合 ----------
print('\n' + '=' * 74)
print('稳健性: 在 B6（独立子集）上重复广义拟合')
print('=' * 74)
N6 = b6['N_params_B'].values.astype(float)
D6 = b6['D_tokens_B'].values.astype(float)
Q6 = b6['Q_score'].values.astype(float)
L6 = b6['val_loss'].values.astype(float)
def f6(p):
    return p[0] + p[1]*N6**(-p[2]) + p[3]*D6**(-p[4]) + p[5]*N6**(-p[6])*Q6**(-p[7]) - L6
r6 = least_squares(f6, x0=[1.5, 0.5, 0.29, 1.33, 0.30, 0.1, 0.16, 0.17],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,-1e8,1e-3,1e-3], [4,1e8,3,1e8,3,1e8,3,10]),
                   max_nfev=100000)
pred6 = r6.fun + L6
r2_6 = 1 - np.sum((L6-pred6)**2)/np.sum((L6-L6.mean())**2)
print(f'  B6 广义拟合: alpha={r6.x[2]:.4f}, gamma={r6.x[6]:.4f}, δ={r6.x[7]:.4f}, R²={r2_6:.5f}')
if r_gen is not None:
    print(f'  B7 广义拟合: alpha={r_gen.x[2]:.4f}, gamma={r_gen.x[6]:.4f}, δ={r_gen.x[7]:.4f}, R²={r2_gen:.5f}')
    print(f'  → B6/B7 一致性: alpha差={abs(r6.x[2]-r_gen.x[2]):.4f}, gamma差={abs(r6.x[6]-r_gen.x[6]):.4f}')