# -*- coding: utf-8 -*-
"""决定性检验：乘子形式的联合拟合 vs 替代形式
用户已选定乘子形式，此处验证其自洽性并确定参数
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))

def prep(df):
    return (df['N_params_B'].values.astype(float),
            df['D_tokens_B'].values.astype(float),
            df['Q_score'].values.astype(float),
            df['val_loss'].values.astype(float))

N7, D7, Q7, L7 = prep(b7)
N6, D6, Q6, L6 = prep(b6)

def report(name, res, N, D, Q, L, npar):
    pred = res.fun + L
    r2 = 1 - np.sum((L-pred)**2)/np.sum((L-L.mean())**2)
    rmse = np.sqrt(np.mean((L-pred)**2))
    n = len(L)
    aic = n*np.log(np.sum((L-pred)**2)/n) + 2*npar
    print(f'  {name}')
    print(f'    参数: {np.round(res.x, 4)}')
    print(f'    R²={r2:.5f}, RMSE={rmse:.5f}, AIC={aic:.2f}')
    return r2, rmse, aic

print('=' * 72)
print('联合拟合（B7 全量 450 点）')
print('=' * 72)

# ---------- 形式1: 乘子 + 线性 h ----------
def m1(p, N, D, Q):
    E, A, a, B, b, d = p
    return E + A*N**(-a)*(1 - d*(1-Q)) + B*D**(-b)
r1 = least_squares(lambda p: m1(p, N7, D7, Q7) - L7,
                   x0=[1.5, 0.5, 0.3, 1.3, 0.3, 0.8],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,0.0], [4,1e8,3,1e8,3,1.0]),
                   max_nfev=60000)
r2_1, rm_1, aic_1 = report('形式1 乘子+线性h: L=E+A*N^-a*(1-δ(1-Q))+B*D^-b', r1, N7, D7, Q7, L7, 6)

# ---------- 形式2: 乘子 + 指数 h ----------
def m2(p, N, D, Q):
    E, A, a, B, b, lam = p
    return E + A*N**(-a)*np.exp(-lam*(1-Q)) + B*D**(-b)
r2_ = least_squares(lambda p: m2(p, N7, D7, Q7) - L7,
                    x0=[1.5, 0.5, 0.3, 1.3, 0.3, 2.0],
                    bounds=([0.1,1e-8,1e-3,1e-8,1e-3,1e-3], [4,1e8,3,1e8,3,20]),
                    max_nfev=60000)
r2_2, rm_2, aic_2 = report('形式2 乘子+指数h: L=E+A*N^-a*exp(-λ(1-Q))+B*D^-b', r2_, N7, D7, Q7, L7, 6)

# ---------- 形式3: 乘子 + 幂律 h ----------
def m3(p, N, D, Q):
    E, A, a, B, b, dd = p
    return E + A*N**(-a)*Q**(-dd) + B*D**(-b)
r3 = least_squares(lambda p: m3(p, N7, D7, Q7) - L7,
                   x0=[1.5, 0.5, 0.3, 1.3, 0.3, 0.5],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,1e-3], [4,1e8,3,1e8,3,10]),
                   max_nfev=60000)
r2_3, rm_3, aic_3 = report('形式3 乘子+幂律h: L=E+A*N^-a*Q^-δ+B*D^-b', r3, N7, D7, Q7, L7, 6)

# ---------- 形式4: 加性缺口（非乘子，作对照） ----------
def m4(p, N, D, Q):
    E, A, a, B, b, c, g = p
    return E + A*N**(-a) + B*D**(-b) - c*(1-Q)*N**(-g)
r4 = least_squares(lambda p: m4(p, N7, D7, Q7) - L7,
                   x0=[1.5, 0.5, 0.3, 1.3, 0.3, 0.3, 0.2],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,1e-8,1e-3], [4,1e8,3,1e8,3,1e8,3]),
                   max_nfev=60000)
r2_4, rm_4, aic_4 = report('形式4 加性缺口(对照): L=E+A*N^-a+B*D^-b-c*(1-Q)*N^-g', r4, N7, D7, Q7, L7, 7)

# ---------- 形式5: 纯可分离（对照） ----------
def m5(p, N, D, Q):
    E, A, a, B, b, c, g = p
    return E + A*N**(-a) + B*D**(-b) - c*(1-Q)
r5 = least_squares(lambda p: m5(p, N7, D7, Q7) - L7,
                   x0=[1.5, 0.5, 0.3, 1.3, 0.3, 0.3, 0.2],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,1e-8,1e-3], [4,1e8,3,1e8,3,1e8,3]),
                   max_nfev=60000)
r2_5, rm_5, aic_5 = report('形式5 可分离(对照): L=E+A*N^-a+B*D^-b-c*(1-Q)', r5, N7, D7, Q7, L7, 7)

print('\n' + '=' * 72)
print('核心比较：alpha 的估计值')
print('=' * 72)
print(f'  B1(Pythia, 无噪声): alpha = 0.3400')
print(f'  形式1 乘子+线性h:   alpha = {r1.x[2]:.4f}')
print(f'  形式2 乘子+指数h:   alpha = {r2_.x[2]:.4f}')
print(f'  形式3 乘子+幂律h:   alpha = {r3.x[2]:.4f}')
print(f'  形式4 加性缺口:     alpha = {r4.x[1]:.4f}, gamma = {r4.x[6]:.4f}')
print(f'  实测 dL/dQ ∝ N^-0.162')

print('\n' + '=' * 72)
print('h(Q) 形式对比（乘子族内部）')
print('=' * 72)
print(f'  线性 h=1-δ(1-Q):      δ={r1.x[5]:.4f}  → h(0)={1-r1.x[5]:.4f}, h(1)=1')
print(f'  指数 h=exp(-λ(1-Q)):  λ={r2_.x[5]:.4f}  → h(0)={np.exp(-r2_.x[5]):.4f}, h(1)=1')
print(f'  幂律 h=Q^-δ:          δ={r3.x[5]:.4f}  → h(0)=∞, h(1)=1')

# ---------- 退化性检验：Q=1 时是否回到经典形式 ----------
print('\n' + '=' * 72)
print('退化性检验（Q=1 时各形式预测 vs B1 经典标度律）')
print('=' * 72)
print(f'  B1 经典: L(N=1B,D=100B) = {1.6898 + 0.3540*1**(-0.34) + 1.2403*100**(-0.28):.4f}')
for nm, p, f in [('形式1', r1.x, lambda p: p[0]+p[1]*1**(-p[2])*(1-p[5]*0)+p[3]*100**(-p[4])),
                 ('形式2', r2_.x, lambda p: p[0]+p[1]*1**(-p[2])*np.exp(-p[5]*0)+p[3]*100**(-p[4])),
                 ('形式3', r3.x, lambda p: p[0]+p[1]*1**(-p[2])*1**(-p[5])+p[3]*100**(-p[4]))]:
    print(f'  {nm}: Q=1 时 L(N=1B,D=100B) = {f(p):.4f}')