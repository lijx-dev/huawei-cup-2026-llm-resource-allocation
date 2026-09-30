# -*- coding: utf-8 -*-
"""B6 校准 → B7 新增点验证（题目规定的质量效应协议）+ 最终统一对比
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
LOGE = np.log
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
def keyf(df): return list(zip(df.N_params_B.round(4), df.D_tokens_B.round(4), df.Q_score.round(4)))
k6 = set(keyf(b6))
b7 = b7.copy()
b7['_new'] = [k not in k6 for k in keyf(b7)]
tr, te = b6, b7[b7._new]
print(f'B6 校准集 n={len(tr)}；B7 新增（不在B6中）n={len(te)}；B7∩B6 n={(~b7._new).sum()}')
print(f'新增点的 N: {sorted(te.N_params_B.unique())}')
print(f'新增点的 Q: {sorted(te.Q_score.unique())}')
print(f'新增点 D: {sorted(te.D_tokens_B.unique())}')

def mk(kind):
    def f(p, N, D, Q):
        if kind == 'P1 挂D·幂律(思路一)':
            return p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))*Q**(-np.exp(p[5]))
        if kind == 'P2 双挂·幂律(思路二)':
            return (p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*Q**(-np.exp(p[5]))
                    + np.exp(p[3])*D**(-np.exp(p[4]))*Q**(-np.exp(p[6])))
        if kind == 'P3 挂D·指数(思路三)':
            return p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[5]*Q)
        if kind == 'P4 改E·有界下限(思路四备选)':
            return p[0] + p[5]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))
        if kind == 'P5 双挂·指数':
            return (p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q)
                    + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q))
        return (p[0] + p[7]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q)
                + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q))
    return f

base5 = [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831)]
CFG = {
 'P1 挂D·幂律(思路一)':        (base5+[LOGE(0.10)],            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10)]),
 'P3 挂D·指数(思路三)':        (base5+[1.0],                  [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
 'P4 改E·有界下限(思路四备选)': (base5+[0.3],                  [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -5],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 5]),
 'P2 双挂·幂律(思路二)':       (base5+[LOGE(0.14), LOGE(0.10)], [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), LOGE(1e-3)],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), LOGE(10)]),
 'P5 双挂·指数':              (base5+[1.5, 1.0],              [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
 'P6 三项全挂·指数':           (base5+[1.5, 1.0, 0.3],        [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20, -5],
                                                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20, 5]),
}

print()
print('=' * 104)
print('B6 校准 → B7 新增点验证（真外推：新增点的 D∈[10,600] 但 Q 网格更密）')
print('=' * 104)
print(f'{"模型":<26}{"k":>3}{"B6拟合RMSE":>12}{"B7新增RMSE":>12}{"新增MAE":>11}{"新增最大误差":>13}{"新增偏差中位":>13}')
out = {}
for nm, (p0, lo, hi) in CFG.items():
    f = mk(nm)
    r = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values, tr.Q_score.values) - tr.val_loss.values,
                      p0, bounds=(lo, hi), max_nfev=60000)
    rss = np.sum(r.fun**2); k = len(p0)
    fit_rmse = np.sqrt(rss/len(tr))
    e = te.val_loss.values - f(r.x, te.N_params_B.values, te.D_tokens_B.values, te.Q_score.values)
    out[nm] = (r.x, fit_rmse, np.sqrt(np.mean(e**2)))
    print(f'{nm:<26}{k:>3}{fit_rmse:>12.5f}{np.sqrt(np.mean(e**2)):>12.5f}{np.mean(np.abs(e)):>11.5f}'
          f'{np.max(np.abs(e)):>13.4f}{np.median(e):>13.4f}')

print()
print('=' * 104)
print('全样本（B6∪B7=450点）最终对比')
print('=' * 104)
b_all = pd.concat([b6, b7[b7._new]])
Na, Da, Qa, La = (b_all.N_params_B.values.astype(float), b_all.D_tokens_B.values.astype(float),
                  b_all.Q_score.values.astype(float), b_all.val_loss.values.astype(float))
n = len(La)
print(f'{"模型":<26}{"k":>3}{"RMSE":>10}{"AIC":>10}{"BIC":>10}{"ΔAIC":>9}')
best = 1e18
rows = []
for nm, (p0, lo, hi) in CFG.items():
    f = mk(nm)
    r = least_squares(lambda p: f(p, Na, Da, Qa) - La, p0, bounds=(lo, hi), max_nfev=60000)
    rss = float(np.sum(r.fun**2)); k = len(p0)
    aic = n*np.log(rss/n)+2*k; bic = n*np.log(rss/n)+k*np.log(n)
    rows.append((nm, k, np.sqrt(rss/n), aic, bic, r.x))
    best = min(best, aic)
for nm, k, rmse, aic, bic, x in rows:
    print(f'{nm:<26}{k:>3}{rmse:>10.5f}{aic:>10.2f}{bic:>10.2f}{aic-best:>9.2f}')
print()
print('参数解读:')
for nm, k, rmse, aic, bic, x in rows:
    if nm.startswith('P1'):   s = f'beta={np.exp(x[4]):.4f}, delta_D={np.exp(x[5]):.4f}'
    elif nm.startswith('P3'): s = f'beta={np.exp(x[4]):.4f}, rho={x[5]:.4f}'
    elif nm.startswith('P4'): s = f'E1={x[5]:.4f}, beta={np.exp(x[4]):.4f}'
    elif nm.startswith('P2'): s = f'alpha={np.exp(x[2]):.4f}, delta_N={np.exp(x[5]):.4f}, delta_D={np.exp(x[6]):.4f}'
    elif nm.startswith('P5'): s = f'rho_N={x[5]:.4f}, rho_D={x[6]:.4f}'
    else:                     s = f'E1={x[7]:.4f}, rho_N={x[5]:.4f}, rho_D={x[6]:.4f}'
    print(f'  {nm:<26} {s}')

print()
print('=' * 104)
print('★ 思路三 vs 思路四 的结构性差异：质量效应在 D→∞ 时的极限')
print('=' * 104)
Dbig = np.array([1e3, 1e4, 1e5, 1e6])
Nfix = np.array([1.0]*4)
print(f'固定 N=1.0，Q=0.2 vs Q=1.0 的损失差 ΔL（质量效应强度）随 D 的变化:')
print(f'{"D":>10}' + ''.join(f'{nm.split()[0]:>12}' for nm, *_ in rows))
for i, d in enumerate(Dbig):
    line = f'{d:>10.0f}'
    for nm, k, rmse, aic, bic, x in rows:
        f = mk(nm)
        lo_ = f(x, Nfix[i:i+1], np.array([d]), np.array([0.2]))[0]
        hi_ = f(x, Nfix[i:i+1], np.array([d]), np.array([1.0]))[0]
        line += f'{lo_-hi_:>12.4f}'
    print(line)
print('  → 只挂D项/双挂（思路一、二、三）：ΔL 随 D 增大而衰减到 0（质量效应会消失）')
print('  → 含改E项（思路四备选、P6）：ΔL 收敛到非零常数（质量效应永久保留）')
print('  → 这是两种思路在数学上唯一的、可被数据判别的结构性差异')
print()
print('  B7 数据实际覆盖的 D 上限 = 600；实测振幅 D 指数 = -0.055（显著但很弱）')
print('  只挂D项预测 -0.280；只改E预测 0.000；实测 -0.055 更靠近 0')
print('  → 在观测区间内质量效应随 D 衰减极慢，与“几乎不随D衰减”更相容；')
print('    但因 D 只覆盖 10~600（不到两个数量级），无法把“衰减极慢”与“完全不衰减”分开')