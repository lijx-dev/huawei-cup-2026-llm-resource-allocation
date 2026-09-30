# -*- coding: utf-8 -*-
"""乘子形式可行性验证
1) B1 非线性拟合 L=E+A*N^-a+B*D^-b  → 真实 alpha
2) B7 归一化质量曲线重合性检验（乘子形式的核心预测）
3) h(Q) 候选形式识别
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))

# ================= 1. B1 非线性拟合 =================
print('=' * 60)
print('1. B1 非线性拟合 L = E + A*N^-a + B*D^-b')
print('=' * 60)
N = b1['N_params_B'].values.astype(float)
D = b1['D_tokens_B'].values.astype(float)
L = b1['val_loss'].values.astype(float)

def model(p, N, D):
    E, A, a, B, b = p
    return E + A * N**(-a) + B * D**(-b)

def resid(p, N, D, L):
    return model(p, N, D) - L

best = None
for x0 in ([1.5, 50, 0.3, 50, 0.3], [1.8, 20, 0.15, 20, 0.15], [1.2, 100, 0.5, 100, 0.5]):
    try:
        r = least_squares(resid, x0=x0, args=(N, D, L),
                          bounds=([0.1, 1e-6, 1e-3, 1e-6, 1e-3], [4, 1e8, 3, 1e8, 3]),
                          max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    except Exception as e:
        print('  fail:', e)

E, A, a, B, b = best.x
pred = model(best.x, N, D)
r2 = 1 - np.sum((L - pred)**2) / np.sum((L - L.mean())**2)
rmse = np.sqrt(np.mean((L - pred)**2))
print(f'  非线性拟合: E={E:.4f}, A={A:.4f}, alpha={a:.4f}, B={B:.4f}, beta={b:.4f}')
print(f'  R²={r2:.5f}, RMSE={rmse:.4f}')
print(f'\n  对比线性log-log拟合: alpha≈0.0540, beta≈0.0620 (R²=0.9439)')
print(f'  → 非线性 alpha={a:.4f}, beta={b:.4f}')
print(f'  → alpha 提升 {a/0.054:.2f} 倍')

# 逐 N 固定看残差，判断 E 是否被合理估计
print(f'\n  不可约损失 E 估计值 = {E:.4f}')
print(f'  B1 中最小 val_loss = {L.min():.4f} (N={N[L.argmin()]:.3f}, D={D[L.argmin()]:.1f})')
print(f'  理论最小可能 L → E = {E:.4f}')

# ================= 2. 归一化质量曲线重合性 =================
print('\n' + '=' * 60)
print('2. B7 归一化质量曲线重合性检验（乘子形式核心预测）')
print('=' * 60)
# 乘子形式: L = E + A*N^-a*h(Q) + B*D^-b
#   → L(Q) - L(Qmax) = A*N^-a * (h(Q) - 1)
#   → 归一化后 (L(Q)-L(Qmax))/(L(Qmin)-L(Qmax)) = (h(Q)-1)/(h(Qmin)-1) 与 N,D 无关
QMAX = 1.0
curves = {}
for (n, d), g in b7.groupby(['N_params_B', 'D_tokens_B']):
    g = g.sort_values('Q_score')
    if QMAX not in set(g['Q_score']):
        continue
    Lmax = g.loc[g['Q_score'] == QMAX, 'val_loss'].values[0]
    Lmin = g.loc[g['Q_score'] == g['Q_score'].min(), 'val_loss'].values[0]
    if abs(Lmin - Lmax) < 1e-9:
        continue
    norm = (g['val_loss'].values - Lmax) / (Lmin - Lmax)
    curves[(n, d)] = (g['Q_score'].values, norm)

print(f'  可用 (N,D) 组数: {len(curves)}')
# 对齐到公共 Q 网格
qgrid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
mat = []
for k, (q, nm) in curves.items():
    if len(q) == len(qgrid) and np.allclose(q, qgrid):
        mat.append(nm)
mat = np.array(mat)
print(f'  对齐到完整 Q 网格的组数: {len(mat)}')
if len(mat) > 0:
    mean_c = mat.mean(axis=0)
    std_c = mat.std(axis=0)
    print(f'\n  Q     归一化均值   跨组标准差   变异系数')
    for i, q in enumerate(qgrid):
        cv = std_c[i] / abs(mean_c[i]) if abs(mean_c[i]) > 1e-9 else 0
        print(f'  {q:.1f}   {mean_c[i]:10.4f}   {std_c[i]:10.4f}   {cv:8.4f}')
    print(f'\n  跨组标准差均值 = {std_c.mean():.4f}')
    print(f'  → {"重合性好，支持乘子形式" if std_c.mean() < 0.08 else "重合性一般，需谨慎"}')

# 对比：若为可分离形式 L=f(N,D)+g(Q)，则 g(Q)=L(Q)-L(Qmax) 本身应跨组重合
print('\n  [对照] 可分离假设下，原始差值 L(Q)-L(Qmax) 的跨组变异:')
raw = []
for k, (q, nm) in curves.items():
    if len(q) == len(qgrid) and np.allclose(q, qgrid):
        g = b7[(b7['N_params_B'] == k[0]) & (b7['D_tokens_B'] == k[1])].sort_values('Q_score')
        Lmax = g.loc[g['Q_score'] == QMAX, 'val_loss'].values[0]
        raw.append(g['val_loss'].values - Lmax)
raw = np.array(raw)
print(f'  Q=0.1 处原始差值: mean={raw[:,0].mean():.4f}, std={raw[:,0].std():.4f}, CV={raw[:,0].std()/abs(raw[:,0].mean()):.4f}')
print(f'  → CV 越小越支持可分离；越大越支持乘子（N,D 依赖）')

# ================= 3. h(Q) 形式识别 =================
print('\n' + '=' * 60)
print('3. h(Q) 候选形式拟合（用归一化均值曲线）')
print('=' * 60)
if len(mat) > 0:
    y = mean_c  # (h(Q)-1)/(h(Qmin)-1)
    qq = qgrid

    def fit_form(name, func, p0, bounds):
        def r(p):
            return func(p, qq) - y
        try:
            res = least_squares(r, x0=p0, bounds=bounds, max_nfev=20000)
            pr = func(res.x, qq)
            r2f = 1 - np.sum((y - pr)**2) / np.sum((y - y.mean())**2)
            rmse_f = np.sqrt(np.mean((y - pr)**2))
            print(f'  {name:28s}: 参数={np.round(res.x,4)}, R²={r2f:.5f}, RMSE={rmse_f:.5f}')
            return r2f, res.x
        except Exception as e:
            print(f'  {name:28s}: fail {e}')
            return -9, None

    # 幂律 h=Q^-d
    fit_form('幂律 h(Q)=Q^-δ', lambda p, q: (q**(-p[0]) - 1) / (qq.min()**(-p[0]) - 1),
             [0.5], ([0.01], [10]))
    # 指数 h=exp(-λ(1-Q))
    fit_form('指数 h(Q)=exp(-λ(1-Q))', lambda p, q: (np.exp(-p[0]*(1-q)) - 1) / (np.exp(-p[0]*(1-qq.min())) - 1),
             [2.0], ([0.01], [20]))
    # 有理 h=1/(1+c(1-Q))
    fit_form('有理 h(Q)=1/(1+c(1-Q))', lambda p, q: (1/(1+p[0]*(1-q)) - 1) / (1/(1+p[0]*(1-qq.min())) - 1),
             [1.0], ([0.01], [50]))
    # 幂律(1-Q) h=(1-δ(1-Q))^k
    fit_form('线性 h(Q)=1-δ(1-Q)', lambda p, q: (p[0]*(q-1)) / (p[0]*(qq.min()-1)),
             [0.8], ([0.01], [1.0]))