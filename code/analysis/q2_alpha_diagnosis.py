# -*- coding: utf-8 -*-
"""诊断：B6/B7 自身的标度律指数，以及 alpha 是否随 Q 变化
核心问题：B1 的 alpha=0.34，而 B6 的 dL/dQ ∝ N^-0.164
若乘子形式 L=E+A*N^-a*h(Q)+B*D^-b 成立，应有 dL/dQ ∝ N^-a
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))

def fit_sl(N, D, L, x0=None):
    def model(p, N, D):
        E, A, a, B, b = p
        return E + A * N**(-a) + B * D**(-b)
    def resid(p):
        return model(p, N, D) - L
    best = None
    for x0_ in ([1.5, 1, 0.3, 1, 0.3], [1.5, 0.1, 0.2, 1, 0.3], [2.0, 5, 0.4, 5, 0.4]):
        try:
            r = least_squares(resid, x0=x0_,
                              bounds=([0.1, 1e-8, 1e-3, 1e-8, 1e-3], [4, 1e8, 3, 1e8, 3]),
                              max_nfev=40000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best

print('=' * 70)
print('诊断A: B7 在每个固定 Q 下拟合经典标度律（含E）')
print('=' * 70)
print(f'{"Q":>5s} {"E":>8s} {"A":>10s} {"alpha":>8s} {"B":>10s} {"beta":>8s} {"R2":>9s} {"RMSE":>8s}')
res_by_q = {}
for q in sorted(b7['Q_score'].unique()):
    sub = b7[b7['Q_score'] == q]
    N = sub['N_params_B'].values.astype(float)
    D = sub['D_tokens_B'].values.astype(float)
    L = sub['val_loss'].values.astype(float)
    r = fit_sl(N, D, L)
    E, A, a, B, b = r.x
    pred = E + A*N**(-a) + B*D**(-b)
    r2 = 1 - np.sum((L-pred)**2)/np.sum((L-L.mean())**2)
    rmse = np.sqrt(np.mean((L-pred)**2))
    res_by_q[q] = (E, A, a, B, b, r2)
    print(f'{q:5.2f} {E:8.4f} {A:10.4f} {a:8.4f} {B:10.4f} {b:8.4f} {r2:9.5f} {rmse:8.5f}')

print('\n  → 若 alpha 随 Q 变化，说明 Q 与 N 的耦合不是简单乘子')

# ---------- 诊断B: 提取 A*N^-a 项（质量乘子作用的项）随 Q 的变化 ----------
print('\n' + '=' * 70)
print('诊断B: 质量相关项 A(Q)*N^-alpha(Q) 的结构')
print('=' * 70)
qs = sorted(res_by_q.keys())
print(f'{"Q":>5s} {"A(Q)":>10s} {"alpha(Q)":>9s} {"A*N^-a @N=1":>14s} {"相对Q=1":>10s}')
A1, a1 = res_by_q[1.0][1], res_by_q[1.0][2]
for q in qs:
    E, A, a, B, b, r2 = res_by_q[q]
    ratio = A / A1
    print(f'{q:5.2f} {A:10.4f} {a:9.4f} {A:14.4f} {ratio:10.4f}')
print(f'\n  Q=1.0 时: A={A1:.4f}, alpha={a1:.4f}  ← 经典标度律（完美质量）')
print(f'  对比 B1(Pythia): alpha=0.3400, A=0.3540')

# ---------- 诊断C: 直接检验 dL/dQ 的 N 依赖 vs alpha ----------
print('\n' + '=' * 70)
print('诊断C: dL/dQ 的 N 依赖指数 vs 各 Q 下的 alpha')
print('=' * 70)
rows = []
for (n, d), g in b7.groupby(['N_params_B', 'D_tokens_B']):
    g = g.sort_values('Q_score')
    sl = np.polyfit(g['Q_score'], g['val_loss'], 1)
    rows.append({'N': n, 'D': d, 'dL_dQ': sl[0]})
s = pd.DataFrame(rows)
gN = s.groupby('N')['dL_dQ'].mean()
slope = np.polyfit(np.log(gN.index.values), np.log(-gN.values), 1)[0]
print(f'  dL/dQ ∝ N^({slope:.4f})  [实测]')
print(f'  B1 的 alpha = 0.3400')
print(f'  Q=1 时 B7 拟合的 alpha = {a1:.4f}')
print(f'  各 Q 下 alpha 范围: {min(res_by_q[q][2] for q in qs):.4f} ~ {max(res_by_q[q][2] for q in qs):.4f}')
print(f'\n  → {"一致" if abs(slope + a1) < 0.06 else "不一致，需要非简单乘子形式"}')

# ---------- 诊断D: 检验 dL/dQ ∝ N^-a 是否在控制 D 后仍成立 ----------
print('\n' + '=' * 70)
print('诊断D: 控制 D 后 dL/dQ 的 N 依赖（排除 D 混淆）')
print('=' * 70)
for d in sorted(b7['D_tokens_B'].unique()):
    sub = s[s['D'] == d].sort_values('N')
    if len(sub) < 4:
        continue
    sl = np.polyfit(np.log(sub['N']), np.log(-sub['dL_dQ']), 1)[0]
    print(f'  D={d:6.0f}: dL/dQ ∝ N^({sl:.4f})')

# ---------- 诊断E: B6 是否与 B7 一致 ----------
print('\n' + '=' * 70)
print('诊断E: B6 独立复核 dL/dQ 的 N 依赖')
print('=' * 70)
rows6 = []
for (n, d), g in b6.groupby(['N_params_B', 'D_tokens_B']):
    g = g.sort_values('Q_score')
    if len(g) < 4:
        continue
    sl = np.polyfit(g['Q_score'], g['val_loss'], 1)
    rows6.append({'N': n, 'dL_dQ': sl[0]})
s6 = pd.DataFrame(rows6)
gN6 = s6.groupby('N')['dL_dQ'].mean()
slope6 = np.polyfit(np.log(gN6.index.values), np.log(-gN6.values), 1)[0]
print(f'  B6: dL/dQ ∝ N^({slope6:.4f})')
print(f'  B7: dL/dQ ∝ N^({slope:.4f})')
print(f'  → B6/B7 一致性: {"好" if abs(slope6-slope) < 0.05 else "差"}')