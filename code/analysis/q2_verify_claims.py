# -*- coding: utf-8 -*-
"""核实文档中引用的若干数值：B2/B1 网格重叠、B4/B5 逐来源偏移、B1 重复性、Q_A 域级离散度"""
import os
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b2 = pd.read_csv(os.path.join(BASE, 'cerebras_training_log.csv'))
b4 = pd.read_csv(os.path.join(BASE, 'scaling_baseline.csv'))
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
print('列名 B2:', list(b2.columns))
print('列名 B4:', list(b4.columns))
print('列名 B5:', list(b5.columns))

# --- B1 基准参数 ---
def m0(p, N, D):
    E, A, a, B, b = p
    return E + A * N ** (-a) + B * D ** (-b)
from scipy.optimize import least_squares
r0 = least_squares(lambda p: m0(p, b1.N_params_B.values.astype(float), b1.D_tokens_B.values.astype(float))
                   - b1.val_loss.values.astype(float),
                   [1.7, .35, .34, 1.24, .28],
                   bounds=([.1, 1e-8, 1e-3, 1e-8, 1e-3], [4, 1e8, 3, 1e8, 3]), max_nfev=200000)
E0, A0, a0, B0, b0 = r0.x
print(f'\nB1 参数: E={E0:.4f} A={A0:.4f} a={a0:.4f} B={B0:.4f} b={b0:.4f}')

# --- B1 重复性 ---
d1 = b1.duplicated(subset=['N_params_B', 'D_tokens_B']).sum()
print(f'B1 重复 (N,D) 行数 = {d1}；唯一 (N,D) = {b1[["N_params_B","D_tokens_B"]].drop_duplicates().shape[0]}')
print(f'B1 N 取值 {b1.N_params_B.nunique()} 个, D 取值 {b1.D_tokens_B.nunique()} 个')

# --- B2 与 B1 网格重叠 ---
k1 = set(zip(np.round(b1.N_params_B, 6), np.round(b1.D_tokens_B, 6)))
k2 = set(zip(np.round(b2.N_params_B, 6), np.round(b2.D_tokens_B, 6)))
print(f'\nB2 n={len(b2)}, 唯一 (N,D)={len(k2)}, 与 B1 交集={len(k1 & k2)}')
print(f'B2 N 范围 [{b2.N_params_B.min()}, {b2.N_params_B.max()}], D 范围 [{b2.D_tokens_B.min()}, {b2.D_tokens_B.max()}]')

# --- B2 零自由度外推 ---
pred2 = m0(r0.x, b2.N_params_B.values.astype(float), b2.D_tokens_B.values.astype(float))
res2 = b2.val_loss.values.astype(float) - pred2
print(f'B2 零自由度: RMSE={np.sqrt(np.mean(res2**2)):.4f}, 偏差中位={np.median(res2):+.4f}, '
      f'偏差均值={res2.mean():+.4f}')
c2 = res2.mean()
print(f'B2 加偏移 c={c2:+.4f} 后 RMSE={np.sqrt(np.mean((res2-c2)**2)):.4f}, '
      f'R²={1-np.sum((res2-c2)**2)/np.sum((b2.val_loss.values.astype(float)-b2.val_loss.values.astype(float).mean())**2):.3f}')

# --- B4 逐来源偏移 ---
print('\n' + '=' * 70)
print('B4 逐模型族偏移')
print('=' * 70)
famcol = [c for c in b4.columns if 'family' in c.lower() or 'model' in c.lower() or 'name' in c.lower()]
print('候选族列:', famcol)
fc = famcol[0] if famcol else None
pred4 = m0(r0.x, b4.N_params_B.values.astype(float), b4.D_tokens_B.values.astype(float))
res4 = b4.val_loss.values.astype(float) - pred4
print(f'B4 n={len(b4)}, 零自由度 RMSE={np.sqrt(np.mean(res4**2)):.4f}, 偏差中位={np.median(res4):+.4f}, 偏差均值={res4.mean():+.4f}')
print(f'B4 逐记录偏差范围 [{res4.min():+.4f}, {res4.max():+.4f}]')
if fc:
    g = b4.assign(r=res4).groupby(fc).r.agg(['size', 'median'])
    print(f'按 [{fc}] 分组的偏移中位（{len(g)} 组）:')
    print(g.round(4).to_string())
    print(f'  → 逐族偏移中位范围 [{g["median"].min():+.4f}, {g["median"].max():+.4f}]')
c4 = res4.mean()
print(f'B4 加单一偏移后 RMSE={np.sqrt(np.mean((res4-c4)**2)):.4f}, R²={1-np.sum((res4-c4)**2)/np.sum((b4.val_loss.values.astype(float)-b4.val_loss.values.astype(float).mean())**2):.3f}')

# --- B5 逐文献偏移 ---
print('\n' + '=' * 70)
print('B5 逐文献偏移')
print('=' * 70)
pcol = [c for c in b5.columns if 'paper' in c.lower() or 'source' in c.lower() or 'ref' in c.lower()]
print('候选文献列:', pcol)
pred5 = m0(r0.x, b5.N_params_B.values.astype(float), b5.D_tokens_B.values.astype(float))
res5 = b5.val_loss.values.astype(float) - pred5
print(f'B5 n={len(b5)}, 零自由度 RMSE={np.sqrt(np.mean(res5**2)):.4f}, 偏差中位={np.median(res5):+.4f}, 偏差均值={res5.mean():+.4f}')
print(f'B5 逐记录偏差范围 [{res5.min():+.4f}, {res5.max():+.4f}]')
if pcol:
    pc = pcol[0]
    g5 = b5.assign(r=res5).groupby(pc).r.agg(['size', 'median'])
    print(f'按 [{pc}] 分组的偏移中位（{len(g5)} 组）:')
    print(g5.round(4).to_string())
    print(f'  → 逐文献偏移中位范围 [{g5["median"].min():+.4f}, {g5["median"].max():+.4f}]')
c5 = res5.mean()
print(f'B5 加单一偏移后 RMSE={np.sqrt(np.mean((res5-c5)**2)):.4f}, R²={1-np.sum((res5-c5)**2)/np.sum((b5.val_loss.values.astype(float)-b5.val_loss.values.astype(float).mean())**2):.3f}')

# --- Q_A 域级离散度 ---
print('\n' + '=' * 70)
print('Q_A 域级综合分的离散度')
print('=' * 70)
for cand in [r'd:\F题\F题\real_attachments\A_data_quality']:
    if os.path.isdir(cand):
        print('A 目录文件:', os.listdir(cand)[:12])
