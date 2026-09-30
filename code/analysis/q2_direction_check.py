# -*- coding: utf-8 -*-
"""问题二关键问题诊断：B6/B7/B8 质量方向矛盾"""
import pandas as pd
import numpy as np
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))

# ---------- 1. B6 与 B8 在相同 (N,D,Q) 点的对比 ----------
print('===== B6 与 B8 在共同点上的对比 =====')
k6 = set(zip(b6['N_params_B'], b6['D_tokens_B'], b6['Q_score']))
k8 = set(zip(b8['N_params_B'], b8['D_tokens_B'], b8['Q_score']))
common = k6 & k8
print(f'B6 点数={len(k6)}, B8 点数={len(k8)}, 共同点={len(common)}')
if common:
    m6 = b6.set_index(['N_params_B', 'D_tokens_B', 'Q_score'])['val_loss']
    m8 = b8.set_index(['N_params_B', 'D_tokens_B', 'Q_score'])['val_loss']
    rows = []
    for k in sorted(common)[:15]:
        rows.append({'N': k[0], 'D': k[1], 'Q': k[2], 'B6_loss': m6[k], 'B8_loss': m8[k]})
    print(pd.DataFrame(rows).to_string(index=False))

# ---------- 2. B8 内部：按 data_type 分组看 Q-loss 关系 ----------
print('\n===== B8 按 data_type 分组 =====')
for dt in b8['data_type'].unique():
    sub = b8[b8['data_type'] == dt]
    r = np.corrcoef(sub['Q_score'], sub['val_loss'])[0, 1]
    print(f'--- {dt}: n={len(sub)}, corr(Q,loss)={r:.4f} ---')
    print(f'  Q范围: {sub["Q_score"].min()}~{sub["Q_score"].max()}')
    print(f'  N范围: {sub["N_params_B"].min()}~{sub["N_params_B"].max()}')
    print(f'  D范围: {sub["D_tokens_B"].min()}~{sub["D_tokens_B"].max()}')
    print(f'  loss范围: {sub["val_loss"].min():.4f}~{sub["val_loss"].max():.4f}')

# ---------- 3. B8 中固定 (N,D)，Q 与 loss 的单调性逐组检查 ----------
print('\n===== B8 固定(N,D)下 Q-loss 单调性（抽样20组） =====')
groups = b8.groupby(['N_params_B', 'D_tokens_B'])
inc = dec = 0
sample_rows = []
for (n, d), g in groups:
    g = g.sort_values('Q_score')
    if len(g) < 3:
        continue
    # Spearman 单调性
    from scipy.stats import spearmanr
    rho = spearmanr(g['Q_score'], g['val_loss']).statistic
    if rho > 0.5:
        inc += 1
    elif rho < -0.5:
        dec += 1
    if len(sample_rows) < 20:
        sample_rows.append({'N': n, 'D': d, 'n_Q': len(g), 'rho(Q,loss)': round(rho, 3),
                            'dt': g['data_type'].iloc[0]})
print(f'共 {len(groups)} 组(N,D)')
print(f'  强正单调(rho>0.5): {inc} 组')
print(f'  强负单调(rho<-0.5): {dec} 组')
print(pd.DataFrame(sample_rows).to_string(index=False))

# ---------- 4. B8 loss 与 N,D 的关系（是否合理） ----------
print('\n===== B8: loss 与 N, D 的关系（固定 Q） =====')
for q in [0.1, 0.5, 1.0]:
    sub = b8[b8['Q_score'] == q]
    if len(sub) < 3:
        continue
    rn = np.corrcoef(np.log(sub['N_params_B']), sub['val_loss'])[0, 1]
    rd = np.corrcoef(np.log(sub['D_tokens_B']), sub['val_loss'])[0, 1]
    print(f'Q={q}: n={len(sub)}, corr(logN,loss)={rn:.3f}, corr(logD,loss)={rd:.3f}, loss均值={sub["val_loss"].mean():.3f}')

# ---------- 5. B6 同样检查 ----------
print('\n===== B6: loss 与 N, D 的关系（固定 Q） =====')
for q in [0.1, 0.5, 1.0]:
    sub = b6[b6['Q_score'] == q]
    if len(sub) < 3:
        continue
    rn = np.corrcoef(np.log(sub['N_params_B']), sub['val_loss'])[0, 1]
    rd = np.corrcoef(np.log(sub['D_tokens_B']), sub['val_loss'])[0, 1]
    print(f'Q={q}: n={len(sub)}, corr(logN,loss)={rn:.3f}, corr(logD,loss)={rd:.3f}, loss均值={sub["val_loss"].mean():.3f}')