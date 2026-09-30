# -*- coding: utf-8 -*-
"""问题二数学分析：Q 效应的规模依赖性 + 替代关系 + B1标度律基础"""
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))

# ---------- 1. B1 经典标度律：对数线性拟合基础 ----------
print('===== B1: 经典标度律数据特征 =====')
# L = E + A*N^-alpha + B*D^-beta  → 先看纯幂律部分
d = b1.dropna(subset=['val_loss', 'N_params_B', 'D_tokens_B']).copy()
d['logN'] = np.log(d['N_params_B']); d['logD'] = np.log(d['D_tokens_B'])
d['logL'] = np.log(d['val_loss'])
# 多元线性回归 logL ~ logN + logD（不含E，作为一阶近似）
X = np.column_stack([np.ones(len(d)), d['logN'], d['logD']])
beta, *_ = np.linalg.lstsq(X, d['logL'].values, rcond=None)
pred = X @ beta
r2 = 1 - np.sum((d['logL'] - pred)**2) / np.sum((d['logL'] - d['logL'].mean())**2)
print(f'logL ~ logN + logD 拟合: 系数={beta.round(4)}, R²={r2:.4f}')
print(f'  → 隐含指数 alpha≈{-beta[1]:.4f}, beta≈{-beta[2]:.4f}')

# 固定 N 时，D 的幂律指数（更干净的估计）
print('\n固定 N 下 logL vs logD 的斜率（-beta 估计）:')
for n in sorted(d['N_params_B'].unique())[:8]:
    sub = d[d['N_params_B'] == n].sort_values('D_tokens_B')
    if len(sub) < 10:
        continue
    sl = np.polyfit(np.log(sub['D_tokens_B']), np.log(sub['val_loss']), 1)
    print(f'  N={n:8.4f}: slope={sl[0]:.4f}  R²={1 - np.sum((np.log(sub["val_loss"]) - np.polyval(sl, np.log(sub["D_tokens_B"])))**2)/np.sum((np.log(sub["val_loss"]) - np.log(sub["val_loss"]).mean())**2):.4f}')

# ---------- 2. B6: Q 效应的规模依赖性（核心） ----------
print('\n===== B6: dloss/dQ 是否依赖 N, D（质量与规模的交互） =====')
rows = []
for (n, dd), g in b6.groupby(['N_params_B', 'D_tokens_B']):
    g = g.sort_values('Q_score')
    if len(g) < 4:
        continue
    sl = np.polyfit(g['Q_score'], g['val_loss'], 1)  # 线性斜率 dL/dQ
    rows.append({'N': n, 'D': dd, 'dL_dQ': sl[0], 'loss_at_Q1': g['val_loss'].iloc[-1],
                 'loss_range': g['val_loss'].max() - g['val_loss'].min()})
sl_df = pd.DataFrame(rows)
print(f'dL/dQ 统计: mean={sl_df["dL_dQ"].mean():.4f}, std={sl_df["dL_dQ"].std():.4f}, '
      f'范围=[{sl_df["dL_dQ"].min():.4f}, {sl_df["dL_dQ"].max():.4f}]')
print(f'dL/dQ 全为负: {(sl_df["dL_dQ"] < 0).all()}')
# dL/dQ 与 N, D 的相关性
print(f'corr(dL_dQ, logN) = {np.corrcoef(sl_df["dL_dQ"], np.log(sl_df["N"]) )[0,1]:.4f}')
print(f'corr(dL_dQ, logD) = {np.corrcoef(sl_df["dL_dQ"], np.log(sl_df["D"]) )[0,1]:.4f}')
print(f'corr(dL_dQ, loss_at_Q1) = {np.corrcoef(sl_df["dL_dQ"], sl_df["loss_at_Q1"])[0,1]:.4f}')
print('\n按 N 分组的平均 dL/dQ:')
print(sl_df.groupby('N')['dL_dQ'].mean().round(4).to_string())

# ---------- 3. 等效替代：Q 提升 0.1 等价于 N 增加多少 ----------
print('\n===== 等效替代分析（B6, 固定 D） =====')
# 在同一 (N,D) 上，Q 提升 dQ 的 loss 变化；在同一 (D,Q) 上，N 变化的 loss 变化
# 先估计 dL/dlogN
rows2 = []
for (dd, q), g in b6.groupby(['D_tokens_B', 'Q_score']):
    g = g.sort_values('N_params_B')
    if len(g) < 4:
        continue
    sl = np.polyfit(np.log(g['N_params_B']), g['val_loss'], 1)  # dL/dlogN
    rows2.append({'D': dd, 'Q': q, 'dL_dlogN': sl[0]})
sl2 = pd.DataFrame(rows2)
print(f'dL/dlogN 统计: mean={sl2["dL_dlogN"].mean():.4f}, std={sl2["dL_dlogN"].std():.4f}')
print('按 Q 分组的平均 dL/dlogN（看质量是否影响规模效应）:')
print(sl2.groupby('Q')['dL_dlogN'].mean().round(4).to_string())

# 替代比：dQ=0.1 的等效 logN 变化 = (dL/dQ * 0.1) / (dL/dlogN)
print('\n替代比 R = (dL/dQ * 0.1) / (dL/dlogN)  → 质量提升0.1 等效于 N 增加 e^R 倍')
for q in [0.1, 0.3, 0.5, 0.8, 1.0]:
    dldq = sl_df[sl_df['N'].between(0.5, 1.5)]['dL_dQ'].mean()
    dldn = sl2[sl2['Q'] == q]['dL_dlogN'].mean()
    if not np.isnan(dldn) and dldn != 0:
        R = (dldq * 0.1) / dldn
        print(f'  Q={q}: dL/dQ均值(小N)={dldq:.4f}, dL/dlogN={dldn:.4f} → 等效 logN 变化={R:.4f}, N 倍数={np.exp(R):.4f}')

# ---------- 4. B8 方向问题的定量确认 ----------
print('\n===== B8 方向问题：与 B6 的逐点对比 =====')
m6 = b6.set_index(['N_params_B', 'D_tokens_B', 'Q_score'])['val_loss']
m8 = b8.set_index(['N_params_B', 'D_tokens_B', 'Q_score'])['val_loss']
common = sorted(set(m6.index) & set(m8.index))
print(f'共同点 {len(common)} 个')
# 在共同点上，计算 B6 与 B8 的秩相关（是否同向）
v6 = np.array([m6[k] for k in common]); v8 = np.array([m8[k] for k in common])
print(f'B6 与 B8 在这些点上的相关系数: {np.corrcoef(v6, v8)[0,1]:.4f}')
# 检验假设：B8 的 loss 是否等于 (常数 - B6的loss)?
diff = v8 - v6
print(f'B8 - B6 的差: mean={diff.mean():.4f}, std={diff.std():.4f}')
# 检验假设：B8 的 Q 是反向的。用 Q'=1-Q 重排后 B8 与 B6 的相关
q_rev = [(k[0], k[1], 1 - k[2]) for k in common]
m8_rev = b8.copy(); m8_rev['Q_score'] = 1 - m8_rev['Q_score']
m8r = m8_rev.set_index(['N_params_B', 'D_tokens_B', 'Q_score'])['val_loss']
v8r = np.array([m8r[k] for k in q_rev if k in m8r.index])
v6c = np.array([m6[(k[0], k[1], 1-k[2])] for k in q_rev if k in m8r.index])
print(f'假设Q反向: 用 Q\'=1-Q 对齐后, B6 与 B8 相关系数 = {np.corrcoef(v6c, v8r)[0,1]:.4f} (n={len(v8r)})')
# 检验假设：B8 loss 与 Q 的关系是否可被 "B8是excess loss" 解释
print(f'\nB8 loss 下界: {b8["val_loss"].min():.4f} (出现次数 {(b8["val_loss"] == b8["val_loss"].min()).sum()})')
print(f'B8 loss 在 Q=1.0 时的均值: {b8[b8["Q_score"]==1.0]["val_loss"].mean():.4f}')
print(f'B8 loss 在 Q=0.05 时的均值: {b8[b8["Q_score"]==0.05]["val_loss"].mean():.4f}')