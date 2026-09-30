# -*- coding: utf-8 -*-
"""分源定参检验：用 B1(真实) 定 alpha,beta，用 B6/B7(半合成) 只定质量效应
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))

N1 = b1['N_params_B'].values.astype(float); D1 = b1['D_tokens_B'].values.astype(float); L1 = b1['val_loss'].values.astype(float)
N7 = b7['N_params_B'].values.astype(float); D7 = b7['D_tokens_B'].values.astype(float)
Q7 = b7['Q_score'].values.astype(float); L7 = b7['val_loss'].values.astype(float)
N6 = b6['N_params_B'].values.astype(float); D6 = b6['D_tokens_B'].values.astype(float)
Q6 = b6['Q_score'].values.astype(float); L6 = b6['val_loss'].values.astype(float)

# ---------- 步骤1: B1 定经典标度律 ----------
def m_b1(p, N, D):
    E, A, a, B, b = p
    return E + A*N**(-a) + B*D**(-b)
r1 = least_squares(lambda p: m_b1(p, N1, D1) - L1, x0=[1.7, 0.35, 0.34, 1.24, 0.28],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3], [4,1e8,3,1e8,3]), max_nfev=100000)
E1, A1, a1, B1_, b1_ = r1.x
pred1 = m_b1(r1.x, N1, D1)
r2_1 = 1 - np.sum((L1-pred1)**2)/np.sum((L1-L1.mean())**2)
print('=' * 72)
print('步骤1: B1 定经典标度律')
print('=' * 72)
print(f'  E={E1:.4f}, A={A1:.4f}, alpha={a1:.4f}, B={B1_:.4f}, beta={b1_:.4f}')
print(f'  R²={r2_1:.6f}, RMSE={np.sqrt(np.mean((L1-pred1)**2)):.6f}')

# ---------- 步骤2: 固定 alpha,beta，用 B7 拟合质量部分 ----------
print('\n' + '=' * 72)
print('步骤2: 固定 alpha,beta（B1值），用 B7 拟合质量乘子')
print('=' * 72)

def fit_fixed_a(df_N, df_D, df_Q, df_L, tag):
    # 乘子: L = E + A*N^-a*h(Q) + B*D^-b,  a,b 固定
    def m_mul(p, N, D, Q):
        E, A, B, d = p
        return E + A*N**(-a1)*Q**(-d) + B*D**(-b1_)
    r = least_squares(lambda p: m_mul(p, df_N, df_D, df_Q) - df_L,
                      x0=[E1, A1, B1_, 0.15],
                      bounds=([0.1,1e-8,1e-8,1e-3], [4,1e8,1e8,10]), max_nfev=100000)
    pred = m_mul(r.x, df_N, df_D, df_Q)
    r2 = 1 - np.sum((df_L-pred)**2)/np.sum((df_L-df_L.mean())**2)
    rmse = np.sqrt(np.mean((df_L-pred)**2))
    print(f'  {tag} 乘子+幂律h（alpha,beta固定为B1值）:')
    print(f'    E={r.x[0]:.4f}, A={r.x[1]:.4f}, B={r.x[2]:.4f}, δ={r.x[3]:.4f}')
    print(f'    R²={r2:.5f}, RMSE={rmse:.5f}')
    return r, r2

r7f, r2_7f = fit_fixed_a(N7, D7, Q7, L7, 'B7')
r6f, r2_6f = fit_fixed_a(N6, D6, Q6, L6, 'B6')

# ---------- 步骤3: 对比（自由 alpha 的乘子形式） ----------
print('\n' + '=' * 72)
print('步骤3: 对比——自由 alpha 的乘子形式')
print('=' * 72)
def m_free(p, N, D, Q):
    E, A, a, B, b, d = p
    return E + A*N**(-a)*Q**(-d) + B*D**(-b)
rf = least_squares(lambda p: m_free(p, N7, D7, Q7) - L7, x0=[1.5, 0.6, 0.22, 1.33, 0.30, 0.17],
                   bounds=([0.1,1e-8,1e-3,1e-8,1e-3,1e-3], [4,1e8,3,1e8,3,10]), max_nfev=100000)
predf = m_free(rf.x, N7, D7, Q7)
r2_f = 1 - np.sum((L7-predf)**2)/np.sum((L7-L7.mean())**2)
print(f'  B7 自由alpha: alpha={rf.x[2]:.4f}, δ={rf.x[5]:.4f}, R²={r2_f:.5f}, RMSE={np.sqrt(np.mean((L7-predf)**2)):.5f}')
print(f'  B7 固定alpha: alpha={a1:.4f}(B1值), δ={r7f.x[3]:.4f}, R²={r2_7f:.5f}, RMSE={np.sqrt(np.mean((L7-m_free([r7f.x[0],r7f.x[1],a1,r7f.x[2],b1_,r7f.x[3]],N7,D7,Q7))**2)):.5f}')

# ---------- 步骤4: 固定alpha后，质量效应的N依赖是否仍被正确预测 ----------
print('\n' + '=' * 72)
print('步骤4: 关键检验——固定alpha=%.4f 后，模型预测的 dL/dQ 的N依赖' % a1)
print('=' * 72)
print(f'  模型预测: dL/dQ ∝ N^(-alpha) = N^(-{a1:.4f})')
print(f'  实测:     dL/dQ ∝ N^(-0.162)')
print(f'  → 偏差 = {abs(a1-0.162):.4f}')
print(f'\n  若改用广义形式（gamma 独立），gamma=0.134，偏差={abs(0.134-0.162):.4f}')

# ---------- 步骤5: 用固定alpha的模型做外推检验（B9/B10 大模型） ----------
print('\n' + '=' * 72)
print('步骤5: 外推检验（B10 大模型，Q=1 假设）')
print('=' * 72)
b10 = pd.read_csv(os.path.join(BASE, 'supplementary_large_baseline.csv'))
b10 = b10.dropna(subset=['N_params_B','D_tokens_B','val_loss'])
N10 = b10['N_params_B'].values.astype(float); D10 = b10['D_tokens_B'].values.astype(float); L10 = b10['val_loss'].values.astype(float)
# 用 B1 参数预测
pred10 = m_b1(r1.x, N10, D10)
mae = np.mean(np.abs(L10-pred10)); mape = np.mean(np.abs((L10-pred10)/L10))*100
print(f'  B1 参数直接外推到 B10: MAE={mae:.4f}, MAPE={mape:.2f}%')
from scipy.stats import spearmanr
print(f'  秩相关 = {spearmanr(L10, pred10).statistic:.4f}')
print(f'  B10 实际 loss 范围: {L10.min():.4f} ~ {L10.max():.4f}')
print(f'  预测 loss 范围: {pred10.min():.4f} ~ {pred10.max():.4f}')