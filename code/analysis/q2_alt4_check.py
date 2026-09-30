# -*- coding: utf-8 -*-
"""思路四备选方案检验（质量改损失下限）+ 质量尺度可比性 + 损失口径对比"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))

print('=' * 74)
print('0) 各来源损失口径与覆盖范围（检验"不可直接堆叠拟合"）')
print('=' * 74)
for tag, df in [('B1 Pythia', b1), ('B6 NQ', b6), ('B7 NQ-expanded', b7), ('B8 NQ-large', b8)]:
    print(f'  {tag:<16} N: {df.N_params_B.min():>7.2f}~{df.N_params_B.max():>7.2f}   '
          f'D: {df.D_tokens_B.min():>6.0f}~{df.D_tokens_B.max():>6.0f}   '
          f'loss: {df.val_loss.min():>7.4f}~{df.val_loss.max():>7.4f}')
print('  → 注：B1 的 loss 量级与 B6/B7/B8 是否同尺度，决定能否合并拟合')

N_, D_, Q_, L_ = b7.N_params_B.values, b7.D_tokens_B.values, b7.Q_score.values, b7.val_loss.values
n = len(L_)


def aic_of(rss, k):
    return n * np.log(rss / n) + 2 * k


print()
print('=' * 74)
print('1) 思路四"备选方案"：质量主要改变损失下限 E(Q)')
print('=' * 74)

# M4a: L = E0 + E1(1-Q) + A N^-a + B D^-b      (6 参数)
def m4a(p):
    E0, E1, A, a, B, b = p
    return E0 + E1 * (1 - Q_) + A * N_ ** (-a) + B * D_ ** (-b)


r = least_squares(lambda p: m4a(p) - L_, x0=[1.5, 0.3, 0.5, 0.26, 1.2, 0.25],
                  bounds=([0.2, -2, 1e-8, 1e-3, 1e-8, 1e-3], [4, 6, 1e8, 3, 1e8, 3]),
                  max_nfev=40000)
rss = np.sum(r.fun ** 2)
print(f'  M4a 线性 E(Q)=E0+E1(1-Q)   RMSE={np.sqrt(rss/n):.5f}  AIC={aic_of(rss,6):.2f}  '
      f'E0={r.x[0]:.4f}, E1={r.x[1]:.4f}, A={r.x[2]:.4f}, a={r.x[3]:.4f}, B={r.x[4]:.4f}, b={r.x[5]:.4f}')

# M4c: L = E0 + E1(1-Q)^g + A N^-a + B D^-b    (7 参数)
def m4c(p):
    E0, E1, g, A, a, B, b = p
    return E0 + E1 * (1 - Q_) ** g + A * N_ ** (-a) + B * D_ ** (-b)


r = least_squares(lambda p: m4c(p) - L_, x0=[1.5, 0.3, 1.0, 0.5, 0.26, 1.2, 0.25],
                  bounds=([0.2, -2, 1e-2, 1e-8, 1e-3, 1e-8, 1e-3], [4, 6, 10, 1e8, 3, 1e8, 3]),
                  max_nfev=40000)
rss = np.sum(r.fun ** 2)
print(f'  M4c 幂式 E(Q)=E0+E1(1-Q)^g RMSE={np.sqrt(rss/n):.5f}  AIC={aic_of(rss,7):.2f}  '
      f'E0={r.x[0]:.4f}, E1={r.x[1]:.4f}, g={r.x[2]:.4f}, a={r.x[3]:.4f}, b={r.x[4]:.4f}')

print('  对照（前次结果）:')
print('    M0 经典(无Q)          RMSE=0.11824  AIC=-1911.53')
print('    M1 质量挂N项          RMSE=0.06448  AIC=-2455.26')
print('    M2 质量挂D项(思路2/3/4) RMSE=0.06954  AIC=-2387.27')
print('    M3 双挂(N+D)          RMSE=0.06202  AIC=-2488.27')

print()
print('=' * 74)
print('2) 直接判定"质量是否只改 E"：振幅是否与 N、D 无关')
print('=' * 74)
rows = []
for (nn, dd), sub in b7.groupby(['N_params_B', 'D_tokens_B']):
    rows.append((nn, dd, sub.val_loss.max() - sub.val_loss.min()))
A = pd.DataFrame(rows, columns=['N', 'D', 'amp'])
y = np.log(A.amp.values)
lN, lD = np.log(A.N.values), np.log(A.D.values)
r2_const = 0.0
X = np.column_stack([np.ones(len(A)), lN, lD])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2_full = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f'  若质量只改 E，振幅 A=E1[(1-Qmin)^g-(1-Qmax)^g] 与 N、D 无关 → 解释力 R²=0')
print(f'  实测：ln A 对 (lnN, lnD) 回归 R² = {r2_full:.4f}')
print(f'  → 振幅有 {r2_full*100:.1f}% 的变异被 N、D 解释，强烈否定"只改 E"')

print()
print('=' * 74)
print('3) 质量尺度可比性：A 端综合分 vs B 端 Q_score')
print('=' * 74)
try:
    dq = pd.read_csv(r'd:\F题\q1_quality_results\domain_Q.csv')
    print('  A 端域级 Q（问题一产出）:')
    print(dq.to_string(index=False))
except Exception as e:
    print('  读取 domain_Q.csv 失败:', e)
try:
    sq = pd.read_csv(r'd:\F题\q1_quality_results\sample_Q.csv')
    num = sq.select_dtypes(include=[np.number])
    print()
    print('  A 端样本级 Q 分布:')
    print(num.describe().round(4).to_string())
except Exception as e:
    print('  读取 sample_Q.csv 失败:', e)
print()
print('  B 端 Q_score 分布:')
print(b7.Q_score.describe().round(4).to_string())
print(f'  B 端 Q 取值集合: {sorted(b7.Q_score.unique())}')
print('  → B 端为等间距设计网格；A 端为 22 指标合成的连续值，两者分布形态不同')