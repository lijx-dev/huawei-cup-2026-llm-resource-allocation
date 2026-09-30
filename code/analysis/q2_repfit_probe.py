# -*- coding: utf-8 -*-
"""核对：文档报告的 E=1.631681 与重拟合得到的 E=1.747333，哪一个是 B6∪B7 的最优？"""
import os
import numpy as np
import pandas as pd
from scipy import optimize

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
d6 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment.csv'))
d7 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment_expanded.csv'))
print('B6 shape', d6.shape, 'cols', list(d6.columns))
print('B7 shape', d7.shape, 'cols', list(d7.columns))
print('B6 N unique', sorted(d6.N_params_B.unique()))
print('B7 N unique', sorted(d7.N_params_B.unique()))
print('B6 D unique', sorted(d6.D_tokens_B.unique()))
print('B7 D unique', sorted(d7.D_tokens_B.unique()))
print('B6 Q unique', sorted(d6.Q_score.unique()))
print('B7 Q unique', sorted(d7.Q_score.unique()))

cat = pd.concat([d6, d7], ignore_index=True)
print('\n拼接后', cat.shape, '去重后', cat.drop_duplicates(subset=['N_params_B','D_tokens_B','Q_score']).shape)
print('完全重复行数:', cat.duplicated(subset=['N_params_B','D_tokens_B','Q_score']).sum())

df = cat.drop_duplicates(subset=['N_params_B','D_tokens_B','Q_score'])
N = df.N_params_B.values.astype(float); D = df.D_tokens_B.values.astype(float)
Q = df.Q_score.values.astype(float); L = df.val_loss.values.astype(float)


def model(th, N, D, Q):
    E, A, a, Bc, b, rN, rD, E1 = th
    return E + A*N**-a*np.exp(-rN*Q) + Bc*D**-b*np.exp(-rD*Q) - E1*Q


def rss(th):
    r = model(th, N, D, Q) - L
    return float(r @ r)


DOC = [1.631681, 0.639571, 0.282713, 1.426335, 0.299794, 0.349709, 0.130121, 0.115652]
print(f'\n文档参数 RSS = {rss(DOC):.10f}  RMSE={np.sqrt(rss(DOC)/len(L)):.6f}')

sol = optimize.least_squares(lambda th: model(th, N, D, Q) - L, DOC, method='lm',
                             xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
print(f'LM 重拟合 RSS = {rss(sol.x):.10f}  RMSE={np.sqrt(rss(sol.x)/len(L)):.6f}')
print('  LM 参数:', np.round(sol.x, 6))

sol2 = optimize.least_squares(lambda th: model(th, N, D, Q) - L, DOC, method='trf',
                              xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
print(f'TRF 重拟合 RSS = {rss(sol2.x):.10f}  RMSE={np.sqrt(rss(sol2.x)/len(L)):.6f}')
print('  TRF 参数:', np.round(sol2.x, 6))

# 文档口径：B6 校准、B7 新增验证（去重后）
print('\n若仅在 B6（360 点）上拟合：')
Nb, Db, Qb, Lb = d6.N_params_B.values.astype(float), d6.D_tokens_B.values.astype(float), d6.Q_score.values.astype(float), d6.val_loss.values.astype(float)
s6 = optimize.least_squares(lambda th: model(th, Nb, Db, Qb) - Lb, DOC, method='lm',
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
print('  B6-only 参数:', np.round(s6.x, 6), f' RSS={rss(s6.x):.10f}')
print('  B6-only 参数在 B6∪B7 上的 RSS =', round(float(((model(s6.x,N,D,Q)-L)**2).sum()), 10))
