# -*- coding: utf-8 -*-
"""复算 §1.3 的源偏移调整列：对每个来源加最优常数偏移 c_s = mean(L_obs - L_pred)"""
import os
import numpy as np
import pandas as pd

B = r'F题\real_attachments\B_scaling_laws'
E, A, al, Bc, be = 1.689797562932505, 0.3539803206546586, 0.3399765819042535, \
    1.240305583545938, 0.2798781285453637


def pred(N, D):
    return E + A * N ** -al + Bc * D ** -be


def report(name, N, D, L):
    L = np.asarray(L, float)
    N = np.asarray(N, float)
    D = np.asarray(D, float)
    P = pred(N, D)
    r = L - P
    rmse0 = np.sqrt((r ** 2).mean())
    bias_med = np.median(r)
    c = r.mean()
    r2 = L - (P + c)
    rmse1 = np.sqrt((r2 ** 2).mean())
    ss_tot = ((L - L.mean()) ** 2).sum()
    R2 = 1 - (r2 ** 2).sum() / ss_tot
    print(f'  {name:<28} n={len(L):>5}  零自由度RMSE={rmse0:.4f}  '
          f'偏差中位={bias_med:+.4f}  偏移c={c:+.4f}  加偏移RMSE={rmse1:.4f}  R2={R2:.3f}')


print('§1.3 源偏移表复算（B1 参数零自由度外推）')
b2 = pd.read_csv(os.path.join(B, 'cerebras_training_log.csv'))
report('B2 (半合成)', b2['N_params_B'], b2['D_tokens_B'], b2['val_loss'])

b4 = pd.read_csv(os.path.join(B, 'scaling_baseline.csv'))
print('B4 列:', list(b4.columns))
c4 = list(b4.columns)
report('B4 (12 模型族)', b4[c4[1]], b4[c4[2]], b4[c4[3]])

b5 = pd.read_csv(os.path.join(B, 'published_scaling_data.csv'))
print('B5 列:', list(b5.columns))
c5 = list(b5.columns)
report('B5 (6 篇文献)', b5[c5[1]], b5[c5[2]], b5[c5[3]])

b10 = pd.read_csv(os.path.join(B, 'supplementary_large_baseline.csv'))
print('B10 列:', list(b10.columns))
c10 = list(b10.columns)
report('B10 (估算)', b10[c10[1]], b10[c10[2]], b10[c10[3]])
