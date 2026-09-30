# -*- coding: utf-8 -*-
"""输出振幅矩阵，用于可视化判定乘子位置"""
import pandas as pd
import numpy as np
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))

piv = b7.groupby(['N_params_B', 'D_tokens_B']).val_loss.agg(
    lambda s: s.max() - s.min()).unstack()
print('振幅矩阵 A(N,D) = max_Q L - min_Q L')
print(piv.round(4).to_string())
print()

lN = np.log(piv.index.values)
lD = np.log(piv.columns.values)

print('沿 N 方向（固定列，行间相对变化）：')
for j, d in enumerate(piv.columns):
    col = piv.iloc[:, j].values
    ratio = col[0] / col[-1]
    print(f'  D={d:>6}: A(N最小)/A(N最大) = {ratio:.2f}')

print('沿 D 方向（固定行，列间相对变化）：')
for i, n in enumerate(piv.index):
    row = piv.iloc[i, :].values
    ratio = row[0] / row[-1]
    print(f'  N={n:>6}: A(D最小)/A(D最大) = {ratio:.2f}')

print()
print('振幅整体量级：min=%.4f  max=%.4f  极差倍数=%.2f' % (piv.values.min(), piv.values.max(), piv.values.max() / piv.values.min()))

print()
print('=== BIC 补充计算 ===')
N_, D_, Q_, L_ = b7.N_params_B.values, b7.D_tokens_B.values, b7.Q_score.values, b7.val_loss.values
n = len(L_)
rmses = {'M0': 0.11824, 'M1': 0.06448, 'M2': 0.06954, 'M3': 0.06202}
ks = {'M0': 5, 'M1': 6, 'M2': 6, 'M3': 7}
print(f'{"模型":<6}{"RSS":>10}{"AIC":>11}{"BIC":>11}')
for m in ['M0', 'M1', 'M2', 'M3']:
    rss = n * rmses[m] ** 2
    aic = n * np.log(rss / n) + 2 * ks[m]
    bic = n * np.log(rss / n) + ks[m] * np.log(n)
    print(f'{m:<6}{rss:>10.4f}{aic:>11.2f}{bic:>11.2f}')
print('ΔAIC(M3 为基准): ', {m: round(n * np.log(n * rmses[m] ** 2 / n) + 2 * ks[m] - (n * np.log(n * rmses["M3"] ** 2 / n) + 2 * ks["M3"]), 2) for m in ['M0', 'M1', 'M2']})
print('ΔBIC(M3 为基准): ', {m: round(n * np.log(n * rmses[m] ** 2 / n) + ks[m] * np.log(n) - (n * np.log(n * rmses["M3"] ** 2 / n) + ks["M3"] * np.log(n)), 2) for m in ['M0', 'M1', 'M2']})