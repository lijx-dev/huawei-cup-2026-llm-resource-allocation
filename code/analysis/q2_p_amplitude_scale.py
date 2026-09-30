# -*- coding: utf-8 -*-
"""配比效应幅度是否随规模变化？
A4–A15 的每个规模切片上，用 p 预测 13 个目标 loss，取预测极差作为"配比效应幅度"，
再看幅度随规模（以 loss 水平为代理）如何变化。
若幅度随规模系统性衰减 → 配比效应与 N/D 有交互，乘子（可分离）形式不成立。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats


def ridge_fit(X, y, lam=1e-3):
    """带截距的岭回归，闭式解"""
    Xc = np.column_stack([np.ones(len(X)), X])
    k = Xc.shape[1]
    R = np.eye(k) * lam
    R[0, 0] = 0.0
    beta = np.linalg.solve(Xc.T @ Xc + R, Xc.T @ y)
    return Xc @ beta

A = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'

SLICES = [('1m(train)', 'train_mixture_1m.csv', 'train_pile_loss_1m.csv'),
          ('1m(test)', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv'),
          ('60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
          ('1B', 'test_mixture_1B.csv', 'test_pile_loss_1B.csv'),
          ('10b(est)', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
          ('70b(est)', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]

rows = []
for tag, mf, lf in SLICES:
    X = pd.read_csv(os.path.join(A, mf)).iloc[:, 1:].values.astype(float)
    Y = pd.read_csv(os.path.join(A, lf)).iloc[:, 1:].values.astype(float)
    targets = list(pd.read_csv(os.path.join(A, lf)).columns[1:])
    n = len(X)
    for j, t in enumerate(targets):
        y = Y[:, j]
        pred = ridge_fit(X, y)
        amp = pred.max() - pred.min()
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({'slice': tag, 'target': t.replace('metric/the_pile_', '').replace('_val_loss', ''),
                     'n': n, 'mean_loss': y.mean(), 'amp': amp, 'rel_amp': amp / y.mean(), 'r2': r2})

df = pd.DataFrame(rows)
df.to_csv(r'd:\F题\q2_p_amplitude_by_slice.csv', index=False, encoding='utf-8-sig')

print('=' * 96)
print('各规模切片上的配比效应幅度（Ridge 预测极差）')
print('=' * 96)
summ = df.groupby('slice').agg(n=('n', 'first'), mean_loss=('mean_loss', 'mean'),
                               amp_median=('amp', 'median'), rel_amp_median=('rel_amp', 'median'),
                               r2_median=('r2', 'median')).reindex([s[0] for s in SLICES])
print(summ.round(4).to_string())

print()
print('=' * 96)
print('幅度随规模的变化：以 loss 水平为规模代理（loss 越低=规模越大）')
print('=' * 96)
x = np.log(df.mean_loss.values)
y = np.log(df.amp.values)
sl, ic, r, p, se = stats.linregress(x, y)
print(f'  log(幅度) = {ic:.4f} {sl:+.4f}·log(均值loss)   R²={r**2:.3f}  p={p:.3e}  n={len(df)}')
print(f'  → 斜率 {sl:+.4f}；若显著为正，说明规模越大（loss 越低）配比效应幅度越小 → 与 N/D 存在交互')

print()
print('  用相对幅度（幅度/损失水平）复核：')
y2 = np.log(df.rel_amp.values)
sl2, ic2, r2_, p2, se2 = stats.linregress(x, y2)
print(f'  log(相对幅度) = {ic2:.4f} {sl2:+.4f}·log(均值loss)   R²={r2_**2:.3f}  p={p2:.3e}')
print(f'  → 斜率 {sl2:+.4f}（相对幅度已扣除损失水平本身的下降）')

print()
print('=' * 96)
print('切片层面的汇总回归（6 个点，稳健性参考）')
print('=' * 96)
s2 = summ.copy()
sl3, ic3, r3, p3, se3 = stats.linregress(np.log(s2.mean_loss.values), np.log(s2.amp_median.values))
print(f'  log(中位幅度) = {ic3:.4f} {sl3:+.4f}·log(中位loss)  R²={r3**2:.3f}  p={p3:.3e}')
for t, v in zip(s2.index, s2.amp_median.values):
    print(f'    {t:<10} 中位loss={s2.loc[t,"mean_loss"]:.3f}  中位幅度={v:.4f}')

print()
print('=' * 96)
print('逐目标看：幅度最大的目标与衰减最明显的目标')
print('=' * 96)
piv = df.pivot_table(index='target', columns='slice', values='amp')
piv = piv.reindex(columns=[s[0] for s in SLICES])
piv['衰减比(70b/1m_test)'] = piv['70b(est)'] / piv['1m(test)']
piv = piv.sort_values('衰减比(70b/1m_test)')
print(piv.round(4).to_string())
