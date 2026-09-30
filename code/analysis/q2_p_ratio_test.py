# -*- coding: utf-8 -*-
"""决定性判据：同配方跨规模损失比值

设 L(N,D,p) 在两个规模切片 s1(小), s2(大) 上取同一配方 p：
  可乘形式 L = F(N,D)·g(p)  ⇒  L_s1(p)/L_s2(p) = F_s1/F_s2 = 常数，与 p 无关
  可加形式 L = F(N,D)+G(p)  ⇒  比值 = (F_s1+G(p))/(F_s2+G(p))，随 G(p) 单调变化，与 p 强相关
因此：比值的变异系数(CV)≈0 且比值与 p 效应无关 ⇒ 可乘；比值随 p 显著变化 ⇒ 可加。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'


def ridge_fit(X, y, lam=1e-3):
    Xc = np.column_stack([np.ones(len(X)), X])
    k = Xc.shape[1]
    R = np.eye(k) * lam
    R[0, 0] = 0.0
    return Xc @ np.linalg.solve(Xc.T @ Xc + R, Xc.T @ y)


def load(mf, lf):
    X = pd.read_csv(os.path.join(A, mf)).iloc[:, 1:].values.astype(float)
    Y = pd.read_csv(os.path.join(A, lf)).iloc[:, 1:].values.astype(float)
    tg = [c.replace('metric/the_pile_', '').replace('_val_loss', '')
          for c in pd.read_csv(os.path.join(A, lf)).columns[1:]]
    return X, Y, tg


PAIRS = [('1m', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv',
          '60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
         ('10b', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv',
          '70b', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]

print('=' * 100)
print('E4 同配方跨规模损失比值：可乘 ⇒ 比值为常数；可加 ⇒ 比值随配比显著变化')
print('=' * 100)
rows = []
for s1, mf1, lf1, s2, mf2, lf2 in PAIRS:
    X1, Y1, tg1 = load(mf1, lf1)
    X2, Y2, tg2 = load(mf2, lf2)
    k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(X1)}
    k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(X2)}
    common = sorted(set(k1) & set(k2))
    print(f'\n  对照: {s1}(n={len(X1)}) vs {s2}(n={len(X2)})，共享配方 {len(common)} 个')
    i1 = [k1[c] for c in common]
    i2 = [k2[c] for c in common]
    for j, t in enumerate(tg1):
        if t not in tg2:
            continue
        jj = tg2.index(t)
        p1 = ridge_fit(X1, Y1[:, j])
        p2 = ridge_fit(X2, Y2[:, jj])
        r = p1[i1] / p2[i2]
        # 该目标的配比效应强度：用较大规模切片的预测损失水平排序代理
        rows.append({'pair': f'{s1}vs{s2}', 'target': t,
                     'ratio_mean': r.mean(), 'ratio_cv': r.std() / r.mean(),
                     'ratio_range': r.max() - r.min(),
                     'corr_ratio_with_p2': stats.pearsonr(r, p2[i2])[0],
                     'loss_s1': Y1[:, j].mean(), 'loss_s2': Y2[:, jj].mean()})
d = pd.DataFrame(rows)

print('\n  汇总（按规模对照）:')
print(d.groupby('pair').agg(
    ratio_mean=('ratio_mean', 'mean'),
    ratio_cv_median=('ratio_cv', 'median'),
    ratio_range_median=('ratio_range', 'median'),
    corr_with_p2_median=('corr_ratio_with_p2', 'median')).round(4).to_string())

print('\n  若"可乘"成立：ratio_cv 应接近 0，corr_ratio_with_p2 应接近 0')
print('  若"可加"成立：ratio_cv 显著>0，且 corr_ratio_with_p2 应显著为负（小损失配方比值偏大）')
print()
print('  逐目标明细（按 ratio_cv 降序，取前 8）:')
print(d.sort_values('ratio_cv', ascending=False).head(8).round(4).to_string(index=False))

print()
print('=' * 100)
print('E5 反向对照：把两个规模切成同一配方集，检验"配比效应是否整体等比缩放"')
print('=' * 100)
for s1, mf1, lf1, s2, mf2, lf2 in PAIRS:
    X1, Y1, tg1 = load(mf1, lf1)
    X2, Y2, tg2 = load(mf2, lf2)
    k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(X1)}
    k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(X2)}
    common = sorted(set(k1) & set(k2))
    i1 = [k1[c] for c in common]
    i2 = [k2[c] for c in common]
    slopes = []
    for j, t in enumerate(tg1):
        if t not in tg2:
            continue
        jj = tg2.index(t)
        p1 = ridge_fit(X1, Y1[:, j])[i1]
        p2 = ridge_fit(X2, Y2[:, jj])[i2]
        # 回归 p1 = a + b*p2，b 为缩放因子，R² 为线性度
        sl, ic, r, p, se = stats.linregress(p2, p1)
        slopes.append({'pair': f'{s1}vs{s2}', 'target': t, 'scale_b': sl, 'R2': r ** 2})
    sd = pd.DataFrame(slopes)
    print(f'\n  {s1} vs {s2}: 斜率 b 的均值={sd.scale_b.mean():.4f} 中位={sd.scale_b.median():.4f}，'
          f'线性 R² 中位={sd.R2.median():.4f}')
    print(f'    可乘形式预测：p1 与 p2 应成过原点的严格比例关系（R²→1，截距→0）')

d.to_csv(r'd:\F题\q2_p_ratio_test.csv', index=False, encoding='utf-8-sig')
print('\n  明细已保存: q2_p_ratio_test.csv')
