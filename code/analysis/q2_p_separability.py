# -*- coding: utf-8 -*-
"""配比效应 p 与规模 (N,D) 的可分离性检验

判定标准：
  可乘（可分离）形式  L = F(N,D) * g(p)   ⇒ 归一化剖面 L(p)/mean_p L 在不同规模切片上一致，
                                            且 相对幅度 (max-min)/mean 跨切片恒定
  可加形式            L = F(N,D) + G(p)   ⇒ 绝对幅度 (max-min) 跨切片恒定，相对幅度随 loss 下降而下降

输出三组证据：
  E1 相对幅度 / 变异系数 CV 随规模的变化
  E2 共享配方上的归一化剖面一致性（若切片间有共同配方）
  E3 秩一致性：同一配方在不同切片上的预测损失排序是否保持
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'
SLICES = [('1m', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv'),
          ('60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
          ('1B', 'test_mixture_1B.csv', 'test_pile_loss_1B.csv'),
          ('10b', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
          ('70b', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]


def ridge_fit(X, y, lam=1e-3):
    Xc = np.column_stack([np.ones(len(X)), X])
    k = Xc.shape[1]
    R = np.eye(k) * lam
    R[0, 0] = 0.0
    return Xc @ np.linalg.solve(Xc.T @ Xc + R, Xc.T @ y)


data = {}
for tag, mf, lf in SLICES:
    X = pd.read_csv(os.path.join(A, mf)).iloc[:, 1:].values.astype(float)
    Y = pd.read_csv(os.path.join(A, lf)).iloc[:, 1:].values.astype(float)
    targets = [c.replace('metric/the_pile_', '').replace('_val_loss', '')
               for c in pd.read_csv(os.path.join(A, lf)).columns[1:]]
    data[tag] = (X, Y, targets, mf)

print('=' * 100)
print('E0 配方行重叠情况（按 17 维配比向量精确匹配）')
print('=' * 100)
keys = {}
for tag, (X, Y, tg, mf) in data.items():
    keys[tag] = {tuple(np.round(r, 6)): i for i, r in enumerate(X)}
    print(f'  {tag:<5} n={len(X):<5} file={mf}')
tags = list(data.keys())
print()
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        inter = len(set(keys[tags[i]]) & set(keys[tags[j]]))
        if inter:
            print(f'  {tags[i]} ∩ {tags[j]} = {inter}')

print()
print('=' * 100)
print('E1 幅度与变异系数随规模的变化（可乘 ⇒ 相对幅度恒定；可加 ⇒ 绝对幅度恒定）')
print('=' * 100)
rows = []
for tag, (X, Y, targets, mf) in data.items():
    for j, t in enumerate(targets):
        y = Y[:, j]
        pred = ridge_fit(X, y)
        rows.append({'slice': tag, 'target': t, 'mean_loss': y.mean(),
                     'amp': pred.max() - pred.min(),
                     'cv': pred.std() / pred.mean(),
                     'rel_amp': (pred.max() - pred.min()) / pred.mean()})
df = pd.DataFrame(rows)

print('\n  逐切片汇总（13 个目标的中位数）:')
s = df.groupby('slice').agg(mean_loss=('mean_loss', 'median'), amp=('amp', 'median'),
                            cv=('cv', 'median'), rel_amp=('rel_amp', 'median'))
s = s.reindex([t for t, _, _ in SLICES])
print(s.round(4).to_string())

print('\n  回归（以 log(中位loss) 为规模代理，loss 越低=规模越大）:')
for col, name, expect in [('amp', '绝对幅度', '可加形式下应无趋势(斜率≈0)'),
                          ('rel_amp', '相对幅度', '可乘形式下应无趋势(斜率≈0)'),
                          ('cv', '变异系数', '可乘形式下应无趋势(斜率≈0)')]:
    x = np.log(s.mean_loss.values)
    yv = np.log(s[col].values)
    sl, ic, r, p, se = stats.linregress(x, yv)
    print(f'    log({name}) = {ic:+.4f} {sl:+.4f}·log(loss)  R²={r**2:.3f}  p={p:.3e}   [{expect}]')

print()
print('=' * 100)
print('E2 共享配方上的归一化剖面一致性（可乘 ⇒ 高相关）')
print('=' * 100)
pairs = []
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        common = sorted(set(keys[tags[i]]) & set(keys[tags[j]]))
        if len(common) < 10:
            continue
        a, b = tags[i], tags[j]
        for k, t in enumerate(data[a][2]):
            if t not in data[b][2]:
                continue
            kk = data[b][2].index(t)
            pa = ridge_fit(data[a][0], data[a][1][:, k])
            pb = ridge_fit(data[b][0], data[b][1][:, kk])
            ia = [keys[a][c] for c in common]
            ib = [keys[b][c] for c in common]
            va = pa[ia] / pa[ia].mean()
            vb = pb[ib] / pb[ib].mean()
            r = np.corrcoef(va, vb)[0, 1]
            pairs.append({'slice_a': a, 'slice_b': b, 'target': t, 'n_common': len(common), 'corr': r})
if pairs:
    pdf = pd.DataFrame(pairs)
    print(pdf.groupby(['slice_a', 'slice_b'])['corr'].agg(['mean', 'median', 'count']).round(4).to_string())
else:
    print('  切片之间没有足够的共享配方，E2 不可用')

print()
print('=' * 100)
print('E3 秩一致性：同一配方在不同切片上的预测损失排序')
print('=' * 100)
if pairs:
    rk = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            common = sorted(set(keys[tags[i]]) & set(keys[tags[j]]))
            if len(common) < 10:
                continue
            a, b = tags[i], tags[j]
            for k, t in enumerate(data[a][2]):
                if t not in data[b][2]:
                    continue
                kk = data[b][2].index(t)
                pa = ridge_fit(data[a][0], data[a][1][:, k])
                pb = ridge_fit(data[b][0], data[b][1][:, kk])
                ia = [keys[a][c] for c in common]
                ib = [keys[b][c] for c in common]
                rho = stats.spearmanr(pa[ia], pb[ib]).correlation
                rk.append({'pair': f'{a}-{b}', 'rho': rho})
    rdf = pd.DataFrame(rk)
    print(rdf.groupby('pair').rho.agg(['mean', 'median', 'count']).round(4).to_string())

df.to_csv(r'd:\F题\q2_p_separability_by_target.csv', index=False, encoding='utf-8-sig')
print('\n  明细已保存: q2_p_separability_by_target.csv')
