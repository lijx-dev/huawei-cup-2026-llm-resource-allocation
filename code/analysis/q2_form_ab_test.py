# -*- coding: utf-8 -*-
"""Form A vs Form B 的判定检验

论文 5.2.3 给出两种配比接入结构：
  Form A（p 只挂数据项）: L = E + A N^-α + B D^-β exp{g_Q + λ_p h(p)}
  Form B（p 作用于全部可约损失）: L = E + [A N^-α + B D^-β exp{g_Q}] exp{λ_p h(p)}

在一个固定切片内 N、D 均不变，两式都退化为  L = u_s + v_s·z(p)，z(p)=exp{λ_p h(p)}：
  Form A:  u_s = E + A N_s^-α      （随规模变化）
  Form B:  u_s = E                 （跨规模恒定）

因此取两个切片 s1(小)、s2(大)，在**同一配方集**上：
  L_{s1}(p) = u1 + (v1/v2)·(L_{s2}(p) - u2)
即 L_{s1} 对 L_{s2} 的线性回归
  斜率 s = v1/v2，截距 a = u1 - (v1/v2)·u2
  Form B: a = E·(1 - s)，故 E_implied = a/(1-s) 应为正且量级合理
  Form A: a = (E + A N1^-α) - s·(E + A N2^-α)，无此约束
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'


def rd(f):
    return pd.read_csv(os.path.join(A, f)).iloc[:, 1:].values.astype(float)


def cols(f):
    return [c.replace('metric/the_pile_', '').replace('_val_loss', '')
            for c in pd.read_csv(os.path.join(A, f)).columns[1:]]


def fit(X, Y, j):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, Y[:, j], rcond=None)
    return Xc @ b


PAIRS = [('1m', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv',
          '60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
         ('1m', 'train_mixture_1m.csv', 'train_pile_loss_1m.csv',
          '10b', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
         ('1m', 'train_mixture_1m.csv', 'train_pile_loss_1m.csv',
          '70b', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]

for s1, mf1, lf1, s2, mf2, lf2 in PAIRS:
    X1, Y1, t1 = rd(mf1), rd(lf1), cols(lf1)
    X2, Y2, t2 = rd(mf2), rd(lf2), cols(lf2)
    k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(X1)}
    k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(X2)}
    common = sorted(set(k1) & set(k2))
    i1 = [k1[c] for c in common]
    i2 = [k2[c] for c in common]
    print('=' * 104)
    print(f'{s1}(n={len(X1)}) vs {s2}(n={len(X2)})，共享配方 {len(common)}')
    print('=' * 104)
    print(f"{'target':<20}{'斜率s':>9}{'截距a':>10}{'R2':>8}"
          f"{'E_implied':>11}{'a符号':>8}{'L2均值':>9}{'L1均值':>9}")
    rows = []
    for j, t in enumerate(t1):
        if t not in t2:
            continue
        p1 = fit(X1, Y1, j)[i1]
        p2 = fit(X2, Y2, t2.index(t))[i2]
        sl, ic, r, pv, se = stats.linregress(p2, p1)
        e_imp = ic / (1 - sl) if abs(1 - sl) > 1e-9 else np.nan
        rows.append((t, sl, ic, r ** 2, e_imp))
        print(f'{t:<20}{sl:>9.4f}{ic:>+10.4f}{r**2:>8.4f}{e_imp:>11.4f}'
              f'{"负" if ic < 0 else "正":>8}{p2.mean():>9.4f}{p1.mean():>9.4f}')
    d = pd.DataFrame(rows, columns=['t', 'slope', 'intercept', 'r2', 'E_implied'])
    n_neg = (d.intercept < 0).sum()
    print(f'\n  斜率中位={d.slope.median():.4f}（Form B 要求 s>1，即小模型损失放大后对应大模型）')
    print(f'  截距中位={d.intercept.median():+.4f}，负截距 {n_neg}/{len(d)} 个')
    print(f'  E_implied 中位={d.E_implied.median():.4f}，'
          f'正值占比={ (d.E_implied>0).mean():.1%}，范围[{d.E_implied.min():.3f},{d.E_implied.max():.3f}]')
    print(f'  Form B 判定：截距应全为负（a=E(1-s)，s>1⇒a<0）→ '
          f'{"支持 Form B" if n_neg == len(d) and (d.slope>1).all() else "不完全支持，需看量级"}')
    print()
