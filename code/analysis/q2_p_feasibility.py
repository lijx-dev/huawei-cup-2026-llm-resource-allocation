# -*- coding: utf-8 -*-
"""配比作用位置判据升级版：两种结构的“反解可行性”检验

同一切片内 N、D 固定，故 L_s(p) = u_s + v_s z(p)。
成对回归 L_1 = a + s L_2 给出 s = v_1/v_2，a = u_1 - s u_2。

Form B（p 挂全部可约损失）：u_s = c_A + E  ->  a = (c_A+E)(1-s)
    反解 E_B = a/(1-s) - c_A；要求 E > 0（不可约损失为正）
Form A（p 只挂数据项）：u_s = c_A + E + A N_s^-alpha
    a = c_A(1-s) + E(1-s) + A(N_1^-alpha - s N_2^-alpha)
    反解 A_A = [a + (E + c_A)(s-1)] / (N_1^-alpha - s N_2^-alpha)

切片为 1M 与 60M 参数量（N_1 = 0.001B, N_2 = 0.06B）。
alpha、E、A 取 B1 的独立估计（0.3400, 1.6898, 0.3540）。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'
X1 = pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
X2 = pd.read_csv(os.path.join(A, 'test_mixture_60m.csv')).iloc[:, 1:].values.astype(float)
Y1df = pd.read_csv(os.path.join(A, 'test_pile_loss_1m.csv'))
Y2df = pd.read_csv(os.path.join(A, 'test_pile_loss_60m.csv'))
tgt = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in Y1df.columns[1:]]
Y1 = Y1df.iloc[:, 1:].values.astype(float)
Y2 = Y2df.iloc[:, 1:].values.astype(float)

k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(X1)}
k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(X2)}
common = sorted(set(k1) & set(k2))
i1 = [k1[c] for c in common]
i2 = [k2[c] for c in common]

ALPHA, E1, A1 = 0.34, 1.6898, 0.3540
N1, N2 = 0.001, 0.06
CA_LIST = [0.0, 0.5, 1.0, 2.0]

rows = []
for j, t in enumerate(tgt):
    l1, l2 = Y1[i1, j], Y2[i2, j]
    s, a, r, pv, se = stats.linregress(l2, l1)
    n = len(l1)
    xm = l2.mean()
    se_a = np.sqrt(np.sum((l1 - (a + s * l2)) ** 2) / (n - 2)) * \
        np.sqrt(1 / n + xm ** 2 / np.sum((l2 - xm) ** 2))
    p_a = 2 * stats.t.sf(abs(a / se_a), n - 2)
    E_B = a / (1 - s)
    coefA = N1 ** -ALPHA - s * N2 ** -ALPHA
    A_imp = {c: (a + (E1 + c) * (s - 1)) / coefA for c in CA_LIST}
    cA_star = A1 * coefA / (s - 1) - E1
    rows.append(dict(t=t, s=s, a=a, se_a=se_a, p_a=p_a, r2=r ** 2,
                     E_B=E_B, coefA=coefA, cA_star=cA_star,
                     **{f'A_c{c}'.replace('.', 'p'): A_imp[c] for c in CA_LIST}))
d = pd.DataFrame(rows)

print('=' * 118)
print('判据一：Form B 的反解可行性 —— E_B = a/(1-s)，要求 E > 0')
print('=' * 118)
print(f"{'target':<20}{'s':>8}{'a':>9}{'p(a=0)':>11}{'E_B':>10}{'可行':>7}")
for _, x in d.iterrows():
    ok = 'YES' if x.E_B > 0 else 'no'
    print(f"{x.t:<20}{x.s:>8.4f}{x.a:>+9.4f}{x.p_a:>11.2e}{x.E_B:>+10.3f}{ok:>7}")
print(f"\n  Form B 不可行（E_B<=0）的目标数：{int((d.E_B<=0).sum())}/{len(d)}")
print(f"  E_B 中位 = {d.E_B.median():+.3f}，范围 [{d.E_B.min():+.3f}, {d.E_B.max():+.3f}]")
print(f"  唯一可行者：{d.loc[d.E_B>0,'t'].tolist()}（E_B = "
      f"{d.loc[d.E_B>0,'E_B'].round(3).tolist()}）")

print('\n' + '=' * 118)
print('判据二：Form A 的反解 A —— 对 A 端源偏移 c_A 的敏感性')
print('=' * 118)
print(f"{'target':<20}{'cA=0':>9}{'cA=0.5':>9}{'cA=1.0':>9}{'cA=2.0':>9}{'cA*使A=0.354':>14}")
for _, x in d.iterrows():
    print(f"{x.t:<20}{x['A_c0p0']:>9.4f}{x['A_c0p5']:>9.4f}{x['A_c1p0']:>9.4f}"
          f"{x['A_c2p0']:>9.4f}{x.cA_star:>+14.4f}")
print()
for c in CA_LIST:
    v = d[f'A_c{str(c).replace(".", "p")}']
    print(f'  c_A={c:<4}: 反解 A 中位={v.median():.4f}  范围[{v.min():.4f},{v.max():.4f}]'
          f'  落在 B1 的 A=0.3540 的 ±50% 内: {int(((v>0.177)&(v<0.531)).sum())}/{len(d)}')
print(f'\n  使反解 A 恰等于 B1 值所需的 c_A：中位={d.cA_star.median():+.4f}  '
      f'范围[{d.cA_star.min():+.4f},{d.cA_star.max():+.4f}]')
print('  （c_A 为 A 端相对 B1 口径的源偏移；其量级应与实测跨来源偏移可比）')

d.to_csv(r'd:\F题\q2_p_feasibility_by_target.csv', index=False)
print('\n已导出 q2_p_feasibility_by_target.csv')

print('\n' + '=' * 118)
print('判据二的敏感性：反解 A 依赖 B1 的 alpha 外推到 1M-60M 尺度（B1 只覆盖 0.07-12B）')
print('=' * 118)
for al in [0.28, 0.30, 0.34, 0.38, 0.42]:
    cf = N1 ** -al - d.s * N2 ** -al
    Ai = (d.a + E1 * (d.s - 1)) / cf
    print(f'  alpha={al:.2f}: coefA 中位={cf.median():.4f}  反解 A 中位={Ai.median():.4f}  '
          f'范围[{Ai.min():.4f},{Ai.max():.4f}]  '
          f'落在 0.3540 的 ±50% 内: {int(((Ai > 0.177) & (Ai < 0.531)).sum())}/13')
print('  （alpha 越大，coefA 越大，反解 A 越小；结论方向不随 alpha 改变）')
