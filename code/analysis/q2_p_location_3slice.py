# -*- coding: utf-8 -*-
"""三切片联立求解 u_s：判定配比 p 的作用位置

同一配比切片内 N、D 固定，故
    L_s(p) = u_s + v_s * z(p),   z(p) 为共同形状（未知，无需知道）
两种结构对 u_s 的预测不同：
    Form A（p 只挂数据项）: u_s = E + A N_s^-a      -> 随规模显著变化
    Form B（p 挂全部可约项）: u_s = E                 -> 跨切片恒定

三个切片共享同一配方集时，成对回归给出
    L_i = a_ij + s_ij * L_j,  s_ij = v_i/v_j,  a_ij = u_i - s_ij u_j
令 v_1 = 1，则 v_2 = 1/s_12, v_3 = 1/s_13，且必须满足 s_23 = s_13/s_12（自洽性检验）。
u 由三个截距线性解出：
    u_3 = (a_12 + s_12 a_23 - a_13) / (s_13 - s_12 s_23)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'

FILES = {
    '1m_test':  ('test_mixture_1m.csv',  'test_pile_loss_1m.csv'),
    '60m_test': ('test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
    '1B_test':  ('test_mixture_1B.csv',  'test_pile_loss_1B.csv'),
    '1m_train': ('train_mixture_1m.csv', 'train_pile_loss_1m.csv'),
    '10b_est':  ('est_mixture_10b.csv',  'est_pile_loss_10b.csv'),
    '70b_est':  ('est_mixture_70b.csv',  'est_pile_loss_70b.csv'),
}


def load(mf, lf):
    X = pd.read_csv(os.path.join(A, mf))
    Y = pd.read_csv(os.path.join(A, lf))
    tgt = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in Y.columns[1:]]
    Xv = X.iloc[:, 1:].values.astype(float)
    Yv = Y.iloc[:, 1:].values.astype(float)
    key = {tuple(np.round(r, 6)): i for i, r in enumerate(Xv)}
    return Xv, Yv, tgt, key


DATA = {k: load(*v) for k, v in FILES.items()}

print('=' * 96)
print('配方集重叠情况')
print('=' * 96)
names = list(DATA)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n = len(set(DATA[names[i]][3]) & set(DATA[names[j]][3]))
        print(f'  {names[i]:>9} ∩ {names[j]:<9} = {n}')

TRIPLES = [('1m_test', '60m_test', '1B_test'),
           ('1m_train', '10b_est', '70b_est')]

for (n1, n2, n3) in TRIPLES:
    X1, Y1, t1, k1 = DATA[n1]
    X2, Y2, t2, k2 = DATA[n2]
    X3, Y3, t3, k3 = DATA[n3]
    common = sorted(set(k1) & set(k2) & set(k3))
    if len(common) < 10:
        print(f'\n### {n1}/{n2}/{n3}: 共享配方仅 {len(common)}，跳过')
        continue
    i1 = [k1[c] for c in common]
    i2 = [k2[c] for c in common]
    i3 = [k3[c] for c in common]
    print('\n' + '=' * 96)
    print(f'三切片联立：{n1} / {n2} / {n3}，共享配方 {len(common)}')
    print('=' * 96)
    print(f"{'target':<18}{'s12':>8}{'s13':>8}{'s23':>8}{'s13/s12':>10}"
          f"{'u1':>9}{'u2':>9}{'u3':>9}{'u2-u1':>9}{'u3-u1':>9}{'R2_12':>8}")
    rec = []
    for t in t1:
        if t not in t2 or t not in t3:
            continue
        L1 = Y1[i1, t1.index(t)]
        L2 = Y2[i2, t2.index(t)]
        L3 = Y3[i3, t3.index(t)]
        s12, a12, r12, _, _ = stats.linregress(L2, L1)
        s13, a13, r13, _, _ = stats.linregress(L3, L1)
        s23, a23, r23, _, _ = stats.linregress(L3, L2)
        den = s13 - s12 * s23
        if abs(den) < 1e-9:
            continue
        u3 = (a12 + s12 * a23 - a13) / den
        u2 = a23 + s23 * u3
        u1 = a12 + s12 * u2
        rec.append((t, s12, s13, s23, u1, u2, u3, r12 ** 2))
        print(f'{t:<18}{s12:>8.4f}{s13:>8.4f}{s23:>8.4f}{s13/s12:>10.4f}'
              f'{u1:>9.3f}{u2:>9.3f}{u3:>9.3f}{u2-u1:>+9.3f}{u3-u1:>+9.3f}{r12**2:>8.4f}')
    d = pd.DataFrame(rec, columns=['t', 's12', 's13', 's23', 'u1', 'u2', 'u3', 'r2'])
    if len(d) == 0:
        continue
    print(f"\n  自洽性 s23 vs s13/s12：中位绝对偏差 "
          f"{np.median(np.abs(d.s23 - d.s13/d.s12)):.4f}（应≈0）")
    print(f"  u1 中位={d.u1.median():.3f}   u2 中位={d.u2.median():.3f}   "
          f"u3 中位={d.u3.median():.3f}")
    print(f"  u2-u1 中位={ (d.u2-d.u1).median():+.3f}（Form B 要求≈0；Form A 要求显著<0）")
    print(f"  u3-u1 中位={ (d.u3-d.u1).median():+.3f}")
    print(f"  跨目标 u1 标准差={d.u1.std():.3f}（若 u_s 为结构常数，跨目标波动应小）")
