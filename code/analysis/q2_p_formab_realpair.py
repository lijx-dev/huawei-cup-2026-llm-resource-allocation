# -*- coding: utf-8 -*-
"""配比 p 的作用位置判定：用真实数据对 (1m, 60m) 的 256 个共同配方

同一切片内 N、D 固定，故 L_s(p) = u_s + v_s z(p)，z 为共同未知形状。
成对回归  L_1 = a + s L_2  给出
    s = v_1/v_2,   a = u_1 - s u_2
两个结构对 u_s 的预测：
    Form A（p 只挂数据项）: u_s = E + A N_s^-a  -> u_1 > u_2，且 a > 0（当 s>1）
    Form B（p 挂全部可约项）: u_s = E            -> a = E(1-s) < 0（当 s>1）
故 a 的符号即为判别量：a<0 支持 Form B，a>0 支持 Form A。
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

print('=' * 92)
print(f'1m vs 60m：共同配方 {len(common)}（同一组配方，无设计混淆，均为真实表）')
print('=' * 92)
print(f"{'target':<18}{'s':>8}{'a':>9}{'R2':>8}{'E_B':>9}{'E_A需A>':>10}"
      f"{'L1均值':>9}{'L2均值':>9}")
rows = []
for j, t in enumerate(tgt):
    l1, l2 = Y1[i1, j], Y2[i2, j]
    s, a, r, pv, se = stats.linregress(l2, l1)
    e_b = a / (1 - s) if abs(1 - s) > 1e-9 else np.nan
    # Form A 要求 a = -E(s-1) + A*(N1^-a - s N2^-a)；以 A 为未知，E 取 1.6
    coefA = 0.001 ** -0.34 - s * 0.06 ** -0.34
    a_req = (a + 1.6 * (s - 1)) / coefA if abs(coefA) > 1e-9 else np.nan
    rows.append((t, s, a, r ** 2, e_b, a_req))
    print(f'{t:<18}{s:>8.4f}{a:>+9.4f}{r**2:>8.4f}{e_b:>9.3f}{a_req:>10.4f}'
          f'{l1.mean():>9.3f}{l2.mean():>9.3f}')
d = pd.DataFrame(rows, columns=['t', 's', 'a', 'r2', 'e_b', 'A_req'])
print()
print(f'  斜率 s 中位 = {d.s.median():.4f}，s>1 的目标 {int((d.s>1).sum())}/{len(d)}')
print(f'  截距 a 中位 = {d.a.median():+.4f}，a>0 的目标 {int((d.a>0).sum())}/{len(d)}')
print(f'  Form B 预测 a = E(1-s) < 0：被 {int((d.a>0).sum())}/{len(d)} 个目标否证')
print(f'  Form A 预测 a = -E(s-1)+A(N1^-a - s N2^-a)：符号与实测一致的目标 '
      f'{int((np.sign(d.a)==np.sign(-1.6*(d.s-1)+0.354*(0.001**-0.34-d.s*0.06**-0.34))).sum())}/{len(d)}')
print(f'  由 a 反解 Form A 的 A 中位 = {d.A_req.median():.4f}（B1 的 A = 0.3540）')

# 配对检验：a 的符号是否显著
w = stats.wilcoxon(d.a, alternative='greater')
print(f"\n  单侧 Wilcoxon（H0: a<=0 即支持 Form B）p = {w.pvalue:.3e}")
tst = stats.ttest_1samp(d.a, 0.0, alternative='greater')
print(f"  单侧 t 检验（H0: a<=0）p = {tst.pvalue:.3e}，均值 a = {d.a.mean():+.4f} ± {d.a.std(ddof=1)/np.sqrt(len(d)):.4f}")
