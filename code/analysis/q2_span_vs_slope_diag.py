# -*- coding: utf-8 -*-
"""诊断：差分回归斜率 0.7622 与"损失跨度比 0.9950"为何看起来矛盾。
   两者都是"配比效应绝对幅度"的度量，若不一致，说明其中一个被极端点主导。
   输出可直接写进论文的稳健幅度比与极端点定位。
"""
import itertools
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'


def rd(stem):
    df = pd.read_csv(f'{BASE}/{stem}.csv')
    return df[[c for c in df.columns if c != 'index']].values


L1, L60 = rd('test_pile_loss_1m'), rd('test_pile_loss_60m')
l1, l60 = L1.mean(1), L60.mean(1)
n = len(l1)
idx = list(itertools.combinations(range(n), 2))
i = np.array([a for a, _ in idx])
j = np.array([b for _, b in idx])
d1, d60 = l1[i] - l1[j], l60[i] - l60[j]

print('== 幅度度量对比（聚合 13 目标均值，256 配比的两两差分，32640 对）==')
print(f'  差分标准差比 sd(dL60)/sd(dL1) = {d60.std()/d1.std():.4f}')
q = lambda x, p: np.percentile(x, p)
for lo, hi in [(0, 100), (1, 99), (5, 95), (10, 90), (25, 75)]:
    r1 = q(d1, hi) - q(d1, lo)
    r60 = q(d60, hi) - q(d60, lo)
    print(f'  {lo:>2d}–{hi:<3d} 分位跨度: 1m={r1:.4f}  60m={r60:.4f}  比值={r60/r1:.4f}')
print(f'  极差(0-100)比 = {(l60.max()-l60.min())/(l1.max()-l1.min()):.4f}  '
      f'<- 该值由单一极端点决定')

print('\n== 极端点定位（为何极差比≈1 而回归斜率≈0.76）==')
k1 = int(np.argmax(d1))
k60 = int(np.argmax(d60))
print(f'  使 dL_1m 最大的配比对 (idx {i[k1]},{j[k1]}): dL_1m={d1[k1]:.4f}, dL_60m={d60[k1]:.4f}')
print(f'  使 dL_60m 最大的配比对 (idx {i[k60]},{j[k60]}): dL_1m={d1[k60]:.4f}, dL_60m={d60[k60]:.4f}')
print(f'  两者是否同一配比对: {k1 == k60}')

s, c = np.polyfit(d1, d60, 1)
res = d60 - (s * d1 + c)
print(f'\n  全样本回归斜率 = {s:.4f}, 截距 = {c:+.6f}, R² = {1-(res**2).sum()/((d60-d60.mean())**2).sum():.5f}')
for thr in [0.5, 1.0, 1.2]:
    m = np.abs(d1) <= thr
    s2, c2 = np.polyfit(d1[m], d60[m], 1)
    print(f'  剔除 |dL_1m|>{thr} 的极端对后（保留 {m.sum()}/{len(m)}）: 斜率={s2:.4f}, 截距={c2:+.6f}')

print('\n== 逐目标幅度比（各目标独立）==')
cols = [c_ for c_ in pd.read_csv(f'{BASE}/test_pile_loss_1m.csv').columns if c_ != 'index']
print(f'  {"目标":<20s}{"斜率":>9s}{"sd比":>9s}{"0-100跨度比":>13s}')
rows = []
for t, col in enumerate(cols):
    a, b = L1[:, t], L60[:, t]
    aa, bb = a[i] - a[j], b[i] - b[j]
    sl = np.polyfit(aa, bb, 1)[0]
    sd_r = bb.std() / aa.std()
    sp_r = (b.max() - b.min()) / (a.max() - a.min())
    nm = col.replace('metric/the_pile_', '').replace('_val_loss', '')
    rows.append((nm, sl, sd_r, sp_r))
    print(f'  {nm:<20s}{sl:>9.4f}{sd_r:>9.4f}{sp_r:>13.4f}')
A = np.array([[r[1], r[2], r[3]] for r in rows])
print(f'  {"均值":<20s}{A[:,0].mean():>9.4f}{A[:,1].mean():>9.4f}{A[:,2].mean():>13.4f}')
print(f'  {"中位数":<20s}{np.median(A[:,0]):>9.4f}{np.median(A[:,1]):>9.4f}{np.median(A[:,2]):>13.4f}')
print(f'  斜率<1: {(A[:,0]<1).sum()}/13    sd比<1: {(A[:,1]<1).sum()}/13    跨度比<1: {(A[:,2]<1).sum()}/13')

print('\n== 不可约损失 E 反解（按 (L60-E)/(L1-E)=slope，假设比例式可约损失）==')
for nm, sl, _, _ in rows:
    t = [r[0] for r in rows].index(nm)
    m1, m60 = L1[:, t].mean(), L60[:, t].mean()
    if sl < 1:
        E = (m60 - sl * m1) / (1 - sl)
        tag = '合理' if 0 < E < m60 else '越界'
        print(f'  {nm:<20s} L@1m={m1:.3f} L@60m={m60:.3f} slope={sl:.4f} -> E={E:>8.3f} [{tag}]')
    else:
        print(f'  {nm:<20s} L@1m={m1:.3f} L@60m={m60:.3f} slope={sl:.4f} -> 不可识别')
