# -*- coding: utf-8 -*-
"""决定性结构检验：配比效应是"作用于与规模无关的项"(Form A) 还是
   "乘在随规模衰减的可约损失上"(Form B)？

   关键性质（无需知道 N、D 的具体数值）：
     Form A: L(p) = E + A N^-α + B·g(p)   ->  任意两配比之差 ΔL 与 N 无关
     Form B: L(p) = E + (A N^-α + B)·g(p) ->  ΔL 随 N 衰减，比例 = (A N60^-α+B)/(A N1^-α+B)
   因此把 60m 的配比间差分对 1m 的差分做回归：
     斜率 ≈ 1  -> Form A（配比效应尺度不变）
     斜率 < 1  -> Form B（配比效应随规模衰减）
"""
import itertools
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'


def mix(s):
    df = pd.read_csv(f'{BASE}/test_mixture_{s}.csv')
    return df[[c for c in df.columns if c != 'index']].values


def loss(s):
    df = pd.read_csv(f'{BASE}/test_pile_loss_{s}.csv')
    return df[[c for c in df.columns if c != 'index']].values


P1, P60 = mix('1m'), mix('60m')
L1, L60 = loss('1m'), loss('60m')
l1, l60 = L1.mean(1), L60.mean(1)
n = len(l1)

idx = list(itertools.combinations(range(n), 2))
i, j = np.array([a for a, _ in idx]), np.array([b for _, b in idx])
d1, d60 = l1[i] - l1[j], l60[i] - l60[j]

s, c = np.polyfit(d1, d60, 1)
res = d60 - (s * d1 + c)
se_s = np.sqrt((res ** 2).sum() / (len(d1) - 2) / ((d1 - d1.mean()) ** 2).sum())
se_c = se_s * np.sqrt((d1 ** 2).mean())

print('== 配比间差分的跨规模回归 (聚合 13 目标均值) ==')
print(f'  ΔL_60m = {s:.5f} * ΔL_1m + {c:+.6f}')
print(f'  斜率 se = {se_s:.5f},  t(slope=1) = {(s - 1) / se_s:+.2f}')
print(f'  截距 se = {se_c:.5f},  t(intercept=0) = {c / se_c:+.2f}')
print(f'  R² = {1 - (res ** 2).sum() / ((d60 - d60.mean()) ** 2).sum():.5f}')
print(f'  ΔL_1m 标准差 {d1.std():.4f},  ΔL_60m 标准差 {d60.std():.4f},  比值 {d60.std()/d1.std():.4f}')

print('\n== 逐目标（13 个有损失域）的斜率 ==')
loss_cols = [c_ for c_ in pd.read_csv(f'{BASE}/test_pile_loss_1m.csv').columns if c_ != 'index']
slopes = []
for t, col in enumerate(loss_cols):
    a, b = L1[:, t], L60[:, t]
    dd1, dd60 = a[i] - a[j], b[i] - b[j]
    ss = np.polyfit(dd1, dd60, 1)[0]
    slopes.append(ss)
    short = col.replace('metric/the_pile_', '').replace('_val_loss', '')
    print(f'  {short:<20s} slope = {ss:.4f}')
slopes = np.array(slopes)
print(f'  13 目标斜率: 均值 {slopes.mean():.4f}, 中位数 {np.median(slopes):.4f}, '
      f'范围 [{slopes.min():.4f}, {slopes.max():.4f}], 斜率<1 的个数 = {(slopes < 1).sum()}/13')

print('\n== 判据 ==')
print('  若斜率≈1 -> 配比效应的绝对幅度与规模无关，支持 Form A（配比作用于与规模无关的分量）')
print('  若斜率<1 -> 配比效应随规模衰减，支持 Form B（配比乘在可约损失上）')
print(f'  实测斜率 {s:.4f}，{"接近 1" if abs(s-1) < 0.05 else "显著偏离 1"}')
