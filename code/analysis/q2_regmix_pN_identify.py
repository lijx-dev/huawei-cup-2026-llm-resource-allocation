# -*- coding: utf-8 -*-
"""检验 regmix 表是否提供 (p, N, L) 联合观测，以及配比效应是否随规模变化。
   决定性问题：配比效应 h(p)=log[L(p)/L(p_ref)] 在 N=1m 与 N=60m 之间是否尺度不变？
   - 若 h 近似尺度不变  -> 配比效应可分离，Form B（作用于全部可约项）方向成立
   - 若 h 随 N 显著变化  -> 存在 p×N 交互，需在模型中显式写交互
"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'


def mix(scale):
    df = pd.read_csv(f'{BASE}/test_mixture_{scale}.csv')
    return df[[c for c in df.columns if c != 'index']].values


def loss(scale):
    df = pd.read_csv(f'{BASE}/test_pile_loss_{scale}.csv')
    return df[[c for c in df.columns if c != 'index']].values


P1, P60, P1B = mix('1m'), mix('60m'), mix('1B')
L1, L60, L1B = loss('1m'), loss('60m'), loss('1B')
P1B_tr = pd.read_csv(f'{BASE}/test_mixture_1B.csv')
Ptr = pd.read_csv(f'{BASE}/train_mixture_1m.csv')[
    [c for c in pd.read_csv(f'{BASE}/train_mixture_1m.csv').columns if c != 'index']].values

print('== 结构 ==')
print(f'  test_1m 配比 {P1.shape}, test_60m 配比 {P60.shape}, test_1B 配比 {P1B.shape}')
print(f'  test_1m 与 test_60m 配比完全相同? {np.allclose(P1, P60)}  (最大差 {np.abs(P1-P60).max():.2e})')
print(f'  test_1m 行和范围 [{P1.sum(1).min():.6f}, {P1.sum(1).max():.6f}]')
print(f'  test_1m 是否 train 的子集? 逐行匹配数 = '
      f'{sum(any(np.allclose(r, t) for t in Ptr) for r in P1)}/{len(P1)}')

# 聚合损失：13 个有损失目标取均值（与 Q1 的口径一致）
l1, l60, l1b = L1.mean(1), L60.mean(1), L1B.mean(1)
print('\n== 规模效应（同一配比）==')
print(f'  N=1m  聚合损失 均值 {l1.mean():.4f}')
print(f'  N=60m 聚合损失 均值 {l60.mean():.4f}')
print(f'  N=1B  聚合损失 均值 {l1b.mean():.4f}   (64 组不同配比)')

# 参考配比：取与 train 均值最接近的 test 行，避免外推
pref = P1.mean(0)
ref = int(np.argmin(((P1 - pref) ** 2).sum(1)))
print(f'\n== 配比效应 h(p) 的尺度不变性（参考配比 index={ref}）==')
h1 = np.log(l1 / l1[ref])
h60 = np.log(l60 / l60[ref])
print(f'  h_1m   : 均值 {h1.mean():+.4f}, 标准差 {h1.std():.4f}, 范围 [{h1.min():+.4f}, {h1.max():+.4f}]')
print(f'  h_60m  : 均值 {h60.mean():+.4f}, 标准差 {h60.std():.4f}, 范围 [{h60.min():+.4f}, {h60.max():+.4f}]')
print(f'  corr(h_1m, h_60m)      = {np.corrcoef(h1, h60)[0,1]:+.4f}')
print(f'  corr(h_1m, h_60m-2h_1m) = {np.corrcoef(h1, h60 - 2 * h1)[0,1]:+.4f}')

# 尺度不变性回归: h_60m = k * h_1m + c ; k=1 表示可分离
k, c = np.polyfit(h1, h60, 1)
res = h60 - (k * h1 + c)
n = len(h1)
se_k = np.sqrt((res ** 2).sum() / (n - 2) / ((h1 - h1.mean()) ** 2).sum())
print(f'  回归 h_60m = {k:.4f} * h_1m + {c:.5f}   (k 的 se={se_k:.4f}, t(k=1)={(k-1)/se_k:+.2f})')
print(f'  若 k≈1 且 c≈0 -> 配比效应与规模可分离（Form B 方向）；偏离则存在 p×N 交互')

# 直接检验"配比效应振幅是否随 N 变化"：逐配比算 ΔL，看 ΔL 与 h_1m 的关系
d = l60 - l1
s, b = np.polyfit(h1, d, 1)
print(f'\n  ΔL = L_60m - L_1m 对 h_1m 的回归斜率 = {s:+.4f}  (若配比效应与 N 无关，则 ΔL 与 h_1m 无关)')
print(f'  corr(h_1m, ΔL) = {np.corrcoef(h1, d)[0,1]:+.4f}')

# 分解：可约损失随 N 下降的幅度 vs 配比效应振幅
red1, red60 = l1 - l1.min(), l60 - l60.min()
print(f'\n== 振幅对比 ==')
print(f'  配比引起的损失跨度: 1m = {l1.max()-l1.min():.4f}, 60m = {l60.max()-l60.min():.4f}, '
      f'比值 = {(l60.max()-l60.min())/(l1.max()-l1.min()):.4f}')
print(f'  绝对损失水平:      1m = {l1.mean():.4f}, 60m = {l60.mean():.4f}, 比值 = {l60.mean()/l1.mean():.4f}')
print(f'  配比跨度/损失水平: 1m = {(l1.max()-l1.min())/l1.mean():.4f}, '
      f'60m = {(l60.max()-l60.min())/l60.mean():.4f}')
