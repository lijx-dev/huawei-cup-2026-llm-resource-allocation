# -*- coding: utf-8 -*-
"""lambda_p 情景网格在代表点上的影响包络（Form A, SET_A 参数）。"""
import numpy as np

E, At, al, Bt, be, rN, rD, E1 = 1.7112, 0.6519, 0.2783, 1.4207, 0.2834, 0.3555, 0.1428, 0.1027
N, D, Q = 1.0, 150.0, 0.5

Nt = At * N ** -al * np.exp(-rN * Q)
Dt = Bt * D ** -be * np.exp(-rD * Q)
L0 = E + Nt + Dt - E1 * Q
print(f'基线 L = {L0:.5f}   N项 = {Nt:.5f}   D项(配比乘子载体) = {Dt:.5f}')
print()
print('h 范围（运行报告第 5 节，13 目标分位均值）: 低 -0.216 / 中位 +0.096 / 高 +0.190')
print()
hs = [-0.216, 0.096, 0.190]
print(f'{"lambda_p":>9} | {"h=-0.216":>22} | {"h=+0.096":>22} | {"h=+0.190":>22}')
print('-' * 86)
for lp in [0.0, 0.5, 1.0, 1.5]:
    row = []
    for h in hs:
        L = E + Nt + Dt * np.exp(lp * h) - E1 * Q
        row.append(f'{(L / L0 - 1) * 100:>+7.3f}%  (dL{L - L0:+.4f})')
    print(f'{lp:>9} | ' + ' | '.join(f'{x:>22}' for x in row))

print()
phis = [lp * h for lp in [0.0, 0.5, 1.0, 1.5] for h in hs]
print(f'phi = lambda_p * h 全网格范围: [{min(phis):+.4f}, {max(phis):+.4f}]')
Lmin = E + Nt + Dt * np.exp(min(phis)) - E1 * Q
Lmax = E + Nt + Dt * np.exp(max(phis)) - E1 * Q
print(f'包络: L in [{Lmin:.5f}, {Lmax:.5f}]  =>  相对变化 [{(Lmin/L0-1)*100:+.3f}%, {(Lmax/L0-1)*100:+.3f}%]')
dmax = max(abs(Lmin - L0), abs(Lmax - L0))
print(f'|dL| 最大 = {dmax:.5f}  (占 L 的 {dmax/L0*100:.3f}%)')
print()

qeff = 0.0339
print(f'对照：质量 0.5->0.6 的模型内 Loss 变化 = -{qeff}  ({-qeff/L0*100:+.3f}%)')
print(f'配比情景包络 / 质量效应 = {dmax/qeff:.2f} 倍')
print()
# 用 lambda_p=1 时 h 的实际支持范围（§4.6(f): |h_p| 95百分位=0.1244, 全范围[-0.0611,+0.2266]）
print('用 §4.6(f) 的实际 h_p 支持范围复核（13目标等权, A6-A7 256 配方）:')
for h in [-0.0611, 0.2266]:
    for lp in [1.0, 1.5]:
        L = E + Nt + Dt * np.exp(lp * h) - E1 * Q
        print(f'  lambda_p={lp}, h={h:+.4f} -> {(L/L0-1)*100:+.3f}%')
