# -*- coding: utf-8 -*-
"""P0/P1 修正核验：(1) §2.7 代表点 L/R 与弹性；(2) ρ_N/ρ_D 比值的稳定性

背景：文档 §2.7 原文在代表点 (N=1.3B, D=50B, Q=0.7) 报 L=2.5343、R=0.9026，
与自由拟合参数不一致（2.5343 实际对应 B6/B7 网格点 N=6.9, D=10, Q=0.8）。
本脚本用参数直接复算 L、R、M、ε，并检验 ρ_N/ρ_D 的分母符号稳定性。
"""
import os
import numpy as np
import pandas as pd
from scipy import optimize, stats

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
d6 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment.csv'))
d7 = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment_expanded.csv'))
df = pd.concat([d6, d7]).drop_duplicates(subset=['N_params_B', 'D_tokens_B', 'Q_score'])
N = df.N_params_B.values.astype(float)
D = df.D_tokens_B.values.astype(float)
Q = df.Q_score.values.astype(float)
L = df.val_loss.values.astype(float)

P = dict(E=1.631681, A=0.639571, a=0.282713, B=1.426335, b=0.299794,
         rN=0.349709, rD=0.130121, E1=0.115652)


def model(th, N, D, Q):
    E, A, a, Bc, b, rN, rD, E1 = th
    return E + A * N ** -a * np.exp(-rN * Q) + Bc * D ** -b * np.exp(-rD * Q) - E1 * Q


def resid(th):
    return model(th, N, D, Q) - L


TH0 = [P['E'], P['A'], P['a'], P['B'], P['b'], P['rN'], P['rD'], P['E1']]
TH = optimize.least_squares(resid, TH0, method='lm').x
print('全样本重拟合参数：', np.round(TH, 6))
print('  (与文档 §2.3 报告值对照)')

E, A, a, Bc, b, rN, rD, E1 = TH


def Lf(n, dd, q):
    return E + A * n ** -a * np.exp(-rN * q) + Bc * dd ** -b * np.exp(-rD * q) - E1 * q


print('\n' + '=' * 84)
print('§2.7 代表点复算  (N=1.3B, D=50B, Q=0.7, λ_p=0)')
print('=' * 84)
for (n, dd, q) in [(1.3, 50, 0.7), (1.0, 50, 0.7), (1.3, 150, 0.7), (1.3, 50, 0.5)]:
    v = Lf(n, dd, q)
    R = v - E
    dLdN = -a * A * n ** (-a - 1) * np.exp(-rN * q)
    dLdD = -b * Bc * dd ** (-b - 1) * np.exp(-rD * q)
    dLdQ = -rN * A * n ** -a * np.exp(-rN * q) - rD * Bc * dd ** -b * np.exp(-rD * q) - E1
    eN = -(a * A * n ** -a * np.exp(-rN * q)) / R
    eD = -(b * Bc * dd ** -b * np.exp(-rD * q)) / R
    eQ = -(q / R) * (rN * A * n ** -a * np.exp(-rN * q)
                     + rD * Bc * dd ** -b * np.exp(-rD * q) + E1)
    print(f'\n  N={n}B D={dd}B Q={q}:  L={v:.4f}   R=L-E={R:.4f}')
    print(f'    M_N={-dLdN:.4f}  M_D={-dLdD:.4f}  M_Q={-dLdQ:.4f}')
    print(f'    ε_N={eN:.4f}  ε_D={eD:.4f}  ε_Q={eQ:.4f}  ∂lnR/∂Q={(1/R)*dLdQ:.4f}')
    print(f'    |dN/dQ|={abs(dLdQ/dLdN):.4f}  ε_Q/ε_N={eQ/eN:.4f}')

print('\n' + '=' * 84)
print('P1：ρ_N/ρ_D 比值的稳定性（分母 ρ_D 的 bootstrap 分布是否跨 0）')
print('=' * 84)
rng = np.random.default_rng(20260924)
n = len(N)
bn = []
for _ in range(600):
    idx = rng.integers(0, n, n)
    try:
        t = optimize.least_squares(lambda th: model(th, N[idx], D[idx], Q[idx]) - L[idx],
                                   TH, method='lm').x
        bn.append(t)
    except Exception:
        pass
bn = np.array(bn)
rN_b, rD_b = bn[:, 5], bn[:, 6]
ratio = rN_b / rD_b
print(f'  bootstrap 次数: {len(bn)}')
print(f'  ρ_N: 中位={np.median(rN_b):.4f}  95%CI=[{np.percentile(rN_b,2.5):.4f},{np.percentile(rN_b,97.5):.4f}]  跨0比例={np.mean(rN_b<=0):.3f}')
print(f'  ρ_D: 中位={np.median(rD_b):.4f}  95%CI=[{np.percentile(rD_b,2.5):.4f},{np.percentile(rD_b,97.5):.4f}]  跨0比例={np.mean(rD_b<=0):.3f}')
print(f'  ρ_N/ρ_D: 中位={np.median(ratio):.4f}  95%CI=[{np.percentile(ratio,2.5):.4f},{np.percentile(ratio,97.5):.4f}]')
print(f'  ρ_N/ρ_D 为负（分母符号翻转）的比例={np.mean(ratio<0):.3f}，'
      f'|ratio|>10 的比例={np.mean(np.abs(ratio)>10):.3f}')
# 分母为负子样本的比值分布
neg = ratio[rD_b <= 0]
print(f'  当 ρ_D<=0 时（{len(neg)} 次）比值范围=[{neg.min():.2f},{neg.max():.2f}]')
pos = ratio[rD_b > 0]
print(f'  当 ρ_D>0  时（{len(pos)} 次）比值范围=[{pos.min():.2f},{pos.max():.2f}]，'
      f'中位={np.median(pos):.3f}，95%CI=[{np.percentile(pos,2.5):.3f},{np.percentile(pos,97.5):.3f}]')
