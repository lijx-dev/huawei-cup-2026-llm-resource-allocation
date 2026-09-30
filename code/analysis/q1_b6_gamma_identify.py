# -*- coding: utf-8 -*-
"""审阅用：B6 质量项 gamma_Q 的可识别性（论文 5.4.2 报 gamma_Q=0.317893, RMSE 0.076087）
   做法：在 gamma 的网格上，固定 gamma 并重拟合其余 5 个参数，画出 RMSE(gamma) 曲线。
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

b6 = pd.read_csv(r'd:\F题\F题\real_attachments\B_scaling_laws\supplementary_NQ_experiment.csv')
N, D, Q, L = (b6[c].values for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
Q0 = 0.5


def rmse_at_gamma(g):
    def r(th):
        E, A, a, B, b = np.exp(th)
        return E + A * N ** -a + B * D ** -b * np.exp(g * (Q - Q0)) - L
    best = np.inf
    for s in range(12):
        rng = np.random.default_rng(s)
        th0 = np.log([1.6, 0.5, 0.28, 1.3, 0.28]) + rng.normal(0, 0.4, 5)
        sol = least_squares(r, th0, method='lm', max_nfev=200000)
        if sol.cost < best:
            best = sol.cost
            bp = np.exp(sol.x)
    return np.sqrt(2 * best / len(L)), bp


print('gamma 网格扫描（固定 gamma，重拟合 E,A,alpha,B,beta）')
print(f"{'gamma':>9}{'RMSE':>11}   {'E':>8}{'A':>8}{'alpha':>8}{'B':>8}{'beta':>8}")
grid = [-1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.317893, 0.4, 0.5, 0.7, 0.9, 1.2]
for g in grid:
    r, p = rmse_at_gamma(g)
    print(f'{g:>9.6f}{r:>11.6f}   {p[0]:>8.4f}{p[1]:>8.4f}{p[2]:>8.5f}{p[3]:>8.4f}{p[4]:>8.5f}')

# 二次结构
print('\n二次质量项 L = E + A N^-a + B D^-b exp{g1(Q-Q0) + g2(Q-Q0)^2}')
def r2(th):
    E, A, a, B, b = np.exp(th[:5])
    g1, g2 = th[5], th[6]
    return E + A * N ** -a + B * D ** -b * np.exp(g1 * (Q - Q0) + g2 * (Q - Q0) ** 2) - L


best = None
for s in range(30):
    rng = np.random.default_rng(s)
    th0 = np.r_[np.log([1.6, 0.5, 0.28, 1.3, 0.28]) + rng.normal(0, 0.3, 5), rng.normal(0.4, 0.5), rng.normal(0, 0.5)]
    sol = least_squares(r2, th0, method='lm', max_nfev=200000)
    if best is None or sol.cost < best.cost:
        best = sol
E, A, a, B, b = np.exp(best.x[:5])
print(f'  E={E:.4f} A={A:.4f} alpha={a:.5f} B={B:.4f} beta={b:.5f} '
      f'g1={best.x[5]:.6f} g2={best.x[6]:.6f}  RMSE={np.sqrt(2*best.cost/len(L)):.6f}')

# 论文口径：B1 参数固定，只拟合 gamma（论文 5.4.2 报 gamma=0.869619 + 来源偏置 0.166609）
print('\nB1 参数固定、只拟合 gamma 与偏置：')
E1, A1, a1, B1_, b1 = 1.689798, 0.353980, 0.339977, 1.240306, 0.279878


def r3(th):
    g, c = th
    return E1 + c + A1 * N ** -a1 + B1_ * D ** -b1 * np.exp(g * (Q - Q0)) - L


sol = least_squares(r3, [0.3, 0.1], method='lm', max_nfev=200000)
print(f'  gamma={sol.x[0]:.6f}  偏置={sol.x[1]:.6f}  RMSE={np.sqrt(2*sol.cost/len(L)):.6f}'
      f'   （论文 gamma=0.869619, 偏置=0.166609）')

print('\n结论判据：若 RMSE(gamma) 在 gamma≈0.3 附近有明显下降，则质量项可识别；'
      '若曲线在 gamma=0 处最低或近乎平坦，则 B6 上的质量项不可识别。')
