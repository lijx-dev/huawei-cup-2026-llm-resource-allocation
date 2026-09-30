# -*- coding: utf-8 -*-
"""审阅用：在 B6/B7/B8 原始附件上重拟合论文 5.2.2 的 M0/MQ 模型，核验
   (1) beta6 与 B1 的 beta=0.27988 的关系（参数塌缩）
   (2) gamma_Q 的符号与量级
   (3) Q 在固定 (N,D) 下与 Loss 的方向
   (4) 论文表5.7 的边际量能否由 B6 拟合参数复现
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(f'{B}/supplementary_NQ_experiment.csv')
b7 = pd.read_csv(f'{B}/supplementary_NQ_experiment_expanded.csv')
b8 = pd.read_csv(f'{B}/supplementary_NQ_experiment_large.csv')

print('=' * 88)
print('[0] 附件形状与 Q 的网格')
print('=' * 88)
for nm, df in [('B6', b6), ('B7', b7), ('B8', b8)]:
    print(f'  {nm}: {df.shape}  N={sorted(df.N_params_B.unique())}')
    print(f'      D={sorted(df.D_tokens_B.unique())}')
    print(f'      Q={sorted(df.Q_score.unique())}')

N6, D6, Q6, L6 = (b6[c].values.astype(float) for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
N7, D7, Q7, L7 = (b7[c].values.astype(float) for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])

print('\n' + '=' * 88)
print('[1] 固定 (N,D) 下 Q 与 Loss 的方向（B6）')
print('=' * 88)
keys = list(zip(N6, D6))
u = sorted(set(keys))
cnt_neg = cnt_pos = 0
for k in u:
    m = np.array([kk == k for kk in keys])
    if m.sum() < 2:
        continue
    r = np.corrcoef(Q6[m], L6[m])[0, 1]
    if np.isnan(r):
        continue
    cnt_neg += r < 0
    cnt_pos += r > 0
print(f'  B6 的 {len(u)} 个 (N,D) 基础组中：corr(Q,Loss) < 0 的 {cnt_neg} 组，> 0 的 {cnt_pos} 组')
print('  => Q 增大 Loss 下降（负相关），与论文“gamma_Q > 0 表示 Q 增大 Loss 下降”的编码自洽')

print('\n' + '=' * 88)
print('[2] 重拟合 M0 / MQ（论文 5.2.2 结构）')
print('=' * 88)
Q0 = 0.5


def fit_m0(N, D, L):
    def r(th):
        E, A, a, B_, b = np.exp(th)
        return E + A * N ** -a + B_ * D ** -b - L
    best = None
    for s in range(24):
        rng = np.random.default_rng(s)
        x0 = np.log([1.6, 0.35, 0.34, 1.24, 0.28]) + rng.normal(0, 0.5, 5)
        sol = least_squares(r, x0, method='lm', max_nfev=400000)
        if best is None or sol.cost < best.cost:
            best = sol
    p = np.exp(best.x)
    res = r(best.x)
    return p, np.sqrt((res ** 2).mean())


def fit_mq(N, D, Q, L, sign=+1):
    def r(th):
        E, A, a, B_, b, g = np.exp(th)
        return E + A * N ** -a + B_ * D ** -b * np.exp(sign * g * (Q - Q0)) - L
    best = None
    for s in range(24):
        rng = np.random.default_rng(s)
        x0 = np.log([1.6, 0.35, 0.34, 1.24, 0.28, 0.3]) + rng.normal(0, 0.5, 6)
        sol = least_squares(r, x0, method='lm', max_nfev=400000)
        if best is None or sol.cost < best.cost:
            best = sol
    p = np.exp(best.x)
    return p, np.sqrt((r(best.x) ** 2).mean())


p0, r0 = fit_m0(N6, D6, L6)
print(f'  M0,B6 : E={p0[0]:.5f} A={p0[1]:.5f} alpha={p0[2]:.5f} B={p0[3]:.5f} beta={p0[4]:.5f}  RMSE={r0:.5f}')
print(f'          论文表5.4: M0 RMSE=0.128762')
print(f'          B1 对照:   E=1.68980 A=0.35398 alpha=0.33998 B=1.24031 beta=0.27988')
print(f'          => beta6/beta_B1 = {p0[4] / 0.27988:.4f}   ({(p0[4] / 0.27988 - 1) * 100:+.1f}%)')

for sg, tag in [(+1, 'exp{+g(Q-Q0)}'), (-1, 'exp{-g(Q-Q0)}')]:
    pq, rq = fit_mq(N6, D6, Q6, L6, sg)
    print(f'  MQ,B6 [{tag}]: E={pq[0]:.5f} A={pq[1]:.5f} alpha={pq[2]:.5f} '
          f'B={pq[3]:.5f} beta={pq[4]:.5f} gamma={sg * pq[5]:+.6f}  RMSE={rq:.5f}   （论文 0.076087, gamma=+0.317893）')

print('\n' + '=' * 88)
print('[3] 用 B6 重拟合参数复算论文表5.7 的边际量（FormA/单通道）')
print('=' * 88)
pq, _ = fit_mq(N6, D6, Q6, L6, +1)
E_, A_, a_, B_, b_, g_ = pq


def marg(N, D, Q=Q0):
    MN = a_ * A_ * N ** (-a_ - 1)
    MD = b_ * B_ * D ** (-b_ - 1) * np.exp(g_ * (Q - Q0))
    MQ = g_ * B_ * D ** (-b_) * np.exp(g_ * (Q - Q0))
    return MN, MD, MQ


print(f"  {'工作点':<12}{'M_N':>10}{'M_D':>11}{'M_Q':>10}{'dN/dQ':>11}")
print(f"  {'论文(0.7,10)':<12}{0.2379:>10.4f}{0.01421:>11.5f}{0.4536:>10.4f}{-1.9067:>11.4f}")
print(f"  {'重拟合':<12}" + ''.join(f'{v:>10.5f}' if i < 3 else f'{v:>11.4f}'
      for i, v in enumerate(marg(0.7, 10.0))))
print(f"  {'论文(0.41,300)':<12}{0.4715:>10.4f}{0.000338:>11.6f}{0.3232:>10.4f}{-0.6855:>11.4f}")
print(f"  {'重拟合':<12}" + ''.join(f'{v:>10.5f}' if i < 3 else f'{v:>11.4f}'
      for i, v in enumerate(marg(0.41, 300.0))))
print(f"  {'论文(6.9,300)':<12}{0.01274:>10.5f}{0.000338:>11.6f}{0.3232:>10.4f}{-25.3648:>11.4f}")
print(f"  {'重拟合':<12}" + ''.join(f'{v:>10.5f}' if i < 3 else f'{v:>11.4f}'
      for i, v in enumerate(marg(6.9, 300.0))))

print('\n' + '=' * 88)
print('[4] B8 方向压力测试复核')
print('=' * 88)
N8, D8, Q8, L8 = (b8[c].values.astype(float) for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
typ = b8['data_type'].values
for t in np.unique(typ):
    m = typ == t
    keys8 = list(zip(N8[m], D8[m]))
    neg = pos = 0
    for k in sorted(set(keys8)):
        mm = np.array([kk == k for kk in keys8])
        if mm.sum() < 2:
            continue
        r = np.corrcoef(Q8[m][mm], L8[m][mm])[0, 1]
        if np.isnan(r):
            continue
        neg += r < 0
        pos += r > 0
    print(f'  B8 [{t}]: {m.sum()} 行, {len(set(keys8))} 个(N,D)组 -> corr(Q,Loss)<0 的 {neg} 组, >0 的 {pos} 组')
print('  论文 5.4.2 称 B8 校准组 90/90、外推组 60/60 组内 Spearman 均为正（与 B6 方向相反）')
