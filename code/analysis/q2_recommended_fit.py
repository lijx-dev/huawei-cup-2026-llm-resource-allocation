# -*- coding: utf-8 -*-
"""推荐模型（通道—加性框架）的完整定参：参数、bootstrap、留出、弹性、等损失替代率
推荐形式：
    L = E + A*N^(-a)*exp(-rho_N*Q) + B*D^(-b)*exp(-rho_D*Q) + E1*(1-Q)
（等价于以 Q*=0 为归一化参考点的写法；Q* 本身不可识别，详见报告）
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import stats

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))

N1 = b1.N_params_B.values.astype(float); D1 = b1.D_tokens_B.values.astype(float)
L1 = b1.val_loss.values.astype(float)
b67 = pd.concat([b6, b7], ignore_index=True).drop_duplicates(
    subset=['N_params_B', 'D_tokens_B', 'Q_score'])
N_, D_, Q_, L_ = (b67.N_params_B.values.astype(float), b67.D_tokens_B.values.astype(float),
                  b67.Q_score.values.astype(float), b67.val_loss.values.astype(float))
n = len(L_)
print(f'B1 n={len(L1)};  B6∪B7 去重后 n={n}')

# ---------- 1) B1 经典基准 ----------
def m0(p, N, D):
    E, A, a, B, b = p
    return E + A * N ** (-a) + B * D ** (-b)

r0 = least_squares(lambda p: m0(p, N1, D1) - L1, x0=[1.7, .35, .34, 1.24, .28],
                   bounds=([.1, 1e-8, 1e-3, 1e-8, 1e-3], [4, 1e8, 3, 1e8, 3]), max_nfev=200000)
E0, A0, a0, B0, b0 = r0.x
print(f'\n[B1 基准·真实] E={E0:.4f} A={A0:.4f} alpha={a0:.4f} B={B0:.4f} beta={b0:.4f} '
      f'RMSE={np.sqrt(np.mean((m0(r0.x,N1,D1)-L1)**2)):.6f}')

# ---------- 2) 推荐模型 ----------
def rec(p, N, D, Q):
    E, A, a, B, b, rN, rD, E1 = p
    return E + A * N ** (-a) * np.exp(-rN * Q) + B * D ** (-b) * np.exp(-rD * Q) + E1 * (1 - Q)

lo = [0.2, 1e-8, 1e-3, 1e-8, 1e-3, 1e-3, 1e-3, 1e-3]
hi = [4.0, 1e8, 3.0, 1e8, 3.0, 10.0, 10.0, 10.0]
p0 = [1.56, 0.51, 0.2643, 1.21, 0.2489, 0.35, 0.13, 0.116]
res = least_squares(lambda p: rec(p, N_, D_, Q_) - L_, p0, bounds=(lo, hi), max_nfev=200000)
x = res.x
rss = np.sum(res.fun ** 2)
rmse = np.sqrt(rss / n)
k = 8
print('\n' + '=' * 78)
print('[推荐模型·半合成校准] L = E + A N^-a exp(-rN Q) + B D^-b exp(-rD Q) + E1(1-Q)')
print('=' * 78)
names = ['E', 'A', 'alpha', 'B', 'beta', 'rho_N', 'rho_D', 'E1']
for nm, v in zip(names, x):
    print(f'  {nm:<7}= {v:.6f}')
print(f'  RMSE = {rmse:.6f}   RSS = {rss:.6f}   AIC = {n*np.log(rss/n)+2*k:.2f}')

# ---------- 3) bootstrap ----------
rng = np.random.default_rng(20260924)
B_ = 400
boot = []
for _ in range(B_):
    idx = rng.integers(0, n, n)
    try:
        rb = least_squares(lambda p: rec(p, N_[idx], D_[idx], Q_[idx]) - L_[idx], x,
                           bounds=(lo, hi), max_nfev=20000)
        boot.append(rb.x)
    except Exception:
        pass
boot = np.array(boot)
print(f'\nbootstrap 成功 {len(boot)}/{B_} 次，95% 百分位区间：')
for i, nm in enumerate(names):
    lo_, hi_ = np.percentile(boot[:, i], [2.5, 97.5])
    sig = '显著非零' if (lo_ > 0 or hi_ < 0) else '含 0'
    print(f'  {nm:<7} = {x[i]:+.4f}  [{lo_:+.4f}, {hi_:+.4f}]  {sig}')
print(f'  rho_N/rho_D = {x[5]/x[6]:.3f}')

# ---------- 4) 留出验证：留一 (N,D) 组 ----------
print('\n' + '=' * 78)
print('[留出验证] 留一 (N,D) 组（每组含 10 个 Q 值），预测该组全部点')
print('=' * 78)
groups = list(b67.groupby(['N_params_B', 'D_tokens_B']))
errs = []
for i in range(len(groups)):
    tr = pd.concat([g for j, (_, g) in enumerate(groups) if j != i])
    te = groups[i][1]
    try:
        rr = least_squares(lambda p: rec(p, tr.N_params_B.values.astype(float),
                                         tr.D_tokens_B.values.astype(float),
                                         tr.Q_score.values.astype(float)) - tr.val_loss.values.astype(float),
                           x, bounds=(lo, hi), max_nfev=20000)
        pr = rec(rr.x, te.N_params_B.values.astype(float), te.D_tokens_B.values.astype(float),
                 te.Q_score.values.astype(float))
        errs.append(np.sqrt(np.mean((pr - te.val_loss.values.astype(float)) ** 2)))
    except Exception:
        pass
print(f'  组数 = {len(groups)}，留一组 RMSE 中位 = {np.median(errs):.6f}，均值 = {np.mean(errs):.6f}')

# ---------- 5) B6 校准 → B7 新增外推 ----------
key = ['N_params_B', 'D_tokens_B', 'Q_score']
m = b6.merge(b7, on=key, how='inner', suffixes=('_6', '_7'))
new7 = b7[~b7.set_index(key).index.isin(b6.set_index(key).index)].copy()
print('\n' + '=' * 78)
print(f'[迁移检验] B6 校准(n={len(b6)}) → B7 新增 {len(new7)} 点（Q=0.5,0.7 内插）')
print('=' * 78)
r6 = least_squares(lambda p: rec(p, b6.N_params_B.values.astype(float),
                                 b6.D_tokens_B.values.astype(float),
                                 b6.Q_score.values.astype(float)) - b6.val_loss.values.astype(float),
                   x, bounds=(lo, hi), max_nfev=40000)
pr = rec(r6.x, new7.N_params_B.values.astype(float), new7.D_tokens_B.values.astype(float),
         new7.Q_score.values.astype(float))
print(f'  B6 自拟合 RMSE = {np.sqrt(np.mean((rec(r6.x, b6.N_params_B.values.astype(float), b6.D_tokens_B.values.astype(float), b6.Q_score.values.astype(float))-b6.val_loss.values.astype(float))**2)):.6f}')
print(f'  B7 新增 RMSE  = {np.sqrt(np.mean((pr-new7.val_loss.values.astype(float))**2)):.6f}')

# ---------- 6) 振幅指数（模型隐含 vs 实测） ----------
print('\n' + '=' * 78)
print('[模型无关判据核对] 质量振幅 A(N,D)=max_Q L - min_Q L 的 log-log 指数')
print('=' * 78)
Ns, Ds = np.array(sorted(b67.N_params_B.unique())), np.array(sorted(b67.D_tokens_B.unique()))
amp = np.array([[b67[(b67.N_params_B == nv) & (b67.D_tokens_B == d)].val_loss.max() -
                 b67[(b67.N_params_B == nv) & (b67.D_tokens_B == d)].val_loss.min() for d in Ds]
                for nv in Ns])
Y = np.log(amp.ravel())
X = np.column_stack([np.ones(amp.size), np.repeat(np.log(Ns), len(Ds)), np.tile(np.log(Ds), len(Ns))])
be, *_ = np.linalg.lstsq(X, Y, rcond=None)
print(f'  实测:  logA = {be[0]:.4f} {be[1]:+.4f} logN {be[2]:+.4f} logD')
print(f'  → 实测 N 指数 {be[1]:+.4f}, D 指数 {be[2]:+.4f}')
# 模型隐含
E_, A_, a_, Bm, b_, rN_, rD_, E1_ = x
amp_pred = (A_ * Ns[:, None] ** (-a_) * (np.exp(-rN_ * Q_.min()) - np.exp(-rN_ * Q_.max()))
            + Bm * Ds[None, :] ** (-b_) * (np.exp(-rD_ * Q_.min()) - np.exp(-rD_ * Q_.max()))
            + E1_ * (Q_.max() - Q_.min()))
Yp = np.log(amp_pred.ravel())
bp, *_ = np.linalg.lstsq(X, Yp, rcond=None)
print(f'  模型隐含: logA = {bp[0]:.4f} {bp[1]:+.4f} logN {bp[2]:+.4f} logD')
print(f'  → 偏差: N {bp[1]-be[1]:+.4f}, D {bp[2]-be[2]:+.4f}')

# ---------- 7) 边际效用与弹性 ----------
print('\n' + '=' * 78)
print('[边际效用与弹性] 代表点 N=1.3B, D=50B, Q=0.7')
print('=' * 78)
Nv, Dv, Qv = 1.3, 50.0, 0.7
eN, eD = np.exp(-rN_ * Qv), np.exp(-rD_ * Qv)
Lv = E_ + A_ * Nv ** (-a_) * eN + Bm * Dv ** (-b_) * eD + E1_ * (1 - Qv)
dLdN = -a_ * A_ * Nv ** (-a_ - 1) * eN
dLdD = -b_ * Bm * Dv ** (-b_ - 1) * eD
dLdQ = -rN_ * A_ * Nv ** (-a_) * eN - rD_ * Bm * Dv ** (-b_) * eD - E1_
R = Lv - E_
print(f'  L={Lv:.5f}  E={E_:.5f}  R=L-E={R:.5f}')
print(f'  M_N=-dL/dN={-dLdN:.6f}   M_D=-dL/dD={-dLdD:.6f}   M_Q=-dL/dQ={-dLdQ:.6f}')
print(f'  eps_N={dLdN*Nv/R:+.5f}   eps_D={dLdD*Dv/R:+.5f}   eps_Q={dLdQ*Qv/R:+.5f}')
print(f'  半弹性 dlnR/dQ = {dLdQ/R:+.5f}')
dNdQ = (dLdQ) / (a_ * A_ * Nv ** (-a_ - 1) * eN)   # = -(dL/dQ)/(dL/dN)，dL/dN 已含负号
print(f'  等损失替代率 dN/dQ|_{{L,D}} = {dNdQ:+.5f}  (Q 提高 1 单位可补偿的参数量，单位 B)')
print(f'  等价地: Q 提高 0.1 可补偿 {abs(dNdQ)*0.1:.4f} B 参数量（相对 N={Nv}B 为 {abs(dNdQ)*0.1/Nv*100:.2f}%）')
print(f'  弹性比 |eps_Q/eps_N| = {abs((dLdQ*Qv/R)/(dLdN*Nv/R)):.3f}（应与上式一致）')

# ---------- 8) 真实数据锚定版 ----------
print('\n' + '=' * 78)
print('[真实数据锚定版] 固定 E,alpha,beta 为 B1 值，只放开 A,B,rho_N,rho_D,E1')
print('=' * 78)
def rec_anchor(p, N, D, Q):
    A, B, rN, rD, E1 = p
    return E0 + A * N ** (-a0) * np.exp(-rN * Q) + B * D ** (-b0) * np.exp(-rD * Q) + E1 * (1 - Q)
ra = least_squares(lambda p: rec_anchor(p, N_, D_, Q_) - L_,
                   [0.5, 1.2, 0.35, 0.13, 0.116],
                   bounds=([1e-8, 1e-8, 1e-3, 1e-3, 1e-3], [1e8, 1e8, 10, 10, 10]), max_nfev=200000)
print(f'  A={ra.x[0]:.5f}  B={ra.x[1]:.5f}  rho_N={ra.x[2]:.5f}  rho_D={ra.x[3]:.5f}  E1={ra.x[4]:.5f}')
print(f'  RMSE = {np.sqrt(np.mean(ra.fun**2)):.6f}   AIC = {n*np.log(np.sum(ra.fun**2)/n)+2*5:.2f}')
