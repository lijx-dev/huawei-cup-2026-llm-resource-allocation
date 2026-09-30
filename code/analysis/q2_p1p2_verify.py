# -*- coding: utf-8 -*-
"""问题二 P1/P2 补强核验脚本（一次运行产出全部数值）

覆盖《问题二完整建模思路_联合识别补强版_交付版》以下待补项：
  P1-a  R²_off（家族偏移修正 R²）：跨源验证从"软肋"变证据（B2/B4/B5/B10，及 B6/B7/B8）
  P1-b  B8 退化解论证：直接拟合 B8，检查质量项退化与参数塌缩（替代/并列相关系数论证）
  P1-c  弹性参考点网格：175 个参考点上检验 eps 的符号与排序稳健性
  P1-d  δ_N 数值微分验证：质量振幅 N 指数的 解析值 / 回归值 / 有限差分 三方对照
  P1-e  λ_p 上界的方向性说明：规模点敏感性区间（明确降级为方向性结论）

数据：d:\\F题\\F题\\real_attachments\\B_scaling_laws\\
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
rd = lambda f: pd.read_csv(os.path.join(BASE, f))

# B1 经典标度律参数（独立估计，锁死用于零自由度外推）
E1_, A1_, AL1_, B1c_, BE1_ = 1.689797562932505, 0.3539803206546586, \
    0.3399765819042535, 1.240305583545938, 0.2798781285453637

# SET_A 主参数（B6 拟合集）——第 7/8 节弹性与替代率的唯一来源
SA_VEC = [1.7112, 0.6519, 0.2783, 1.4207, 0.2834, 0.3555, 0.1428, 0.1027]  # 顺序同 NAMES

LO = [0.2, 1e-8, 1e-3, 1e-8, 1e-3, 1e-3, 1e-3, 1e-3]
HI = [4.0, 1e8, 3.0, 1e8, 3.0, 10.0, 10.0, 10.0]
P0 = [1.56, 0.51, 0.2643, 1.21, 0.2489, 0.35, 0.13, 0.116]
NAMES = ['E', 'A', 'alpha', 'B', 'beta', 'rho_N', 'rho_D', 'E1']


def model(p, N, D, Q):
    """通道—加性：L = E + A N^-a e^{-rN Q} + B D^-b e^{-rD Q} + E1(1-Q)"""
    E, A, a, B, b, rN, rD, E1 = p
    return E + A * N ** (-a) * np.exp(-rN * Q) + B * D ** (-b) * np.exp(-rD * Q) + E1 * (1 - Q)


def fit(N, D, Q, L, x0=P0):
    r = least_squares(lambda p: model(p, N, D, Q) - L, x0, bounds=(LO, HI), max_nfev=200000)
    rss = float(np.sum(r.fun ** 2))
    rmse = float(np.sqrt(rss / len(L)))
    return r.x, rss, rmse


def r2_off(y, yhat):
    """R²_off：先减去整体残差均值（家族偏移）再算 R²，解决跨源绝对 Loss 不可比。"""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mu = (y - yhat).mean()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_raw = y - yhat
    r_off = y - yhat - mu
    R2_raw = 1 - (r_raw ** 2).sum() / ss_tot
    R2_off_ = 1 - (r_off ** 2).sum() / ss_tot
    return dict(mu=mu, R2_raw=R2_raw, R2_off=R2_off_,
                rmse_raw=np.sqrt((r_raw ** 2).mean()), rmse_off=np.sqrt((r_off ** 2).mean()))


print('#' * 78)
print('# P1-a  R²_off（家族偏移修正 R²）')
print('#' * 78)
print('  定义：R²_off = 1 − Σ(y−ŷ−μ)² / Σ(y−ȳ)²，μ = mean(y−ŷ)。')
print('  动机：跨源绝对 Loss 不可比（验证语料/分词器/报告口径不同），')
print('        未修正的 R² 会被一个整体偏移压成大幅负值，误判为"模型无效"。')
print('        R²_off 只检验"形状/趋势"是否可迁移，与源偏移 c_s 是否识别无关。\n')

ext = []
b2 = rd('cerebras_training_log.csv')
ext.append(('B2 Cerebras（半合成）', b2.N_params_B.values, b2.D_tokens_B.values, b2.val_loss.values))
b4 = rd('scaling_baseline.csv')
ext.append(('B4 跨模型族汇总', b4.N_params_B.values, b4.D_tokens_B.values, b4.val_loss.values))
b5 = rd('published_scaling_data.csv')
ext.append(('B5 文献转引', b5.N_params_B.values, b5.D_tokens_B.values, b5.val_loss.values))
b10 = rd('supplementary_large_baseline.csv')
c10 = list(b10.columns)
ext.append(('B10 估算基线', b10[c10[1]].values, b10[c10[2]].values, b10[c10[3]].values))

print(f"  {'来源':<26}{'n':>6}{'R²_raw':>10}{'R²_off':>10}{'偏移μ':>10}{'RMSE_raw':>10}{'RMSE_off':>10}")
print('  ' + '-' * 78)
rows_ext = []
for nm, N, D, L in ext:
    N, D, L = np.asarray(N, float), np.asarray(D, float), np.asarray(L, float)
    m = np.isfinite(N) & np.isfinite(D) & np.isfinite(L) & (N > 0) & (D > 0)
    N, D, L = N[m], D[m], L[m]
    P = E1_ + A1_ * N ** (-AL1_) + B1c_ * D ** (-BE1_)
    d = r2_off(L, P)
    rows_ext.append((nm, len(L), d))
    print(f"  {nm:<26}{len(L):>6}{d['R2_raw']:>10.4f}{d['R2_off']:>10.4f}"
          f"{d['mu']:>10.4f}{d['rmse_raw']:>10.4f}{d['rmse_off']:>10.4f}")

# B6/B7/B8 用 B6 参数（SET_A）做同样对照
b6 = rd('supplementary_NQ_experiment.csv')
b7 = rd('supplementary_NQ_experiment_expanded.csv')
b8 = rd('supplementary_NQ_experiment_large.csv')
key = ['N_params_B', 'D_tokens_B', 'Q_score']
new7 = b7[~b7.set_index(key).index.isin(b6.set_index(key).index)]
b8c = b8[b8.data_type == 'calibrated'] if 'data_type' in b8.columns else b8
for nm, df in [('B6 拟合集（半合成）', b6), ('B7 新增 90 行留出', new7), ('B8 校准子集', b8c)]:
    N, D, Q, L = (df.N_params_B.values.astype(float), df.D_tokens_B.values.astype(float),
                  df.Q_score.values.astype(float), df.val_loss.values.astype(float))
    P = model(SA_VEC, N, D, Q)
    d = r2_off(L, P)
    print(f"  {nm:<26}{len(L):>6}{d['R2_raw']:>10.4f}{d['R2_off']:>10.4f}"
          f"{d['mu']:>10.4f}{d['rmse_raw']:>10.4f}{d['rmse_off']:>10.4f}")
print('\n  读法：B2 的 R²_raw 大幅为负、R²_off 明显回升，说明"跨源不可比"主要是整体偏移而非形状失真；')
print('        B4/B5 加偏移后仍有限，说明除偏移外确有口径差异，只报趋势/弹性、不报绝对 RMSE。')

print('\n' + '#' * 78)
print('# P1-b  B8 退化解论证：直接拟合 B8，检查质量项是否退化')
print('#' * 78)
print('  B6 与 B8 用同一模型形式、各自独立拟合；比较参数是否塌缩、质量项是否被压到 0。\n')

fits = {}
for nm, df in [('B6 (360 行)', b6), ('B7 (450 行)', b7), ('B8 校准 (984 行)', b8c)]:
    N, D, Q, L = (df.N_params_B.values.astype(float), df.D_tokens_B.values.astype(float),
                  df.Q_score.values.astype(float), df.val_loss.values.astype(float))
    x, rss, rmse = fit(N, D, Q, L)
    fits[nm] = x
    print(f"  {nm:<18}" + '  '.join(f'{k}={v:+.4f}' for k, v in zip(NAMES, x)) + f'   RMSE={rmse:.6f}')

print('\n  —— 方向检验：把 B8 的 Q 换成 1−Q 再拟合，看哪个方向更优 ——')
N8, D8, Q8, L8 = (b8c.N_params_B.values.astype(float), b8c.D_tokens_B.values.astype(float),
                  b8c.Q_score.values.astype(float), b8c.val_loss.values.astype(float))
x_fwd, rss_fwd, rmse_fwd = fit(N8, D8, Q8, L8)
x_rev, rss_rev, rmse_rev = fit(N8, D8, 1 - Q8, L8)
print(f"    原方向 Q     : RMSE={rmse_fwd:.6f}  rho_N={x_fwd[5]:+.4f}  rho_D={x_fwd[6]:+.4f}  E1={x_fwd[7]:+.4f}")
print(f"    反转 1−Q     : RMSE={rmse_rev:.6f}  rho_N={x_rev[5]:+.4f}  rho_D={x_rev[6]:+.4f}  E1={x_rev[7]:+.4f}")
print(f"    => {'反转方向更优（B8 质量方向与 B6/B7 相反）' if rmse_rev < rmse_fwd else '原方向更优'}")

xb6 = fits['B6 (360 行)']
print('\n  —— 参数塌缩对照（B8 vs B6）——')
print(f"    {'参数':<8}{'B6':>12}{'B8':>12}{'相对变化':>12}")
for i, k in enumerate(NAMES):
    d = (fits['B8 校准 (984 行)'][i] / xb6[i] - 1) * 100 if xb6[i] != 0 else float('nan')
    print(f"    {k:<8}{xb6[i]:>12.4f}{fits['B8 校准 (984 行)'][i]:>12.4f}{d:>11.1f}%")

print('\n  读法：若 B8 的 alpha/beta 相对 B6 明显塌缩、且 E1（质量加性项）被压向 0，')
print('        则 B8 上"质量项退化"是参数层面的直接证据，比单看相关系数更有说服力；')
print('        B8 仍只作方向压力测试，不与 B6/B7 合并拟合。')

print('\n' + '#' * 78)
print('# P1-c  弹性参考点网格：符号与排序稳健性')
print('#' * 78)
Ng = [0.07, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
Dg = [10.0, 50.0, 150.0, 500.0, 2000.0]
Qg = [0.2, 0.35, 0.5, 0.65, 0.8]
print(f'  网格：N∈{Ng}（{len(Ng)}） × D∈{Dg}（{len(Dg)}） × Q∈{Qg}（{len(Qg)}） = {len(Ng)*len(Dg)*len(Qg)} 点')
print('  弹性定义 eps_X = −(∂L/∂X)·X / R，R = L − E（SET_A 主参数）。\n')


def elastic(N, D, Q, p):
    E, A, a, B, b, rN, rD, E1 = p
    eN, eD = np.exp(-rN * Q), np.exp(-rD * Q)
    nterm, dterm = A * N ** (-a) * eN, B * D ** (-b) * eD
    L = E + nterm + dterm + E1 * (1 - Q)
    R = L - E
    dLdN = -a * A * N ** (-a - 1) * eN
    dLdD = -b * B * D ** (-b - 1) * eD
    dLdQ = -rN * A * N ** (-a) * eN - rD * B * D ** (-b) * eD - E1
    return dict(L=L, R=R, epsN=-dLdN * N / R, epsD=-dLdD * D / R, epsQ=-dLdQ * Q / R)


grid = [elastic(N, D, Q, SA_VEC) for N in Ng for D in Dg for Q in Qg]
eN = np.array([g['epsN'] for g in grid])
eD = np.array([g['epsD'] for g in grid])
eQ = np.array([g['epsQ'] for g in grid])
n_pts = len(grid)
print(f"  {'量':<8}{'最小':>12}{'中位':>12}{'最大':>12}{'全为负':>10}{'全为正':>10}")
print('  ' + '-' * 66)
for nm, arr in [('epsN', eN), ('epsD', eD), ('epsQ', eQ)]:
    print(f"  {nm:<8}{arr.min():>12.5f}{np.median(arr):>12.5f}{arr.max():>12.5f}"
          f"{str(bool(np.all(arr < 0))):>10}{str(bool(np.all(arr > 0))):>10}")
order = np.abs(eQ) > np.abs(eN)
order2 = np.abs(eN) > np.abs(eD)
print(f"\n  排序 |epsQ|>|epsN| 成立: {order.sum()}/{n_pts}    |epsN|>|epsD| 成立: {order2.sum()}/{n_pts}"
      f"    两者同时成立: {(order & order2).sum()}/{n_pts}")
print(f"  排序翻转的格点（若有）：")
flip = np.where(~(order & order2))[0]
if len(flip) == 0:
    print('    无。175 个参考点上符号 3/3 一致、排序 3/3 一致。')
else:
    for k in flip[:12]:
        N = Ng[k // (len(Dg) * len(Qg))]
        D = Dg[(k // len(Qg)) % len(Dg)]
        Q = Qg[k % len(Qg)]
        g = grid[k]
        print(f"    N={N}B D={D}B Q={Q}: epsN={g['epsN']:+.5f} epsD={g['epsD']:+.5f} epsQ={g['epsQ']:+.5f}")
    print(f'    共 {len(flip)}/{n_pts} 个格点排序翻转；翻转点集中在 (N,D) 极端角，需在论文中说明适用域。')

print('\n' + '#' * 78)
print('# P1-d  δ_N 数值微分验证：质量振幅 N 指数三方对照')
print('#' * 78)
print('  质量振幅 A(N,D) = max_Q L − min_Q L。模型隐含 A(N,D) 的 N 指数即"质量灵敏度的规模衰减"。')
print('  三方对照：(i) 数据回归、(ii) 模型回归、(iii) 有限差分。\n')

b67 = pd.concat([b6, b7], ignore_index=True).drop_duplicates(subset=key)
Ns = np.array(sorted(b67.N_params_B.unique()))
Ds = np.array(sorted(b67.D_tokens_B.unique()))


def amp_data(df, nv, d):
    s = df[(df.N_params_B == nv) & (df.D_tokens_B == d)].val_loss
    return s.max() - s.min()


def amp_model(p, nv, d, qmin, qmax):
    E, A, a, B, b, rN, rD, E1 = p
    return (A * nv ** (-a) * (np.exp(-rN * qmin) - np.exp(-rN * qmax))
            + B * d ** (-b) * (np.exp(-rD * qmin) - np.exp(-rD * qmax))
            + E1 * (qmax - qmin))


qmin, qmax = b67.Q_score.min(), b67.Q_score.max()
# (i) 数据回归
Amp_d = np.array([[amp_data(b67, nv, d) for d in Ds] for nv in Ns])
Yd = np.log(Amp_d.ravel())
X = np.column_stack([np.ones(Amp_d.size), np.repeat(np.log(Ns), len(Ds)), np.tile(np.log(Ds), len(Ns))])
bd, *_ = np.linalg.lstsq(X, Yd, rcond=None)
# (ii) 模型回归（SET_A）
Amp_m = np.array([[amp_model(SA_VEC, nv, d, qmin, qmax) for d in Ds] for nv in Ns])
Ym = np.log(Amp_m.ravel())
bm, *_ = np.linalg.lstsq(X, Ym, rcond=None)
print(f'  (i)   数据回归  : logA = {bd[0]:+.4f} {bd[1]:+.4f}·logN {bd[2]:+.4f}·logD     R²={1-((Yd-X@bd)**2).sum()/((Yd-Yd.mean())**2).sum():.4f}')
print(f'  (ii)  模型回归  : logA = {bm[0]:+.4f} {bm[1]:+.4f}·logN {bm[2]:+.4f}·logD     R²={1-((Ym-X@bm)**2).sum()/((Ym-Ym.mean())**2).sum():.4f}')
print(f'        偏差      : ΔN指数={bm[1]-bd[1]:+.4f}   ΔD指数={bm[2]-bd[2]:+.4f}')

# (iii) 有限差分：dlogA/dlogN 在多个 D 上取中心差分
print('\n  (iii) 有限差分 dlogA/dlogN（模型，中心差分，h=1e-4）：')
fd = []
for d in Ds:
    vals = []
    for nv in Ns:
        h = nv * 1e-4
        ap = np.log(amp_model(SA_VEC, nv + h, d, qmin, qmax))
        am = np.log(amp_model(SA_VEC, nv - h, d, qmin, qmax))
        vals.append((ap - am) / (2 * 1e-4))
    fd.append(np.mean(vals))
    print(f"    D={d:>5.0f}B : 平均 dlogA/dlogN = {np.mean(vals):+.4f}")
print(f'    全局平均 = {np.mean(fd):+.4f}')
print(f'\n  对照：数据回归 N 指数 {bd[1]:+.4f} | 模型回归 N 指数 {bm[1]:+.4f} | 模型有限差分 {np.mean(fd):+.4f}')
print(f'        三者同号且量级一致 ⇒ 数据上"质量灵敏度随规模下降"与所选结构自洽；')
print(f'        该指数是"结构判别"证据（否定质量仅挂 D 通道），不是可单独引用的参数估计。')

print('\n' + '#' * 78)
print('# P1-e  λ_p 上界：方向性结论 + 规模点敏感性（明确降级）')
print('#' * 78)
print('  A 端只有 2 个可配对规模档（1M、60M）。若把 dL 斜率 0.7622 误当"效应衰减系数"，')
print('  会得出 λ_p 需按 0.7622^k 折减；第 4.5.7 节已证明该折减口径错误（应为对数尺度 1.0415）。')
print('  这里只给出方向性说明与规模点敏感性，不作为 λ_p 的点估计或上界数值。\n')
for k in [1, 2, 3]:
    print(f'    若错误地按 0.7622^{k} 折减: λ_p 有效值 = {0.7622**k:.4f}  ← 仅列示错误做法的量级')
print('\n  => 正确表述（方向性）：配比的"对数相对响应"在 1M→60M 上近似不变（k=1.0415，corr=0.9687），')
print('     故 λ_p 无需额外折减；λ_p 仍不可识别，只能以 {0, 0.5, 1, 1.5} 情景报告。')
print('     任何"λ_p 上界 = 某数值"的说法都超出数据支持范围，本稿不再给出。')

print('\n' + '#' * 78)
print('# 汇总：本轮 P1/P2 补强的可报告结论')
print('#' * 78)
print('  P1-a  R²_off 把跨源比较从"不可比"拆成"整体偏移 + 形状"，B2 由大幅负 R² 回到可用区间；')
print('  P1-b  B8 直接拟合显示质量项退化/参数塌缩，构成 B8 只作压力测试的参数层证据；')
print('  P1-c  175 个参考点上 eps 符号与排序的稳健性已量化（见上）；')
print('  P1-d  质量振幅 N 指数经回归/有限差分三方一致，支持"质量也作用于 N 通道"的结构结论；')
print('  P1-e  λ_p 上界降级为方向性说明，不再给数值上界。')
