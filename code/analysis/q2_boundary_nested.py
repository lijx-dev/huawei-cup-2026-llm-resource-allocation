# -*- coding: utf-8 -*-
"""数据边界深查 + 嵌套模型检验
1. B3 是否全为插值；B2/B4/B5 的单位与D口径；B6/B7/B8 的N,D范围
2. 跨源偏移修正后的验证
3. 嵌套模型 F 检验 + bootstrap CI：Q 究竟挂哪个通道
4. 各模型预测的振幅 N/D 指数 vs 实测
"""
import pandas as pd
import numpy as np
import os, glob
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
LOGE = np.log

print('=' * 100)
print('1) B3 轨迹：interpolated 标记语义核查')
print('=' * 100)
f = os.path.join(BASE, 'training_trajectories', 'pythia_0.070542B_trajectory.csv')
d = pd.read_csv(f)
print(d.head(8).to_string())
print('...')
print(d.tail(4).to_string())
print(f'interpolated 取值分布: {d.interpolated.value_counts().to_dict()}')
print(f'step 范围 {d.step.min()}~{d.step.max()}, D 范围 {d.D_tokens_B.min():.4f}~{d.D_tokens_B.max():.4f}')
print(f'D 是否单调: {d.D_tokens_B.is_monotonic_increasing}')
# 与B1同N的点对照
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
sub = b1[b1.N_params_B.round(6) == 0.070542]
print(f'\nB1 中 N=0.070542 的 D 点数={len(sub)}, D范围 {sub.D_tokens_B.min():.4f}~{sub.D_tokens_B.max():.4f}')
print(f'B1 该N下的 (steps,D,val_loss) 前6行:')
print(sub[['steps', 'D_tokens_B', 'val_loss']].head(6).to_string())
print(f'轨迹前6行 (step,D,val_loss):')
print(d[['step', 'D_tokens_B', 'val_loss']].head(6).to_string())
# 合并核对：轨迹点是否落在B1点之间
m = pd.merge(d, sub, on=['D_tokens_B'], how='inner', suffixes=('_tr', '_b1'))
print(f'\n轨迹与B1在相同D上的重合点数={len(m)}')
if len(m):
    print(m[['D_tokens_B', 'val_loss_tr', 'val_loss_b1']].head(10).to_string())
    print(f'重合点 val_loss 最大差={np.abs(m.val_loss_tr-m.val_loss_b1).max():.6f}')
print(f'\n轨迹 D 步长（前10个差）: {np.diff(d.D_tokens_B.values)[:10]}')
print(f'轨迹 step 步长（前10个差）: {np.diff(d.step.values)[:10]}')
print('  → 若 interpolated=1 表示该行为插值生成，则轨迹不能作为独立观测证据')

print()
print('=' * 100)
print('2) B1/B2/B6/B7/B8 的 N、D 覆盖对照（单位与口径）')
print('=' * 100)
for tag, fn in [('B1', 'pythia_training_log_existing.csv'), ('B2', 'cerebras_training_log.csv'),
                ('B6', 'supplementary_NQ_experiment.csv'), ('B7', 'supplementary_NQ_experiment_expanded.csv'),
                ('B8', 'supplementary_NQ_experiment_large.csv'), ('B10', 'supplementary_large_baseline.csv')]:
    df = pd.read_csv(os.path.join(BASE, fn))
    print(f'  {tag}: n={len(df):>5}  N {df.N_params_B.min():>9.4f}~{df.N_params_B.max():>9.2f}B   '
          f'D {df.D_tokens_B.min():>9.4f}~{df.D_tokens_B.max():>9.2f}B   '
          f'N唯一={df.N_params_B.nunique():>3} D唯一={df.D_tokens_B.nunique():>4}'
          + (f'  loss {df.val_loss.min():.3f}~{df.val_loss.max():.3f}' if 'val_loss' in df else ''))
print()
b2 = pd.read_csv(os.path.join(BASE, 'cerebras_training_log.csv'))
print('B2 与 B1 在相同 (N,D) 上的 val_loss 直接对照（同(N,D)是否同loss）:')
k1 = b1.set_index(['N_params_B', 'D_tokens_B']).val_loss
k2 = b2.set_index(['N_params_B', 'D_tokens_B']).val_loss
common = k1.index.intersection(k2.index)
print(f'  共同 (N,D) 数={len(common)}')
if len(common):
    diff = (k2.loc[common] - k1.loc[common])
    print(f'  loss 差: 均值={diff.mean():+.4f}, 中位={diff.median():+.4f}, 范围 {diff.min():+.4f}~{diff.max():+.4f}')
else:
    print('  无共同 (N,D) → 无法直接对照，只能比趋势')

print()
print('=' * 100)
print('3) 跨源偏移修正后的验证（识别问题2 的处理方式）')
print('=' * 100)
def fit_anchor(df):
    N, D, L = df.N_params_B.values.astype(float), df.D_tokens_B.values.astype(float), df.val_loss.values.astype(float)
    r = least_squares(lambda p: p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4])) - L,
                      [1.5, LOGE(0.35), LOGE(0.34), LOGE(1.24), LOGE(0.28)],
                      bounds=([0, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                              [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]), max_nfev=80000)
    return r.x[0], np.exp(r.x[1]), np.exp(r.x[2]), np.exp(r.x[3]), np.exp(r.x[4])
E1, A1, al1, B1_, be1 = fit_anchor(b1)
print(f'  B1 锚点: E={E1:.4f} A={A1:.4f} alpha={al1:.4f} B={B1_:.4f} beta={be1:.4f}')
print()
print('  逐源：允许 1 个源偏移 c（L_pred = c + E + A N^-a + B D^-b），其余参数全部锁定为 B1 值')
for tag, fn in [('B2', 'cerebras_training_log.csv'), ('B4', 'scaling_baseline.csv'),
                ('B5', 'published_scaling_data.csv'), ('B10', 'supplementary_large_baseline.csv')]:
    df = pd.read_csv(os.path.join(BASE, fn))
    N, D, L = df.N_params_B.values.astype(float), df.D_tokens_B.values.astype(float), df.val_loss.values.astype(float)
    base = E1 + A1*N**(-al1) + B1_*D**(-be1)
    c = np.mean(L - base)
    e = L - base - c
    print(f'  {tag}: 偏移 c={c:+.4f}  → 修正后 RMSE={np.sqrt(np.mean(e**2)):.4f}, '
          f'MAE={np.mean(np.abs(e)):.4f}, R²={1-np.sum(e**2)/np.sum((L-L.mean())**2):.4f}')
print('  → 加一个源偏移后 B4/B5 残差与 B1 同量级；B2 仍偏大，说明其 D 口径/数据分布差异不止一个常数')
print()
print('  B5 逐文献来源偏移:')
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
for s, sub in b5.groupby('source'):
    N, D, L = sub.N_params_B.values.astype(float), sub.D_tokens_B.values.astype(float), sub.val_loss.values.astype(float)
    base = E1 + A1*N**(-al1) + B1_*D**(-be1)
    c = np.mean(L - base); e = L - base - c
    print(f'    {s:<24} c={c:+.4f}  修正后 RMSE={np.sqrt(np.mean(e**2)):.4f}')

print()
print('=' * 100)
print('4) 嵌套模型：Q 挂哪个通道（含 F 检验与 bootstrap）')
print('=' * 100)
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
N7, D7, Q7, L7 = (b7.N_params_B.values.astype(float), b7.D_tokens_B.values.astype(float),
                  b7.Q_score.values.astype(float), b7.val_loss.values.astype(float))
n = len(L7)

def M0(p, N, D, Q):   # 无质量项（基线）
    return p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))
def M_N(p, N, D, Q):  # 只挂N
    return p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q) + np.exp(p[3])*D**(-np.exp(p[4]))
def M_D(p, N, D, Q):  # 只挂D
    return p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[5]*Q)
def M_E(p, N, D, Q):  # 只改E（有界下限）
    return p[0] + p[5]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))
def M_ND(p, N, D, Q): # 双挂
    return p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q) + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q)
def M_END(p, N, D, Q):# 三项全挂
    return p[0] + p[7]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q) + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q)

base5 = [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831)]
CFG = {
 'M0 无质量项':        (M0,  base5,                    [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]),
 'M_E 改E(有界下限)':   (M_E, base5+[0.3],             [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -5],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 5]),
 'M_N 只挂N':          (M_N, base5+[1.5],             [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
 'M_D 只挂D':          (M_D, base5+[1.0],             [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
 'M_ND 双挂':          (M_ND, base5+[1.5, 1.0],       [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
 'M_END 三项全挂':      (M_END, base5+[1.5, 1.0, 0.3],[0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20, -5],
                                                      [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20, 5]),
}
res = {}
print(f'{"模型":<22}{"k":>3}{"RSS":>12}{"RMSE":>10}{"AIC":>10}{"BIC":>10}')
for nm, (f, p0, lo, hi) in CFG.items():
    r = least_squares(lambda p: f(p, N7, D7, Q7) - L7, p0, bounds=(lo, hi), max_nfev=60000)
    rss = float(np.sum(r.fun**2)); k = len(p0)
    res[nm] = (r.x, rss, k)
    print(f'{nm:<22}{k:>3}{rss:>12.6f}{np.sqrt(rss/n):>10.5f}'
          f'{n*np.log(rss/n)+2*k:>10.2f}{n*np.log(rss/n)+k*np.log(n):>10.2f}')

def ftest(m_small, m_big):
    rss_s, k_s = res[m_small][1], res[m_small][2]
    rss_b, k_b = res[m_big][1], res[m_big][2]
    df1, df2 = k_b - k_s, n - k_b
    F = ((rss_s - rss_b)/df1) / (rss_b/df2)
    from scipy.stats import f as fdist
    return F, df1, df2, 1 - fdist.cdf(F, df1, df2)

print()
print('F 检验（H0: 小模型足够）:')
for a, b, lab in [('M0 无质量项', 'M_E 改E(有界下限)', '加入质量改E'),
                  ('M0 无质量项', 'M_D 只挂D', '加入质量挂D'),
                  ('M0 无质量项', 'M_N 只挂N', '加入质量挂N'),
                  ('M_E 改E(有界下限)', 'M_END 三项全挂', '改E之上再加双挂'),
                  ('M_D 只挂D', 'M_ND 双挂', '挂D之上再加挂N'),
                  ('M_N 只挂N', 'M_ND 双挂', '挂N之上再加挂D'),
                  ('M_ND 双挂', 'M_END 三项全挂', '双挂之上再加改E')]:
    F, d1, d2, p = ftest(a, b)
    print(f'  {lab:<20} F({d1},{d2})={F:>9.3f}  p={p:.3e}  {"拒绝小模型" if p<0.01 else "不能拒绝"}')

print()
print('拟合出的质量参数:')
x = res['M_N 只挂N'][0];  print(f'  M_N 只挂N : rho_N={x[5]:.4f}  (N通道)')
x = res['M_D 只挂D'][0];  print(f'  M_D 只挂D : rho_D={x[5]:.4f}  (D通道)')
x = res['M_ND 双挂'][0];  print(f'  M_ND 双挂 : rho_N={x[5]:.4f}, rho_D={x[6]:.4f}')
x = res['M_END 三项全挂'][0]; print(f'  M_END 全挂: E1={x[7]:.4f}, rho_N={x[5]:.4f}, rho_D={x[6]:.4f}')

print()
print('bootstrap（按 (N,D) 组重抽样，120 次）:')
rng = np.random.default_rng(20240924)
groups = list(b7.groupby(['N_params_B', 'D_tokens_B']))
def boot_fit(name, idx):
    sub = pd.concat([groups[i][1] for i in idx])
    f, p0, lo, hi = CFG[name]
    r = least_squares(lambda p: f(p, sub.N_params_B.values, sub.D_tokens_B.values,
                                  sub.Q_score.values) - sub.val_loss.values,
                      res[name][0], bounds=(lo, hi), max_nfev=4000)
    return r.x
for name, cols in [('M_ND 双挂', [5, 6]), ('M_END 三项全挂', [5, 6, 7])]:
    B_ = []
    for _ in range(120):
        idx = rng.integers(0, len(groups), len(groups))
        try:
            B_.append(boot_fit(name, idx))
        except Exception:
            pass
    B_ = np.array(B_)
    print(f'  {name}:')
    for c in cols:
        nm = {5: 'rho_N', 6: 'rho_D', 7: 'E1'}[c]
        lo_, hi_ = np.percentile(B_[:, c], [2.5, 97.5])
        print(f'    {nm:>6} = {np.mean(B_[:, c]):+.4f}  95%CI [{lo_:+.4f}, {hi_:+.4f}]  '
              f'{"显著非零" if lo_*hi_ > 0 else "含0"}')

print()
print('=' * 100)
print('5) 各模型预测的振幅 log-log 指数 vs 实测')
print('=' * 100)
Ns, Ds = np.array(sorted(b7.N_params_B.unique())), np.array(sorted(b7.D_tokens_B.unique()))
obs = np.array([[np.log(b7[(b7.N_params_B == nv) & (b7.D_tokens_B == d)].val_loss.max() -
                      b7[(b7.N_params_B == nv) & (b7.D_tokens_B == d)].val_loss.min())
                 for d in Ds] for nv in Ns])
X = np.array([[1.0, np.log(nv), np.log(d)] for nv in Ns for d in Ds])
y = obs.ravel()
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f'  实测: logA = {coef[0]:.4f} {coef[1]:+.4f} logN {coef[2]:+.4f} logD')
Qmin, Qmax = Q7.min(), Q7.max()
for nm in ['M0 无质量项', 'M_E 改E(有界下限)', 'M_N 只挂N', 'M_D 只挂D', 'M_ND 双挂', 'M_END 三项全挂']:
    p = res[nm][0]
    pred = np.array([f(p, np.array([nv]), np.array([d]), np.array([Qmin]))[0] -
                     f(p, np.array([nv]), np.array([d]), np.array([Qmax]))[0]
                     for nv in Ns for d in Ds])
    pred = np.abs(pred)
    ok = pred > 1e-9
    if ok.sum() < 10:
        print(f'  {nm:<22} 预测振幅≈0（无质量效应）')
        continue
    c2, *_ = np.linalg.lstsq(X[ok], np.log(pred[ok]), rcond=None)
    print(f'  {nm:<22} 预测: logA = {c2[0]:+.4f} {c2[1]:+.4f} logN {c2[2]:+.4f} logD   '
          f'偏差(N)={c2[1]-coef[1]:+.3f}, 偏差(D)={c2[2]-coef[2]:+.3f}')
print('  → 谁预测的 N 指数（接近 -0.143）和 D 指数（接近 -0.055）更准，谁的结构更符合数据')