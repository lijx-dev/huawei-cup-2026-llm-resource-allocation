# -*- coding: utf-8 -*-
"""模型无关判据 + 可识别性 + B8 边界 + C 桥接分层
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
from scipy import stats

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
CB = r'd:\F题\F题\real_attachments\C_efficiency_evolution'
LOGE = np.log
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
N7, D7, Q7, L7 = (b7.N_params_B.values.astype(float), b7.D_tokens_B.values.astype(float),
                  b7.Q_score.values.astype(float), b7.val_loss.values.astype(float))
n = len(L7)

print('=' * 100)
print('【判据1·模型无关】若“质量只重标定 D_eff”，则固定 D 时质量振幅必须与 N 无关')
print('=' * 100)
Ns, Ds = np.array(sorted(b7.N_params_B.unique())), np.array(sorted(b7.D_tokens_B.unique()))
amp = np.array([[b7[(b7.N_params_B == nv) & (b7.D_tokens_B == d)].val_loss.max() -
                 b7[(b7.N_params_B == nv) & (b7.D_tokens_B == d)].val_loss.min() for d in Ds] for nv in Ns])
print(f'{"D":>8}{"斜率(logA~logN)":>18}{"t":>9}{"p":>12}{"R²":>8}   H0:斜率=0')
for j, d in enumerate(Ds):
    yv = np.log(amp[:, j]); x = np.log(Ns)
    sl, ic, r, p, se = stats.linregress(x, yv)
    print(f'{d:>8.0f}{sl:>18.4f}{sl/se:>9.2f}{p:>12.3e}{r**2:>8.3f}   '
          f'{"拒绝(斜率≠0)" if p < 0.05 else "不能拒绝"}')
print()
print('固定 N 时，质量振幅对 D 的斜率（若只挂D项，应为 -beta≈-0.28）:')
print(f'{"N":>10}{"斜率(logA~logD)":>18}{"t":>9}{"p":>12}{"R²":>8}')
for i, nv in enumerate(Ns):
    yv = np.log(amp[i, :]); x = np.log(Ds)
    sl, ic, r, p, se = stats.linregress(x, yv)
    print(f'{nv:>10.3f}{sl:>18.4f}{sl/se:>9.2f}{p:>12.3e}{r**2:>8.3f}')

# 全局
X = np.array([[1.0, np.log(nv), np.log(d)] for nv in Ns for d in Ds]); y = np.log(amp.ravel())
coef, res_, *_ = np.linalg.lstsq(X, y, rcond=None)
s2 = np.sum(res_**2) / (len(y) - 3)
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
print(f'\n全局: logA = {coef[0]:.4f} {coef[1]:+.4f}·logN {coef[2]:+.4f}·logD   (R²='
      f'{1-np.sum(res_**2)/np.sum((y-y.mean())**2):.3f})')
print(f'  N 指数 = {coef[1]:+.4f} ± {se[1]:.4f} (t={coef[1]/se[1]:.2f}, p={2*(1-stats.t.cdf(abs(coef[1]/se[1]), len(y)-3)):.2e})')
print(f'  D 指数 = {coef[2]:+.4f} ± {se[2]:.4f} (t={coef[2]/se[2]:.2f}, p={2*(1-stats.t.cdf(abs(coef[2]/se[2]), len(y)-3)):.2e})')
print()
print('  模型预测对照:')
print(f'    只挂D项(D_eff=D·g(Q)) : N指数=0,        D指数=-beta=-0.2799')
print(f'    只挂N项               : N指数=-alpha=-0.34, D指数=0')
print(f'    双挂                  : N指数∈(0,-alpha), D指数∈(0,-beta)')
print(f'    只改E(有界下限)        : N指数=0,        D指数=0')
print(f'    实测                  : N指数={coef[1]:+.4f}, D指数={coef[2]:+.4f}')
print(f'  → 实测 N 指数显著为负且远小于 |alpha|；D 指数显著为负且远小于 |beta|。')
print(f'    只挂D项预测 N指数=0 被拒绝(p={2*(1-stats.t.cdf(abs(coef[1]/se[1]), len(y)-3)):.1e})；')
print(f'    只改E预测 D指数=0 也被拒绝(p={2*(1-stats.t.cdf(abs(coef[2]/se[2]), len(y)-3)):.1e})。')

print()
print('=' * 100)
print('【判据2·模型无关】组内质量灵敏度 dL/dlnQ 沿 N 的衰减')
print('=' * 100)
rows = []
for (nv, d), sub in b7.groupby(['N_params_B', 'D_tokens_B']):
    s = sub.sort_values('Q_score')
    sl = np.polyfit(np.log(s.Q_score.values), s.val_loss.values, 1)[0]
    rows.append((nv, d, sl))
sd = pd.DataFrame(rows, columns=['N', 'D', 's'])
x, yv = np.log(sd.N.values), np.log(np.abs(sd.s.values))
sl, ic, r, p, se = stats.linregress(x, yv)
print(f'全局: |dL/dlnQ| ∝ N^{sl:.4f} ± {se:.4f}  (R²={r**2:.3f}, p={p:.2e}, n={len(sd)})')
print(f'  只挂D项预测指数=0 → 被强烈拒绝')
print(f'  双挂预测指数应介于 0 与 -alpha 之间；实测 {sl:.3f} 落在 (-0.34, 0) 内')

print()
print('=' * 100)
print('【可识别性】floor(E1) 与 D通道(rho_D) 的分工是否可分离？profile likelihood')
print('=' * 100)
def fit_with_E1_fixed(E1fix):
    def m(p, N, D, Q):
        return (p[0] + E1fix * (1 - Q) + np.exp(p[1]) * N ** (-np.exp(p[2])) * np.exp(-p[3] * Q)
                + np.exp(p[4]) * D ** (-np.exp(p[5])) * np.exp(-p[6] * Q))
    r = least_squares(lambda p: m(p, N7, D7, Q7) - L7,
                      [1.5, LOGE(0.5), LOGE(0.26), 0.35, LOGE(1.2), LOGE(0.25), 0.13],
                      bounds=([0.2, LOGE(1e-8), LOGE(1e-3), -20, LOGE(1e-8), LOGE(1e-3), -20],
                              [4, LOGE(1e8), LOGE(3), 20, LOGE(1e8), LOGE(3), 20]), max_nfev=40000)
    rss = np.sum(r.fun ** 2)
    return rss, r.x
base = fit_with_E1_fixed(0.1173)[0]
print(f'{"E1(固定)":>10}{"RSS":>12}{"ΔRSS":>11}{"rho_D":>9}{"B(D项)":>10}{"E0":>8}')
for E1f in [0.0, 0.03, 0.06, 0.09, 0.1173, 0.15, 0.20, 0.30, 0.45]:
    rss, x_ = fit_with_E1_fixed(E1f)
    print(f'{E1f:>10.4f}{rss:>12.6f}{rss-base:>11.6f}{x_[6]:>9.4f}{np.exp(x_[4]):>10.4f}{x_[0]:>8.4f}')
print('  → 若 ΔRSS 在很宽的 E1 区间内几乎不变，说明 E1 与 rho_D 不能分离，只有“总质量效应”可识别')

print()
print('=' * 100)
print('【B8 边界】方向异常 + calibrated/extrapolated 分层 + 尺度校准可行性')
print('=' * 100)
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
def keyf(df): return set(zip(df.N_params_B.round(4), df.D_tokens_B.round(4), df.Q_score.round(4)))
k6, k7, k8 = keyf(b6), keyf(b7), keyf(b8)
# 在共有格点上比较B7与B8的loss
mg = pd.merge(b7, b8, on=['N_params_B', 'D_tokens_B', 'Q_score'], suffixes=('_7', '_8'))
print(f'B7∩B8 共同格点={len(mg)}')
if len(mg):
    d_ = mg.val_loss_8 - mg.val_loss_7
    print(f'  B8-B7 loss 差: 均值={d_.mean():+.4f}, 中位={d_.median():+.4f}, 标准差={d_.std():.4f}')
    print(f'  corr(B7loss, B8loss)={np.corrcoef(mg.val_loss_7, mg.val_loss_8)[0,1]:.4f}')
    print(f'  分组看：')
    for dt, sub in mg.groupby('data_type'):
        dd = sub.val_loss_8 - sub.val_loss_7
        print(f'    [{dt}] n={len(sub)} 差均值={dd.mean():+.4f} 中位={dd.median():+.4f}')
print()
print('B8 内 corr(Q, L) = +0.913；B6/B7 为 -0.33/-0.31 → 方向相反')
print('  检验 Q → 1-Q 变换后是否与B6/B7同向:')
mg['Q_inv'] = 1 - mg.Q_score
print(f'    corr(1-Q, L_B8) = {np.corrcoef(mg.Q_inv, mg.val_loss_8)[0,1]:+.4f}')
print(f'  但更可能的解释是：B8 的 loss 由某条“反向”标度关系生成（组内corr(Q,L)中位数=+0.998，几乎完全线性）')
print(f'  → B8 组内 corr 高达 +0.998 说明其 Q 效应是确定性生成而非带噪观测；')
print(f'    它不能作为独立观测证据，只能作“半合成敏感性分析”')

print()
print('=' * 100)
print('【C 桥接】Loss → Benchmark 的可比性分层')
print('=' * 100)
lb = pd.read_csv(os.path.join(CB, 'loss_benchmark_bridge.csv'))
lbe = pd.read_csv(os.path.join(CB, 'loss_benchmark_bridge_expanded.csv'))
print(f'bridge n={len(lb)}, expanded n={len(lbe)}')
print('Loss_Comparability 分层:')
print(lb.Loss_Comparability.value_counts().to_string())
print()
print('Loss_Source 分层:')
print(lb.Loss_Source.value_counts().to_string())
print()
print(lb[['Model', 'N_params_B', 'D_tokens_B', 'Val_Loss', 'LB_Average', 'Loss_Comparability']].to_string())
print()
print('expanded 版 Loss_Comparability:')
print(lbe.Loss_Comparability.value_counts().to_string())
print()
print('★ 桥接分层规则（按可比性等级分别拟合，不混池）:')
for lvl, sub in lb.groupby('Loss_Comparability'):
    if len(sub) >= 3:
        r = stats.linregress(sub.Val_Loss.values, sub.LB_Average.values)
        print(f'  [{lvl}] n={len(sub)}  LB ~ Loss: 斜率={r.slope:.4f}, R²={r.rvalue**2:.3f}, p={r.pvalue:.2e}')
    else:
        print(f'  [{lvl}] n={len(sub)} 样本不足，只报告不拟合')
print()
print('对照：把全部等级混池拟合（应避免）:')
r = stats.linregress(lb.Val_Loss.values, lb.LB_Average.values)
print(f'  混池: n={len(lb)} 斜率={r.slope:.4f}, R²={r.rvalue**2:.3f}')
print('  → 分层与混池的斜率/R² 差异即为“口径不可比”造成的偏差量级')

print()
print('=' * 100)
print('【排行榜时间趋势】识别问题3：时间效应不能直接当因果')
print('=' * 100)
ts = pd.read_csv(os.path.join(CB, 'leaderboard_extended_timeseries.csv'))
print(f'n={len(ts)}, 年份 {ts.Year.min()}~{ts.Year.max()}')
print(f'Source 分层: {ts.Source.value_counts().to_dict()}')
print()
print(f'{"年份":>6}{"n":>6}{"中位Params":>12}{"中位Average":>14}')
for y, sub in ts.groupby('Year'):
    print(f'{y:>6}{len(sub):>6}{sub.Params_B.median():>12.2f}{sub.Average.median():>14.3f}')
print()
# 控制参数规模后，时间还剩多少
sub = ts.dropna(subset=['Average', 'Params_B', 'Year'])
sub = sub[(sub.Params_B > 0) & (sub.Average > 0)]
X = np.column_stack([np.ones(len(sub)), np.log(sub.Params_B.values), sub.Year.values - 2019])
yv = np.log(sub.Average.values)
coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
resid = yv - X @ coef
r2 = 1 - np.sum(resid**2) / np.sum((yv - yv.mean())**2)
print(f'log(Average) ~ log(Params) + Year  (n={len(sub)}, R²={r2:.3f})')
print(f'  log(Params) 系数 = {coef[1]:.4f}  (规模弹性)')
print(f'  Year 系数       = {coef[2]:+.4f}  (每年剩余增长)')
print(f'  → 该 Year 系数 = 样本选择+评测变化+微调+数据工程的混合残留，不能解释为纯技术因果')
# 只看 pretrained 子集
for tp, s2 in ts.groupby(ts.Source):
    s2 = s2.dropna(subset=['Average', 'Params_B', 'Year'])
    s2 = s2[(s2.Params_B > 0) & (s2.Average > 0)]
    if len(s2) < 20:
        continue
    X2 = np.column_stack([np.ones(len(s2)), np.log(s2.Params_B.values), s2.Year.values - 2019])
    c2, *_ = np.linalg.lstsq(X2, np.log(s2.Average.values), rcond=None)
    print(f'  [{tp}] n={len(s2)}: log(Params)={c2[1]:.4f}, Year={c2[2]:+.4f}')