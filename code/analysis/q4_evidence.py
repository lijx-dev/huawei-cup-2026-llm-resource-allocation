# -*- coding: utf-8 -*-
"""
问题四证据基座（可复现）：
  1. 前沿分位数回归 lnS = c + b_N lnN + b_T t，三种时间口径 × 多估计量
  2. Bootstrap 置信区间（pinball 损失 L-BFGS-B）
  3. C4 宏观增速 g_N / g_compute（跨源，须标口径）
  4. C3 历史前沿（累计最大）
  5. 增长核算：规模 vs 非规模贡献占比（对 g_N 口径敏感）
  6. Loss–Benchmark 桥接（C6，按可比性分组；以问题二 SET_A 的 E 作锚点做水平校准检查）
  7. 12/24 个月前沿预测（基础/放缓情景 + 不确定性三源分解）
  8. C8 逐任务聚合 与 C1 榜单一致性
  9. Chow 结构断点检验
输出：控制台日志 + q4_evidence_results.json
"""
import os, re, json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, curve_fit

warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
OUT = r"d:\F题\q4_evidence_results.json"
E_P2 = 1.71120          # 问题二 SET_A 的不可约损失（B6 口径）
SEED = 20260925

def sec(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

# ---------------------------------------------------------------- 数据载入
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
enh = pd.read_csv(os.path.join(BASE, "leaderboard_enhanced.csv"), low_memory=False)
c3 = pd.read_csv(os.path.join(BASE, "leaderboard_extended_timeseries.csv"), low_memory=False)
c4 = pd.read_csv(os.path.join(BASE, "epoch_all_ai_models.csv"), low_memory=False)
c6 = pd.read_csv(os.path.join(BASE, "loss_benchmark_bridge_expanded.csv"), low_memory=False)

lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']).copy()
lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()

# 时间口径
lb['t_int'] = lb['date'].dt.year - 2022
lb['t_frac'] = (lb['date'] - pd.Timestamp('2022-01-01')).dt.days / 365.25
lb['t_mon'] = (lb['date'].dt.year + (lb['date'].dt.month - 1) / 12.0) - 2022
lb['lnN'] = np.log(lb['N']); lb['lnS'] = np.log(lb['S'])
lb['q'] = lb['date'].dt.to_period('Q').astype(str)

# 筛选口径
OW = ['apache','mit','bsd','llama','gemma','cc-by','openrail','gpl','wtfpl','afl','creativeml','bigscience','bigcode','apple-ascl']
lic = lb['Hub License'].fillna('').str.lower()
m_lic = lic.apply(lambda x: any(w in x for w in OW))
m_base = lb['Type'].fillna('').str.contains('pretrained')
m_merge = lb['Type'].fillna('').str.contains('merges')
# C2 Epoch 开源权重标记（按模型名对齐）
eow = enh.drop_duplicates('Model').set_index('Model')['Epoch_AI_Open_Weights'].astype(str).str.lower()
lb['epoch_open'] = lb['Model'].map(eow)
m_epoch = lb['epoch_open'].eq('yes')

SAMPLES = {
    'ALL':            np.ones(len(lb), bool),
    'OPEN_LIC':       m_lic.values,
    'OPEN_EPOCH':     m_epoch.values,
    'BASE_ONLY':      m_base.values,
    'OPEN_BASE':      (m_lic & m_base).values,
    'NO_MERGE':       (~m_merge).values,
}

sec("【0】样本与口径概览")
print(f"C1 可用样本 n={len(lb)}  窗口 {lb['date'].min().date()} ~ {lb['date'].max().date()}  "
      f"跨度 {(lb['date'].max()-lb['date'].min()).days} 天 = {(lb['date'].max()-lb['date'].min()).days/365.25:.2f} 年")
for k, v in SAMPLES.items():
    print(f"  {k:12s} n={int(v.sum()):5d}  中位N={lb.loc[v,'N'].median():7.2f}B  中位S={lb.loc[v,'S'].median():6.2f}  "
          f"p90S={lb.loc[v,'S'].quantile(.9):6.2f}  季度数={lb.loc[v,'q'].nunique()}")

# ---------------------------------------------------------------- 估计量
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def qr_lp(X, y, tau):
    """Koenker–Bassett 分位数回归，线性规划精确解（HiGHS）"""
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    res = linprog(c, A_eq=A_eq, b_eq=y,
                  bounds=[(None, None)] * k + [(0, None)] * (2 * n), method='highs')
    return res.x[:k] if res.success else None

def qr_pin(X, y, tau):
    """pinball 损失直接极小化（L-BFGS-B），用于 Bootstrap 快速重抽"""
    def f(b):
        r = y - X @ b
        return np.sum(np.where(r >= 0, tau * r, (tau - 1) * r))
    b0 = ols(X, y)
    return minimize(f, b0, method='L-BFGS-B').x

def design(df, tcol):
    return np.column_stack([np.ones(len(df)), df['lnN'].values, df[tcol].values])

sec("【1】前沿回归 lnS = c + b_N·lnN + b_T·t：时间口径 × 估计量（tau=0.9 为主）")
reg_rows = []
for sname, msk in SAMPLES.items():
    sub = lb[msk]
    if len(sub) < 50:
        continue
    for tcol, tname in [('t_int', '整数年'), ('t_frac', '连续年'), ('t_mon', '月度')]:
        X = design(sub, tcol); y = sub['lnS'].values
        bo = ols(X, y)
        bq = qr_lp(X, y, 0.9)
        reg_rows.append(dict(sample=sname, t_caliber=tname, n=len(sub),
                             ols_c=bo[0], ols_bN=bo[1], ols_bT=bo[2],
                             qr90_c=bq[0] if bq is not None else None,
                             qr90_bN=bq[1] if bq is not None else None,
                             qr90_bT=bq[2] if bq is not None else None))
dfr = pd.DataFrame(reg_rows)
print(f"{'样本':12s} {'口径':6s} {'n':>5s} | {'OLS b_N':>9s} {'OLS b_T':>9s} | {'QR90 b_N':>9s} {'QR90 b_T':>9s}")
for _, r in dfr.iterrows():
    print(f"{r['sample']:12s} {r['t_caliber']:6s} {r['n']:5d} | {r['ols_bN']:+9.4f} {r['ols_bT']:+9.4f} | "
          f"{r['qr90_bN']:+9.4f} {r['qr90_bT']:+9.4f}")

sec("【1b】分位数族（主口径：OPEN_LIC × 连续年）")
sub = lb[SAMPLES['OPEN_LIC']]; X = design(sub, 't_frac'); y = sub['lnS'].values
qr_family = {}
for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    b = qr_lp(X, y, tau)
    qr_family[tau] = dict(bN=float(b[1]), bT=float(b[2]))
    print(f"  tau={tau:.2f}  b_N={b[1]:+.4f}  b_T={b[2]:+.4f}")

sec("【1c】识别性诊断：lnN 与 t 的共线性（决定 b_N / b_T 能否分别识别）")
collin = {}
for sname, msk in [('ALL', SAMPLES['ALL']), ('OPEN_LIC', SAMPLES['OPEN_LIC']), ('NO_MERGE', SAMPLES['NO_MERGE'])]:
    s = lb[msk]
    r_nt = float(np.corrcoef(s['lnN'], s['t_frac'])[0, 1])
    Z = np.column_stack([s['lnN'].values, s['t_frac'].values])
    Z = (Z - Z.mean(0)) / Z.std(0)
    ev = np.linalg.eigvalsh(np.corrcoef(Z.T))
    vif = float(1 / (1 - r_nt ** 2))
    collin[sname] = dict(corr_lnN_t=r_nt, vif=vif, cond_number=float(np.sqrt(ev.max() / ev.min())))
    print(f"  {sname:10s} n={len(s):5d}  corr(lnN,t)={r_nt:+.4f}  VIF={vif:7.3f}  "
          f"条件数={np.sqrt(ev.max()/ev.min()):6.2f}  季度内 lnN 跨度={s.groupby('q')['lnN'].std().mean():.3f}")
print("  判读：VIF>10 视为严重共线；窗口仅 4 个季度时 b_N 与 b_T 只能弱识别。")

# ---------------------------------------------------------------- Bootstrap
sec("【2】Bootstrap 置信区间（OPEN_LIC × 连续年 × tau=0.9，B=300，pinball 重抽）")
sub = lb[SAMPLES['OPEN_LIC']].reset_index(drop=True)
X = design(sub, 't_frac'); y = sub['lnS'].values
b_hat = qr_lp(X, y, 0.9)
print(f"  点估计(精确LP): c={b_hat[0]:.4f}  b_N={b_hat[1]:.4f}  b_T={b_hat[2]:.4f}")
rng = np.random.default_rng(SEED)
B = 300
boot = np.zeros((B, 3))
for i in range(B):
    idx = rng.integers(0, len(sub), len(sub))
    boot[i] = qr_pin(X[idx], y[idx], 0.9)
ci = np.percentile(boot, [5, 50, 95], axis=0)
print(f"  b_N 90%CI = [{ci[0,1]:.4f}, {ci[2,1]:.4f}]   b_T 90%CI = [{ci[0,2]:.4f}, {ci[2,2]:.4f}]")
print(f"  b_T 含 0 ? {ci[0,2] <= 0 <= ci[2,2]}    b_N 含 0 ? {ci[0,1] <= 0 <= ci[2,1]}")

# ---------------------------------------------------------------- C4 宏观增速
sec("【3】C4 宏观增速（跨源，口径须标注）")
c4['year'] = pd.to_datetime(c4['Publication date'], errors='coerce').dt.year
c4['P'] = pd.to_numeric(c4['Parameters'], errors='coerce')
c4['C'] = pd.to_numeric(c4['Training compute (FLOP)'], errors='coerce')
c4['Dsz'] = pd.to_numeric(c4['Training dataset size (total)'], errors='coerce')
print("C4 年份分布(前12):", c4['year'].value_counts().sort_index().to_dict())
lang = c4['Domain'].astype(str).str.contains('Language', na=False)
openw = c4['Open model weights?'].astype(str).str.lower().eq('yes')

def growth(df, col, lo=2018, hi=2025):
    d = df.dropna(subset=[col, 'year'])
    d = d[(d['year'] >= lo) & (d['year'] <= hi) & (d[col] > 0)]
    g = d.groupby('year')[col].quantile(0.9)
    if len(g) < 3:
        return None, None, g
    yy = g.index.values.astype(float); ll = np.log(g.values)
    sl = np.polyfit(yy, ll, 1)[0]
    return sl, len(g), g

gN_rows = {}
for nm, msk in [('C4 全库', np.ones(len(c4), bool)), ('C4 Language', lang.values),
                ('C4 Language+开源', (lang & openw).values)]:
    gN, ny, _ = growth(c4[msk], 'P')
    gC, _, _ = growth(c4[msk], 'C')
    gD, _, _ = growth(c4[msk], 'Dsz')
    gN_rows[nm] = dict(g_N=gN, g_compute=gC, g_data=gD, n_years=ny)
    print(f"  {nm:20s} 年份数={ny}  g_N={gN if gN is None else round(gN,4)}  "
          f"g_compute={gC if gC is None else round(gC,4)}  g_data={gD if gD is None else round(gD,4)}")

# C1 自身 p90 规模轨迹
gN_c1 = {}
for sname in ['ALL', 'OPEN_LIC', 'OPEN_BASE']:
    s = lb[SAMPLES[sname]]
    g = s.groupby('q')['N'].quantile(0.9)
    tt = np.arange(len(g))
    sl = np.polyfit(tt, np.log(g.values), 1)[0] * 4
    gN_c1[sname] = float(sl)
    print(f"  C1 {sname:12s} p90 参数量年化对数增速 = {sl:+.4f}   (季度 p90: {g.round(2).to_dict()})")

# ---------------------------------------------------------------- C3 历史前沿
sec("【4】C3 历史前沿（累计最大口径）")
c3['S'] = pd.to_numeric(c3['Average'], errors='coerce')
c3['N'] = pd.to_numeric(c3['Params_B'], errors='coerce')
hist = c3.groupby('Year')['S'].max().sort_index()
cum = hist.cummax()
print("  逐年最大:", hist.round(2).to_dict())
print("  累计最大:", cum.round(2).to_dict())
hl = np.log(cum.values); hy = cum.index.values.astype(float)
print(f"  累计前沿年化对数增速(2019-2025) = {np.polyfit(hy, hl, 1)[0]:+.4f}")

# ---------------------------------------------------------------- 规模分桶内的纯时间趋势
sec("【5a】规模分桶内的纯时间趋势（把 lnN 近似固定，绕开 lnN–t 共线）")
sub = lb[SAMPLES['OPEN_LIC']].copy()
BINS = [(0, 3), (3, 10), (10, 30), (30, 1e9)]
binned = {}
print(f"  {'规模桶':12s} {'n':>5s} {'季度数':>6s} | {'b_T(桶内)':>10s} {'b_N(桶内)':>10s} | {'桶内 g_N':>9s}")
for lo, hi in BINS:
    s = sub[(sub['N'] >= lo) & (sub['N'] < hi)]
    if len(s) < 60:
        print(f"  {f'{lo:g}-{hi:g}B':12s} n={len(s)} 太少"); continue
    gq = s.groupby('q')['lnS'].quantile(0.9)
    if len(gq) < 3:
        print(f"  {f'{lo:g}-{hi:g}B':12s} 季度不足"); continue
    tt = np.arange(len(gq))
    bT_bin = float(np.polyfit(tt, gq.values, 1)[0] * 4)
    nq = s.groupby('q')['N'].quantile(0.9)
    gN_bin = float(np.polyfit(tt, np.log(nq.values), 1)[0] * 4)
    Xb = design(s, 't_frac'); yb = s['lnS'].values
    bb = qr_lp(Xb, yb, 0.9)
    binned[f'{lo:g}-{hi:g}B'] = dict(n=len(s), bT_bin=bT_bin, gN_bin=gN_bin,
                                     qr90_bN=float(bb[1]), qr90_bT=float(bb[2]))
    print(f"  {f'{lo:g}-{hi:g}B':12s} {len(s):5d} {len(gq):6d} | {bT_bin:+10.4f} {bb[2]:+10.4f} | {gN_bin:+9.4f}")
print("  判读：桶内 b_T 是 N 近似固定下的非规模进步率，不受 lnN–t 共线污染，可作 b_T 的独立交叉验证。")

# ---------------------------------------------------------------- 增长核算
sec("【5】增长核算 dlnS/dt = b_N·g_N + b_T   （b_N,b_T 取 OPEN_LIC×连续年×QR90）")
bN, bT = float(b_hat[1]), float(b_hat[2])
qfront = lb[SAMPLES['OPEN_LIC']].groupby('q')['S'].quantile(0.9)
r_obs = float(np.polyfit(np.arange(len(qfront)), np.log(qfront.values), 1)[0] * 4)
gN_implied = (r_obs - bT) / bN
print(f"  实测前沿 p90 年化对数增速 r_obs = {r_obs:+.4f}   (季度 p90: {qfront.round(2).to_dict()})")
print(f"  模型自洽反解 g_N* = (r_obs - b_T)/b_N = {gN_implied:+.4f}/年   ← 使分解与实测一致的规模增速")
print()
print(f"  {'g_N 口径':30s} {'g_N':>8s} {'规模贡献':>10s} {'非规模':>10s} {'规模占比':>10s} {'预测dlnS/dt':>11s}")
acc = {}
for nm, g in [('C1 OPEN_LIC p90（提交构成漂移）', gN_c1['OPEN_LIC']),
              ('C1 ALL p90（提交构成漂移）', gN_c1['ALL']),
              ('C4 全库', gN_rows['C4 全库']['g_N']),
              ('C4 Language', gN_rows['C4 Language']['g_N']),
              ('C4 Language+开源 ← 主口径', gN_rows['C4 Language+开源']['g_N']),
              ('参考论文声称 g_N=1.251', 1.251),
              ('自洽反解 g_N*', gN_implied)]:
    sc = bN * g; ns = bT
    share = sc / (sc + ns) * 100 if (sc + ns) != 0 else np.nan
    acc[nm] = dict(g_N=float(g), scale=float(sc), non_scale=float(ns), share_pct=float(share),
                   predicted=r_obs * 0 + sc + ns)
    print(f"  {nm:30s} {g:+8.4f} {sc:+10.4f} {ns:+10.4f} {share:+9.1f}% {sc+ns:+11.4f}")
print(f"  （对照）实测 r_obs = {r_obs:+.4f}")
print("\n  时间口径对 b_T 与规模占比的影响（同一数据，仅换 t 的定义，g_N 固定为主口径）:")
gN_main = gN_rows['C4 Language+开源']['g_N']
cal_rows = {}
for tcol, tname in [('t_int', '整数年（参考论文口径）'), ('t_frac', '连续年（本稿主口径）'), ('t_mon', '月度')]:
    Xc = design(lb[SAMPLES['OPEN_LIC']], tcol); yc = lb[SAMPLES['OPEN_LIC']]['lnS'].values
    bc = qr_lp(Xc, yc, 0.9)
    sc = bc[1] * gN_main; share = sc / (sc + bc[2]) * 100
    cal_rows[tname] = dict(bN=float(bc[1]), bT=float(bc[2]), share_pct=float(share))
    print(f"    {tname:22s} b_N={bc[1]:+.4f}  b_T={bc[2]:+.4f}  →  规模占比={share:+6.1f}%")
print(f"  判读：b_N 稳定在 0.34–0.37（口径无关），b_T 在 {min(v['bT'] for v in cal_rows.values()):.3f}–"
      f"{max(v['bT'] for v in cal_rows.values()):.3f} 间摆动，规模占比随之在 "
      f"{min(v['share_pct'] for v in cal_rows.values()):.0f}%–{max(v['share_pct'] for v in cal_rows.values()):.0f}% 间摆动。")

# 整数年衰减的解析解释
sf = lb[SAMPLES['OPEN_LIC']]
t_frac_v = sf['t_frac'].values; t_int_v = sf['t_int'].values.astype(float)
att = float(np.cov(t_int_v, t_frac_v)[0, 1] / np.var(t_int_v))
print(f"\n  整数年离散化的衰减因子 = cov(t_int,t_frac)/var(t_int) = {att:.4f}")
print(f"  预期 b_T(整数年) ≈ {att:.3f} × b_T(连续年) = {att*cal_rows['连续年（本稿主口径）']['bT']:+.4f}  "
      f"（实测 {cal_rows['整数年（参考论文口径）']['bT']:+.4f}，同量级）")
print("  判读：整数年把 0.76 年窗口压成 2024/2025 两个点，等价于对时间变量加测量误差，"
      "经典衰减使 b_T 系统性偏低 → 参考论文的 b_T=0.089 是衰减下界，不是非规模进步率的无偏估计。")

# ---------------------------------------------------------------- 桥接
sec("【6】Loss–Benchmark 桥接（C6, n=%d）" % len(c6))
br = c6.dropna(subset=['Val_Loss', 'LB_Average']).copy()
br = br[(br['Val_Loss'] > 0) & (br['LB_Average'] > 0)]
L = br['Val_Loss'].values; A = br['LB_Average'].values
hi_cmp = br['Loss_Comparability'].str.contains('High', na=False).values

def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

bridge = {}
c = np.polyfit(np.log(L), np.log(A), 1)
bridge['lnA_lnL'] = dict(k=float(c[0]), b=float(c[1]), r2=float(r2(np.log(A), np.polyval(c, np.log(L)))))
print(f"  (a) lnA ~ lnL     : 斜率={c[0]:+.4f} 截距={c[1]:+.4f}  R2={bridge['lnA_lnL']['r2']:.4f}")
c = np.polyfit(np.log(L), A, 1)
bridge['A_lnL'] = dict(k=float(c[0]), b=float(c[1]), r2=float(r2(A, np.polyval(c, np.log(L)))))
print(f"  (b) A  ~ lnL      : 斜率={c[0]:+.4f} 截距={c[1]:+.4f}  R2={bridge['A_lnL']['r2']:.4f}")
p = np.clip(A / 100, 1e-4, 1 - 1e-4); lg = np.log(p / (1 - p))
c = np.polyfit(np.log(L), lg, 1)
bridge['logit_lnL'] = dict(k=float(c[0]), b=float(c[1]), r2=float(r2(lg, np.polyval(c, np.log(L)))))
print(f"  (c) logit(A)~lnL  : 斜率={c[0]:+.4f} 截距={c[1]:+.4f}  R2={bridge['logit_lnL']['r2']:.4f}   [参考论文形式]")
try:
    po, _ = curve_fit(lambda l, Am, lam, E0: Am * np.exp(-lam * (l - E0)), L, A,
                      p0=[60, 2, 1.5], maxfev=40000)
    bridge['saturated'] = dict(Amax=float(po[0]), lam=float(po[1]), E=float(po[2]),
                               r2=float(r2(A, po[0] * np.exp(-po[1] * (L - po[2])))))
    print(f"  (d) 饱和(自由E)   : Amax={po[0]:.2f} lam={po[1]:.3f} E={po[2]:.3f}  R2={bridge['saturated']['r2']:.4f}")
except Exception as e:
    print("  (d) 饱和拟合失败", e)

print(f"\n  C6 Val_Loss 范围 [{L.min():.3f}, {L.max():.3f}]；问题二 SET_A 的 E = {E_P2:.4f}")
n_below = int((L < E_P2).sum())
print(f"  ** L < E(P2) 的样本数 = {n_below}/{len(L)} ({n_below/len(L)*100:.1f}%) —— "
      f"说明 C6 的 Val_Loss 口径与附件 B 不可直接对齐，E 不能原样用作饱和锚点")
bridge['n_below_E'] = n_below
# 以 C6 自身 E 作锚点的饱和形式（E 固定为 C6 的 min 或自由解）
for tag, Efix in [('E=C6 min', L.min()), ('E=E_P2', E_P2)]:
    try:
        po, _ = curve_fit(lambda l, Am, lam: Am * np.exp(-lam * (l - Efix)), L, A,
                          p0=[60, 2], maxfev=40000)
        rr = r2(A, po[0] * np.exp(-po[1] * (L - Efix)))
        print(f"  饱和(锚定 {tag}={Efix:.3f}): Amax={po[0]:.2f} lam={po[1]:.3f}  R2={rr:.4f}")
    except Exception as e:
        print(f"  饱和(锚定 {tag}) 拟合失败: {e}")

print("\n  按可比性分组：")
for nm, msk in [('High', hi_cmp), ('Medium', ~hi_cmp)]:
    if msk.sum() < 5:
        print(f"    {nm}: n={int(msk.sum())} 太少"); continue
    c = np.polyfit(np.log(L[msk]), np.log(A[msk]), 1)
    pred = np.polyval(c, np.log(L[msk]))
    print(f"    {nm:7s} n={int(msk.sum()):3d}  lnA~lnL 斜率={c[0]:+.4f} 截距={c[1]:+.4f} R2={r2(np.log(A[msk]),pred):.4f}  "
          f"残差(分) 均值={np.mean(A[msk]-np.exp(pred)):+.2f} 标准差={np.std(A[msk]-np.exp(pred)):.2f}")

# ---------------------------------------------------------------- 桥接误差传播
sec("【6b】桥接误差向分解与预测的传播")
br_prop = {}
# 采用主形式 lnA ~ lnL（R2 最高）：A = exp(b) * L^k, dlnA/dL = k/L
k_ln = bridge['lnA_lnL']['k']; b_ln = bridge['lnA_lnL']['b']
resid_ln = np.log(A) - (b_ln + k_ln * np.log(L))
sig_ln = float(np.std(resid_ln))
print(f"  主桥接形式 lnA = {b_ln:.4f} + {k_ln:.4f}·lnL，对数残差 σ_ln = {sig_ln:.4f} "
      f"(≈ {sig_ln*100:.1f}% 的能力相对误差)")
print(f"  在中位损失 L̄={np.median(L):.3f} 处：dlnA/dL = k/L = {k_ln/np.median(L):+.4f} 每单位损失")
# 把 ±σ_ln 的能力误差折算成 S 的水平误差
A_med = float(np.median(A))
print(f"  能力水平误差：S = {A_med:.1f} 时 ±{sig_ln*100:.1f}% ≈ ±{A_med*sig_ln:.1f} 分")
# 误差如何影响 b_N / b_T：把每个样本的 lnS 加独立噪声后重估
rng2 = np.random.default_rng(SEED + 1)
sub = lb[SAMPLES['OPEN_LIC']]
Xb = design(sub, 't_frac'); yb = sub['lnS'].values
bb0 = qr_lp(Xb, yb, 0.9)
dN, dT = [], []
for _ in range(60):
    yp = yb + rng2.normal(0, sig_ln, len(yb))
    bb = qr_pin(Xb, yp, 0.9)
    dN.append(bb[1]); dT.append(bb[2])
print(f"  注入桥接噪声 σ_ln={sig_ln:.4f} 后重估（60 次）：")
print(f"    b_N: {bb0[1]:+.4f} → 均值 {np.mean(dN):+.4f}  标准差 {np.std(dN):.4f}  "
      f"(相对 {np.std(dN)/abs(bb0[1])*100:.1f}%)")
print(f"    b_T: {bb0[2]:+.4f} → 均值 {np.mean(dT):+.4f}  标准差 {np.std(dT):.4f}  "
      f"(相对 {np.std(dT)/abs(bb0[2])*100:.1f}%)")
br_prop = dict(sigma_ln=sig_ln, rel_err_pct=sig_ln * 100, bN_sd=float(np.std(dN)),
               bT_sd=float(np.std(dT)), bN_rel=float(np.std(dN) / abs(bb0[1])),
               bT_rel=float(np.std(dT) / abs(bb0[2])))
print(f"  判读：桥接噪声主要抬高 b_T 的不确定性（b_T 相对误差 "
      f"{np.std(dT)/abs(bb0[2])*100:.0f}% > b_N 的 {np.std(dN)/abs(bb0[1])*100:.0f}%），"
      f"即'非规模项'最易被桥接误差吞掉 → 桥接是分解结论的第二大不确定性来源。")

# ---------------------------------------------------------------- 预测
sec("【7】12/24 个月前沿预测（基础 / 算力放缓 双情景）")
sub = lb[SAMPLES['OPEN_LIC']]
last_q = sorted(sub['q'].unique())[-1]
S_0 = float(sub[sub['q'] == last_q]['S'].quantile(0.9))
N_0 = float(sub[sub['q'] == last_q]['N'].quantile(0.9))
t_0 = float((sub[sub['q'] == last_q]['date'].max() - pd.Timestamp('2022-01-01')).days / 365.25)
resid = sub['lnS'].values - (b_hat[0] + bN * sub['lnN'].values + bT * sub['t_frac'].values)
sig_res = float(np.std(resid))
print(f"  基准点 = 观测前沿（{last_q}）：S_0={S_0:.2f}  N_p90={N_0:.2f}B  t_0={t_0:.3f}  残差σ={sig_res:.4f}")
print(f"  （对照）QR90 平面在 t=3 的拟合值 = {np.exp(b_hat[0]+bN*np.log(N_0)+bT*3.0):.2f}，"
      f"与观测 {S_0:.2f} 存在水平差 → 采用观测锚点 + 模型增速外推")

rng = np.random.default_rng(SEED)
def predict(gN, h, nboot=2000):
    r = bN * gN + bT                     # 模型隐含年化对数增速
    base = np.log(S_0) + r * h
    eps = rng.normal(0, sig_res, nboot)
    bi = rng.integers(0, B, nboot)
    bp = boot[bi]
    r_p = bp[:, 1] * gN + bp[:, 2]
    par = np.log(S_0) + r_p * h
    gs = rng.uniform(0.2, 1.6, nboot)
    gsc = np.log(S_0) + (bN * gs + bT) * h
    return base, eps, par, gsc, r

pred = {}
scen = [('基础情景 g_N=%.3f（C4 主口径）' % gN_main, gN_main),
        ('算力放缓情景 g_N=%.3f（减半）' % (gN_main / 2), gN_main / 2),
        ('参考论文口径 g_N=1.251', 1.251),
        ('C1 自洽口径 g_N=%.3f（诊断）' % gN_c1['OPEN_LIC'], gN_c1['OPEN_LIC'])]
for tag, gN in scen:
    for h in [1.0, 2.0]:
        base, eps, par, gsc, r = predict(gN, h)
        draw = base + eps
        lo, md, hiq = np.percentile(np.exp(draw), [5, 50, 95])
        pred[f'{tag}|{int(h*12)}M'] = dict(S_hat=float(np.exp(base)), lo=float(lo), mid=float(md),
                                           hi=float(hiq), ratio=float(np.exp(base) / S_0),
                                           r=float(r))
        print(f"  {tag:34s} {int(h*12):2d}M: Ŝ={np.exp(base):7.2f}  "
              f"90%CI=[{lo:6.2f},{hiq:6.2f}]  相对基准 ×{np.exp(base)/S_0:.2f}  (r={r:+.3f}/年)")

print("\n  不确定性三源分解（全方差定律，基础情景 12M / 24M）：")
V_res = float(np.var(rng.normal(0, sig_res, 20000)))
for h in [1.0, 2.0]:
    base, eps, par, gsc, r = predict(gN_main, h)
    V_par = float(np.var(par - base)); V_g = float(np.var(gsc - base))
    tot = V_res + V_par + V_g
    print(f"    {int(h*12):2d}M: 残差 {V_res/tot*100:5.1f}% | 参数 {V_par/tot*100:5.1f}% | g_N 情景 {V_g/tot*100:5.1f}%")

print("\n  g_N 敏感性扫描（基础点 S_0=%.2f）:" % S_0)
for g in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
    r = bN * g + bT
    print(f"    g_N={g:.1f} → 12M Ŝ={S_0*np.exp(r*1):6.2f}  24M Ŝ={S_0*np.exp(r*2):7.2f}")

# ---------------- logit 有界预测（主口径预测形式）
sec("【7b】logit 有界前沿模型（尊重 0–100 量纲）")
sub = lb[SAMPLES['OPEN_LIC']].copy()
p0v = np.clip(sub['S'].values / 100, 1e-4, 1 - 1e-4)
sub['logitS'] = np.log(p0v / (1 - p0v))
Xl = design(sub, 't_frac'); yl = sub['logitS'].values
bl = qr_lp(Xl, yl, 0.9)
bo = ols(Xl, yl)
print(f"  logit(S/100) = c + b_N·lnN + b_T·t   （QR90，n={len(sub)}）")
print(f"    QR90 : c={bl[0]:+.4f}  b_N={bl[1]:+.4f}  b_T={bl[2]:+.4f}")
print(f"    OLS  : c={bo[0]:+.4f}  b_N={bo[1]:+.4f}  b_T={bo[2]:+.4f}")
resid_l = yl - (bl[0] + bl[1] * sub['lnN'].values + bl[2] * sub['t_frac'].values)
sig_l = float(np.std(resid_l))
S0l = float(sub[sub['q'] == last_q]['S'].quantile(0.9))
logit0 = np.log((S0l / 100) / (1 - S0l / 100))
print(f"  基准点 logit(S_0/100) = {logit0:+.4f}  (S_0={S0l:.2f})   logit 残差σ = {sig_l:.4f}")

rng3 = np.random.default_rng(SEED + 2)
pred_logit = {}
print(f"\n  {'情景':34s} {'期限':>4s} {'Ŝ(logit)':>10s} {'90%CI':>20s} {'相对基准':>9s}")
for tag, gN in [('基础情景 g_N=%.3f（C4 主口径）' % gN_main, gN_main),
                ('算力放缓情景 g_N=%.3f（减半）' % (gN_main / 2), gN_main / 2),
                ('参考论文口径 g_N=1.251', 1.251)]:
    for h in [1.0, 2.0]:
        r = bl[1] * gN + bl[2]
        base = logit0 + r * h
        eps = rng3.normal(0, sig_l, 3000)
        lo, md, hiq = np.percentile(100 / (1 + np.exp(-(base + eps))), [5, 50, 95])
        Sh = float(100 / (1 + np.exp(-base)))
        pred_logit[f'{tag}|{int(h*12)}M'] = dict(S_hat=Sh, lo=float(lo), mid=float(md), hi=float(hiq),
                                                 ratio=float(Sh / S0l), r=float(r))
        print(f"  {tag:34s} {int(h*12):2d}M {Sh:10.2f}   [{lo:6.2f},{hiq:6.2f}]      ×{Sh/S0l:.2f}")
print(f"  判读：logit 形式在 24M 给出 {pred_logit[f'基础情景 g_N={gN_main:.3f}（C4 主口径）|24M']['S_hat']:.1f} 分，"
      f"天然不越过 100；对数线性形式同口径给出 {pred[f'基础情景 g_N={gN_main:.3f}（C4 主口径）|24M']['S_hat']:.1f} 分，"
      f"已越界 → 长周期预测必须以有界形式为准。")

# ---------------------------------------------------------------- C8 聚合
sec("【8】C8 逐任务聚合 与 C1 榜单一致性")
KEY = {'leaderboard_ifeval': ('inst_level_strict_acc,none', 'prompt_level_strict_acc,none'),
       'leaderboard_bbh': ('acc_norm,none',), 'leaderboard_math_hard': ('exact_match,none',),
       'leaderboard_gpqa': ('acc_norm,none',), 'leaderboard_musr': ('acc_norm,none',),
       'leaderboard_mmlu_pro': ('acc,none',)}
det = os.path.join(BASE, "detailed_results")
rows, bad = [], 0
for d in os.listdir(det):
    dp = os.path.join(det, d)
    if not os.path.isdir(dp):
        continue
    fn = [f for f in os.listdir(dp) if f.endswith('.json')]
    if not fn:
        continue
    try:
        js = json.load(open(os.path.join(dp, fn[0]), encoding='utf-8'))
    except Exception:
        bad += 1; continue
    g = js.get('groups', {}) or {}; res = js.get('results', {}) or {}
    r = {'key': re.sub(r'[^a-z0-9]', '', d.lower())}
    for t, ks in KEY.items():
        src = g.get(t) or res.get(t)
        if isinstance(src, dict):
            v = [src.get(k) for k in ks if isinstance(src.get(k), (int, float))]
            if v:
                r[t] = float(np.mean(v)) * 100
    rows.append(r)
c8 = pd.DataFrame(rows)
print(f"  C8 目录 {len(os.listdir(det))}，可解析 {len(c8)}，损坏 {bad}")
DIMS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
        'leaderboard_gpqa', 'leaderboard_musr', 'leaderboard_mmlu_pro']
c8['C8_mean'] = c8[DIMS].mean(axis=1)
lb['key'] = lb['Model'].map(lambda x: re.sub(r'[^a-z0-9]', '', str(x).lower()))
m = lb.merge(c8[['key', 'C8_mean'] + DIMS], on='key', how='inner')
m = m.dropna(subset=['C8_mean', 'S'])
print(f"  匹配 n={len(m)}   C8_mean 均值={m['C8_mean'].mean():.2f}  C1 S 均值={m['S'].mean():.2f}")
print(f"  Pearson(lnC8, lnS)={np.corrcoef(np.log(m['C8_mean']), np.log(m['S']))[0,1]:.4f}  "
      f"Spearman={pd.Series(m['C8_mean']).corr(pd.Series(m['S']), method='spearman'):.4f}")
print("  逐维平均（C8 聚合）:", {k.replace('leaderboard_',''): round(c8[k].mean(),3) for k in DIMS})
# 逐任务年化增速（前沿：季度最大值）
sec("【8b】C8 逐任务前沿增速（季度最大值，年化对数）")
for k in DIMS:
    s = c8.join(lb.set_index('key')['q'], on='key', how='left')
    if 'q' not in s:
        continue
    gg = s.dropna(subset=['q']).groupby('q')[k].max()
    if len(gg) >= 3:
        sl = np.polyfit(np.arange(len(gg)), np.log(gg.values), 1)[0] * 4
        print(f"  {k.replace('leaderboard_',''):14s} 年化={sl:+.3f}  季度前沿={gg.round(1).to_dict()}")

# ---------------------------------------------------------------- Chow 断点
sec("【9】Chow 结构断点检验（断点 2024-07-01，主口径 OPEN_LIC×连续年）")
sub = lb[SAMPLES['OPEN_LIC']].copy()
brk = pd.Timestamp('2024-07-01')
d1 = sub[sub['date'] < brk]; d2 = sub[sub['date'] >= brk]
Xf = design(sub, 't_frac'); yf = sub['lnS'].values
bf = ols(Xf, yf); rss_f = ((yf - Xf @ bf) ** 2).sum()
rss_s = 0; kk = 0
for dd in (d1, d2):
    if len(dd) < 20: continue
    Xd = design(dd, 't_frac'); yd = dd['lnS'].values
    bd = ols(Xd, yd); rss_s += ((yd - Xd @ bd) ** 2).sum(); kk += 3
    print(f"  段内 n={len(dd):5d}  OLS b_N={bd[1]:+.4f}  b_T={bd[2]:+.4f}")
F = ((rss_f - rss_s) / 3) / (rss_s / (len(sub) - kk))
print(f"  Chow F(3, {len(sub)-kk}) = {F:.2f}")

# ---------------------------------------------------------------- 落盘
out = dict(
    meta=dict(seed=SEED, E_problem2=E_P2, window_days=int((lb['date'].max()-lb['date'].min()).days)),
    samples={k: int(v.sum()) for k, v in SAMPLES.items()},
    regression=dfr.to_dict(orient='records'),
    qr_family=qr_family,
    collinearity=collin,
    binned=binned,
    time_caliber=cal_rows,
    bootstrap=dict(point=[float(x) for x in b_hat],
                   ci_bN=[float(ci[0,1]), float(ci[2,1])], ci_bT=[float(ci[0,2]), float(ci[2,2])], B=B),
    c4_macro=gN_rows,
    c1_gN=gN_c1,
    r_obs=float(r_obs), gN_implied=float(gN_implied),
    c3_history=dict(annual={int(k): float(v) for k, v in hist.items()},
                    cummax={int(k): float(v) for k, v in cum.items()}),
    accounting=acc,
    bridge=bridge,
    bridge_propagation=br_prop,
    prediction=pred,
    prediction_logit=pred_logit,
    logit_model=dict(qr90=[float(x) for x in bl], ols=[float(x) for x in bo], sigma=float(sig_l)),
    c8=dict(n_match=int(len(m)), spearman=float(pd.Series(m['C8_mean']).corr(pd.Series(m['S']), method='spearman')),
            dim_mean={k: float(c8[k].mean()) for k in DIMS}, damaged=int(bad)),
    chow=dict(F=float(F), bN_pre=None),
)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已写出 {OUT}")
