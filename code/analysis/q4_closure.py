# -*- coding: utf-8 -*-
"""
增长核算闭合性诊断（问题四的数学核心）：
  恒等式 r = b_N·g_N + b_T 是否成立，取决于 g_N 在哪个点上度量。
  本脚本对比三种 g_N 度量：
    (i)  全样本 p90(N) 的增速        —— 与 r_obs 不同源，恒等式不闭合
    (ii) 前沿模型自身的 N 增速        —— 与 r_obs 同源，应闭合
    (iii)C4 普查增速                 —— 外部来源，仅作情景输入
"""
import pandas as pd, numpy as np, os, warnings
from scipy.optimize import linprog
warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"

def qr_lp(X, y, tau):
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A = np.hstack([X, np.eye(n), -np.eye(n)])
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method='highs')
    return r.x[:k] if r.success else None

lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']); lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
lb['t'] = (lb['date'] - pd.Timestamp('2022-01-01')).dt.days / 365.25
lic = lb['Hub License'].fillna('').str.lower()
OW = ['apache','mit','bsd','llama','gemma','cc-by','openrail','gpl','wtfpl','afl','creativeml','bigscience','bigcode','apple-ascl']
sub = lb[lic.apply(lambda x: any(w in x for w in OW))].copy()
print(f"主口径样本 OPEN_LIC n={len(sub)}")

X = np.column_stack([np.ones(len(sub)), np.log(sub['N'].values), sub['t'].values])
y = np.log(sub['S'].values)
b = qr_lp(X, y, 0.9); bN, bT = b[1], b[2]
print(f"前沿回归 QR90: c={b[0]:.4f}  b_N={bN:+.4f}  b_T={bT:+.4f}")

qs = sorted(sub['q'].unique())
print(f"\n季度: {qs}")
# ---- 前沿点定义1：每季度 p90(S) 水平 与 其对应 N（取最接近 p90S 的模型）
print("\n【口径 i / ii】逐季度前沿点与其 N")
rows = []
for q in qs:
    s = sub[sub['q'] == q]
    S90 = s['S'].quantile(0.9)
    near = s.iloc[(s['S'] - S90).abs().argmin()]
    top = s.loc[s['S'].idxmax()]
    rows.append(dict(q=q, n=len(s), S90=S90, N_at_S90=near['N'], Smax=top['S'], N_at_Smax=top['N'],
                     N90=s['N'].quantile(0.9), t_mid=s['t'].mean()))
fr = pd.DataFrame(rows).set_index('q')
print(fr.round(3).to_string())

tt = fr['t_mid'].values
def slope(v, t):
    return float(np.polyfit(t, np.log(np.asarray(v, float)), 1)[0])

r_obs = slope(fr['S90'].values, tt)
gN_S90 = slope(fr['N_at_S90'].values, tt)
gN_Smax = slope(fr['N_at_Smax'].values, tt)
gN_N90 = slope(fr['N90'].values, tt)
r_obs_max = slope(fr['Smax'].values, tt)
print(f"\n  观测前沿增速 r_obs（p90(S) 轨迹）        = {r_obs:+.4f}/年")
print(f"  观测前沿增速 r_obs（max(S) 轨迹）        = {r_obs_max:+.4f}/年")
print(f"  口径 ii  g_N（前沿点自身的 N）           = {gN_S90:+.4f}/年  → b_N·g_N+b_T = {bN*gN_S90+bT:+.4f}  闭合差 {bN*gN_S90+bT-r_obs:+.4f}")
print(f"  口径 ii' g_N（max(S) 点的 N）            = {gN_Smax:+.4f}/年  → b_N·g_N+b_T = {bN*gN_Smax+bT:+.4f}  闭合差 {bN*gN_Smax+bT-r_obs_max:+.4f}")
print(f"  口径 i   g_N（全样本 p90(N)）            = {gN_N90:+.4f}/年  → b_N·g_N+b_T = {bN*gN_N90+bT:+.4f}  闭合差 {bN*gN_N90+bT-r_obs:+.4f}")

print("\n【结论】恒等式 r = b_N·g_N + b_T 的闭合性")
print(f"  同源口径（前沿点自身 N）闭合差 = {bN*gN_S90+bT-r_obs:+.4f}，相对 r_obs 为 "
      f"{(bN*gN_S90+bT-r_obs)/r_obs*100:+.1f}%")
print(f"  异源口径（全样本 p90(N)）闭合差 = {bN*gN_N90+bT-r_obs:+.4f}，相对 r_obs 为 "
      f"{(bN*gN_N90+bT-r_obs)/r_obs*100:+.1f}%")

# ---- 自洽反解：使恒等式闭合所需的 g_N
gN_star = (r_obs - bT) / bN
print(f"\n  自洽反解 g_N* = (r_obs - b_T)/b_N = {gN_star:+.4f}/年")
print(f"  → 与 C4 普查 g_N=+0.9759 的差 = {gN_star-0.9759:+.4f}")
print(f"  → 与 C1 全样本 p90 g_N={gN_N90:+.4f} 的差 = {gN_star-gN_N90:+.4f}")

# ---- 贡献占比（三种 g_N）
print("\n【贡献占比】规模占比 = b_N·g_N / (b_N·g_N + b_T)")
for tag, g in [('自洽反解 g_N*', gN_star), ('C4 主口径', 0.9759), ('C4 全库', 0.9340),
               ('参考论文 1.251', 1.251), ('C1 前沿点自身 N', gN_S90), ('C1 全样本 p90(N)', gN_N90)]:
    sc = bN * g; ns = bT; tot = sc + ns
    print(f"  {tag:20s} g_N={g:+8.4f}  规模={sc:+8.4f}  非规模={ns:+7.4f}  合计={tot:+8.4f}  规模占比={sc/tot*100:+7.1f}%")

# ---- 时间窗口长度对 b_T 的影响（子窗口稳健性）
print("\n【窗口稳健性】按提交日期滚动子窗口的 QR90 b_N / b_T")
cuts = [pd.Timestamp('2024-06-08'), pd.Timestamp('2024-08-01'), pd.Timestamp('2024-10-01'),
        pd.Timestamp('2024-12-01'), pd.Timestamp('2025-01-01')]
for c0 in cuts:
    s = sub[sub['date'] >= c0]
    if len(s) < 100: continue
    Xs = np.column_stack([np.ones(len(s)), np.log(s['N'].values), s['t'].values])
    bs = qr_lp(Xs, np.log(s['S'].values), 0.9)
    print(f"  起点 {c0.date()}  n={len(s):5d}  b_N={bs[1]:+.4f}  b_T={bs[2]:+.4f}")
print("  判读：窗口越短，b_T 越大且越不稳定 → 0.76 年窗口对 b_T 的识别力弱，必须给区间而非点值。")
