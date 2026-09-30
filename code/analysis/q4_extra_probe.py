# -*- coding: utf-8 -*-
"""
问题四补充核验：
  A. C4 内部自洽性：Training compute 是否等于 6ND；g_C 与 g_N+g_D 的关系
  B. C4 主口径下的算力/数据/参数三增速与 Chinchilla 关系
  C. C8 综合能力口径：等权 / 分位 / 标准化三种加权下的前沿，及 lnN、t 回归
  D. C6 桥接：按可比性等级分层 + High 子集的小样本区间（bootstrap）
  E. 能力前沿的"腿"一致性：C1(季度) / C3(年度) / C8(季度) 三源增速对照
"""
import os, json, re, warnings
import numpy as np, pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
SEED = 20260925
def sec(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

def qr_lp(X, y, tau):
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    r = linprog(c, A_eq=A_eq, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method='highs')
    return r.x[:k] if r.success else None

c4 = pd.read_csv(os.path.join(BASE, "epoch_all_ai_models.csv"), low_memory=False)
c4['year'] = pd.to_datetime(c4['Publication date'], errors='coerce').dt.year
for c, nm in [('Parameters', 'P'), ('Training compute (FLOP)', 'C'),
              ('Training dataset size (total)', 'D')]:
    c4[nm] = pd.to_numeric(c4[c], errors='coerce')

# ---------------- A. C4 自洽性：C =? 6ND
sec("【A】C4 内部自洽性：Training compute 与 6ND 的一致性")
d = c4.dropna(subset=['P', 'C', 'D'])
d = d[(d['P'] > 0) & (d['C'] > 0) & (d['D'] > 0)]
ratio = d['C'] / (6 * d['P'] * d['D'])
print(f"  三字段齐全 n={len(d)}")
print(f"  C/(6ND) 分位: p10={ratio.quantile(.1):.3f}  p50={ratio.median():.3f}  p90={ratio.quantile(.9):.3f}")
lr = np.log(ratio)
print(f"  ln[C/(6ND)] 均值={lr.mean():+.4f} 标准差={lr.std():.4f}  → 中位偏离 6ND 约 {np.exp(lr.median()):.2f}×")
ok = (ratio > 0.5) & (ratio < 2)
print(f"  落在 0.5–2 倍带内: {int(ok.sum())}/{len(d)} = {ok.mean()*100:.1f}%")
print("  判读：C4 的 compute 字段与 6ND 同量级但非恒等，说明它是独立估算量；"
      "因此 g_C 与 g_N+g_D 不构成恒等式，三个增速须各自标口径报告。")

# ---------------- B. 主口径三增速
sec("【B】C4 主口径（Language + 开源权重）三增速与 Chinchilla 关系")
lang = c4['Domain'].astype(str).str.contains('Language', na=False)
openw = c4['Open model weights?'].astype(str).str.lower().eq('yes')
sub = c4[(lang & openw).values]
def growth(df, col, lo=2018, hi=2025, q=0.9):
    x = df.dropna(subset=[col, 'year']); x = x[(x['year'] >= lo) & (x['year'] <= hi) & (x[col] > 0)]
    g = x.groupby('year')[col].quantile(q)
    if len(g) < 3: return None, g
    return float(np.polyfit(g.index.values.astype(float), np.log(g.values), 1)[0]), g
gN, sN = growth(sub, 'P'); gC, sC = growth(sub, 'C'); gD, sD = growth(sub, 'D')
print(f"  g_N = {gN:+.4f}/年   g_C = {gC:+.4f}/年   g_D = {gD:+.4f}/年")
print(f"  检验 g_C ?= g_N + g_D :  {gC:+.4f}  vs  {gN+gD:+.4f}   差 {gC-(gN+gD):+.4f}")
print(f"  检验 g_N ?= g_C/2（D∝N 的 Chinchilla 等比例扩张）:  {gN:+.4f} vs {gC/2:+.4f}")
print(f"  实测 D/N 比值年化变化 = g_D - g_N = {gD-gN:+.4f}  → "
      f"{'数据增速快于参数，D/N 上升' if gD>gN else '参数增速快于数据，D/N 下降'}")
print(f"  p90 逐年: N={sN.round(2).to_dict()}")

# ---------------- C. C8 综合能力口径
sec("【C】C8 逐任务聚合：三种综合能力口径下的前沿与分解")
KEY = {'leaderboard_ifeval': ('inst_level_strict_acc,none', 'prompt_level_strict_acc,none'),
       'leaderboard_bbh': ('acc_norm,none',), 'leaderboard_math_hard': ('exact_match,none',),
       'leaderboard_gpqa': ('acc_norm,none',), 'leaderboard_musr': ('acc_norm,none',),
       'leaderboard_mmlu_pro': ('acc,none',)}
det = os.path.join(BASE, "detailed_results")
rows, bad = [], 0
for dd in os.listdir(det):
    dp = os.path.join(det, dd)
    if not os.path.isdir(dp): continue
    fn = [f for f in os.listdir(dp) if f.endswith('.json')]
    if not fn: continue
    try: js = json.load(open(os.path.join(dp, fn[0]), encoding='utf-8'))
    except Exception: bad += 1; continue
    g = js.get('groups', {}) or {}; res = js.get('results', {}) or {}
    r = {'key': re.sub(r'[^a-z0-9]', '', dd.lower()), 'raw': dd}
    for t, ks in KEY.items():
        src = g.get(t) or res.get(t)
        if isinstance(src, dict):
            v = [src.get(k) for k in ks if isinstance(src.get(k), (int, float))]
            if v: r[t] = float(np.mean(v)) * 100
    rows.append(r)
c8 = pd.DataFrame(rows)
DIMS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
        'leaderboard_gpqa', 'leaderboard_musr', 'leaderboard_mmlu_pro']
c8 = c8.dropna(subset=DIMS).copy()
c8['equal'] = c8[DIMS].mean(axis=1)
# 难度加权：以全样本各维均值的倒数作权重（越难权重越大）
w = 1.0 / c8[DIMS].mean(axis=0)
w = w / w.sum()
c8['hardw'] = (c8[DIMS] * w).sum(axis=1)
# 标准化（z 后取均值）——消除量纲差异
z = (c8[DIMS] - c8[DIMS].mean(axis=0)) / c8[DIMS].std(axis=0)
c8['zmean'] = z.mean(axis=1)
print(f"  可解析且六维齐全 n={len(c8)}（损坏 {bad}）")
print(f"  权重(难度): { {k.replace('leaderboard_',''): round(float(v),4) for k,v in w.items()} }")

lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']); lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['key'] = lb['Model'].map(lambda x: re.sub(r'[^a-z0-9]', '', str(x).lower()))
lb['t'] = (lb['date'] - pd.Timestamp('2022-01-01')).dt.days / 365.25
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
lb['lnN'] = np.log(lb['N'])
m = lb.merge(c8[['key', 'equal', 'hardw', 'zmean'] + DIMS], on='key', how='inner')
m = m.dropna(subset=['equal']).copy()
print(f"  与 C1 匹配 n={len(m)}   相关: equal~S Spearman={m['equal'].corr(m['S'],method='spearman'):.4f}  "
      f"hardw~S={m['hardw'].corr(m['S'],method='spearman'):.4f}")

print(f"\n  {'口径':10s} {'QR90 b_N':>10s} {'QR90 b_T':>10s} {'OLS b_N':>9s} {'OLS b_T':>9s} | 季度前沿年化增速")
front = {}
for col, nm in [('S', 'C1汇总分'), ('equal', 'C8等权'), ('hardw', 'C8难度加权'), ('zmean', 'C8标准化')]:
    mm = m.dropna(subset=[col])
    mm = mm[(mm[col] > 0)]
    y = np.log(mm[col].values) if col != 'zmean' else mm[col].values
    X = np.column_stack([np.ones(len(mm)), mm['lnN'].values, mm['t'].values])
    bq = qr_lp(X, y, 0.9); bo = np.linalg.lstsq(X, y, rcond=None)[0]
    gq = mm.groupby('q')[col].max() if col != 'zmean' else mm.groupby('q')[col].max()
    tt = np.arange(len(gq))
    sl = float(np.polyfit(tt, np.log(np.clip(gq.values, 1e-6, None)), 1)[0] * 4)
    front[nm] = dict(qr_bN=float(bq[1]), qr_bT=float(bq[2]), ols_bN=float(bo[1]),
                     ols_bT=float(bo[2]), frontier_g=sl)
    print(f"  {nm:10s} {bq[1]:+10.4f} {bq[2]:+10.4f} {bo[1]:+9.4f} {bo[2]:+9.4f} | {sl:+.4f}")

# 逐任务前沿增速排序
print("\n  逐任务季度前沿年化对数增速（max 口径）:")
task_g = {}
for k in DIMS:
    gq = m.groupby('q')[k].max()
    sl = float(np.polyfit(np.arange(len(gq)), np.log(gq.values), 1)[0] * 4)
    task_g[k.replace('leaderboard_', '')] = sl
for k, v in sorted(task_g.items(), key=lambda x: -x[1]):
    print(f"    {k:12s} {v:+.4f}")
print(f"  判读：逐任务增速极差 {max(task_g.values())-min(task_g.values()):.3f}"
      f"（最快 {max(task_g,key=task_g.get)} {max(task_g.values()):+.3f} vs "
      f"最慢 {min(task_g,key=task_g.get)} {min(task_g.values()):+.3f}）→ 单一汇总分会掩盖任务间分化，"
      f"综合能力口径必须显式声明加权方式。")

# ---------------- D. C6 桥接分层
sec("【D】C6 桥接：可比性分层与 High 子集 bootstrap 区间")
c6 = pd.read_csv(os.path.join(BASE, "loss_benchmark_bridge_expanded.csv"), low_memory=False)
br = c6.dropna(subset=['Val_Loss', 'LB_Average']).copy()
br = br[(br['Val_Loss'] > 0) & (br['LB_Average'] > 0)]
print(f"  Loss_Comparability 取值: {br['Loss_Comparability'].value_counts().to_dict()}")
print(f"  Loss_Source 取值: {br['Loss_Source'].value_counts().to_dict()}")
L = br['Val_Loss'].values; A = br['LB_Average'].values
hi = br['Loss_Comparability'].astype(str).str.contains('High', na=False).values
def fit(msk, tag):
    if msk.sum() < 4: print(f"    {tag}: n={int(msk.sum())} 太少"); return None
    c = np.polyfit(np.log(L[msk]), np.log(A[msk]), 1)
    pred = np.polyval(c, np.log(L[msk]))
    r2 = 1 - ((np.log(A[msk]) - pred) ** 2).sum() / ((np.log(A[msk]) - np.log(A[msk]).mean()) ** 2).sum()
    sd = float(np.std(np.log(A[msk]) - pred))
    print(f"    {tag:22s} n={int(msk.sum()):3d}  k={c[0]:+.4f}  b={c[1]:+.4f}  R2={r2:+.4f}  "
          f"对数残差σ={sd:.4f} (≈{sd*100:.1f}% 能力相对误差)")
    return dict(n=int(msk.sum()), k=float(c[0]), b=float(c[1]), r2=float(r2), sigma=float(sd))
print("  lnA ~ lnL 分层:")
res = {}
for tag, msk in [('全部', np.ones(len(L), bool)), ('High 可比', hi), ('Medium/其他', ~hi)]:
    res[tag] = fit(msk, tag)

if hi.sum() >= 6:
    rng = np.random.default_rng(SEED)
    ks = []
    for _ in range(2000):
        i = rng.integers(0, hi.sum(), hi.sum())
        ks.append(np.polyfit(np.log(L[hi][i]), np.log(A[hi][i]), 1)[0])
    print(f"    High 子集斜率 k 的 bootstrap 90%CI = [{np.percentile(ks,5):+.4f}, {np.percentile(ks,95):+.4f}]"
          f"  含 0 ? {np.percentile(ks,5)<=0<=np.percentile(ks,95)}")
    print(f"    High 子集损失范围 [{L[hi].min():.3f}, {L[hi].max():.3f}]  "
          f"→ 外推区间外无信息，斜率不可用于远端外推")

# ---------------- E. 三腿一致性
sec("【E】能力前沿三腿增速对照")
c3 = pd.read_csv(os.path.join(BASE, "leaderboard_extended_timeseries.csv"), low_memory=False)
c3['S'] = pd.to_numeric(c3['Average'], errors='coerce')
hist = c3.groupby('Year')['S'].max().sort_index()
cum = hist.cummax()
g_c3 = float(np.polyfit(cum.index.values.astype(float), np.log(cum.values), 1)[0])
qfront = lb.groupby('q')['S'].quantile(.9)
g_c1 = float(np.polyfit(np.arange(len(qfront)), np.log(qfront.values), 1)[0] * 4)
print(f"  腿1 C1 季度 p90 前沿年化增速        = {g_c1:+.4f}   (窗口 {lb['date'].min().date()}~{lb['date'].max().date()})")
print(f"  腿2 C3 年度累计最大前沿年化增速      = {g_c3:+.4f}   (2019–2025，长历史)")
print(f"  腿3 机制腿 b_N·g_N + b_T           = 见主脚本（结构分解）")
print(f"  判读：C3 长历史增速 {g_c3:+.3f} 远低于 C1 近期 {g_c1:+.3f}，"
      f"比值 {g_c1/g_c3:.2f}× → 近两年前沿加速，长周期预测不宜直接用近窗口斜率外推 24 个月。")

out = dict(c4_consistency=dict(n=int(len(d)), median_ratio=float(ratio.median()),
                               in_band_pct=float(ok.mean() * 100)),
           c4_growth=dict(g_N=gN, g_C=gC, g_D=gD, g_C_minus_sum=float(gC - (gN + gD))),
           c8_frontier=front, c8_task_growth={k: float(v) for k, v in task_g.items()},
           bridge_layers=res, legs=dict(c1=g_c1, c3=g_c3))
json.dump(out, open(r"d:\F题\q4_extra_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n已写出 d:\\F题\\q4_extra_results.json")
