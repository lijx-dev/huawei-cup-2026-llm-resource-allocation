# -*- coding: utf-8 -*-
"""
问题四「三条证据腿」合成：
  腿1 机制腿：问题三 L*(C) 的预算弹性 ε_C × C4 算力增速 g_C → 桥接 lnA~lnL 的斜率 k
  腿2 近期前沿腿：C1 季度 p90(S) 轨迹的年化增速
  腿3 长历史腿：C3 年度累计最大前沿的年化增速
按逆方差合成，给出 12/24 个月前沿预测。
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
SEED = 20260925

# ---------- 腿1 机制腿
eps_C = -0.1404134235356952          # 问题三解析预算弹性 dlnL/dlnC
k_br, b_br = -2.7353, 4.8038         # C6 桥接 lnA ~ lnL
sig_br = 0.5825                      # 桥接对数残差 σ
gC = 1.2875                          # C4 Language+开源 算力增速
g_mech = k_br * eps_C * gC
print(f"【腿1 机制腿】dlnS/dt = k · ε_C · g_C = {k_br:.4f} × ({eps_C:.4f}) × {gC:.4f} = {g_mech:+.4f}/年")
print(f"  等价写法：L ∝ C^{{ε_C}}，C ∝ e^{{g_C t}}，S ∝ L^{{k}} → dlnS/dt = {g_mech:+.4f}")

# ---------- 腿2 近期前沿腿
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']); lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
lic = lb['Hub License'].fillna('').str.lower()
OW = ['apache','mit','bsd','llama','gemma','cc-by','openrail','gpl','wtfpl','afl','creativeml','bigscience','bigcode','apple-ascl']
sub = lb[lic.apply(lambda x: any(w in x for w in OW))].copy()
qs = sorted(sub['q'].unique())
qfront = sub.groupby('q')['S'].quantile(0.9).reindex(qs)
r2 = float(np.polyfit(np.arange(len(qfront)), np.log(qfront.values), 1)[0] * 4)
rng = np.random.default_rng(SEED)
bs = []
for _ in range(4000):
    i = rng.integers(0, len(qfront), len(qfront))
    v = np.log(qfront.values[i])
    if len(np.unique(i)) < 3: continue
    bs.append(np.polyfit(np.arange(len(qfront))[i], v, 1)[0] * 4)
sd2 = float(np.std(bs))
print(f"\n【腿2 近期前沿腿】C1 OPEN_LIC 季度 p90(S) 年化增速 = {r2:+.4f}   季度值={qfront.round(2).to_dict()}")
print(f"  季度 bootstrap 标准差 = {sd2:.4f}（仅 4 个季度点，自由度极低）")

# ---------- 腿3 长历史腿
c3 = pd.read_csv(os.path.join(BASE, "leaderboard_extended_timeseries.csv"), low_memory=False)
c3['S'] = pd.to_numeric(c3['Average'], errors='coerce')
cum = c3.groupby('Year')['S'].max().sort_index().cummax()
r3 = float(np.polyfit(cum.index.values.astype(float), np.log(cum.values), 1)[0])
bs3 = []
yy = cum.index.values.astype(float); vv = np.log(cum.values)
for _ in range(4000):
    i = rng.integers(0, len(vv), len(vv))
    if len(np.unique(i)) < 3: continue
    bs3.append(np.polyfit(yy[i], vv[i], 1)[0])
sd3 = float(np.std(bs3))
print(f"\n【腿3 长历史腿】C3 累计最大前沿年化增速 = {r3:+.4f}   逐年={cum.to_dict()}")
print(f"  年度 bootstrap 标准差 = {sd3:.4f}（7 个年度点）")

# ---------- 腿1 的不确定性
sd1_br = abs(k_br) * abs(eps_C) * gC * (sig_br / 2)   # 桥接斜率不确定性的保守折算
sd1 = float(np.sqrt((abs(eps_C) * gC * sig_br) ** 2 + (abs(k_br) * gC * 0.02) ** 2))
print(f"\n【腿1 不确定性】σ(g_mech) ≈ {sd1:.4f}（桥接对数残差 {sig_br:.3f} 与弹性不确定 0.02 的合成）")

# ---------- 逆方差合成
legs = {'腿1 机制': (g_mech, sd1), '腿2 近期前沿': (r2, sd2), '腿3 长历史': (r3, sd3)}
w = {k: 1 / v[1] ** 2 for k, v in legs.items()}
W = sum(w.values())
g_comb = sum(legs[k][0] * w[k] for k in legs) / W
sd_comb = float(np.sqrt(1 / W))
print(f"\n【逆方差合成】")
for k, (g, s) in legs.items():
    print(f"  {k:12s} 增速={g:+.4f}  σ={s:.4f}  权重={w[k]/W*100:5.1f}%")
print(f"  合成增速 = {g_comb:+.4f}/年   σ = {sd_comb:.4f}   相对标准差 = {sd_comb/abs(g_comb)*100:.1f}%")
print(f"  腿间极差 = {max(v[0] for v in legs.values())-min(v[0] for v in legs.values()):.4f} "
      f"→ 腿间差异远大于单腿统计误差，合成权重不可掩盖系统差异")

# ---------- 预测
S0 = float(sub[sub['q'] == qs[-1]]['S'].quantile(0.9))
logit0 = np.log((S0 / 100) / (1 - S0 / 100))
print(f"\n【预测】基准点 {qs[-1]} p90(S) = {S0:.2f}")
print(f"  {'口径':22s} {'12M':>8s} {'24M':>8s} {'90%CI(24M)':>18s}")
rng2 = np.random.default_rng(SEED + 9)
for tag, g in [('腿合成（机制+近期+长史）', g_comb), ('腿1 机制', g_mech), ('腿2 近期前沿', r2), ('腿3 长历史', r3)]:
    lo_l = logit0 + (g - 1.645 * sd_comb) * 2
    hi_l = logit0 + (g + 1.645 * sd_comb) * 2
    s12 = 100 / (1 + np.exp(-(logit0 + g)))
    s24 = 100 / (1 + np.exp(-(logit0 + 2 * g)))
    print(f"  {tag:22s} {s12:8.2f} {s24:8.2f}   [{100/(1+np.exp(-hi_l)):6.2f},{100/(1+np.exp(-lo_l)):6.2f}]")
print("  说明：logit 有界形式保证不越 0–100；线性外推在 24M 已越界（见主脚本）。")

json.dump(dict(legs={k: dict(g=v[0], sd=v[1], w=w[k] / W) for k, v in legs.items()},
               g_comb=g_comb, sd_comb=sd_comb, S0=S0,
               bridge=dict(k=k_br, b=b_br, sigma=sig_br), eps_C=eps_C, gC=gC),
          open(r"d:\F题\q4_legs_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n已写出 d:\\F题\\q4_legs_results.json")
