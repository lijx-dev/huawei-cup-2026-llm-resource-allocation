# -*- coding: utf-8 -*-
"""
问题四有界前沿预测（最终版）：
  问题：ln 尺度回归给 dlnS/dt = b_N·g_N + b_T，而 logit 尺度回归给另一个量级，
        两个函数形式隐含的增速相差约 2.8 倍 —— 必须先统一，再做外推。
  方案：采用 logistic 有界形式 S(t)=100/(1+A e^{-r t})，A=(100-S_0)/S_0，
        标定 r 使 t=0 处的 dlnS/dt 恰等于目标增速 g_target：r = g_target/(1-S_0/100)。
        这样 (i) 短周期与 ln 尺度增速一致；(ii) 长周期自动饱和于 100，不越界。
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
SEED = 20260925

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
S0 = float(sub[sub['q'] == qs[-1]]['S'].quantile(0.9))
print(f"基准点 {qs[-1]}：S_0 = {S0:.2f}（OPEN_LIC p90）")

# ---- 目标增速（三条腿 + 合成，来自 q4_legs.py）
g_mech, sd_mech = 0.4945, 0.1267
g_rec,  sd_rec  = 0.6326, 0.0699
g_hist, sd_hist = 0.2540, 0.1908
w = np.array([1/sd_mech**2, 1/sd_rec**2, 1/sd_hist**2]); w = w/w.sum()
g_comb = g_mech*w[0] + g_rec*w[1] + g_hist*w[2]
sd_comb = float(np.sqrt(1/sum([1/sd_mech**2, 1/sd_rec**2, 1/sd_hist**2])))
print(f"腿权重: 机制 {w[0]*100:.1f}% | 近期前沿 {w[1]*100:.1f}% | 长历史 {w[2]*100:.1f}%")
print(f"合成增速 g_comb = {g_comb:+.4f}/年  σ = {sd_comb:.4f}")

def logistic_path(g_target, S0=40.42, h=2.0, n=2400):
    A = (100 - S0) / S0
    r = g_target / (1 - S0 / 100)          # 标定：t=0 处 dlnS/dt = g_target
    t = np.linspace(0, h, n)
    S = 100 / (1 + A * np.exp(-r * t))
    return S, r

def path_at(g_target, h):
    S, r = logistic_path(g_target, S0, h)
    return float(S[-1]), float(r)

# ---- 与「无界对数线性」并列对照
bN, bT, gN_C4 = 0.3356, 0.5140, 0.9759
g_acct = bN * gN_C4 + bT
print(f"\n对照：ln 尺度增长核算 g_acct = b_N·g_N + b_T = {g_acct:+.4f}/年")
print(f"      logit 尺度回归隐含的 ln 尺度增速 ≈ p(1-p)·(b_N'+b_T'·... ) ≈ "
      f"{(S0/100)*(1-S0/100)*1.2289:+.4f}/年  ← 与 g_acct 相差 "
      f"{g_acct/((S0/100)*(1-S0/100)*1.2289):.1f} 倍，两个函数形式不自洽，须择一并声明")

print("\n" + "=" * 100)
print("【主表】12/24 个月前沿预测（有界 logistic，基准点 S_0=%.2f）" % S0)
print("=" * 100)
print(f"  {'情景':38s} {'g_target':>9s} {'12M':>7s} {'24M':>7s} {'90%CI(24M)':>17s}")
rows = []
for tag, g in [('腿合成（机制21%+近期70%+长史9%）', g_comb),
               ('腿1 机制（问题三 L*(C) + 桥接）', g_mech),
               ('腿2 近期前沿（C1 季度 p90）', g_rec),
               ('腿3 长历史（C3 累计最大）', g_hist),
               ('增长核算（b_N·g_N+b_T，C4 g_N）', g_acct),
               ('参考论文口径（整数年 + g_N=1.251）', 0.5456)]:
    s12, r = path_at(g, 1.0); s24, _ = path_at(g, 2.0)
    lo, _ = path_at(g - 1.645 * sd_comb, 2.0); hi, _ = path_at(g + 1.645 * sd_comb, 2.0)
    rows.append(dict(tag=tag, g=g, s12=s12, s24=s24, lo=lo, hi=hi, r=r))
    print(f"  {tag:38s} {g:+9.4f} {s12:7.2f} {s24:7.2f}   [{lo:6.2f},{hi:6.2f}]")

print("\n【算力放缓情景】两种定义")
print(f"  A 整体增速减半（参考论文式，粗放）")
print(f"  B 仅规模/算力分量减半：g' = g - 0.5·b_N·g_N = g - {0.5*bN*gN_C4:.4f}（依据增长核算分解）")
print(f"  {'情景':38s} {'g_target':>9s} {'12M':>7s} {'24M':>7s}")
for tag, g in [('基础 腿合成', g_comb), ('放缓A 腿合成×0.5', g_comb * 0.5),
               ('放缓B 腿合成−0.5·b_N·g_N', g_comb - 0.5 * bN * gN_C4)]:
    s12, _ = path_at(g, 1.0); s24, _ = path_at(g, 2.0)
    print(f"  {tag:38s} {g:+9.4f} {s12:7.2f} {s24:7.2f}")
    rows.append(dict(tag=tag, g=g, s12=s12, s24=s24, lo=None, hi=None, r=None))

print("\n【无界 vs 有界对照（说明为什么必须用有界形式）】")
for tag, g in [('腿合成', g_comb), ('增长核算', g_acct), ('参考论文口径', 0.5456)]:
    lin12 = S0 * np.exp(g * 1); lin24 = S0 * np.exp(g * 2)
    s12, _ = path_at(g, 1.0); s24, _ = path_at(g, 2.0)
    print(f"  {tag:10s} 无界 12M={lin12:7.2f} 24M={lin24:8.2f}   |   有界 12M={s12:6.2f} 24M={s24:6.2f}"
          f"   {'← 无界已越 100' if lin24 > 100 else ''}")

print("\n【不确定性三源分解（腿合成，24M）】")
# 残差 σ（ln 尺度）→ 折算到 S 的水平不确定
sig_res = 0.4724
S24, _ = path_at(g_comb, 2.0)
v_res = (S24 * sig_res) ** 2
v_par = (S24 * sd_comb * 2 * (1 - S24 / 100) / (1 - S0 / 100)) ** 2
v_scen = ((S24 - path_at(g_comb * 0.5, 2.0)[0]) / 1.645) ** 2
tot = v_res + v_par + v_scen
print(f"  24M 点估计 = {S24:.2f}")
print(f"    残差/模型误差 {v_res/tot*100:5.1f}%  |  参数误差 {v_par/tot*100:5.1f}%  "
      f"|  情景（放缓）误差 {v_scen/tot*100:5.1f}%")
print("  判读：长期预测的不确定性由『模型残差』与『情景设定』主导，参数统计误差占比很小，"
      "因此把 90%CI 只算参数区间会严重低估风险。")

json.dump(dict(S0=S0, g_comb=g_comb, sd_comb=sd_comb, weights=w.tolist(), rows=rows,
               g_acct=g_acct, g_mech=g_mech, g_rec=g_rec, g_hist=g_hist,
               unc_24M=dict(res=v_res/tot, par=v_par/tot, scen=v_scen/tot)),
          open(r"d:\F题\q4_forecast_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n已写出 d:\\F题\\q4_forecast_results.json")
