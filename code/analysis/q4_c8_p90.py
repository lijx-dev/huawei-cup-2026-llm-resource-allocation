# -*- coding: utf-8 -*-
"""C8 综合分与逐维 p90 前沿增速（与主脚本统计量口径一致）。"""
import pandas as pd, numpy as np, os, re, json, warnings
warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
KEY = {'leaderboard_ifeval': ('inst_level_strict_acc,none', 'prompt_level_strict_acc,none'),
       'leaderboard_bbh': ('acc_norm,none',), 'leaderboard_math_hard': ('exact_match,none',),
       'leaderboard_gpqa': ('acc_norm,none',), 'leaderboard_musr': ('acc_norm,none',),
       'leaderboard_mmlu_pro': ('acc,none',)}
det = os.path.join(BASE, "detailed_results"); rows = []
for dd in os.listdir(det):
    dp = os.path.join(det, dd)
    if not os.path.isdir(dp): continue
    fn = [f for f in os.listdir(dp) if f.endswith('.json')]
    if not fn: continue
    try: js = json.load(open(os.path.join(dp, fn[0]), encoding='utf-8'))
    except Exception: continue
    g = js.get('groups', {}) or {}; res = js.get('results', {}) or {}
    r = {'key': re.sub(r'[^a-z0-9]', '', dd.lower())}
    for t, ks in KEY.items():
        src = g.get(t) or res.get(t)
        if isinstance(src, dict):
            v = [src.get(k) for k in ks if isinstance(src.get(k), (int, float))]
            if v: r[t] = float(np.mean(v)) * 100
    rows.append(r)
c8 = pd.DataFrame(rows); DIMS = list(KEY.keys()); c8 = c8.dropna(subset=DIMS).copy()
c8['equal'] = c8[DIMS].mean(axis=1)
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb['key'] = lb['Model'].map(lambda x: re.sub(r'[^a-z0-9]', '', str(x).lower()))
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
m = lb.merge(c8[['key', 'equal'] + DIMS], on='key', how='inner').dropna(subset=['equal'])
n_all = len(m)
m = m[m['q'] != 'NaT'].copy()
print(f"C8 六维 vs C1 汇总分（同模型，n={n_all}，其中日期有效 {len(m)}），季度 p90 前沿年化增速：")
res = {}
for col in ['S', 'equal'] + DIMS:
    g = m.groupby('q')[col].quantile(0.9)
    sl = float(np.polyfit(np.arange(len(g)), np.log(g.values), 1)[0] * 4)
    res[col] = dict(g=sl, series={k: float(v) for k, v in g.items()})
    print(f"  {col.replace('leaderboard_', ''):10s} 年化={sl:+.4f}  季度p90={g.round(1).to_dict()}")
print(f"\n匹配样本 n={len(m)}")
print(f"  C1 汇总分 与 C8 等权 的季度 p90 水平对照: "
      f"{m.groupby('q')['S'].quantile(.9).round(1).to_dict()} vs "
      f"{m.groupby('q')['equal'].quantile(.9).round(1).to_dict()}")
print(f"  逐模型水平差 (C8等权 - C1汇总分): 均值={float((m['equal']-m['S']).mean()):+.2f}  "
      f"中位={float((m['equal']-m['S']).median()):+.2f}  标准差={float((m['equal']-m['S']).std()):.2f}")
print(f"  逐模型水平差 与 C1汇总分 的相关: Pearson={float(np.corrcoef(m['equal']-m['S'], m['S'])[0,1]):+.4f}")
json.dump(res, open(r"d:\F题\q4_c8_p90_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
