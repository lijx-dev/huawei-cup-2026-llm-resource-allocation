# -*- coding: utf-8 -*-
"""溯源：参考论文 g_N=1.251 的可能出处（C4 参数量/算力在不同分位与年份窗口下的年化增速）。"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
c4 = pd.read_csv(os.path.join(BASE, "epoch_all_ai_models.csv"), low_memory=False)
c4['year'] = pd.to_datetime(c4['Publication date'], errors='coerce').dt.year
for c, nm in [('Parameters', 'P'), ('Training compute (FLOP)', 'C'),
              ('Training dataset size (total)', 'D')]:
    c4[nm] = pd.to_numeric(c4[c], errors='coerce')
lang = c4['Domain'].astype(str).str.contains('Language', na=False)
openw = c4['Open model weights?'].astype(str).str.lower().eq('yes')

def g(df, col, lo, hi, q):
    x = df.dropna(subset=[col, 'year']); x = x[(x['year'] >= lo) & (x['year'] <= hi) & (x[col] > 0)]
    gg = x.groupby('year')[col].quantile(q)
    if len(gg) < 3: return None
    return float(np.polyfit(gg.index.values.astype(float), np.log(gg.values), 1)[0])

print(f"{'子集':22s} {'字段':4s} {'窗口':12s} {'分位':>5s} {'年化增速':>9s}")
hits = []
for nm, msk in [('全库', np.ones(len(c4), bool)), ('Language', lang.values), ('Language+开源', (lang & openw).values)]:
    for col in ['P', 'C', 'D']:
        for lo, hi in [(2018, 2025), (2019, 2025), (2020, 2025), (2021, 2025), (2018, 2024), (2020, 2024)]:
            for q in [0.5, 0.75, 0.9, 0.95]:
                v = g(c4[msk], col, lo, hi, q)
                if v is None: continue
                if abs(v - 1.251) < 0.06:
                    hits.append((nm, col, f'{lo}-{hi}', q, v))
                    print(f"{nm:22s} {col:4s} {f'{lo}-{hi}':12s} {q:5.2f} {v:+9.4f}   ← 接近 1.251")
print(f"\n接近 1.251（±0.06）的组合数 = {len(hits)}")
# 参考论文 n=2493 对应哪个筛选
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lic = lb['Hub License'].fillna('').str.lower()
OW = ['apache','mit','bsd','llama','gemma','cc-by','openrail','gpl','wtfpl','afl','creativeml','bigscience','bigcode','apple-ascl']
print(f"\nC1 行数={len(lb)}  许可开源={int(lic.apply(lambda x: any(w in x for w in OW)).sum())}  参考论文 n=2493")
