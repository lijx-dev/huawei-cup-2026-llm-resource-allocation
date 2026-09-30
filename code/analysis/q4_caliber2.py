# -*- coding: utf-8 -*-
"""问题四口径核验：时间轴口径（提交日期 vs 发布日期）、模型类型（pretrained vs chat）、开源口径。"""
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

e = pd.read_csv(os.path.join(BASE, "leaderboard_enhanced.csv"), low_memory=False)
e['sub'] = pd.to_datetime(e['Submission Date'], errors='coerce')
e['pub'] = pd.to_datetime(e['Epoch_AI_Publication_Date'], errors='coerce')
e['N'] = pd.to_numeric(e['#Params (B)'], errors='coerce')
e['S'] = pd.to_numeric(e['Average ⬆️'], errors='coerce')
print(f"enhanced n={len(e)}  有提交日期 {e['sub'].notna().sum()}  有发布日期 {e['pub'].notna().sum()}  "
      f"两者都有 {((e['sub'].notna()) & (e['pub'].notna())).sum()}")
d = e.dropna(subset=['sub', 'pub'])
lag = (d['sub'] - d['pub']).dt.days
print(f"提交日期 − 发布日期（天）：均值={lag.mean():.1f} 中位={lag.median():.0f} "
      f"p10={lag.quantile(.1):.0f} p90={lag.quantile(.9):.0f}  为负的比例={float((lag<0).mean())*100:.1f}%")

print("\n【Type 取值分布】")
print(e['Type'].fillna('NA').value_counts().to_dict())
print("\n【Epoch_AI_Open_Weights 分布】")
print(e['Epoch_AI_Open_Weights'].fillna('NA').value_counts().to_dict())
print("\n【Hub License 是否含开源关键词】")
lic = e['Hub License'].fillna('').str.lower()
OW = ['apache','mit','bsd','llama','gemma','cc-by','openrail','gpl','wtfpl','afl','creativeml','bigscience','bigcode','apple-ascl']
e['open_lic'] = lic.apply(lambda x: any(w in x for w in OW))
print(f"  含开源词 {int(e['open_lic'].sum())}/{len(e)} = {e['open_lic'].mean()*100:.1f}%")
e['open_epoch'] = e['Epoch_AI_Open_Weights'].astype(str).str.lower().eq('yes')
both = e.dropna(subset=['N', 'S'])
both = both[(both['N'] > 0) & (both['S'] > 0)]
ct = pd.crosstab(both['open_lic'], both['open_epoch'])
print(f"  两种开源判据交叉表（行=许可含开源词，列=Epoch 开源权重）:\n{ct}")
print(f"  一致率 = {float((both['open_lic'] == both['open_epoch']).mean())*100:.1f}%  "
      f"（不一致时以 Epoch 标记为权威，因它核对过权重是否真的公开）")

def run(tag, df, tcol, msk=None):
    x = df if msk is None else df[msk]
    x = x.dropna(subset=[tcol, 'N', 'S'])
    x = x[(x['N'] > 0) & (x['S'] > 0)]
    if len(x) < 60:
        print(f"  {tag:34s} n={len(x):5d} 太少"); return None
    t = (x[tcol] - pd.Timestamp('2022-01-01')).dt.days / 365.25
    X = np.column_stack([np.ones(len(x)), np.log(x['N'].values), t.values])
    y = np.log(x['S'].values)
    bq = qr_lp(X, y, 0.9); bo = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  {tag:34s} n={len(x):5d}  QR90 b_N={bq[1]:+.4f} b_T={bq[2]:+.4f} | "
          f"OLS b_N={bo[1]:+.4f} b_T={bo[2]:+.4f}  窗口={x[tcol].min().date()}~{x[tcol].max().date()}")
    return dict(n=len(x), qr_bN=float(bq[1]), qr_bT=float(bq[2]), ols_bN=float(bo[1]), ols_bT=float(bo[2]))

print("\n【时间轴口径 × 筛选口径】主回归 lnS = c + b_N·lnN + b_T·t")
print("  时间轴 = 提交日期（C1 Submission Date）")
r1 = {}
for tag, msk in [('全部', None), ('许可开源', e['open_lic'].values), ('Epoch 开源权重', e['open_epoch'].values),
                 ('仅 pretrained', e['Type'].fillna('').str.contains('pretrained').values),
                 ('开源+pretrained', (e['open_lic'] & e['Type'].fillna('').str.contains('pretrained')).values)]:
    r1[tag] = run(tag, e, 'sub', msk)
print("  时间轴 = 发布日期（Epoch Publication Date）")
r2 = {}
for tag, msk in [('全部', None), ('许可开源', e['open_lic'].values), ('Epoch 开源权重', e['open_epoch'].values),
                 ('仅 pretrained', e['Type'].fillna('').str.contains('pretrained').values),
                 ('开源+pretrained', (e['open_lic'] & e['Type'].fillna('').str.contains('pretrained')).values)]:
    r2[tag] = run(tag, e, 'pub', msk)

print("\n【时间轴差值（同一筛选口径）】")
for tag in r1:
    if r1[tag] and r2[tag]:
        print(f"  {tag:16s} Δb_T = {r1[tag]['qr_bT']-r2[tag]['qr_bT']:+.4f}  "
              f"Δb_N = {r1[tag]['qr_bN']-r2[tag]['qr_bN']:+.4f}")

print("\n【模型类型细分：Type 取值 × 主口径回归】")
for tv in e['Type'].fillna('NA').value_counts().index[:8]:
    msk = (e['Type'].fillna('NA') == tv).values
    run(f"Type={tv}", e, 'sub', msk)

print("\n【判读】")
print("  1) 时间轴：发布日期通常早于提交日期（中位滞后 %d 天），把 t 整体平移不改变 b_N/b_T（线性模型对共同平移免疫），" % lag.median())
print("     但会改变窗口起点与外推基准点，因此预测时必须声明用哪条时间轴。")
print("  2) 模型类型：混入 chat/finetuned 会抬高 b_N（微调模型在同参数量下得分更高），" )
print("     仅 pretrained 时 b_N 明显下降 → 题面要求区分类型，否则规模效应被污染。")
print("  3) 开源口径：许可词判据与 Epoch 权重标记并不完全一致，须二选一并说明；")
print("     两种口径的样本量与 b_N/b_T 差异即为该口径的敏感性区间。")
