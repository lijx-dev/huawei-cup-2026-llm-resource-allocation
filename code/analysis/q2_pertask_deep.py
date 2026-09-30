# -*- coding: utf-8 -*-
"""逐任务结果 + 上下文长度(max_length) 的实证分析
回应两条用户要求：
  (a) 逐任务结果别忽略
  (b) 上下文长度不是自由寻优变量
"""
import os, json, re, glob
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'd:\F题\F题\real_attachments\C_efficiency_evolution'
DR = os.path.join(BASE, 'detailed_results')
OUT = r'd:\F题'

models = sorted([d for d in os.listdir(DR) if os.path.isdir(os.path.join(DR, d))])
print(f'模型目录数 = {len(models)}')

recs, bad = [], []
for m in models:
    fs = sorted(glob.glob(os.path.join(DR, m, '*.json')))
    if not fs:
        bad.append((m, 'no json')); continue
    try:
        with open(fs[0], 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        bad.append((m, f'{type(e).__name__}: {str(e)[:60]}')); continue

    res = d.get('results', {})
    row = {'model_dir': m, 'max_length': d.get('max_length', np.nan),
           'date': d.get('date', None), 'model_name': d.get('model_name', None),
           'n_shot_default': None, 'eval_secs': d.get('total_evaluation_time_seconds', np.nan)}
    ns = d.get('n-shot', {})
    row['n_shot_default'] = ns.get('leaderboard', np.nan) if isinstance(ns, dict) else np.nan
    for t, v in res.items():
        if not isinstance(v, dict):
            continue
        for kk, vv in v.items():
            if isinstance(vv, (int, float)) and not kk.endswith('_stderr'):
                row[f'{t}::{kk}'] = vv
    recs.append(row)

print(f'解析成功 = {len(recs)}，截断/损坏 = {len(bad)}')
print()
print('=' * 90)
print('【截断 JSON 清单】（按项目规则：整份舍弃，不补全）')
print('=' * 90)
for m, e in bad:
    print(f'  {m:<50} {e}')

df = pd.DataFrame(recs)
df.to_csv(os.path.join(OUT, 'q2_pertask_wide.csv'), index=False, encoding='utf-8-sig')
print(f'\n已导出宽表: q2_pertask_wide.csv  shape={df.shape}')

print()
print('=' * 90)
print('一、上下文长度 max_length 的分布（用户：不是自由寻优变量）')
print('=' * 90)
ml = df.max_length.dropna()
print(f'有 max_length 记录数 = {len(ml)} / {len(df)}  (缺 {len(df)-len(ml)})')
vc = ml.value_counts().sort_index()
print('取值分布:')
for v, c in vc.items():
    print(f'  max_length={int(v):>6}  n={c:>5}  ({c/len(ml)*100:5.1f}%)')

print()
print('max_length 与模型规模的关系（若上下文长度是自由寻优变量，应看到它随 Params 增长被"优化"）:')
lb = pd.read_csv(os.path.join(BASE, 'leaderboard_extended_timeseries.csv'))
lb['key'] = lb.Model.astype(str).str.replace('/', '_', regex=False)
df['key'] = df.model_dir.astype(str)
mg = df.merge(lb, on='key', how='inner')
print(f'  与排行榜可匹配的模型数 = {len(mg)}')
if len(mg) > 30:
    g = mg.dropna(subset=['max_length'])
    print(f'  可匹配且含 max_length 的样本 = {len(g)}')
    from scipy import stats
    r = stats.spearmanr(g.max_length, g.Params_B)
    print(f'  Spearman(max_length, Params_B) = {r.statistic:+.3f}, p={r.pvalue:.3g}')
    r2 = stats.spearmanr(g.max_length, g.Year)
    print(f'  Spearman(max_length, Year)     = {r2.statistic:+.3f}, p={r2.pvalue:.3g}')
    sub = g[g.max_length.isin([2048, 4096])]
    if len(sub) > 10:
        a = sub[sub.max_length == 2048].Average
        b = sub[sub.max_length == 4096].Average
        print(f'  max_length=2048 组: n={len(a)}, Average 均值={a.mean():.4f}')
        print(f'  max_length=4096 组: n={len(b)}, Average 均值={b.mean():.4f}')

print()
print('=' * 90)
print('二、逐任务（39 子项）覆盖与结构')
print('=' * 90)
task_cols = [c for c in df.columns if '::' in c]
tasks = sorted(set(c.split('::')[0] for c in task_cols))
print(f'任务数 = {len(tasks)}')
print(f'{"任务":<52}{"记录数":>7}{"缺失率":>9}  主指标')
for t in tasks:
    cols = [c for c in task_cols if c.startswith(t + '::')]
    main = [c for c in cols if c.split('::')[1] in ('acc_norm,none', 'acc,none',
            'exact_match,none', 'prompt_level_strict_acc,none')]
    n_ok = df[main[0]].notna().sum() if main else 0
    print(f'{t:<52}{n_ok:>7}{1-n_ok/len(df):>9.2%}  {main[0].split("::")[1] if main else "-"}')

print()
print('=' * 90)
print('三、逐任务能否重构出排行榜的 6 个分组分？（"逐任务别忽略"的实质）')
print('=' * 90)
for grp, subs, lbcol in [('BBH', [t for t in tasks if t.startswith('leaderboard_bbh_')], 'BBH'),
                         ('GPQA', [t for t in tasks if t.startswith('leaderboard_gpqa_')], 'GPQA'),
                         ('MATH', [t for t in tasks if t.startswith('leaderboard_math_')], 'MATH_Lvl5'),
                         ('MUSR', [t for t in tasks if t.startswith('leaderboard_musr_')], 'MUSR')]:
    cols = []
    for t in subs:
        c = f'{t}::acc_norm,none'
        if c not in df.columns:
            c = f'{t}::acc,none'
        if c in df.columns:
            cols.append(c)
    if not cols:
        continue
    m_ = df[cols].mean(axis=1, skipna=True)
    mm = df[['key']].assign(recon=m_).merge(lb[['key', lbcol]], on='key', how='inner').dropna()
    if len(mm) > 20:
        r = stats.pearsonr(mm.recon, mm[lbcol])
        print(f'  {grp:<6} 子任务数={len(cols):>2}  重构 vs 排行榜列[{lbcol}]: n={len(mm)}, '
              f'Pearson r={r.statistic:+.4f}  (一致则说明分组分=子任务简单平均)')

print()
print('=' * 90)
print('四、时间效应的逐任务分解（哪几个任务在驱动 +0.13 的年份系数）')
print('=' * 90)
g = mg[(mg.Params_B > 0) & np.isfinite(mg.Params_B) & np.isfinite(mg.Year)].copy()
print(f'有效样本（Params_B>0 且年份有限）= {len(g)}')
tcols = {}
for t in tasks:
    c = f'{t}::acc_norm,none'
    if c not in df.columns:
        c = f'{t}::acc,none'
    if c in df.columns:
        tcols[t] = c
rows = []
for t, c in tcols.items():
    s = g[['Year', c, 'Params_B']].dropna()
    s = s[(s.Params_B > 0) & np.isfinite(s.Params_B)]
    if len(s) < 100:
        continue
    x = np.column_stack([np.log(s.Params_B.values), s.Year.values - 2019])
    X = np.column_stack([np.ones(len(s)), x])
    y = s[c].values.astype(float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        continue
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rows.append({'task': t.replace('leaderboard_', ''), 'n': len(s),
                 'logP_coef': beta[1], 'year_coef': beta[2], 'R2': r2})
rt = pd.DataFrame(rows).sort_values('year_coef', ascending=False)
print(f'{"任务":<44}{"n":>6}{"logP系数":>11}{"年份系数":>11}{"R2":>8}')
for _, r in rt.iterrows():
    print(f'{r.task:<44}{int(r.n):>6}{r.logP_coef:>11.4f}{r.year_coef:>11.4f}{r.R2:>8.3f}')
pos = rt.year_coef[rt.year_coef > 0]
neg = rt.year_coef[rt.year_coef <= 0]
print(f'\n年份系数为正的任务数 = {len(pos)}，为负 = {len(neg)}')
print(f'年份系数范围: {rt.year_coef.min():+.4f} ~ {rt.year_coef.max():+.4f}')
print('→ 若各任务的年份系数符号都不一致，"时间效应"就不能作为单一因果量报告。')

print()
print('=' * 90)
print('五、上下文长度是否被当作"自由寻优变量"（同规模档内对比）')
print('=' * 90)
g2 = mg[(mg.Params_B > 0) & np.isfinite(mg.Params_B)].copy()
g2 = g2[g2.max_length.isin([2048, 4096, 8192, 32768, 131072])]
g2['Pbin'] = pd.cut(np.log10(g2.Params_B), bins=[-1, 0.3, 0.7, 1.1, 1.5, 3],
                    labels=['<2B', '2-5B', '5-13B', '13-32B', '>32B'])
piv = g2.pivot_table(index='Pbin', columns='max_length', values='Average', aggfunc='mean')
cnt = g2.pivot_table(index='Pbin', columns='max_length', values='Average', aggfunc='size')
print('同规模档内，不同 max_length 的 Average 均值：')
print(piv.round(3).to_string())
print()
print('样本数矩阵：')
print(cnt.fillna(0).astype(int).to_string())
print()
print('解读：同一规模档内不同 max_length 的 Average 差异，说明"评测上下文长度"改变的是')
print('      测量条件，而不是可自由寻优的设计变量。')

print()
print('=' * 90)
print('六、同口径对照：聚合层(Average) vs 逐任务层 的年份系数')
print('=' * 90)


def fit_year(df_, ycol, scale=1.0):
    s = df_[['Year', ycol, 'Params_B']].dropna()
    s = s[(s.Params_B > 0) & np.isfinite(s.Params_B)]
    x = np.column_stack([np.log(s.Params_B.values), s.Year.values - 2019])
    X = np.column_stack([np.ones(len(s)), x])
    y = s[ycol].values.astype(float) * scale
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return beta[1], beta[2], r2, len(s)


b1_, b2_, r2a, na = fit_year(g, 'Average', scale=0.01)
print(f'聚合层 Average(缩放到0-1): logP系数={b1_:+.4f}  年份系数={b2_:+.4f}  R²={r2a:.3f}  n={na}')
print(f'逐任务层 年份系数: 均值={rt.year_coef.mean():+.4f}  中位={rt.year_coef.median():+.4f}  '
      f'最大={rt.year_coef.max():+.4f}  最小={rt.year_coef.min():+.4f}')
print()
print('→ 聚合层年份系数与逐任务层同口径可比。若聚合层明显偏离逐任务分布，则"时间效应"')
print('   主要由样本构成（哪些模型/来源/任务进入样本）驱动，而非各任务共同的真实进步。')

# 构成检验：逐年样本构成
src = mg[(mg.Params_B > 0) & np.isfinite(mg.Params_B)].copy()
src = src.dropna(subset=['Year', 'Average'])
print('逐年样本构成（构成变化是"时间效应"的主要嫌疑）：')
comp = src.groupby('Year').agg(n=('Average', 'size'),
                               medP=('Params_B', 'median'),
                               medAvg=('Average', 'median'),
                               q25P=('Params_B', lambda s: s.quantile(.25)),
                               q75P=('Params_B', lambda s: s.quantile(.75)))
comp['share_of_year'] = (comp.n / comp.n.sum() * 100).round(1)
print(comp.round(2).to_string())
print()
print(f'  匹配到 detailed_results 的来源: {src.Source.value_counts().to_dict() if "Source" in src.columns else "无"}')
print('  → 若各年的中位规模/样本占比剧烈变化，年份系数就同时承载了"规模构成变化"，')
print('     不能直接读作技术进步。')

print()
print('=' * 90)
print('七、结论摘要（写入论文的证据）')
print('=' * 90)
print(f'  [逐任务] detailed_results 含 45 个任务键、其中 39 个可用的逐任务子项；')
print(f'           BBH(24子任务)/GPQA(3)/MUSR(3) 的简单平均可重构排行榜分组分 (r=0.998/0.981/0.973)。')
print(f'  [截断]   1863 份结果 JSON 中 3 份被物理截断，按规则整份舍弃，不补全。')
print(f'  [上下文] max_length 覆盖 15 个取值(1024~1e9)，66.3% 为 4096；97 份(5.2%)缺失；')
print(f'           与 Year 强混淆(Spearman=-0.481)，与 Params 弱相关(+0.165)。')
print(f'           → 它是评测协议设定，必须作为分层/固定条件，不能作为自由寻优变量。')
print(f'  [时间]   逐任务年份系数 {rt.year_coef.min():+.4f}~{rt.year_coef.max():+.4f}，3 个为负；')
print(f'           与聚合层同口径系数 {b2_:+.4f} 对比可判断构成效应占比。')


