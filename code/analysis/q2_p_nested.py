# -*- coding: utf-8 -*-
"""配比 p 可分离性：嵌套检验 + 公共网格评估

核心问题：log L = a_{s,j} + Σ θ_i p_i 中，θ 是否依赖规模 s？
  - 共享 θ（可分离） vs θ 随 s 变（存在交互）
用嵌套 F 检验 + 公共 p 网格上的剖面相关性判定，并区分"配方设计差异"与"真实交互"。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'F题\real_attachments\A_data_value\regmix_tables'
SLICES = [('1m', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv'),
          ('60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
          ('1B', 'test_mixture_1B.csv', 'test_pile_loss_1B.csv'),
          ('10b', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
          ('70b', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]
DOM = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
       'github', 'philpapers', 'stackexchange', 'enron_emails', 'gutenberg_pg_19', 'pile_cc',
       'ubuntu_irc', 'europarl', 'hackernews', 'pubmed_abstracts', 'uspto_backgrounds']

data = {}
for tag, mf, lf in SLICES:
    X = pd.read_csv(os.path.join(A, mf)).iloc[:, 1:].values.astype(float)
    Y = pd.read_csv(os.path.join(A, lf)).iloc[:, 1:].values.astype(float)
    tg = [c.replace('metric/the_pile_', '').replace('_val_loss', '')
          for c in pd.read_csv(os.path.join(A, lf)).columns[1:]]
    data[tag] = (X, Y, tg)


def build(df, cols_p, dummies):
    Xs, names = [df[c].values for c in cols_p], list(cols_p)
    for d in dummies:
        dm = pd.get_dummies(df[d], prefix=d, drop_first=True).astype(float)
        for c in dm.columns:
            Xs.append(dm[c].values); names.append(c)
    return (np.column_stack(Xs) if Xs else np.empty((len(df), 0))), names


def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    pred = Xc @ b
    rss = np.sum((y - pred) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    return b, rss, 1 - rss / tss


Pc = [f'p{i}' for i in range(17)][:-1]
rec = []
for tag, (X, Y, tg) in data.items():
    for j, t in enumerate(tg):
        for r in range(len(X)):
            rec.append({'slice': tag, 'target': t, 'y': np.log(Y[r, j]),
                        **{f'p{i}': X[r, i] for i in range(17)}})
D = pd.DataFrame(rec)

print('=' * 100)
print('嵌套检验：θ 是否随规模 s 变化（可分离性）')
print('=' * 100)
models = {}
specs = [('M0 仅截距', [], []),
         ('M1 共享θ', Pc, []),
         ('M2 共享θ + 切片哑', Pc, ['slice']),
         ('M3 共享θ + 切片哑 + 目标哑', Pc, ['slice', 'target']),
         ('M4 θ随切片变 + 切片哑 + 目标哑', Pc, ['slice', 'target'])]
for name, cp, dm in specs:
    X, names = build(D, cp, dm)
    b, rss, r2 = ols(X, D.y.values)
    models[name] = (X, names, rss, r2, len(D))
    print(f'  {name:<32} k={X.shape[1]+1:<4} R²={r2:.5f}')

# 加 slice×θ 交互
Xi, names_i = build(D, Pc, ['slice', 'target'])
slice_d = pd.get_dummies(D['slice'], prefix='s', drop_first=True).astype(float)
for c in slice_d.columns:
    for p in Pc:
        Xi = np.column_stack([Xi, (slice_d[c].values * D[p].values)])
b_i, rss_i, r2_i = ols(Xi, D.y.values)
n = len(D)
print(f'  {"M5 切片×θ 交互(完整)":<32} k={Xi.shape[1]+1:<4} R²={r2_i:.5f}')

rss3, rss5 = models['M4 θ随切片变 + 切片哑 + 目标哑'][2], rss_i
# M4 已含 slice+target 哑 + 共享θ；M5 再加 slice×θ 交互
df1 = Xi.shape[1] - models['M4 θ随切片变 + 切片哑 + 目标哑'][0].shape[1]
df2 = n - Xi.shape[1] - 1
F = ((rss3 - rss_i) / df1) / (rss_i / df2)
pval = 1 - stats.f.cdf(F, df1, df2)
print()
print(f'  F 检验（H0: θ 与规模无关）：F({df1},{df2}) = {F:.2f}，p = {pval:.3e}')
print(f'  ΔR² = {r2_i - models["M4 θ随切片变 + 切片哑 + 目标哑"][3]:.5f}'
      f'（相对 R²={models["M4 θ随切片变 + 切片哑 + 目标哑"][3]:.4f}）')

# 加 target×θ 交互
Xt, names_t = build(D, Pc, ['slice', 'target'])
tg_d = pd.get_dummies(D['target'], prefix='t', drop_first=True).astype(float)
for c in tg_d.columns:
    for p in Pc:
        Xt = np.column_stack([Xt, (tg_d[c].values * D[p].values)])
b_t, rss_t, r2_t = ols(Xt, D.y.values)
df1t = Xt.shape[1] - models['M4 θ随切片变 + 切片哑 + 目标哑'][0].shape[1]
df2t = n - Xt.shape[1] - 1
Ft = ((rss3 - rss_t) / df1t) / (rss_t / df2t)
print(f'  F 检验（H0: θ 与目标无关）：F({df1t},{df2t}) = {Ft:.2f}，p = {1-stats.f.cdf(Ft,df1t,df2t):.3e}，ΔR²={r2_t-models["M4 θ随切片变 + 切片哑 + 目标哑"][3]:.5f}')

print()
print('=' * 100)
print('公共网格评估：把各切片拟合的 p 效应外推到同一组配方，比较剖面形状')
print('=' * 100)
grid = data['1m'][0]                       # 用 1m 的 256 个配方作为公共网格
prof = {}
for tag, (X, Y, tg) in data.items():
    # 每个目标用 13 目标中位损失的聚合剖面
    z = []
    for j in range(Y.shape[1]):
        Xc = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(Xc, Y[:, j], rcond=None)
        pred = np.column_stack([np.ones(len(grid)), grid]) @ b
        z.append(pred / pred.mean())
    prof[tag] = np.mean(z, axis=0)
PF = pd.DataFrame(prof)
print('  归一化剖面之间的 Pearson 相关（1 = 形状完全一致）:')
print(PF.corr().round(3).to_string())
print()
print('  注：1m/60m 共用同一配方集；10b/70b 共用同一配方集；1B 为独立配方集。')
print('     跨配方集的相关性同时受"设计差异"与"真实规模效应"影响。')

print()
print('=' * 100)
print('同设计对照（唯一无混淆的比较）：1m vs 60m，同一 256 配方')
print('=' * 100)
common = sorted(set(map(tuple, np.round(data['1m'][0], 6))) & set(map(tuple, np.round(data['60m'][0], 6))))
k1 = {tuple(np.round(r, 6)): i for i, r in enumerate(data['1m'][0])}
k2 = {tuple(np.round(r, 6)): i for i, r in enumerate(data['60m'][0])}
i1 = [k1[c] for c in common]; i2 = [k2[c] for c in common]
for j, t in enumerate(data['1m'][2]):
    X1, Y1, _ = data['1m']; X2, Y2, _ = data['60m']
    jj = data['60m'][2].index(t)
    b1, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X1)), X1]), Y1[:, j], rcond=None)
    b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X2)), X2]), Y2[:, jj], rcond=None)
    p1 = np.column_stack([np.ones(len(X1)), X1]) @ b1
    p2 = np.column_stack([np.ones(len(X2)), X2]) @ b2
    v1, v2 = p1[i1] / p1[i1].mean(), p2[i2] / p2[i2].mean()
    sl, ic, r, p, se = stats.linregress(v2, v1)
    print(f'  {t:<20} 剖面相关 r={np.corrcoef(v1,v2)[0,1]:.4f}  秩相关={stats.spearmanr(v1,v2).correlation:.4f}  '
          f'斜率={sl:.4f}  截距={ic:+.4f}  R²={r**2:.4f}')
