# -*- coding: utf-8 -*-
"""配比 p 的可分离性：共享参数对数线性拟合

模型（可乘形式的等价对数线性表述）：
    log L_{s,j}(p) = a_{s,j} + Σ_i θ_i · p_i
其中 s 为规模切片、j 为评测目标域、a_{s,j} 吸收所有规模与目标的水平效应，
θ 描述配比 p 的"形状"效应。

判据：
  (1) θ 是否与规模 s 无关  → 可分离（p 效应与 N/D 无关，仅乘一个水平因子）
  (2) θ 是否与目标 j 无关  → 是否可用单一 g(p) 统一 13 个目标
  (3) 共享 θ 的 R² 与"逐切片 θ"的 R² 差距 → 可分离假设的代价
"""
import os
import numpy as np
import pandas as pd

A = r'F题\real_attachments\A_data_value\regmix_tables'
SLICES = [('1m', 'test_mixture_1m.csv', 'test_pile_loss_1m.csv'),
          ('60m', 'test_mixture_60m.csv', 'test_pile_loss_60m.csv'),
          ('1B', 'test_mixture_1B.csv', 'test_pile_loss_1B.csv'),
          ('10b', 'est_mixture_10b.csv', 'est_pile_loss_10b.csv'),
          ('70b', 'est_mixture_70b.csv', 'est_pile_loss_70b.csv')]


def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    pred = Xc @ b
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return b, pred, r2


rec = []
for tag, mf, lf in SLICES:
    X = pd.read_csv(os.path.join(A, mf)).iloc[:, 1:].values.astype(float)
    Y = pd.read_csv(os.path.join(A, lf)).iloc[:, 1:].values.astype(float)
    tg = [c.replace('metric/the_pile_', '').replace('_val_loss', '')
          for c in pd.read_csv(os.path.join(A, lf)).columns[1:]]
    for j, t in enumerate(tg):
        for r in range(len(X)):
            rec.append({'slice': tag, 'target': t, 'y': np.log(Y[r, j]), **{f'p{i}': X[r, i] for i in range(17)}})
D = pd.DataFrame(rec)
P = [f'p{i}' for i in range(17)]
print(f'样本量 {len(D)}（{D.slice.nunique()} 切片 × {D.target.nunique()} 目标 × 配方）')

# 参考域剔除一个以消除与截距的共线性（p 和为 1）
Pc = P[:-1]


def fit(df, cols_p, extra_dummies):
    """extra_dummies: list of column names to add as dummies (e.g. ['slice','target'])"""
    Xs = [df[c].values for c in cols_p]
    names = list(cols_p)
    for dcol in extra_dummies:
        dm = pd.get_dummies(df[dcol], prefix=dcol, drop_first=True).astype(float)
        for c in dm.columns:
            Xs.append(dm[c].values)
            names.append(c)
    if not Xs:
        y = df.y.values
        r2 = 0.0
        return {}, r2
    Xm = np.column_stack(Xs)
    b, pred, r2 = ols(Xm, df.y.values)
    return dict(zip(names, b)), r2


print()
print('=' * 100)
print('拟合对比（因变量 log L）')
print('=' * 100)
configs = [
    ('M0  仅截距', [], []),
    ('M1  + 共享θ(17域，p与规模/目标无关)', Pc, []),
    ('M2  + 共享θ + 切片哑变量 a_s', Pc, ['slice']),
    ('M3  + 共享θ + 切片哑 + 目标哑 a_{s,j}', Pc, ['slice', 'target']),
    ('M4  切片×θ（θ随规模变）', Pc, ['slice']),
]
res = {}
for name, cp, ed in configs:
    coef, r2 = fit(D, cp, ed)
    res[name] = (coef, r2)
    print(f'  {name:<45} R²={r2:.4f}')

# M4: slice-specific theta
print()
print('  逐切片单独拟合 θ（不含共享约束）:')
slice_r2 = {}
for tag, _, _ in SLICES:
    sub = D[D.slice == tag]
    coef, r2 = fit(sub, Pc, ['target'])
    slice_r2[tag] = r2
    print(f'    {tag:<5} R²={r2:.4f}')

# M5: per-target theta
print()
print('  逐目标单独拟合 θ（含切片哑变量）:')
tgt_r2 = {}
for t in D.target.unique():
    sub = D[D.target == t]
    coef, r2 = fit(sub, Pc, ['slice'])
    tgt_r2[t] = r2
print('    R² 中位=%.4f  最小=%.4f  最大=%.4f' % (np.median(list(tgt_r2.values())),
                                                 min(tgt_r2.values()), max(tgt_r2.values())))

# 关键对比：共享θ vs 逐切片θ
print()
print('=' * 100)
print('关键对比：θ 是否随规模变化（可分离性检验）')
print('=' * 100)
coef_shared = res['M3  + 共享θ + 切片哑 + 目标哑 a_{s,j}'][0]
r2_shared = res['M3  + 共享θ + 切片哑 + 目标哑 a_{s,j}'][1]
print(f'  共享 θ（全部 5 个规模切片共用一组 θ）: R²={r2_shared:.4f}')
print('  若各切片单独拟合 θ 只带来极小提升 → θ 与规模无关 → p 效应可分离')

# 计算各切片 θ 的相关性
thetas = {}
for tag, _, _ in SLICES:
    sub = D[D.slice == tag]
    coef, r2 = fit(sub, Pc, ['target'])
    thetas[tag] = np.array([coef.get(c, 0.0) for c in Pc])
th = pd.DataFrame(thetas, index=Pc)
print()
print('  各切片 θ 之间的相关矩阵:')
print(th.corr().round(3).to_string())
print()
print('  各切片 θ 的符号一致率（与共享 θ 同号的比例）:')
for tag in th.columns:
    sgn = np.mean([np.sign(th.loc[c, tag]) == np.sign(coef_shared[c]) for c in Pc])
    print(f'    {tag:<5} 同号率={sgn:.2%}')

print()
print('  共享 θ（p 的形状效应，越大越"该域占比高则损失高"）:')
domains = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
           'github', 'philpapers', 'stackexchange', 'enron_emails', 'gutenberg_pg_19', 'pile_cc',
           'ubuntu_irc', 'europarl', 'hackernews', 'pubmed_abstracts', 'uspto_backgrounds']
tab = pd.DataFrame({'domain': domains,
                    'theta': [coef_shared[c] for c in Pc] + [np.nan]})
tab = tab.sort_values('theta')
print(tab.round(4).to_string(index=False))
