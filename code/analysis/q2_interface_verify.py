# -*- coding: utf-8 -*-
"""核验《问题二重新运行报告》宣称已冻结的接口：
   1) p0 = A4 训练配方 14 组算术均值（4 个无 Loss 域并入 other，uspto 为参考）
   2) M2 凸二次 log-linear 在 A4 上的训练 R²、在 A6–A7(1M) 上的相对响应 Spearman
   并对照报告表格中的数值。
"""
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
MERGED = ['nih_exporter', 'philpapers', 'enron_emails', 'europarl']
REF = 'uspto_backgrounds'
S6 = ['arxiv', 'freelaw', 'dm_mathematics', 'github', 'ubuntu_irc', 'hackernews']


def load(tag):
    mx = pd.read_csv(f'{BASE}/{tag}_mixture.csv' if 'train' in tag else f'{BASE}/{tag}_mixture.csv')
    ls = pd.read_csv(f'{BASE}/{tag}_loss.csv' if 'train' in tag else f'{BASE}/{tag}_loss.csv')
    return mx, ls


tr_mx = pd.read_csv(f'{BASE}/train_mixture_1m.csv')
tr_ls = pd.read_csv(f'{BASE}/train_pile_loss_1m.csv')
te_mx = pd.read_csv(f'{BASE}/test_mixture_1m.csv')
te_ls = pd.read_csv(f'{BASE}/test_pile_loss_1m.csv')

cols = [c for c in tr_mx.columns if c != 'index']
loss_cols = [c for c in tr_ls.columns if c != 'index']
short = lambda c: c.replace('train_the_pile_', '').replace('metric/the_pile_', '').replace('_val_loss', '')

# ---------- 1) p0 ----------
raw = tr_mx[cols].values
names = [short(c) for c in cols]
P14 = np.column_stack([raw[:, [i for i, n in enumerate(names) if n not in MERGED]],
                       raw[:, [i for i, n in enumerate(names) if n in MERGED]].sum(1)])
n14 = [n for n in names if n not in MERGED] + ['other']
p0 = P14.mean(0)
p0 = p0 / p0.sum()

reported = {'arxiv': 0.1130, 'freelaw': 0.0968, 'pubmed_central': 0.1138, 'wikipedia_en': 0.0751,
            'dm_mathematics': 0.0255, 'github': 0.1107, 'stackexchange': 0.0994,
            'gutenberg_pg_19': 0.0469, 'pile_cc': 0.1195, 'ubuntu_irc': 0.0196,
            'hackernews': 0.0117, 'pubmed_abstracts': 0.0662, 'uspto_backgrounds': 0.0756,
            'other': 0.0263}
print('== p0 核验（14 组，归一化后）==')
print(f'{"配比组":<20s}{"复算":>10s}{"报告":>10s}{"差":>10s}')
mx_dev = 0.0
for n in n14:
    v = p0[n14.index(n)]
    r = reported[n]
    mx_dev = max(mx_dev, abs(v - r))
    print(f'{n:<20s}{v:>10.5f}{r:>10.5f}{v-r:>+10.5f}')
print(f'  和 = {p0.sum():.6f}   最大偏差 = {mx_dev:.5f}')

# ---------- 2) M2 拟合与核验 ----------
X14_tr = np.column_stack([raw[:, [i for i, n in enumerate(names) if n not in MERGED]],
                          raw[:, [i for i, n in enumerate(names) if n in MERGED]].sum(1)])
keep = [i for i, n in enumerate(n14) if n != REF]
qcol = [keep.index(n14.index(s)) for s in S6]          # 6 个二次项在 13 维坐标中的列位置
x_tr = (X14_tr - p0)[:, keep]
x_tr = np.column_stack([x_tr, x_tr[:, qcol] ** 2])
L_tr = tr_ls[loss_cols].values


def fit_m2(X, Y):
    Z = np.column_stack([np.ones(len(X)), X[:, :13], X[:, 13:]])
    lb = np.r_[[-np.inf] * 14, [0.0] * 6]
    ub = np.r_[[np.inf] * 14, [np.inf] * 6]
    coef = np.column_stack([lsq_linear(Z, np.log(Y[:, t]), bounds=(lb, ub)).x
                            for t in range(Y.shape[1])])
    return Z, coef


Ztr, C = fit_m2(x_tr, L_tr)
pred_tr = np.exp(Ztr @ C)
r2 = 1 - ((pred_tr - L_tr) ** 2).sum(0) / ((L_tr - L_tr.mean(0)) ** 2).sum(0)
print(f'\n== M2 训练核验 ==')
print(f'  A4 训练绝对 Loss 平均 R² = {r2.mean():.4f}   (报告: 0.735;  旧稿: 0.751)')

# 测试切片 A6–A7(1M) 相对响应 Spearman
Xt = np.column_stack([te_mx[cols].values[:, [i for i, n in enumerate(names) if n not in MERGED]],
                      te_mx[cols].values[:, [i for i, n in enumerate(names) if n in MERGED]].sum(1)])
xt = (Xt - p0)[:, keep]
xt = np.column_stack([xt, xt[:, qcol] ** 2])
Zt = np.column_stack([np.ones(len(xt)), xt])
L_te = te_ls[loss_cols].values
pred_te = np.exp(Zt @ C)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return np.corrcoef(ra, rb)[0, 1]


sp = [spearman(np.log(pred_te[:, t] / pred_te[0, t]), np.log(L_te[:, t] / L_te[0, t]))
      for t in range(L_te.shape[1])]
print(f'  A6–A7(1M) 相对配比响应平均 Spearman = {np.mean(sp):.4f}   (报告: 0.873)')
print(f'  A6–A7(1M) 绝对 Loss 平均 R² = '
      f'{(1-((pred_te-L_te)**2).sum(0)/((L_te-L_te.mean(0))**2).sum(0)).mean():.4f}   (报告: 0.723)')

# ---------- 3) 参考点 p0 处的 h_p 是否恒为 0 ----------
print(f'\n== 接口自洽性 ==')
print(f'  h_p(p0) 定义上为 0；p0 行和 = {p0.sum():.6f}，非负 = {(p0 >= 0).all()}')
print(f'  参考域 {REF} 的 p0 = {p0[n14.index(REF)]:.4f}')
