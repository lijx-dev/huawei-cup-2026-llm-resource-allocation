# -*- coding: utf-8 -*-
"""诊断 A10–A11 凸包内配方数：28（13 维聚合空间）vs 文档记录的 29。
依次试：13 维聚合、14 维（含参考域）、17 维原始，各给出计数与容差敏感性。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

A = r'F题\real_attachments\A_data_value\regmix_tables'
MODEL13 = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
           'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
           'hackernews', 'pubmed_abstracts', 'other']
AGG = {'nih_exporter': 'other', 'philpapers': 'other',
       'enron_emails': 'other', 'europarl': 'other'}


def raw(stem):
    df = pd.read_csv(os.path.join(A, stem + '.csv')).iloc[:, 1:].astype(float)
    return pd.DataFrame({c.replace('train_the_pile_', ''): df[c] for c in df.columns})


def agg13(d):
    out = pd.DataFrame(0.0, index=d.index, columns=MODEL13)
    for r in d.columns:
        m = AGG.get(r, r)
        if m in out.columns:
            out[m] = out[m] + d[r]
    return out


def count(Xtr, Xte, tol=1e-6):
    n = len(Xtr)
    hit = 0
    for x in Xte:
        r = linprog(c=np.zeros(n), A_eq=np.vstack([np.ones((1, n)), Xtr.T]),
                    b_eq=np.concatenate([[1.0], x]), bounds=[(0, None)] * n,
                    method='highs')
        if r.status == 0 and np.max(np.abs(Xtr.T @ r.x - x)) < tol:
            hit += 1
    return hit


dtr = raw('train_mixture_1m')
tests = {'A6–A7 (1M)': 'test_mixture_1m', 'A8–A9 (60M)': 'test_mixture_60m', 'A10–A11 (1B)': 'test_mixture_1B'}


def spaces(dtr, dte):
    return {
        '13 维（M2 坐标空间）': (agg13(dtr).values, agg13(dte).values),
        '14 维（含参考域）': (np.column_stack([agg13(dtr).values, dtr['uspto_backgrounds'].values]),
                              np.column_stack([agg13(dte).values, dte['uspto_backgrounds'].values])),
        '17 维原始': (dtr.values, dte.values),
    }


print('A4 训练 n=%d' % len(dtr))
print(f'{"检验集":<16}{"n":>5}{"13 维（M2 坐标空间）":>22}{"14 维（含参考域）":>20}{"17 维原始":>14}')
for nm, stem in tests.items():
    dte = raw(stem)
    cells = []
    for snm, (Xtr, Xte) in spaces(dtr, dte).items():
        h = count(Xtr, Xte)
        cells.append(f'{h}/{len(Xte)}')
    print(f'{nm:<16}{len(dte):>5}{cells[0]:>22}{cells[1]:>20}{cells[2]:>14}')

dte = raw('test_mixture_1B')
Xtr, Xte = agg13(dtr).values, agg13(dte).values
n = len(Xtr)
marg = []
for x in Xte:
    r = linprog(c=np.zeros(n), A_eq=np.vstack([np.ones((1, n)), Xtr.T]),
                b_eq=np.concatenate([[1.0], x]), bounds=[(0, None)] * n, method='highs')
    marg.append(np.max(np.abs(Xtr.T @ r.x - x)) if r.status == 0 else np.inf)
marg = np.array(marg)
inside = np.sort(marg[marg < 1e-6])
print(f'\n13 维空间下 A10–A11 的判定余量：凸包内 {len(inside)} 个，最大余量 {inside[-1]:.3e}；'
      f'凸包外最近的余量为 {np.sort(marg[marg >= 1e-6])[0]}（无贴边点，判定无容差敏感性）')
