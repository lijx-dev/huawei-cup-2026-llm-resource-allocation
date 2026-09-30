# -*- coding: utf-8 -*-
"""复算 M2 配比模型，判定实跑报告的 0.667/0.834 与稿内 0.751/0.873 是否口径差异

同时输出 pooled R² 与 逐目标 R² 均值、逐目标 Spearman 均值，
并用已保存系数 mix_final_model_quad.csv 复核。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

A = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
M2 = pd.read_csv(r'd:\F题\q1_quality_results\mix_final_model_quad.csv', index_col=0)
tgt = list(M2.columns)
MODEL13 = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
           'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
           'hackernews', 'pubmed_abstracts', 'other']
AGG = {'nih_exporter': 'other', 'philpapers': 'other',
       'enron_emails': 'other', 'europarl': 'other'}
QUAD = [i[:-2] for i in M2.index if i.endswith('^2')]


def to13(df):
    d = pd.DataFrame({c.replace('train_the_pile_', ''): df[c] for c in df.columns})
    out = pd.DataFrame(index=d.index)
    for m in MODEL13:
        out[m] = 0.0
    for raw in d.columns:
        m = AGG.get(raw, raw)
        if m in out.columns:
            out[m] = out[m] + d[raw]
    return out


def design(p13):
    X = pd.DataFrame(index=p13.index)
    for m in MODEL13:
        X[m] = p13[m].values
    for m in QUAD:
        X[m + '^2'] = p13[m].values ** 2
    return X


# ---- 数据 ----
ptr = to13(pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).iloc[:, 1:].astype(float))
pte = to13(pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].astype(float))
ltr = pd.read_csv(os.path.join(A, 'train_pile_loss_1m.csv')).iloc[:, 1:].astype(float)
lte = pd.read_csv(os.path.join(A, 'test_pile_loss_1m.csv')).iloc[:, 1:].astype(float)
print('train mixture', ptr.shape, ' test mixture', pte.shape)
print('train loss   ', ltr.shape, ' test loss   ', lte.shape)
print('loss 列名样例:', list(ltr.columns)[:6])

# 对齐 loss 列到 13 目标：列名形如 metric/the_pile_{domain}_val_loss
def norm_loss(df):
    mp = {}
    for c in df.columns:
        s = c.replace('metric/', '')
        s = s.replace('train_the_pile_', '').replace('the_pile_', '')
        s = s.replace('_val_loss', '').replace('val_loss_', '')
        mp[s] = c
    return mp


def pick(df, t, mp):
    return df[mp[t]].values.astype(float) if t in mp else None


mtr, mte = norm_loss(ltr), norm_loss(lte)
print('loss 归一化域名:', sorted(mtr.keys()))
assert set(mtr) == set(mte), '训练/检验 loss 列不一致'
assert set(tgt) <= set(mtr), f'目标缺失: {set(tgt) - set(mtr)}'


Ytr = np.column_stack([pick(ltr, t, mtr) for t in tgt])
Yte = np.column_stack([pick(lte, t, mte) for t in tgt])
ok = ~np.isnan(Ytr).any(axis=0) & ~np.isnan(Yte).any(axis=0)
print('可用目标数:', ok.sum(), '/', len(tgt))
tgt_ok = [t for t, o in zip(tgt, ok) if o]
Ytr, Yte = Ytr[:, ok], Yte[:, ok]

Xtr, Xte = design(ptr), design(pte)


def metrics(Ytrue, Ypred, tag):
    res = Ytrue - Ypred
    pooled = 1 - np.sum(res ** 2) / np.sum((Ytrue - Ytrue.mean()) ** 2)
    r2s, rhos = [], []
    for j in range(Ytrue.shape[1]):
        yt, yp = Ytrue[:, j], Ypred[:, j]
        r2s.append(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))
        rhos.append(stats.spearmanr(yt, yp).statistic)
    print(f'{tag:<26} pooled R²={pooled:.4f}   逐目标R²均值={np.mean(r2s):.4f}   '
          f'Spearman均值={np.mean(rhos):.4f}')
    return np.mean(r2s), np.mean(rhos)


print('\n' + '=' * 88)
print('【A】用已保存系数 mix_final_model_quad.csv（文档所用"冻结"M2）')
print('=' * 88)
B = M2.loc[['intercept'] + MODEL13 + [m + '^2' for m in QUAD], tgt_ok].values.T  # (13目标, 20特征)
Xtr_c = np.column_stack([np.ones(len(Xtr)), Xtr[MODEL13 + [m + '^2' for m in QUAD]].values])
Xte_c = np.column_stack([np.ones(len(Xte)), Xte[MODEL13 + [m + '^2' for m in QUAD]].values])
metrics(np.log(Ytr), Xtr_c @ B.T, '训练集 (A4 512)')
metrics(np.log(Yte), Xte_c @ B.T, '检验集 (A5 1M 256)')

print('\n' + '=' * 88)
print('【B】按书面规格重新拟合（模拟实跑报告的做法）')
print('=' * 88)
Bt = np.linalg.lstsq(Xtr_c, np.log(Ytr), rcond=None)[0]
metrics(np.log(Ytr), Xtr_c @ Bt, '训练集 (A4 512)')
metrics(np.log(Yte), Xte_c @ Bt, '检验集 (A5 1M 256)')

print('\n' + '=' * 88)
print('【C】逐目标明细（检验集，用已保存系数）')
print('=' * 88)
pred = Xte_c @ B.T
for j, t in enumerate(tgt_ok):
    yt, yp = np.log(Yte[:, j]), pred[:, j]
    r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    rho = stats.spearmanr(yt, yp).statistic
    print(f'  {t:<20} R²={r2:>7.4f}   Spearman={rho:>7.4f}')
