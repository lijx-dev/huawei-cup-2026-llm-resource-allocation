# -*- coding: utf-8 -*-
"""实测问题一冻结配比模型 M2 的 h_p 量级，并重建 §2.6.4 配比情景敏感性表

问题一最终模型 M2（13 域凸二次 log-linear）:
    log L_v = a_v + sum_k b_{k,v} p_k + sum_{k in 6主域} g_{k,v} p_k^2
    参考域 uspto_backgrounds 被剔除；nih_exporter/enron_emails/europarl/philpapers 聚合为 other

h_{p,v}(p) = log L_v(p) - log L_v(p0),  p0 = A4 训练配方逐域算术均值
"""
import os
import numpy as np
import pandas as pd

A = r'F题\real_attachments\A_data_value\regmix_tables'
M2 = pd.read_csv(r'd:\F题\q1_quality_results\mix_final_model_quad.csv', index_col=0)
inter = M2.loc['intercept']
quad = M2.loc[[i for i in M2.index if i.endswith('^2')]]
lin = M2.loc[[i for i in M2.index if (not i.endswith('^2')) and i != 'intercept']]
tgt = list(M2.columns)
print('13 个 Loss 目标:', tgt)
print('线性项:', list(lin.index))
print('二次项:', [i[:-2] for i in quad.index])

RAW = ['train_the_pile_' + d for d in
       ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en',
        'dm_mathematics', 'github', 'philpapers', 'stackexchange', 'enron_emails',
        'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc', 'europarl', 'hackernews',
        'pubmed_abstracts', 'uspto_backgrounds']]
MODEL13 = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
           'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
           'hackernews', 'pubmed_abstracts', 'other']
AGG = {'nih_exporter': 'other', 'philpapers': 'other',
       'enron_emails': 'other', 'europarl': 'other'}


def to13(df17):
    d = pd.DataFrame({c.replace('train_the_pile_', ''): df17[c] for c in df17.columns})
    out = pd.DataFrame(index=d.index)
    for m in MODEL13:
        out[m] = 0.0
    for raw in d.columns:
        m = AGG.get(raw, raw)
        if m in out.columns:
            out[m] = out[m] + d[raw]
    return out


MAIN6 = [i[:-2] for i in quad.index]
I13 = {m: k for k, m in enumerate(MODEL13)}
I6 = [I13[m] for m in MAIN6]


def h_p(p13, p0_13):
    """13 目标 x n 的 h_p 矩阵"""
    dp = p13 - p0_13
    dq = (p13 ** 2 - p0_13 ** 2)[:, I6]
    H = np.zeros((len(p13), len(tgt)))
    for j, t in enumerate(tgt):
        b = lin[t].reindex(MODEL13).values
        g = quad[t].values
        H[:, j] = dp @ b + dq @ g
    return H


P13_tr = to13(pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).iloc[:, 1:].astype(float))
P13_te = to13(pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].astype(float))
print(f'\nA4 训练配方 n={len(P13_tr)}，检验配方 n={len(P13_te)}')
print('训练配方行和范围: %.6f ~ %.6f' % (P13_tr.sum(axis=1).min(), P13_tr.sum(axis=1).max()))

p0 = P13_tr.values.mean(axis=0)
print('p0 = A4 训练配方逐域算术均值，行和 = %.6f' % p0.sum())

H = h_p(P13_te.values, p0)
hp13 = H.mean(axis=1)
print('\n' + '=' * 84)
print('h_{p,v} 在 A4 检验配方（256 条）上的分布')
print('=' * 84)
print(f"{'target':<20}{'min':>9}{'p05':>9}{'median':>9}{'p95':>9}{'max':>9}")
for j, t in enumerate(tgt):
    v = H[:, j]
    print(f'{t:<20}{v.min():>+9.4f}{np.percentile(v,5):>+9.4f}{np.median(v):>+9.4f}'
          f'{np.percentile(v,95):>+9.4f}{v.max():>+9.4f}')
print('\n13 目标等权汇总 h_p（文档 §2.6.1 定义）:')
print(f'  min={hp13.min():+.5f}  p05={np.percentile(hp13,5):+.5f}  '
      f'median={np.median(hp13):+.5f}  p95={np.percentile(hp13,95):+.5f}  max={hp13.max():+.5f}')
print(f'  |h_p| 的 p95 = {np.percentile(np.abs(hp13),95):.5f}，max = {np.abs(hp13).max():.5f}')
print(f'  逐目标 |h_p| 的 p95（取 13 目标中位）= '
      f'{np.median([np.percentile(np.abs(H[:, j]), 95) for j in range(len(tgt))]):.5f}')

# ---------------- 情景敏感性表 ----------------
# 口径：拟合器参数化为 E + E1*(1-Q) = E + E1 - E1*Q，
# 被识别的常数项是 E~ = E + E1 = 1.7473（见文档 §2.3 口径警告）。
# 此处统一用 E~，否则基线会漏掉 +E1 而偏低 0.1157（约 4.6%）。
PA = dict(E=1.631681, A=0.639571, a=0.282713, B=1.426335, b=0.299794,
          rN=0.349709, rD=0.130121, E1=0.115652)
PA['Et'] = PA['E'] + PA['E1']
N, D, Q = 1.0, 150.0, 0.5


def loss(phi, form):
    AN = PA['A'] * N ** -PA['a'] * np.exp(-PA['rN'] * Q)
    BD = PA['B'] * D ** -PA['b'] * np.exp(-PA['rD'] * Q)
    if form == 'A':
        return PA['Et'] + AN + BD * np.exp(phi) - PA['E1'] * Q
    return PA['Et'] + (AN + BD - PA['E1'] * Q) * np.exp(phi)


base = loss(0.0, 'A')
print('\n' + '=' * 84)
print(f'配比情景敏感性（工作点 N={N}B, D={D}B, Q={Q}；基线 L={base:.5f}）')
print('  lambda_p 与 h_p 只以乘积 phi=lambda_p*h_p 进入，故情景是一维的')
print('=' * 84)
print(f"{'phi':>8}{'FormA L':>11}{'FormB L':>11}{'A-B':>10}{'A变化%':>10}{'B变化%':>10}")
for phi in [0.0, 0.05, 0.10, 0.15, 0.20]:
    la, lb = loss(phi, 'A'), loss(phi, 'B')
    print(f'{phi:>8.2f}{la:>11.5f}{lb:>11.5f}{la-lb:>+10.5f}'
          f'{(la/base-1)*100:>9.3f}%{(lb/base-1)*100:>9.3f}%')

print('\n按实测 h_p 上界映射到 lambda_p 情景：')
hp95 = np.percentile(np.abs(hp13), 95)
for lp in [0.5, 1.0, 1.5]:
    phi = lp * hp95
    la, lb = loss(phi, 'A'), loss(phi, 'B')
    print(f'  lambda_p={lp}: phi={phi:.4f} -> FormA {(la/base-1)*100:+.3f}%, '
          f'FormB {(lb/base-1)*100:+.3f}%')
print(f'\n  逐目标口径（|h_p| 的 p95 中位 = '
      f'{np.median([np.percentile(np.abs(H[:, j]), 95) for j in range(len(tgt))]):.4f}）:')
for lp in [0.5, 1.0, 1.5]:
    phi = lp * np.median([np.percentile(np.abs(H[:, j]), 95) for j in range(len(tgt))])
    la = loss(phi, 'A')
    print(f'    lambda_p={lp}: FormA {(la/base-1)*100:+.3f}%')

print('\n' + '=' * 84)
print('lambda_p=1 时，遍历 A4 检验配方实测 h_p 全范围的情景影响')
print('=' * 84)
for form in ['A', 'B']:
    lo, hi = loss(hp13.min(), form), loss(hp13.max(), form)
    print(f'  Form {form}: h_p in [{hp13.min():+.5f}, {hp13.max():+.5f}] -> '
          f'相对变化 [{(lo/base-1)*100:+.3f}%, {(hi/base-1)*100:+.3f}%]')
