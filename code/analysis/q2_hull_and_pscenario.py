# -*- coding: utf-8 -*-
"""A4 配方凸包支持检查 + 配比情景（等权主口径）

解决两件事：
  (1) 文档第 10 节第 3 条引用的"A6–A7 / A8–A9 / A10–A11 凸包内配方数"此前无脚本可复现。
      本脚本用线性可行性检查（LP）给出：x 是否可写成 A4 训练配方的凸组合。
  (2) 运行报告第 5 节的配比情景此前用"逐目标分位"（h 达 ±0.22），而文档 §2.2 声明的主口径是
      "13 目标等权汇总 h_p"。两者相差约 2 倍。本脚本按主口径（先等权、后取分位）重算，
      并把逐目标口径降为敏感性上包络。

模型与参数：
  M2: log L_v(p) = a_v + sum_k b_{k,v} p_k + sum_{k in 6主域} g_{k,v} p_k^2
  h_{p,v}(p) = log L_v(p) - log L_v(p0),  p0 = A4 训练配方逐域算术均值
  配比接入：Form A -> L = Et + AN + BD*exp(lambda_p*h);  Form B -> L = Et + (AN+BD-E1*Q)*exp(...)
  B 端参数锁 SET_A（B6 拟合集，唯一计算输入，见 P1-5）。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

A = r'F题\real_attachments\A_data_value\regmix_tables'
M2 = pd.read_csv(r'd:\F题\q1_quality_results\mix_final_model_quad.csv', index_col=0)
quad = M2.loc[[i for i in M2.index if i.endswith('^2')]]
lin = M2.loc[[i for i in M2.index if (not i.endswith('^2')) and i != 'intercept']]
tgt = list(M2.columns)

MODEL13 = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
           'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
           'hackernews', 'pubmed_abstracts', 'other']
AGG = {'nih_exporter': 'other', 'philpapers': 'other',
       'enron_emails': 'other', 'europarl': 'other'}
MAIN6 = [i[:-2] for i in quad.index]
I13 = {m: k for k, m in enumerate(MODEL13)}
I6 = [I13[m] for m in MAIN6]


def to13(path):
    df = pd.read_csv(path).iloc[:, 1:].astype(float)
    d = pd.DataFrame({c.replace('train_the_pile_', ''): df[c] for c in df.columns})
    out = pd.DataFrame(0.0, index=d.index, columns=MODEL13)
    for raw in d.columns:
        m = AGG.get(raw, raw)
        if m in out.columns:
            out[m] = out[m] + d[raw]
    return out


def h_mat(p13, p0_13):
    dp = p13 - p0_13
    dq = (p13 ** 2 - p0_13 ** 2)[:, I6]
    H = np.zeros((len(p13), len(tgt)))
    for j, t in enumerate(tgt):
        H[:, j] = dp @ lin[t].reindex(MODEL13).values + dq @ quad[t].values
    return H


def in_hull(x, X, tol=1e-7):
    """LP 可行性：是否存在 lam>=0, sum(lam)=1, X^T lam = x"""
    n, k = X.shape[0], X.shape[0]
    A_eq = np.vstack([np.ones((1, n)), X.T])
    b_eq = np.concatenate([[1.0], x])
    r = linprog(c=np.zeros(n), A_eq=A_eq, b_eq=b_eq,
                bounds=[(0, None)] * n, method='highs')
    return bool(r.status == 0) and np.max(np.abs(X.T @ r.x - x)) < 1e-6


P_tr, P_te = to13(os.path.join(A, 'train_mixture_1m.csv')), to13(os.path.join(A, 'test_mixture_1m.csv'))
Xtr, Xte = P_tr.values, P_te.values
p0 = Xtr.mean(axis=0)
print(f'A4 训练配方 n={len(Xtr)}；A6–A7 检验配方 n={len(Xte)}；p0 行和={p0.sum():.6f}')

print('\n' + '=' * 78)
print('【1】A4 配方凸包支持检查（LP 线性可行性，13 维聚合空间）')
print('=' * 78)
mask = np.array([in_hull(Xte[i], Xtr) for i in range(len(Xte))])
print(f'  A6–A7（1M）256 个检验配方中，落在 A4 凸包内: {mask.sum()} / {len(mask)}  '
      f'({100*mask.mean():.1f}%)')

H = h_mat(Xte, p0)
hp13 = H.mean(axis=1)


def q(v, name):
    print(f'  {name:<26}n={len(v):>4}  p05={np.percentile(v,5):+.5f}  '
          f'median={np.median(v):+.5f}  p95={np.percentile(v,95):+.5f}  '
          f'min={v.min():+.5f}  max={v.max():+.5f}')


print('\n' + '=' * 78)
print('【2】等权汇总 h_p 的分布（主口径：先对 13 目标等权平均，再取分位）')
print('=' * 78)
q(hp13, '全部 256 配方')
q(hp13[mask], '凸包内')
q(hp13[~mask], '凸包外')

print('\n  对照——逐目标口径（对每个目标取分位后再跨目标取中位，非主口径）：')
per = np.array([np.percentile(H[:, j], [5, 50, 95]) for j in range(len(tgt))])
print(f'    逐目标 p05 中位={np.median(per[:,0]):+.5f}  中位={np.median(per[:,1]):+.5f}  '
      f'p95 中位={np.median(per[:,2]):+.5f}')
print(f'    逐目标 |h| 的 p95 中位={np.median([np.percentile(np.abs(H[:,j]),95) for j in range(len(tgt))]):.5f}')

# ---------------- 配比情景：SET_A 主参数 ----------------
E, Aa, al, Bb, be, rN, rD, E1 = 1.7112, 0.6519, 0.2783, 1.4207, 0.2834, 0.3555, 0.1428, 0.1027
N, D, Q = 1.0, 150.0, 0.5
AN = Aa * N ** -al * np.exp(-rN * Q)
BD = Bb * D ** -be * np.exp(-rD * Q)
Et = E
print('\n' + '=' * 78)
print(f'【3】配比情景（SET_A 主参数；N={N}B, D={D}B, Q={Q}）')
print(f'    Et={Et:.5f}  AN={AN:.5f}  BD={BD:.5f}  基线 L={Et+AN+BD-E1*Q:.5f}')
print('=' * 78)


def loss(phi, form):
    if form == 'A':
        return Et + AN + BD * np.exp(phi) - E1 * Q
    return Et + (AN + BD - E1 * Q) * np.exp(phi)


base = loss(0.0, 'A')
lo_in, md_in, hi_in = np.percentile(hp13[mask], [5, 50, 95])
lo, md, hi = np.percentile(hp13, [5, 50, 95])
print('\n  (a) 主口径：等权 h_p 的 p05 / 中位 / p95（凸包内）')
print(f"    {'λ_p':>6}{'h=-'+f'{abs(lo_in):.4f}':>14}{'h='+f'{md_in:+.4f}':>14}{'h='+f'{hi_in:+.4f}':>14}")
for lp in [0.0, 0.5, 1.0, 1.5]:
    cells = [f'{(loss(lp*h, "A")/base-1)*100:+.3f}%' for h in (lo_in, md_in, hi_in)]
    print(f'    {lp:>6.1f}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}')

print('\n  (b) 全 256 配方（含凸包外，作外推上包络）')
print(f"    {'λ_p':>6}{'p05':>14}{'中位':>14}{'p95':>14}")
for lp in [0.0, 0.5, 1.0, 1.5]:
    cells = [f'{(loss(lp*h, "A")/base-1)*100:+.3f}%' for h in (lo, md, hi)]
    print(f'    {lp:>6.1f}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}')

print('\n  (c) 作用位置对照（λ_p=1；Form A 只乘 D 项，Form B 乘全部可变项，κ 为中间分配）')


def loss_kappa(phi, k):
    return Et + AN * ((1 - k) + k * np.exp(phi)) + BD * np.exp(phi) - E1 * Q


for hnm, hval in [('中位', md_in), ('p95', hi_in)]:
    cells = [f'{nm}={loss_kappa(hval, k):.5f}({(loss_kappa(hval,k)/base-1)*100:+.3f}%)'
             for nm, k in [('FormA', 0.0), ('κ=0.5', 0.5), ('FormB', 1.0)]]
    print(f'    h={hval:+.4f}({hnm}): ' + '  '.join(cells))

print('\n  (d) 旧"逐目标口径"（仅作对照，说明此前表偏高）')
per95 = np.median([np.percentile(np.abs(H[:, j]), 95) for j in range(len(tgt))])
for lp in [0.5, 1.0, 1.5]:
    print(f'    λ_p={lp}: h=+{per95:.4f} -> FormA {(loss(lp*per95,"A")/base-1)*100:+.3f}%  '
          f'(主口径同 λ_p 的 p95 为 {(loss(lp*hi_in,"A")/base-1)*100:+.3f}%)')

print('\n' + '=' * 78)
print('【4】结论')
print('=' * 78)
print(f'  1. A6–A7 仅 {mask.sum()}/{len(mask)} 个配方在 A4 凸包内，配比响应验证主要是外推检查。')
print(f'  2. 等权主口径下 h_p 的 p05–p95 为 [{lo:.4f}, {hi:.4f}]（凸包内 [{lo_in:.4f}, {hi_in:.4f}]）；')
print('     此前第 5 节按逐目标分位给出的 ±0.22 属敏感性上包络，不得作为主结果。')
print('  3. λ_p 仍不可识别，只以 {0,0.5,1,1.5} 情景报告；不得作为优化变量。')
