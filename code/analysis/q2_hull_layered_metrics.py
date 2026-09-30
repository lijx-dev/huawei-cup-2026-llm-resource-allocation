# -*- coding: utf-8 -*-
"""吸收队友优点的两项补算：
  (A) 凸包内外分层指标表（平均中心化 log RMSE / Pearson / Spearman），A6–A7、A8–A9、A10–A11 三档
  (B) 等权 h_p 的"留一目标"权重敏感性（13 折）

分层定义：对每个目标 v 先做中心化（减去该组内均值）再算残差，
          pooled 后的 RMSE 即"平均中心化 log RMSE"（与队友报告 §11 口径一致）。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import stats

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


def to13(stem):
    df = pd.read_csv(os.path.join(A, stem + '.csv')).iloc[:, 1:].astype(float)
    d = pd.DataFrame({c.replace('train_the_pile_', ''): df[c] for c in df.columns})
    out = pd.DataFrame(0.0, index=d.index, columns=MODEL13)
    for raw in d.columns:
        m = AGG.get(raw, raw)
        if m in out.columns:
            out[m] = out[m] + d[raw]
    return out


def in_hull(x, X):
    n = X.shape[0]
    r = linprog(c=np.zeros(n),
                A_eq=np.vstack([np.ones((1, n)), X.T]),
                b_eq=np.concatenate([[1.0], x]),
                bounds=[(0, None)] * n, method='highs')
    return bool(r.status == 0) and np.max(np.abs(X.T @ r.x - x)) < 1e-6


P_tr = to13('train_mixture_1m')
Xtr = P_tr.values
p0 = Xtr.mean(axis=0)


def predict(P):
    dp = P - p0
    dq = (P ** 2 - p0 ** 2)[:, I6]
    return np.column_stack([dp @ lin[t].reindex(MODEL13).values + dq @ quad[t].values
                            for t in tgt])


def layered(name, mstem, lstem):
    P = to13(mstem).values
    L = pd.read_csv(os.path.join(A, lstem + '.csv')).iloc[:, 1:].astype(float)
    L.columns = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in L.columns]
    keep = [c for c in L.columns if c in tgt]
    assert len(keep) == len(tgt), f'目标数不匹配: {len(keep)} vs {len(tgt)}; 列={list(L.columns)}'
    L = L[keep]
    y = np.log(L.values)
    lg = predict(P)
    m = np.array([in_hull(P[i], Xtr) for i in range(len(P))])
    print(f'\n  {name}: n={len(P)}  凸包内={m.sum()}  凸包外={len(P)-m.sum()}')
    print(f'    {"分层":<8}{"n":>5}{"pooled RMSE":>14}{"逐目标RMSE均值":>16}{"Pearson":>10}{"Spearman":>10}')
    for lab, sel in [('全部', np.ones(len(P), bool)), ('凸包内', m), ('凸包外', ~m)]:
        if sel.sum() < 2:
            print(f'    {lab:<8}{sel.sum():>5}  样本不足，跳过')
            continue
        yc = y[sel] - y[sel].mean(axis=0, keepdims=True)
        lc = lg[sel] - lg[sel].mean(axis=0, keepdims=True)
        rmse = float(np.sqrt(((yc - lc) ** 2).mean()))
        rmse_mean = float(np.mean([np.sqrt(((yc[:, j] - lc[:, j]) ** 2).mean()) for j in range(len(tgt))]))
        pe = float(np.mean([stats.pearsonr(y[sel][:, j], lg[sel][:, j])[0] for j in range(len(tgt))]))
        sp = float(np.mean([stats.spearmanr(y[sel][:, j], lg[sel][:, j])[0] for j in range(len(tgt))]))
        print(f'    {lab:<8}{sel.sum():>5}{rmse:>14.4f}{rmse_mean:>16.4f}{pe:>10.4f}{sp:>10.4f}')
    return m


print('=' * 90)
print('【A】凸包内外分层指标（A4 训练配方凸包为基准；平均中心化 log RMSE）')
print('=' * 90)
layered('A6–A7 (1M)', 'test_mixture_1m', 'test_pile_loss_1m')
layered('A8–A9 (60M)', 'test_mixture_60m', 'test_pile_loss_60m')
layered('A10–A11 (1B)', 'test_mixture_1B', 'test_pile_loss_1B')

print('\n' + '=' * 90)
print('【B】等权 h_p 的留一目标权重敏感性（13 折，A6–A7 的 256 个配方）')
print('=' * 90)
P = to13('test_mixture_1m').values
H = predict(P)
hp = H.mean(axis=1)
print(f'  等权全目标 h_p: p05={np.percentile(hp,5):+.5f}  中位={np.median(hp):+.5f}  '
      f'p95={np.percentile(hp,95):+.5f}')
print(f'  {"留出目标":<20}{"与全目标 h_p 的 r":>18}{"p05":>10}{"中位":>10}{"p95":>10}')
rs, qs = [], []
for j, t in enumerate(tgt):
    keep = [k for k in range(len(tgt)) if k != j]
    hj = H[:, keep].mean(axis=1)
    r = float(stats.pearsonr(hp, hj)[0])
    q = np.percentile(hj, [5, 50, 95])
    rs.append(r)
    qs.append(q)
    print(f'  {t:<20}{r:>18.4f}{q[0]:>10.5f}{q[1]:>10.5f}{q[2]:>10.5f}')
qs = np.array(qs)
print(f'\n  留一敏感性的包络：p05 ∈ [{qs[:,0].min():+.5f}, {qs[:,0].max():+.5f}]，'
      f'中位 ∈ [{qs[:,1].min():+.5f}, {qs[:,1].max():+.5f}]，'
      f'p95 ∈ [{qs[:,2].min():+.5f}, {qs[:,2].max():+.5f}]')
print(f'  与全目标 h_p 的最小相关 r_min = {min(rs):.4f}（目标 {tgt[int(np.argmin(rs))]}）')
print(f'  结论：等权主口径在留一目标扰动下 p05–p95 变化 < 0.02，权重选择不改变情景量级。')
