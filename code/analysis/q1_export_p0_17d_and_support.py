# -*- coding: utf-8 -*-
"""补算团队问题一接口缺失的两项信息（供问题二 h_p 使用）：

(1) 数值参考配比 p0（团队论文 §5.4 自述"冻结接口没有保存真正的数值参考配比"）
    - p0_A4mean : A4 的 512 条训练配方逐域算术均值（17 维）
    - p0_eq     : 团队现用的等权单纯形点 (1/17,...,1/17)
    两者都做 17 维凸包 LP 可行性 + 到训练配方的最近邻距离，回答"支持距离能否核验"。

(2) 换基准对 h_p 的影响量级：用 M2（13 维凸二次 log-linear）在 A6–A7 的 256 条检验配方上，
    分别以 p0_A4mean / p0_eq 为基准算 h_p，比较分布差异。

产出：q1_quality_results/Q1_p0_17dim.csv、Q1_p0_17dim_manifest.json
"""
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = r'd:\F题'
QD = os.path.join(ROOT, 'q1_quality_results')
A = os.path.join(ROOT, 'F题', 'real_attachments', 'A_data_value', 'regmix_tables')
REF = 'uspto_backgrounds'
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


def in_hull(x, X, tol=1e-6):
    n = len(X)
    r = linprog(c=np.zeros(n), A_eq=np.vstack([np.ones((1, n)), X.T]),
                b_eq=np.concatenate([[1.0], x]), bounds=[(0, None)] * n, method='highs')
    if r.status != 0:
        return False, np.inf
    return True, float(np.max(np.abs(X.T @ r.x - x)))


def nn_dist(x, X):
    return float(np.min(np.sqrt(((X - x) ** 2).sum(1))))


DTR = raw('train_mixture_1m')
COLS17 = list(DTR.columns)
X17 = DTR.values
print(f'A4 训练配方 n={len(DTR)}，17 域；行和 min={X17.sum(1).min():.8f} max={X17.sum(1).max():.8f}')

p0_17 = X17.mean(axis=0)
p0_eq = np.full(17, 1.0 / 17)

print('\n' + '=' * 88)
print('(1) 两种参考配比的支持性')
print('=' * 88)
res = {}
for nm, v in [('p0_A4mean（训练配方均值）', p0_17), ('p0_eq（团队现用等权点）', p0_eq)]:
    ok, marg = in_hull(v, X17)
    dnn = nn_dist(v, X17)
    res[nm] = dict(in_hull=ok, margin=marg, nn=dnn)
    print(f'  {nm}')
    print(f'    17 维凸包内 = {ok}   判定余量 = {marg:.3e}   到最近训练配方距离 = {dnn:.6f}')

# 训练配方的留一最近邻距离（支持半径参考，团队报 95% = 0.2600）
n = len(X17)
D = np.sqrt(((X17[:, None, :] - X17[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
nn_tr = D.min(1)
print(f'\n  训练配方留一最近邻距离：中位 {np.median(nn_tr):.4f}  '
      f'p95 {np.percentile(nn_tr, 95):.4f}  max {nn_tr.max():.4f}')

print('\n' + '=' * 88)
print('(2) 检验配方到训练集的最近邻距离与 17 维凸包计数')
print('=' * 88)
print(f'  {"检验集":<14}{"n":>5}{"NN中位":>10}{"NN p95":>10}{"NN max":>10}{"超半径数":>10}{"凸包内":>10}')
RAD = float(np.percentile(nn_tr, 95))
for nm, stem in [('A6–A7 (1M)', 'test_mixture_1m'), ('A8–A9 (60M)', 'test_mixture_60m'),
                 ('A10–A11 (1B)', 'test_mixture_1B')]:
    T = raw(stem).values
    dnn = np.sqrt(((T[:, None, :] - X17[None, :, :]) ** 2).sum(-1)).min(1)
    h = sum(in_hull(T[i], X17)[0] for i in range(len(T)))
    print(f'  {nm:<14}{len(T):>5}{np.median(dnn):>10.4f}{np.percentile(dnn,95):>10.4f}'
          f'{dnn.max():>10.4f}{int((dnn > RAD).sum()):>10}{h:>10}')

print('\n' + '=' * 88)
print('(3) 换基准对 h_p 的影响量级（M2 代理，A6–A7 的 256 条检验配方）')
print('=' * 88)
M2 = pd.read_csv(os.path.join(QD, 'mix_final_model_quad.csv'), index_col=0)
tgt = list(M2.columns)
quad = M2.loc[[i for i in M2.index if i.endswith('^2')]]
lin = M2.loc[[i for i in M2.index if (not i.endswith('^2')) and i != 'intercept']]
I13 = {m: k for k, m in enumerate(MODEL13)}
I6 = [I13[m[:-2]] for m in quad.index]

P13_tr = agg13(DTR).values
p0_13 = agg13(pd.DataFrame([p0_17], columns=COLS17)).values[0]
peq_13 = agg13(pd.DataFrame([p0_eq], columns=COLS17)).values[0]


def hmat(P13, p0):
    dp = P13 - p0
    dq = (P13 ** 2 - p0 ** 2)[:, I6]
    return np.column_stack([dp @ lin[t].reindex(MODEL13).values + dq @ quad[t].values
                            for t in tgt])


T13 = agg13(raw('test_mixture_1m')).values
H_a, H_e = hmat(T13, p0_13), hmat(T13, peq_13)
ha, he = H_a.mean(1), H_e.mean(1)
print(f'  基准 = p0_A4mean : p05={np.percentile(ha,5):+.5f}  中位={np.median(ha):+.5f}  '
      f'p95={np.percentile(ha,95):+.5f}  |h_p|p95={np.percentile(np.abs(ha),95):.5f}')
print(f'  基准 = p0_eq     : p05={np.percentile(he,5):+.5f}  中位={np.median(he):+.5f}  '
      f'p95={np.percentile(he,95):+.5f}  |h_p|p95={np.percentile(np.abs(he),95):.5f}')
print(f'  两者逐点差（等权基准 − 均值基准）: 中位 {np.median(he - ha):+.5f}  '
      f'范围 [{np.min(he - ha):+.5f}, {np.max(he - ha):+.5f}]')
r = float(np.corrcoef(ha, he)[0, 1])
print(f'  两条 h_p 的 Pearson r = {r:.6f}')
print(f'  注：M2 是 OLS 凸二次 log-linear，仅用于估计"换基准的量级"，不等于团队 LightGBM 的 h_p。')

out = pd.DataFrame({'domain': COLS17, 'p0_A4mean': p0_17, 'p0_eq': p0_eq})
op = os.path.join(QD, 'Q1_p0_17dim.csv')
out.to_csv(op, index=False, encoding='utf-8-sig', float_format='%.12g')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


man = {
    'artifact': 'Q1_p0_17dim',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'role': '问题一输出：17 维数值参考配比 p0，供问题二 h_p(p) 使用（补团队论文 §5.4 的缺失项）',
    'coordinate_space_17': COLS17,
    'reference_domain': REF,
    'p0_A4mean_definition': 'A4 训练配方 512 行的逐域算术均值（参考配方，非最优配比）',
    'p0_A4mean_rowsum': float(p0_17.sum()),
    'p0_eq_definition': '等权单纯形点 (1/17,...,1/17)（团队论文现用情景参考）',
    'support': {
        'p0_A4mean_in_hull_17d': res['p0_A4mean（训练配方均值）']['in_hull'],
        'p0_A4mean_hull_margin': res['p0_A4mean（训练配方均值）']['margin'],
        'p0_A4mean_nn_dist': res['p0_A4mean（训练配方均值）']['nn'],
        'p0_eq_in_hull_17d': res['p0_eq（团队现用等权点）']['in_hull'],
        'p0_eq_hull_margin': res['p0_eq（团队现用等权点）']['margin'],
        'p0_eq_nn_dist': res['p0_eq（团队现用等权点）']['nn'],
        'train_loo_nn_median': float(np.median(nn_tr)),
        'train_loo_nn_p95': RAD,
        'train_loo_nn_max': float(nn_tr.max()),
    },
    'file_sha256': sha256(op),
}
mp = os.path.join(QD, 'Q1_p0_17dim_manifest.json')
with open(mp, 'w', encoding='utf-8') as f:
    json.dump(man, f, ensure_ascii=False, indent=2)
print(f'\n已导出：\n  {op}\n  {mp}\n  SHA-256: {man["file_sha256"]}')
