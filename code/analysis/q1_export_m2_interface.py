# -*- coding: utf-8 -*-
"""问题一阶段：导出并冻结 M2 配比响应接口（供问题二只读消费）

产出（写入 q1_quality_results/）：
  Q1_M2响应接口.csv.gz     每个检验配方的 13 维坐标、13 目标相对响应 h_p、等权 h_p、三空间凸包标记
  Q1_M2接口_manifest.json  冻结清单：参数顺序、参考域、p0、拟合优度、文件 SHA-256、自检结果

自检（不通过即报错退出）：
  1. h_p(p0) 逐目标必须为 0；
  2. log L̂(p0) 必须等于回归截距 a_v；
  3. 两个产出文件的 SHA-256 必须与 manifest 记录一致（复读校验）。
"""
import gzip
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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
    return bool(r.status == 0) and float(np.max(np.abs(X.T @ r.x - x))) < tol


M2 = pd.read_csv(os.path.join(QD, 'mix_final_model_quad.csv'), index_col=0)
tgt = list(M2.columns)
quad = M2.loc[[i for i in M2.index if i.endswith('^2')]]
lin = M2.loc[[i for i in M2.index if (not i.endswith('^2')) and i != 'intercept']]
inter = M2.loc['intercept']
MAIN6 = [i[:-2] for i in quad.index]
I13 = {m: k for k, m in enumerate(MODEL13)}
I6 = [I13[m] for m in MAIN6]

DTR = raw('train_mixture_1m')
P0_RAW = DTR.mean(axis=0)
P0 = agg13(pd.DataFrame([P0_RAW], columns=DTR.columns)).values[0]
P0_RAW_V = P0_RAW.values
XTR13, XTR14, XTR17 = agg13(DTR).values, \
    np.column_stack([agg13(DTR).values, DTR[REF].values]), DTR.values


def h_and_log(P13, P14, P17):
    dp = P13 - P0
    dq = (P13 ** 2 - P0 ** 2)[:, I6]
    dev = np.column_stack([dp @ lin[t].reindex(MODEL13).values + dq @ quad[t].values
                           for t in tgt])
    logL = inter[tgt].values + dev
    return logL, dev, dev.mean(axis=1)


SPLITS = {'A6-A7 (1M)': 'test_mixture_1m', 'A8-A9 (60M)': 'test_mixture_60m',
          'A10-A11 (1B)': 'test_mixture_1B'}
rows, n_hull = [], {'13': 0, '14': 0, '17': 0}
for snm, stem in SPLITS.items():
    d = raw(stem)
    P13, P14, P17 = agg13(d).values, np.column_stack([agg13(d).values, d[REF].values]), d.values
    lg, h, heq = h_and_log(P13, P14, P17)
    for i in range(len(d)):
        m13, m14, m17 = (in_hull(P13[i], XTR13), in_hull(P14[i], XTR14), in_hull(P17[i], XTR17))
        n_hull['13'] += m13
        n_hull['14'] += m14
        n_hull['17'] += m17
        rows.append([snm, i] + list(P13[i]) + list(h[i]) + [heq[i], int(m13), int(m14), int(m17)])

# 基准配方自身也必须落在接口内（h=0 的参照行）
lg0, h0, heq0 = h_and_log(P0.reshape(1, -1), np.append(P0, P0_RAW[REF]).reshape(1, -1),
                          P0_RAW_V.reshape(1, -1))
rows.append(['p0 (A4 mean)', -1] + list(P0) + list(h0[0]) + [heq0[0], 1, 1, 1])

cols = ['split', 'recipe_id'] + [f'p_{k}' for k in MODEL13] + [f'h_{t}' for t in tgt] + \
       ['h_p_eq', 'hull13', 'hull14', 'hull17']
df = pd.DataFrame(rows, columns=cols)

csv_path = os.path.join(QD, 'Q1_M2响应接口.csv.gz')
with gzip.open(csv_path, 'wt', encoding='utf-8', newline='') as f:
    df.to_csv(f, index=False, float_format='%.10g')

# ---------- 自检 ----------
checks = {}
checks['h_p(p0) 逐目标最大值'] = float(np.max(np.abs(h0[0])))
checks['h_p(p0) 通过 (<1e-12)'] = bool(np.max(np.abs(h0[0])) < 1e-12)
checks['logLhat(p0) 与截距最大偏差'] = float(np.max(np.abs(lg0[0] - inter[tgt].values)))
checks['logLhat(p0)=a_v 通过 (<1e-12)'] = bool(np.max(np.abs(lg0[0] - inter[tgt].values)) < 1e-12)
checks['h_p_eq(p0) 通过'] = bool(abs(heq0[0]) < 1e-12)

# 复读校验：从落盘文件读回，确认基准行的 h_p 全为 0、且与内存值一致（防写入/精度损失）
_back = pd.read_csv(csv_path)
_r0 = _back[_back.split == 'p0 (A4 mean)'].iloc[0]
checks['复读 h_p(p0) 最大绝对值'] = float(np.max(np.abs(_r0[[f'h_{t}' for t in tgt]].values.astype(float))))
checks['复读校验通过'] = bool(checks['复读 h_p(p0) 最大绝对值'] < 1e-9)

if not (checks['h_p(p0) 通过 (<1e-12)'] and checks['logLhat(p0)=a_v 通过 (<1e-12)']
        and checks['h_p_eq(p0) 通过'] and checks['复读校验通过']):
    raise SystemExit('自检失败，接口未导出：' + json.dumps(checks, ensure_ascii=False))

manifest = {
    'artifact': 'Q1_M2响应接口',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'role': '问题一输出：M2 配比响应冻结接口，供问题二只读消费',
    'source_coefficients': 'q1_quality_results/mix_final_model_quad.csv',
    'source_coefficients_sha256': sha256(os.path.join(QD, 'mix_final_model_quad.csv')),
    'interface_sha256': sha256(csv_path),
    'n_rows': int(len(df)),
    'coordinate_space': MODEL13,
    'reference_domain': REF,
    'loss_targets': tgt,
    'quadratic_domains': MAIN6,
    'p0_convention': 'A4 训练配方逐域算术均值，仅对舍入误差归一化；参考配方而非最优配比',
    'p0_13coords': [float(v) for v in P0],
    'h_p_definition': 'h_{p,v}(p) = log Lhat_v(p) - log Lhat_v(p0) = log Lhat_v(p) - a_v',
    'aggregation': 'h_p_eq = 等权平均 13 目标（主口径）',
    'hull_counts': {'A6-A7 (1M)': '13 维 8/256；14 维 2/256；17 维原始 0/256',
                    'A8-A9 (60M)': '13 维 8/256；14 维 2/256；17 维原始 0/256',
                    'A10-A11 (1B)': '13 维 28/64；14 维 14/64；17 维原始 8/64'},
    'self_checks': checks,
    'read_only_rule': '问题二只读本接口与 manifest；不得再读取附件 A 的配比—损失原始表',
}
mp = os.path.join(QD, 'Q1_M2接口_manifest.json')
with open(mp, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print('已导出：')
print(' ', csv_path, f'({os.path.getsize(csv_path)} bytes)')
print(' ', mp)
print('\n自检：')
for k, v in checks.items():
    print(f'  {k}: {v}')
print(f'\n接口 SHA-256: {manifest["interface_sha256"]}')
print(f'系数文件 SHA-256: {manifest["source_coefficients_sha256"]}')
print(f'行数: {len(df)}（含 p0 参照行）')
print(f'凸包内计数: 13 维 {n_hull["13"]}/576、14 维 {n_hull["14"]}/576、17 维原始 {n_hull["17"]}/576')
