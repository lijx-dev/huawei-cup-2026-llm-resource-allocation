# -*- coding: utf-8 -*-
"""核验 §2.6.3 的留一最近邻距离阈值，以及 §1.1 各附件表的规模"""
import os
import numpy as np
import pandas as pd

A = r'F题\real_attachments\A_data_value\regmix_tables'
B = r'F题\real_attachments\B_scaling_laws'
R = r'F题\real_attachments'

X = pd.read_csv(os.path.join(A, 'train_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
print(f'A4 训练配方 n={X.shape[0]}, 维度={X.shape[1]}')
n = len(X)
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
nn = D.min(axis=1)
print('留一最近邻距离（17 维欧氏）:')
print(f'  min={nn.min():.6f}  median={np.median(nn):.6f}  '
      f'p95={np.percentile(nn,95):.6f}  max={nn.max():.6f}')

# 检验配方到训练集的支持距离
T = pd.read_csv(os.path.join(A, 'test_mixture_1m.csv')).iloc[:, 1:].values.astype(float)
Dt = np.sqrt(((T[:, None, :] - X[None, :, :]) ** 2).sum(-1))
nnt = Dt.min(axis=1)
print(f'\nA4 检验配方到训练集的最近邻距离: median={np.median(nnt):.6f}  '
      f'p95={np.percentile(nnt,95):.6f}  max={nnt.max():.6f}')

print('\n' + '=' * 70)
print('§1.1 各附件表规模核验')
print('=' * 70)
for f in sorted(os.listdir(B)):
    p = os.path.join(B, f)
    if f.lower().endswith('.csv'):
        try:
            d = pd.read_csv(p)
            print(f'  {f:<48} n={len(d):<7} cols={len(d.columns)}')
        except Exception as e:
            print(f'  {f:<48} 读取失败: {e}')
    elif os.path.isdir(p):
        fs = [x for x in os.listdir(p) if x.lower().endswith('.csv')]
        tot = 0
        for x in fs:
            try:
                tot += len(pd.read_csv(os.path.join(p, x)))
            except Exception:
                pass
        print(f'  {f}/ 目录: {len(fs)} 个 csv，合计 {tot} 行')
