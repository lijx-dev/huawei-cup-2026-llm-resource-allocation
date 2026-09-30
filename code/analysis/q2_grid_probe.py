# -*- coding: utf-8 -*-
"""B6/B7/B8 网格结构与 Q-L 方向探查"""
import pandas as pd
import numpy as np
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'


def load(fn):
    return pd.read_csv(os.path.join(BASE, fn))


b6 = load('supplementary_NQ_experiment.csv')
b7 = load('supplementary_NQ_experiment_expanded.csv')
b8 = load('supplementary_NQ_experiment_large.csv')

for tag, df in [('B6', b6), ('B7', b7), ('B8', b8)]:
    print(f'===== {tag} =====')
    print('N:', sorted(df.N_params_B.unique()))
    print('D:', sorted(df.D_tokens_B.unique()))
    print('Q:', sorted(df.Q_score.unique()))
    print(f'loss range: {df.val_loss.min():.4f} ~ {df.val_loss.max():.4f}')
    g = df.groupby(['N_params_B', 'D_tokens_B']).size()
    print(f'(N,D) 组数={len(g)}, 每组点数={sorted(set(g.values))}')
    if 'data_type' in df.columns:
        print('data_type:', df.data_type.value_counts().to_dict())
        for dt, sub in df.groupby('data_type'):
            print(f'   [{dt}] loss {sub.val_loss.min():.3f}~{sub.val_loss.max():.3f}, '
                  f'N={sorted(sub.N_params_B.unique())}, D={sorted(sub.D_tokens_B.unique())}')
    cors = []
    for (n, d), sub in df.groupby(['N_params_B', 'D_tokens_B']):
        if sub.Q_score.nunique() >= 4:
            cors.append(np.corrcoef(sub.Q_score, sub.val_loss)[0, 1])
    cors = np.array(cors)
    print(f'组内 corr(Q,L): 中位数={np.median(cors):.3f}, 负值占比={np.mean(cors < 0):.2f}, 组数={len(cors)}')
    print()