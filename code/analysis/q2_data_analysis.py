# -*- coding: utf-8 -*-
"""问题二数学分析：数据规律探查"""
import pandas as pd
import numpy as np
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'

# ---------- 1. B1 Pythia：经典标度律的数据支撑 ----------
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
print('===== B1 Pythia 训练日志 =====')
print('N 取值:', sorted(b1['N_params_B'].unique()))
print('D 取值数:', b1['D_tokens_B'].nunique(), '范围:', b1['D_tokens_B'].min(), '~', b1['D_tokens_B'].max())
print('val_loss 范围:', b1['val_loss'].min(), '~', b1['val_loss'].max())
print('每个N的D覆盖数:')
print(b1.groupby('N_params_B')['D_tokens_B'].agg(['count', 'min', 'max']).to_string())
# C=6ND 是否成立
b1['C_calc'] = 6 * b1['N_params_B'] * 1e9 * b1['D_tokens_B'] * 1e9 / 1e21
ratio = (b1['C_calc'] / b1['C_FLOPs_1e21']).replace([np.inf, -np.inf], np.nan).dropna()
print(f'C_FLOPs / (6ND) 比值: median={ratio.median():.4f}')

# ---------- 2. B6/B7/B8：Q 与 loss 的关系（关键） ----------
print('\n===== B6/B7/B8 质量实验：Q vs loss =====')
for tag, fn in [('B6', 'supplementary_NQ_experiment.csv'),
                ('B7', 'supplementary_NQ_experiment_expanded.csv'),
                ('B8', 'supplementary_NQ_experiment_large.csv')]:
    df = pd.read_csv(os.path.join(BASE, fn))
    print(f'\n--- {tag} shape={df.shape} ---')
    print('Q 取值:', sorted(df['Q_score'].unique()))
    print('N 取值:', sorted(df['N_params_B'].unique()))
    print('D 取值:', sorted(df['D_tokens_B'].unique()))
    if 'data_type' in df.columns:
        print('data_type 分布:', df['data_type'].value_counts().to_dict())
    # 固定 N,D 时 Q 与 loss 的单调性
    g = df[(df['N_params_B'] == df['N_params_B'].iloc[0]) & (df['D_tokens_B'] == df['D_tokens_B'].iloc[0])]
    g = g.sort_values('Q_score')
    print(f'固定 N={g["N_params_B"].iloc[0]}, D={g["D_tokens_B"].iloc[0]} 时:')
    print(g[['Q_score', 'val_loss']].to_string(index=False))
    # 全局：Q 与 loss 的相关系数
    r = np.corrcoef(df['Q_score'], df['val_loss'])[0, 1]
    print(f'全局 corr(Q, loss) = {r:.4f}')

# ---------- 3. B4/B5：跨族/文献 ----------
print('\n===== B4 scaling_baseline =====')
b4 = pd.read_csv(os.path.join(BASE, 'scaling_baseline.csv'))
print('族数:', b4['family'].nunique(), '族:', sorted(b4['family'].unique()))
print('N范围:', b4['N_params_B'].min(), '~', b4['N_params_B'].max())
print('D范围:', b4['D_tokens_B'].min(), '~', b4['D_tokens_B'].max())
print(b4.groupby('family').size().to_string())

print('\n===== B5 published_scaling_data =====')
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
print('来源:', b5['source'].value_counts().to_dict())
print('族:', sorted(b5['family'].unique()))

# ---------- 4. B9/B10：大模型 ----------
print('\n===== B9/B10 大模型 =====')
b9 = pd.read_csv(os.path.join(BASE, 'supplementary_large_models.csv'))
b10 = pd.read_csv(os.path.join(BASE, 'supplementary_large_baseline.csv'))
print('B9 N范围:', b9['N_params_B'].min(), '~', b9['N_params_B'].max(), '| 行数:', len(b9))
print('B10 N范围:', b10['N_params_B'].min(), '~', b10['N_params_B'].max(), '| 行数:', len(b10))
print('B10 val_loss 范围:', b10['val_loss'].min(), '~', b10['val_loss'].max())
print('B9 accessibility 分布:', b9['accessibility'].value_counts().to_dict())