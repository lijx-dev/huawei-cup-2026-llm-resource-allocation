# -*- coding: utf-8 -*-
"""附件B（问题二）数据结构探查"""
import pandas as pd
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
files = [
    ('B1', 'pythia_training_log_existing.csv'),
    ('B2', 'cerebras_training_log.csv'),
    ('B4', 'scaling_baseline.csv'),
    ('B5', 'published_scaling_data.csv'),
    ('B6', 'supplementary_NQ_experiment.csv'),
    ('B7', 'supplementary_NQ_experiment_expanded.csv'),
    ('B8', 'supplementary_NQ_experiment_large.csv'),
    ('B9', 'supplementary_large_models.csv'),
    ('B10', 'supplementary_large_baseline.csv'),
    ('B11', 'open_model_family_metadata.csv'),
    ('B12', 'pythia_checkpoint_index.csv'),
]
for tag, fn in files:
    p = os.path.join(BASE, fn)
    df = pd.read_csv(p)
    print(f'===== {tag} {fn}  shape={df.shape} =====')
    print('列:', list(df.columns))
    print(df.head(3).to_string()[:1200])
    print()

# B3 轨迹
t = pd.read_csv(os.path.join(BASE, 'training_trajectories', 'pythia_0.070542B_trajectory.csv'))
print(f'===== B3 trajectory shape={t.shape} =====')
print('列:', list(t.columns))
print(t.head(3).to_string())