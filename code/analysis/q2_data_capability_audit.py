# -*- coding: utf-8 -*-
"""盘点问题二可用数据的"可识别性供给"：
   对每个附件表输出 行数 / 列名 / 是否含 N、D、Q、p(配比)、L 字段，
   重点判断是否存在 (p, N, L) 与 (N, Q, L) 的联合观测。
"""
import os
import pandas as pd

ROOT = r'd:\F题\F题\real_attachments'
TARGETS = [
    r'A_data_value\regmix_tables\test_mixture_1m.csv',
    r'A_data_value\regmix_tables\test_mixture_60m.csv',
    r'A_data_value\regmix_tables\test_mixture_1B.csv',
    r'A_data_value\regmix_tables\train_mixture_1m.csv',
    r'A_data_value\regmix_tables\test_pile_loss_1m.csv',
    r'A_data_value\regmix_tables\test_pile_loss_60m.csv',
    r'A_data_value\regmix_tables\test_pile_loss_1B.csv',
    r'A_data_value\regmix_tables\train_pile_loss_1m.csv',
    r'B_scaling_laws\scaling_baseline.csv',
    r'B_scaling_laws\pythia_training_log_existing.csv',
    r'B_scaling_laws\pythia_checkpoint_index.csv',
    r'B_scaling_laws\cerebras_training_log.csv',
    r'B_scaling_laws\published_scaling_data.csv',
    r'B_scaling_laws\supplementary_NQ_experiment.csv',
    r'B_scaling_laws\supplementary_NQ_experiment_expanded.csv',
    r'B_scaling_laws\supplementary_NQ_experiment_large.csv',
    r'B_scaling_laws\supplementary_large_baseline.csv',
    r'B_scaling_laws\open_model_family_metadata.csv',
]

for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f'[MISS] {rel}')
        continue
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f'[ERR ] {rel}: {e}')
        continue
    cols = list(df.columns)
    print(f'\n=== {rel}')
    print(f'    shape = {df.shape}')
    print(f'    cols  = {cols}')
    # 前两行样例（截断）
    with pd.option_context('display.max_columns', 40, 'display.width', 200):
        print('    head  =')
        print(df.head(2).to_string(max_colwidth=18))
