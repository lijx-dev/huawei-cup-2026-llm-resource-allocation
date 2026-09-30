# -*- coding: utf-8 -*-
"""识别问题1：A端综合分 vs B端Q_score 是否同一尺度；B各文件Loss口径与来源
"""
import pandas as pd
import numpy as np
import os
import glob

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'

print('=' * 78)
print('一、附件B各文件：形状 / 列名 / Loss口径线索')
print('=' * 78)
for fn in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, fn)
    if not fn.endswith('.csv'):
        continue
    df = pd.read_csv(p)
    print(f'--- {fn}  shape={df.shape}')
    print('    cols:', list(df.columns))
    # 找 loss / 语料 / tokenizer 相关列
    key = [c for c in df.columns if any(k in c.lower() for k in
           ['loss', 'corpus', 'tokenizer', 'eval', 'source', 'data_type', 'note', 'q_', 'quality'])]
    if key:
        print('    关键列:', key)
        for c in key:
            v = df[c].dropna().unique()
            if len(v) <= 12:
                print(f'      {c}: {sorted(map(str, v))[:12]}')
            else:
                print(f'      {c}: {len(v)} 个取值, 例 {v[:5]}')

print()
print('=' * 78)
print('二、B1/B2/B4/B5 的 Loss 水平对照（识别问题2：跨源可比性）')
print('=' * 78)
specs = [('B1', 'pythia_training_log_existing.csv'),
         ('B2', 'cerebras_training_log.csv'),
         ('B4', 'scaling_baseline.csv'),
         ('B5', 'published_scaling_data.csv'),
         ('B9', 'supplementary_large_models.csv'),
         ('B10', 'supplementary_large_baseline.csv')]
for tag, fn in specs:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print(f'{tag}: 缺失'); continue
    df = pd.read_csv(p)
    lc = [c for c in df.columns if 'loss' in c.lower()]
    print(f'{tag} ({fn}) n={len(df)}')
    for c in lc:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        print(f'   {c}: min={s.min():.4f} med={s.median():.4f} max={s.max():.4f}')
    if 'N_params_B' in df.columns:
        print(f'   N范围: {df.N_params_B.min()} ~ {df.N_params_B.max()}')

print()
print('=' * 78)
print('三、B6/B7/B8 的 Q_score 分布（B端质量尺度）')
print('=' * 78)
for tag, fn in [('B6', 'supplementary_NQ_experiment.csv'),
                ('B7', 'supplementary_NQ_experiment_expanded.csv'),
                ('B8', 'supplementary_NQ_experiment_large.csv')]:
    df = pd.read_csv(os.path.join(BASE, fn))
    q = df.Q_score.values.astype(float)
    print(f'{tag}: n={len(df)}, Q取值={len(np.unique(q))} 个, '
          f'min={q.min():.3f} max={q.max():.3f} mean={q.mean():.3f} std={q.std():.4f}')
    print(f'     Q 唯一值: {np.sort(np.unique(q))}')
    print(f'     loss: {df.val_loss.min():.4f} ~ {df.val_loss.max():.4f}')
    print(f'     与Q的相关: corr(Q,L)={np.corrcoef(q, df.val_loss.values)[0,1]:.3f}, '
          f'spearman={pd.Series(q).corr(pd.Series(df.val_loss.values), method="spearman"):.3f}')

print()
print('=' * 78)
print('四、A端质量综合分（问题一产物）的分布')
print('=' * 78)
d = pd.read_csv(r'd:\F题\q1_quality_results\domain_Q.csv')
print('domain_Q.csv cols:', list(d.columns))
print(d.to_string()[:1500])
print()
a1 = pd.read_csv(r'd:\F题\q1_quality_results\A1_vs_full_domain_Q.csv')
print('A1_vs_full_domain_Q.csv:'); print(a1.to_string())