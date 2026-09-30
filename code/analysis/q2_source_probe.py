# -*- coding: utf-8 -*-
"""识别问题2：跨源Loss口径；B8半合成/外推标记；逐任务结构；B3轨迹
"""
import pandas as pd
import numpy as np
import os, glob

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'

print('=' * 78)
print('一、B8 的 calibrated / extrapolated 分层')
print('=' * 78)
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))
for dt, sub in b8.groupby('data_type'):
    q, l = sub.Q_score.values, sub.val_loss.values
    print(f'[{dt}] n={len(sub)}, N={sorted(sub.N_params_B.unique())[:8]}... '
          f'({sub.N_params_B.nunique()}个), D={sub.D_tokens_B.nunique()}个, '
          f'Q={sorted(sub.Q_score.unique())}')
    print(f'     loss {l.min():.4f}~{l.max():.4f}, corr(Q,L)={np.corrcoef(q,l)[0,1]:+.3f}')
    g = sub.groupby(['N_params_B', 'D_tokens_B'])
    cors = [np.corrcoef(s.Q_score, s.val_loss)[0, 1] for _, s in g if s.Q_score.nunique() >= 4]
    cors = np.array(cors)
    print(f'     组内 corr(Q,L): 中位数={np.median(cors):+.3f}, 正值占比={np.mean(cors>0):.2f} (n={len(cors)})')
# 与B6/B7共有的点
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
def key(df): return set(zip(df.N_params_B.round(6), df.D_tokens_B.round(6), df.Q_score.round(6)))
k6, k7, k8 = key(b6), key(b7), key(b8)
print(f'\nB6∩B7={len(k6&k7)}, B6∩B8={len(k6&k8)}, B7∩B8={len(k7&k8)}')
print(f'B7-B6 独有={len(k7-k6)}, B8 与 B7 重叠={len(k7&k8)}/{len(k8)}')

print()
print('=' * 78)
print('二、跨源 Loss 口径偏移（识别问题2）：同(N,D)下的损失差')
print('=' * 78)
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b2 = pd.read_csv(os.path.join(BASE, 'cerebras_training_log.csv'))
b4 = pd.read_csv(os.path.join(BASE, 'scaling_baseline.csv'))
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
print('B1 vs B2：各自拟合经典律后的不可约损失 E')
from scipy.optimize import least_squares
def fitE(df, tag):
    N, D, L = df.N_params_B.values.astype(float), df.D_tokens_B.values.astype(float), df.val_loss.values.astype(float)
    r = least_squares(lambda p: p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4])) - L,
                      [1.5, np.log(0.35), np.log(0.34), np.log(1.24), np.log(0.28)],
                      bounds=([0, np.log(1e-8), np.log(1e-3), np.log(1e-8), np.log(1e-3)], [4, np.log(1e8), np.log(3), np.log(1e8), np.log(3)]),
                      max_nfev=60000)
    E, A, a, B, b = r.x[0], np.exp(r.x[1]), np.exp(r.x[2]), np.exp(r.x[3]), np.exp(r.x[4])
    pred = E + A*N**(-a) + B*D**(-b)
    print(f'  {tag}: n={len(df)}  E={E:.4f}  A={A:.4f} a={a:.4f}  B={B:.4f} b={b:.4f}  '
          f'RMSE={np.sqrt(np.mean((L-pred)**2)):.4f}')
    return E, A, a, B, b
for df, tag in [(b1, 'B1 Pythia'), (b2, 'B2 Cerebras'), (b4, 'B4 baseline'), (b5, 'B5 published')]:
    try:
        fitE(df, tag)
    except Exception as e:
        print(f'  {tag}: 拟合失败 {e}')

print()
print('=' * 78)
print('三、B5 published 的来源分层（文献来源 → 损失口径不同）')
print('=' * 78)
for s, sub in b5.groupby('source'):
    print(f'  {s:<24} n={len(sub):>3}  loss {sub.val_loss.min():.3f}~{sub.val_loss.max():.3f}  '
          f'收敛标记={sub.is_converged.unique()}')
print()
print('B4 baseline family:')
for s, sub in b4.groupby('family'):
    print(f'  {s:<24} n={len(sub):>3}  loss {sub.val_loss.min():.3f}~{sub.val_loss.max():.3f}')

print()
print('=' * 78)
print('四、B3 轨迹文件')
print('=' * 78)
tr = glob.glob(os.path.join(BASE, 'training_trajectories', '*.csv'))
print(f'共 {len(tr)} 个轨迹文件')
for f in tr[:20]:
    d = pd.read_csv(f)
    print(f'  {os.path.basename(f):<44} n={len(d):>4} cols={list(d.columns)}')

print()
print('=' * 78)
print('五、B9 大模型元数据（含 publication_date → 与问题三时间趋势衔接）')
print('=' * 78)
b9 = pd.read_csv(os.path.join(BASE, 'supplementary_large_models.csv'))
print(b9.head(12).to_string())
print(f'\nN范围 {b9.N_params_B.min()}~{b9.N_params_B.max()}, 年份分布:')
print(b9.publication_date.astype(str).str[:4].value_counts().sort_index().to_string())

print()
print('=' * 78)
print('六、C 附件逐任务结构（Loss→评测 桥接分层）')
print('=' * 78)
CB = r'd:\F题\F题\real_attachments\C_efficiency_evolution'
for d in sorted(os.listdir(CB)):
    p = os.path.join(CB, d)
    if os.path.isdir(p):
        fs = os.listdir(p)
        print(f'  {d}: {fs[:8]}')
print()
print('C 根目录:', [f for f in sorted(os.listdir(CB)) if os.path.isfile(os.path.join(CB, f))])