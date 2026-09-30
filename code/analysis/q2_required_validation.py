# -*- coding: utf-8 -*-
"""必用数据核验：B2(跨族) B3(轨迹) B4/B5(跨族/文献) B9/B10(外推) + C的Loss-Benchmark桥接
"""
import pandas as pd
import numpy as np
import os, glob
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
CB = r'd:\F题\F题\real_attachments\C_efficiency_evolution'
LOGE = np.log

b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
N1, D1, L1 = (b1.N_params_B.values.astype(float), b1.D_tokens_B.values.astype(float),
              b1.val_loss.values.astype(float))
r1 = least_squares(lambda p: p[0] + np.exp(p[1]) * N1 ** (-np.exp(p[2])) + np.exp(p[3]) * D1 ** (-np.exp(p[4])) - L1,
                   [1.5, LOGE(0.35), LOGE(0.34), LOGE(1.24), LOGE(0.28)],
                   bounds=([0, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                           [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]), max_nfev=80000)
E, A, al, B, be = r1.x[0], np.exp(r1.x[1]), np.exp(r1.x[2]), np.exp(r1.x[3]), np.exp(r1.x[4])
print(f'[B1 锚点] E={E:.4f} A={A:.4f} alpha={al:.4f} B={B:.4f} beta={be:.4f}')

print()
print('=' * 100)
print('1) B1 拟合是否只用早停前数据 / 是否有偏离点（题目要求检查）')
print('=' * 100)
print(f'B1 run_id 数={b1.run_id.nunique()}（每行唯一）, N 取值数={b1.N_params_B.nunique()}, '
      f'D 取值数={b1.D_tokens_B.nunique()}')
g = b1.groupby(['N_params_B', 'D_tokens_B']).size()
print(f'(N,D) 组合数={len(g)}, 每组记录数分布: {sorted(set(g.values))}')
print(f'  → B1 是 {b1.N_params_B.nunique()} 个 N × {b1.D_tokens_B.nunique()} 个 D 的完整网格，'
      f'无重复(N,D)；因此不能“随机拆点”，须按 N 或按 D 整列留出')
print(f'  N 取值: {sorted(b1.N_params_B.unique())}')
print(f'  D 取值数: {b1.D_tokens_B.nunique()}, 范围 {b1.D_tokens_B.min()}~{b1.D_tokens_B.max()}')
print(f'  但 train_loss 有 {b1.train_loss.notna().sum()} 条，val_loss 有 {b1.val_loss.notna().sum()} 条')
print('  残差最大的 10 个点:')
res = L1 - (E + A * N1 ** (-al) + B * D1 ** (-be))
idx = np.argsort(-np.abs(res))[:10]
print(b1.iloc[idx][['run_id', 'N_params_B', 'D_tokens_B', 'steps', 'val_loss']].assign(
    残差=res[idx]).to_string())

print()
print('=' * 100)
print('2) 必用验证集：B2 跨族 / B3 轨迹 / B4 跨族 / B5 文献 —— 用 B1 参数的零自由度外推')
print('=' * 100)
def eval_set(df, tag, Ncol='N_params_B', Dcol='D_tokens_B', Lcol='val_loss'):
    N, D, L = df[Ncol].values.astype(float), df[Dcol].values.astype(float), df[Lcol].values.astype(float)
    pred = E + A * N ** (-al) + B * D ** (-be)
    e = L - pred
    print(f'  {tag:<34} n={len(df):>5}  RMSE={np.sqrt(np.mean(e**2)):.4f}  '
          f'MAE={np.mean(np.abs(e)):.4f}  偏差中位={np.median(e):+.4f}')
    return e

b2 = pd.read_csv(os.path.join(BASE, 'cerebras_training_log.csv'))
b4 = pd.read_csv(os.path.join(BASE, 'scaling_baseline.csv'))
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
eval_set(b2, 'B2 Cerebras (跨族)')
eval_set(b4, 'B4 scaling_baseline (跨族)')
eval_set(b5, 'B5 published (文献)')
for s, sub in b5.groupby('source'):
    eval_set(sub, f'   B5/{s}')
for s, sub in b4.groupby('family'):
    eval_set(sub, f'   B4/{s}')

print()
print('  B3 轨迹（8条，各500点，interpolated 标记）:')
trs = sorted(glob.glob(os.path.join(BASE, 'training_trajectories', '*.csv')))
for f in trs:
    d = pd.read_csv(f)
    N, D, L = d.N_params_B.values.astype(float), d.D_tokens_B.values.astype(float), d.val_loss.values.astype(float)
    pred = E + A * N ** (-al) + B * D ** (-be)
    e = L - pred
    ip = d.interpolated.value_counts().to_dict() if 'interpolated' in d.columns else {}
    print(f'    {os.path.basename(f):<38} RMSE={np.sqrt(np.mean(e**2)):.4f}  '
          f'偏差中位={np.median(e):+.4f}  interpolated={ip}')
print('  → 轨迹内 D 单调增，属于插值检验；整条轨迹留出才算真外推')

print()
print('=' * 100)
print('3) B9/B10 外推（100B~10T，超出 B1 的 12B 上限）')
print('=' * 100)
b10 = pd.read_csv(os.path.join(BASE, 'supplementary_large_baseline.csv'))
N10, D10, L10 = (b10.N_params_B.values.astype(float), b10.D_tokens_B.values.astype(float),
                 b10.val_loss.values.astype(float))
pred10 = E + A * N10 ** (-al) + B * D10 ** (-be)
e10 = L10 - pred10
print(f'  B10: n={len(b10)}, RMSE={np.sqrt(np.mean(e10**2)):.4f}, MAE={np.mean(np.abs(e10)):.4f}, '
      f'偏差中位={np.median(e10):+.4f}')
print(f'  N 范围 {N10.min():.0f}~{N10.max():.0f}B（B1 只到 {N1.max():.2f}B）→ 纯外推')
print(f'  收敛标记 is_converged: {b10.is_converged.value_counts().to_dict()}')
b9 = pd.read_csv(os.path.join(BASE, 'supplementary_large_models.csv'))
print(f'  B9: 只有元数据(无val_loss)，N {b9.N_params_B.min():.0f}~{b9.N_params_B.max():.0f}B, '
      f'年份 {b9.publication_date.astype(str).str[:4].min()}~{b9.publication_date.astype(str).str[:4].max()}')
print(f'  → B9 是模型清单（供问题三），不能作损失验证；B10 是估算/外推基线')

print()
print('=' * 100)
print('4) C 附件：Loss → Benchmark 桥接分层')
print('=' * 100)
for fn in ['loss_benchmark_bridge.csv', 'loss_benchmark_bridge_expanded.csv',
           'leaderboard_cleaned.csv', 'leaderboard_enhanced.csv',
           'leaderboard_extended_timeseries.csv', 'epoch_all_ai_models.csv',
           'model_architecture_metadata.csv']:
    p = os.path.join(CB, fn)
    if os.path.exists(p):
        d = pd.read_csv(p)
        print(f'--- {fn} shape={d.shape}')
        print('    cols:', list(d.columns))
        print(d.head(2).to_string()[:600])
        print()
parq = glob.glob(os.path.join(CB, 'data', '*.parquet'))
if parq:
    d = pd.read_parquet(parq[0])
    print(f'--- data/{os.path.basename(parq[0])} shape={d.shape}')
    print('    cols:', list(d.columns))
    print(d.head(2).to_string()[:900])