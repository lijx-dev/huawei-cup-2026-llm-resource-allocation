# -*- coding: utf-8 -*-
"""问题二补充分析：B8方向验证 + Q-N耦合形式 + B4/B5/B9/B10作用"""
import pandas as pd
import numpy as np
import os

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b1 = pd.read_csv(os.path.join(BASE, 'pythia_training_log_existing.csv'))
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b8 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_large.csv'))
b4 = pd.read_csv(os.path.join(BASE, 'scaling_baseline.csv'))
b5 = pd.read_csv(os.path.join(BASE, 'published_scaling_data.csv'))
b9 = pd.read_csv(os.path.join(BASE, 'supplementary_large_models.csv'))
b10 = pd.read_csv(os.path.join(BASE, 'supplementary_large_baseline.csv'))

# ---------- 1. B8 方向验证（避免浮点索引） ----------
print('===== B8 方向验证 =====')
for df in (b6, b8):
    df['Qr'] = df['Q_score'].round(2)
    df['Nr'] = df['N_params_B'].round(3)
    df['Dr'] = df['D_tokens_B'].round(1)
m6 = b6.set_index(['Nr', 'Dr', 'Qr'])['val_loss']
m8 = b8.set_index(['Nr', 'Dr', 'Qr'])['val_loss']
common = sorted(set(m6.index) & set(m8.index))
print(f'共同点 {len(common)}')
v6 = np.array([m6[k] for k in common]); v8 = np.array([m8[k] for k in common])
print(f'B6 vs B8 相关系数 = {np.corrcoef(v6, v8)[0,1]:.4f}')

# 假设1：B8 的 Q 反向（Q'=1-Q）
pairs_rev = []
for (n, d, q) in common:
    k2 = (n, d, round(1 - q, 2))
    if k2 in m8.index and k2 in m6.index:
        pairs_rev.append((m6[k2], m8[(n, d, q)]))
if pairs_rev:
    a6, a8 = np.array(pairs_rev).T
    print(f'假设Q反向(Q\'=1-Q): 对齐点数={len(pairs_rev)}, corr={np.corrcoef(a6, a8)[0,1]:.4f}')

# 假设2：B8 与 B6 在同一(N,D,Q)下的 loss 差异是否为常数
diff = v8 - v6
print(f'B8-B6 差值: mean={diff.mean():.3f}, std={diff.std():.3f} → {"近似常数偏移" if diff.std() < 0.3 else "非常数偏移"}')

# 假设3：B8 是否 = 常数 - B6 的单调变换（即完全反向）
print(f'假设完全反向: corr(B6, -B8) = {np.corrcoef(v6, -v8)[0,1]:.4f}')

# ---------- 2. B6 中 Q 效应的 N 依赖形式 ----------
print('\n===== B6: dL/dQ 随 N 的衰减形式 =====')
rows = []
for (n, dd), g in b6.groupby(['N_params_B', 'D_tokens_B']):
    g = g.sort_values('Q_score')
    if len(g) < 4:
        continue
    sl = np.polyfit(g['Q_score'], g['val_loss'], 1)
    rows.append({'N': n, 'D': dd, 'dL_dQ': sl[0]})
s = pd.DataFrame(rows)
gN = s.groupby('N')['dL_dQ'].mean()
print(gN.round(4).to_string())
ln = np.log(gN.index.values); ly = np.log(-gN.values)
slope = np.polyfit(ln, ly, 1)
print(f'\n拟合 log(-dL/dQ) ~ log(N): 斜率={slope[0]:.4f}, 截距={slope[1]:.4f}')
print(f'  → dL/dQ ∝ N^({slope[0]:.3f})，即 Q 的边际效应随 N 以幂律衰减')
r2 = 1 - np.sum((ly - np.polyval(slope, ln))**2) / np.sum((ly - ly.mean())**2)
print(f'  R² = {r2:.4f}')
# 与 -0.163 (dL/dlogN) 比较：若 dL/dQ ∝ N^-a，则 Q 与 N 的交互形式
print(f'\n对比: dL/dlogN ≈ -0.163（近似常数，与 N 无关）')
print(f'      dL/dQ ∝ N^{slope[0]:.3f}（随 N 衰减）')
print(f'  → 交互项 ∂²L/∂Q∂N 的量级 = {slope[0]:.3f} × |dL/dQ|/N')

# ---------- 3. B4/B5 跨族验证的数据结构 ----------
print('\n===== B4/B5 跨族/文献验证结构 =====')
print(f'B4: {b4["family"].nunique()} 族, {len(b4)} 行, N∈[{b4["N_params_B"].min()}, {b4["N_params_B"].max()}]')
print(f'B5: {b5["source"].nunique()} 来源, {len(b5)} 行, N∈[{b5["N_params_B"].min()}, {b5["N_params_B"].max()}]')
# B4 是否与 B1 的 Pythia 重叠
py4 = b4[b4['family'] == 'Pythia']
print(f'B4 中 Pythia 族: {len(py4)} 行, N={sorted(py4["N_params_B"].unique())}')
print(f'B1 中 N: {sorted(b1["N_params_B"].unique())}')
# 检查 B4 的 Pythia 与 B1 是否一致（交叉验证点）
merged = pd.merge(py4, b1, on=['N_params_B', 'D_tokens_B'], suffixes=('_b4', '_b1'))
print(f'B4-B1 匹配点: {len(merged)}')
if len(merged):
    print(f'  val_loss 差异: mean={np.abs(merged["val_loss_b4"] - merged["val_loss_b1"]).mean():.4f}')
    print(merged[['N_params_B', 'D_tokens_B', 'val_loss_b4', 'val_loss_b1']].head(8).to_string(index=False))

# ---------- 4. B9/B10 外推边界 ----------
print('\n===== B9/B10 大模型外推 =====')
print(f'B9: N∈[{b9["N_params_B"].min()}, {b9["N_params_B"].max()}] B params')
print(f'B10: N∈[{b10["N_params_B"].min()}, {b10["N_params_B"].max()}] B params')
print(f'B1 最大 N = {b1["N_params_B"].max()} B → B9/B10 是 {b10["N_params_B"].min()/b1["N_params_B"].max():.1f}x 以上的外推')
# B10 的 loss 与 N,D 的关系是否符合标度律
b10c = b10.dropna(subset=['N_params_B', 'D_tokens_B', 'val_loss'])
X = np.column_stack([np.ones(len(b10c)), np.log(b10c['N_params_B']), np.log(b10c['D_tokens_B'])])
y = np.log(b10c['val_loss']).values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
print(f'B10 上 logL~logN+logD: 系数={beta.round(4)}, R²={r2:.4f}')
print(f'  → 大模型上幂律指数 alpha≈{-beta[1]:.4f}, beta≈{-beta[2]:.4f}')
print(f'  对比 B1 的 alpha≈0.054, beta≈0.062')
# B10 中 D 与 N 的耦合（是否 Chinchilla 最优）
b10c = b10c.copy()
b10c['ratio'] = b10c['D_tokens_B'] / b10c['N_params_B']
print(f'B10 D/N 比值范围: {b10c["ratio"].min():.1f} ~ {b10c["ratio"].max():.1f}')