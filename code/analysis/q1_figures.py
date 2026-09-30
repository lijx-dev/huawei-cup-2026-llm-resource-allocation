# -*- coding: utf-8 -*-
"""问题一 · 论文图表生成"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
OUT = r'd:\F题\q1_quality_results'
BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'

def load_mix(name):
    df = pd.read_csv(os.path.join(BASE, name)).set_index('index')
    df.columns = [c.replace('train_the_pile_', '') for c in df.columns]
    return df

def load_loss(name):
    df = pd.read_csv(os.path.join(BASE, name)).set_index('index')
    df.columns = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in df.columns]
    return df

train_mix = load_mix('train_mixture_1m.csv')
train_loss = load_loss('train_pile_loss_1m.csv')
test_mix = {s: load_mix(f'test_mixture_{s}.csv') for s in ['1m', '60m', '1B']}
test_loss = {s: load_loss(f'test_pile_loss_{s}.csv') for s in ['1m', '60m', '1B']}
LOSS_COLS = list(train_loss.columns)
MIX_COLS = list(train_mix.columns)
NO_LOSS = ['nih_exporter', 'enron_emails', 'europarl', 'philpapers']
LOSS_DOMS = [c for c in MIX_COLS if c not in NO_LOSS]
REF = 'uspto_backgrounds'

def normalize(df):
    return df.div(df.sum(axis=1), axis=0)

def build_base(mix):
    m = normalize(mix)
    X = m[LOSS_DOMS].copy()
    X['other'] = m[NO_LOSS].sum(axis=1)
    return X.drop(columns=[REF])

def add_quad(X):
    X2 = X.copy()
    for f in ['dm_mathematics', 'github', 'freelaw', 'arxiv', 'ubuntu_irc', 'hackernews']:
        X2[f + '^2'] = X[f] ** 2
    return X2

Xtr2 = add_quad(build_base(train_mix))
Xte2 = {s: add_quad(build_base(test_mix[s])) for s in test_mix}
y_log_tr = np.log(train_loss[LOSS_COLS])

def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    return beta

betas = {j: ols(Xtr2, y_log_tr[j]) for j in LOSS_COLS}

# ---------- 图1：域级Q（4种赋权） ----------
domQ = pd.read_csv(os.path.join(OUT, 'domain_Q.csv'))
fig, ax = plt.subplots(figsize=(9, 5))
sub = domQ[domQ['source'] == 'A1抽样'].sort_values('Q_group_mean')
x = np.arange(len(sub))
w = 0.2
for i, col, lab in [(0, 'Q_group_mean', '三族等权(主)'), (1, 'Q_critic_mean', 'CRITIC'), (2, 'Q_equal_mean', '全局等权')]:
    ax.bar(x + (i - 1) * w, sub[col], w, label=lab)
ax.set_xticks(x); ax.set_xticklabels(sub['domain'])
ax.set_ylabel('域级质量评分 Q'); ax.set_ylim(0.35, 0.85)
ax.set_title('图1  七个质量域的域级质量评分（不同赋权方案）')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig1_domain_Q.png'), dpi=150); plt.close()

# ---------- 图2：配比模型检验散点（预测 vs 实际，三尺度） ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, s in zip(axes, ['1m', '60m', '1B']):
    all_pred, all_true = [], []
    logoff = None
    if s != '1m':
        comm = sorted(set(test_loss[s].index) & set(test_loss['1m'].index))
        logoff = pd.Series({j: np.log(test_loss[s].loc[comm, j]).mean() - np.log(test_loss['1m'].loc[comm, j]).mean()
                            for j in LOSS_COLS})
    for j in LOSS_COLS:
        Xc = np.column_stack([np.ones(len(Xte2[s])), Xte2[s].values])
        pred = np.exp(Xc @ betas[j] + (logoff[j] if logoff is not None else 0.0))
        all_pred.append(pred); all_true.append(test_loss[s][j].values)
    p = np.concatenate(all_pred); t = np.concatenate(all_true)
    rho = spearmanr(p, t).statistic
    ax.scatter(t, p, s=6, alpha=0.35)
    lim = [min(p.min(), t.min()) * 0.95, max(p.max(), t.max()) * 1.05]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('实际 Loss'); ax.set_ylabel('预测 Loss')
    ax.set_title(f'{s} 检验集  ρ={rho:.3f}')
    ax.grid(alpha=0.3)
plt.suptitle('图2  配比模型（M2）跨尺度检验：预测 vs 实际交叉熵 Loss')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig2_mix_validation.png'), dpi=150); plt.close()

# ---------- 图3：领域影响（边际效应 + 二次饱和） ----------
avg2 = pd.Series(pd.read_csv(os.path.join(OUT, 'mix_avg_model_quad.csv'), index_col=0).iloc[:, 0])
lin = avg2.drop(['intercept'] + [f for f in avg2.index if f.endswith('^2')])
quad = avg2[[f for f in avg2.index if f.endswith('^2')]]
base_p = pd.Series(1.0 / len(LOSS_DOMS), index=LOSS_DOMS)
marg = {}
for f in lin.index:
    if f == 'other':
        continue
    marg[f] = lin[f] + (2 * quad.get(f + '^2', 0.0) * base_p[f] if f in base_p.index else 0.0)
marg_s = pd.Series(marg).sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ['#c0392b' if v > 0 else '#2e86c1' for v in marg_s.values]
ax.barh(marg_s.index, marg_s.values, color=colors)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('均匀配比处边际效应 d(log L̄)/dp_k（负=增配降Loss，蓝=有利）')
ax.set_title('图3  各领域配比的边际效应（M2 凸二次模型，均匀配比基准）')
for i, (k, v) in enumerate(marg_s.items()):
    ax.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:+.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig3_domain_marginal.png'), dpi=150); plt.close()

# ---------- 图4：饱和效应曲线（单域集中） ----------
fig, ax = plt.subplots(figsize=(8, 5))
def logL_avg(p_dict):
    vec = pd.Series(0.0, index=list(lin.index) + [f for f in quad.index])
    for f in lin.index:
        vec[f] = p_dict.get(f, 0.0)
    for f in quad.index:
        k = f.replace('^2', '')
        vec[f] = p_dict.get(k, 0.0) ** 2
    return float(avg2['intercept'] + vec.values @ avg2[1:].values)
uni = {d: 1.0 / len(LOSS_DOMS) for d in LOSS_DOMS}
base_val = logL_avg(uni)
for dom in ['dm_mathematics', 'ubuntu_irc', 'hackernews', 'arxiv', 'github']:
    xs = np.arange(0.02, 0.92, 0.02)
    ys = []
    for pv in xs:
        p = {x: (1 - pv) / (len(LOSS_DOMS) - 1) for x in LOSS_DOMS}
        p[dom] = pv
        ys.append(logL_avg(p) - base_val)
    ax.plot(xs, ys, label=dom, lw=1.8)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('该域配比占比 p'); ax.set_ylabel('Δ log L̄（相对均匀配比）')
ax.set_title('图4  单域集中的饱和效应（凸二次模型，集中配比均有害）')
ax.legend(ncol=2); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4_saturation.png'), dpi=150); plt.close()

# ---------- 图5：质量评价指标相关热力图（冲突诊断） ----------
corr = pd.read_csv(os.path.join(OUT, 'indicator_corr.csv'), index_col=0)
fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
ax.set_title('图5  22 个质量指标的相关系数矩阵（负相关=指标级冲突）')
plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig5_indicator_corr.png'), dpi=150); plt.close()

print('图表已生成:', os.listdir(OUT))
