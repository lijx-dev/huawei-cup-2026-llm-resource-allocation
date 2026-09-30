# -*- coding: utf-8 -*-
"""
问题一 · 子任务3 补充2：
(1) 数据级可加性验证：log-linear vs 带二次项/交互项模型（证明可加性来自数据）
(2) 域均值基准对照：配比模型的增量价值
(3) 外推逐域诊断：10B/70B Spearman低的原因定位
(4) 最终模型选定与输出汇总
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
OUT = r'd:\F题\q1_quality_results'

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
est_mix = {s: load_mix(f'est_mixture_{s}.csv') for s in ['10b', '70b']}
est_loss = {s: load_loss(f'est_pile_loss_{s}.csv') for s in ['10b', '70b']}

MIX_COLS = list(train_mix.columns)
LOSS_COLS = list(train_loss.columns)
NO_LOSS = ['nih_exporter', 'enron_emails', 'europarl', 'philpapers']
LOSS_DOMS = [c for c in MIX_COLS if c not in NO_LOSS]
REF = 'uspto_backgrounds'

def normalize(df):
    return df.div(df.sum(axis=1), axis=0)

def build_X(mix):
    m = normalize(mix)
    X = m[LOSS_DOMS].copy()
    X['other'] = m[NO_LOSS].sum(axis=1)
    return X.drop(columns=[REF])

Xtr = build_X(train_mix)
Xte = {s: build_X(test_mix[s]) for s in test_mix}
Xest = {s: build_X(est_mix[s]) for s in est_mix}
FEATS = list(Xtr.columns)

def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    pred = Xc @ beta
    r2 = 1 - np.sum((y.values - pred) ** 2) / np.sum((y.values - y.values.mean()) ** 2)
    return beta, r2, pred

# ---------- (1) 数据级可加性验证 ----------
print('========== 可加性验证（对13域log-loss平均） ==========')
y_log = np.log(train_loss[LOSS_COLS]).mean(axis=1)  # 平均log-loss作为目标
# M0: 域均值基准
y_mean = y_log.mean()
r2_dom0 = 1 - np.sum((y_log - y_log.mean()) ** 2) / np.sum((y_log - y_mean) ** 2)
print(f'M0 常数基准 R²={r2_dom0:.4f}（应=0，对照用）')
# M1: 线性（基线log-linear）
beta1, r2_1, _ = ols(Xtr, y_log)
print(f'M1 log-linear(13特征) R²={r2_1:.4f}')
# M2: 加二次项（主要域）
top4 = ['dm_mathematics', 'github', 'freelaw', 'arxiv']
X2 = Xtr.copy()
for f in top4:
    X2[f + '^2'] = Xtr[f] ** 2
for a, b in [('dm_mathematics', 'hackernews'), ('github', 'freelaw')]:
    X2[f'{a}x{b}'] = Xtr[a] * Xtr[b]
beta2, r2_2, _ = ols(X2, y_log)
print(f'M2 线性+二次项/交互(19特征) R²={r2_2:.4f}  增量={r2_2-r2_1:+.4f}')

# 分解：仅二次项 vs 仅交互项
X_quad = Xtr.copy()
for f in top4:
    X_quad[f + '^2'] = Xtr[f] ** 2
_, r2_q, _ = ols(X_quad, y_log)
X_int = Xtr.copy()
for a, b in [('dm_mathematics', 'hackernews'), ('github', 'freelaw')]:
    X_int[f'{a}x{b}'] = Xtr[a] * Xtr[b]
_, r2_i, _ = ols(X_int, y_log)
print(f'仅二次项 R²={r2_q:.4f}（增量{r2_q-r2_1:+.4f}）  仅交互项 R²={r2_i:.4f}（增量{r2_i-r2_1:+.4f}）')

# 8折CV（numpy实现）
def cv_r2(Xf, y, folds=8, seed=0):
    idx = np.arange(len(y)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    r2s = []
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        Xc_tr = np.column_stack([np.ones(len(tr)), Xf.iloc[tr].values])
        Xc_te = np.column_stack([np.ones(len(te)), Xf.iloc[te].values])
        beta, *_ = np.linalg.lstsq(Xc_tr, y.iloc[tr].values, rcond=None)
        pred = Xc_te @ beta
        r2s.append(1 - np.sum((y.iloc[te].values - pred) ** 2) / np.sum((y.iloc[te].values - y.iloc[te].values.mean()) ** 2))
    return np.mean(r2s)
print(f'M1 8折CV R²={cv_r2(Xtr, y_log):.4f}')
print(f'M2 8折CV R²={cv_r2(X2, y_log):.4f}')

# ---------- (2) 域均值基准对照（每域独立） ----------
print('\n========== 配比信息增量价值（每域） ==========')
inc = {}
for j in LOSS_COLS:
    y = np.log(train_loss[j])
    r2_dom = 1 - np.sum((y - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2)  # =0 定义
    beta, r2_p, _ = ols(Xtr, y)
    inc[j] = r2_p
print('配比模型对每域的R²（域均值基准为0，R²即配比增量解释力）:')
print('  ' + ', '.join(f'{j}={inc[j]:.3f}' for j in LOSS_COLS))
print('  平均=%.3f' % np.mean(list(inc.values())))

# ---------- (3) 外推逐域诊断 ----------
print('\n========== 外推逐域诊断（log-linear模型，尺度校正后） ==========')
for s in ['10b', '70b']:
    comm = sorted(set(est_loss[s].index) & set(train_loss.index))
    logoff = pd.Series({j: np.log(est_loss[s].loc[comm, j]).mean() - np.log(train_loss.loc[comm, j]).mean()
                        for j in LOSS_COLS})
    per_dom = []
    for j in LOSS_COLS:
        beta, _, _ = ols(Xtr, np.log(train_loss[j]))
        Xc = np.column_stack([np.ones(len(Xest[s])), Xest[s].values])
        pred = np.exp(Xc @ beta + logoff[j])
        rho = spearmanr(pred, est_loss[s][j].values).statistic
        mape = np.mean(np.abs(pred - est_loss[s][j].values) / est_loss[s][j].values)
        per_dom.append((j, rho, mape))
    per_dom.sort(key=lambda x: x[1])
    print(f'外推[{s}]:')
    for j, rho, mape in per_dom:
        print(f'  {j:20s} Spearman={rho:+.3f} MAPE={mape:.3f}')

# 外推误差与配比的关系：残差是否集中在高配比域
print('\n========== 外推残差模式（10B） ==========')
s = '10b'
comm = sorted(set(est_loss[s].index) & set(train_loss.index))
logoff = pd.Series({j: np.log(est_loss[s].loc[comm, j]).mean() - np.log(train_loss.loc[comm, j]).mean()
                    for j in LOSS_COLS})
resid = pd.DataFrame(index=est_loss[s].index)
for j in LOSS_COLS:
    beta, _, _ = ols(Xtr, np.log(train_loss[j]))
    Xc = np.column_stack([np.ones(len(Xest[s])), Xest[s].values])
    pred = np.exp(Xc @ beta + logoff[j])
    resid[j] = np.log(est_loss[s][j]) - np.log(pred)
r = resid.mean(axis=1)
dom_share = normalize(est_mix[s])[LOSS_DOMS].idxmax(axis=1)
print('残差按主导域分组的均值（正=预测偏低，实际偏高）:')
print(pd.DataFrame({'mean_resid_log': r.groupby(dom_share).mean().round(4)}).to_string())

# ---------- (4) 最终输出汇总 ----------
print('\n========== 最终模型系数（log-linear，13域独立，相对参考域uspto） ==========')
coef_all = {}
for j in LOSS_COLS:
    beta, r2, _ = ols(Xtr, np.log(train_loss[j]))
    coef_all[j] = beta
coef_df = pd.DataFrame(coef_all, index=['intercept'] + FEATS)
print(coef_df.round(3).to_string())
coef_df.to_csv(os.path.join(OUT, 'mix_final_coef.csv'), encoding='utf-8-sig')

# 域难度排序（截距）
print('\n域难度基准（intercept, log-loss，越大越难）:')
print(coef_df.loc['intercept'].sort_values(ascending=False).round(3).to_string())

# 全域平均模型系数（用于报告"配比→平均性能"）
beta_avg, r2_avg, _ = ols(Xtr, y_log)
avg_ser = pd.Series(beta_avg, index=['intercept'] + FEATS)
print('\n13域平均log-loss的配比模型:')
print(avg_ser.round(4).to_string())
avg_ser.to_csv(os.path.join(OUT, 'mix_avg_model.csv'), encoding='utf-8-sig')
print('\n补充分析2完成')
