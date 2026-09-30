# -*- coding: utf-8 -*-
"""
问题一 · 子任务3 最终模型选择：
M1 log-linear(线性) vs M2 log-linear+二次项（主域），在检验集1M/60M/1B与外推集上对比
考虑问题三配比优化的可解性，检查M2的凸性
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

Xtr1 = build_base(train_mix)
Xtr2 = add_quad(Xtr1)
Xte1 = {s: build_base(test_mix[s]) for s in test_mix}
Xte2 = {s: add_quad(Xte1[s]) for s in test_mix}
Xes1 = {s: build_base(est_mix[s]) for s in est_mix}
Xes2 = {s: add_quad(Xes1[s]) for s in est_mix}
y_log_tr = np.log(train_loss[LOSS_COLS])

def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    return beta

def fit_eval(name, Xtr, Xte, Xes):
    betas = {j: ols(Xtr, y_log_tr[j]) for j in LOSS_COLS}
    print(f'\n===== {name} =====')
    # 训练R²
    r2s = []
    for j in LOSS_COLS:
        Xc = np.column_stack([np.ones(len(Xtr)), Xtr.values])
        pred = Xc @ betas[j]
        y = y_log_tr[j].values
        r2s.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f'训练R²均值={np.mean(r2s):.4f}')
    # 检验（同尺度1m + 尺度校正60m/1B）
    for s in ['1m', '60m', '1B']:
        if s == '1m':
            errs, rhos, mapes = [], [], []
            for j in LOSS_COLS:
                Xc = np.column_stack([np.ones(len(Xte[s])), Xte[s].values])
                pred = np.exp(Xc @ betas[j])
                yt = test_loss[s][j].values
                errs.append(np.mean(np.abs(pred - yt)))
                rhos.append(spearmanr(pred, yt).statistic)
                mapes.append(np.mean(np.abs(pred - yt) / yt))
            print(f'检验[1m]: MAE={np.mean(errs):.4f} MAPE={np.mean(mapes):.3f} 秩相关={np.mean(rhos):.4f}')
        else:
            comm = sorted(set(test_loss[s].index) & set(test_loss['1m'].index))
            logoff = pd.Series({j: np.log(test_loss[s].loc[comm, j]).mean() - np.log(test_loss['1m'].loc[comm, j]).mean()
                                for j in LOSS_COLS})
            errs, rhos, mapes = [], [], []
            for j in LOSS_COLS:
                Xc = np.column_stack([np.ones(len(Xte[s])), Xte[s].values])
                pred = np.exp(Xc @ betas[j] + logoff[j])
                yt = test_loss[s][j].values
                errs.append(np.mean(np.abs(pred - yt)))
                rhos.append(spearmanr(pred, yt).statistic)
                mapes.append(np.mean(np.abs(pred - yt) / yt))
            print(f'检验[{s}]: 校正后MAE={np.mean(errs):.4f} MAPE={np.mean(mapes):.3f} 秩相关={np.mean(rhos):.4f}')
    # 外推
    for s in ['10b', '70b']:
        comm = sorted(set(est_loss[s].index) & set(train_loss.index))
        logoff = pd.Series({j: np.log(est_loss[s].loc[comm, j]).mean() - np.log(train_loss.loc[comm, j]).mean()
                            for j in LOSS_COLS})
        errs, rhos, mapes = [], [], []
        for j in LOSS_COLS:
            Xc = np.column_stack([np.ones(len(Xes[s])), Xes[s].values])
            pred = np.exp(Xc @ betas[j] + logoff[j])
            yt = est_loss[s][j].values
            errs.append(np.mean(np.abs(pred - yt)))
            rhos.append(spearmanr(pred, yt).statistic)
            mapes.append(np.mean(np.abs(pred - yt) / yt))
        print(f'外推[{s}]: 校正后MAE={np.mean(errs):.4f} MAPE={np.mean(mapes):.3f} 秩相关={np.mean(rhos):.4f}')
    return betas

betas1 = fit_eval('M1 log-linear(线性)', Xtr1, Xte1, Xes1)
betas2 = fit_eval('M2 log-linear+二次项(6主域)', Xtr2, Xte2, Xes2)

# 二次模型的Hessian（对平均log-loss）——凸性检查
Xc2 = np.column_stack([np.ones(len(Xtr2)), Xtr2.values])
beta_avg2 = ols(Xtr2, y_log_tr.mean(axis=1))
n_feat = len(Xtr2.columns)
H = np.zeros((n_feat, n_feat))
for i, f in enumerate(Xtr2.columns):
    if f.endswith('^2'):
        k = f.replace('^2', '')
        idx = list(Xtr2.columns).index(f)
        H[idx, idx] = 2 * beta_avg2[1 + idx]
print('\n===== M2凸性检查（平均log-loss Hessian对角，二次项系数） =====')
for f in Xtr2.columns:
    if f.endswith('^2'):
        k = list(Xtr2.columns).index(f)
        print(f'  {f}: 二次系数={beta_avg2[1+k]:+.4f} （>0凸, <0凹）')
print('负二次系数=目标函数在log空间非凸，需在配比优化(问题三)中处理')

# 保存M1最终系数（主交付模型）
betas1_df = pd.DataFrame(betas1, index=['intercept'] + list(Xtr1.columns))
betas1_df.to_csv(os.path.join(OUT, 'mix_final_model_loglinear.csv'), encoding='utf-8-sig')
print('\n已保存M1最终系数: mix_final_model_loglinear.csv')
print('模型选择完成')
