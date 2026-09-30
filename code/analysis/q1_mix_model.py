# -*- coding: utf-8 -*-
"""
问题一 · 子任务3：17域配比 -> 13域交叉熵Loss 定量关系建模
- 成分数据共线性处理：参考域法（drop one）+ other聚合（4个无loss域）
- 模型族对比：线性 vs log-linear（RegMix风格）
- 训练: A4/A5(1M)  检验: A6-A11(1M/60M/1B)  外推稳健性: A12-A15(10B/70B)
- 质量Q引入论证：6个可映射域Q作为特征做消融实验
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations
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
NO_LOSS = ['nih_exporter', 'enron_emails', 'europarl', 'philpapers']  # 17域中无loss的4域
LOSS_DOMS = [c for c in MIX_COLS if c not in NO_LOSS]  # 13域
assert set(LOSS_DOMS) == set(LOSS_COLS)

def normalize(df):
    """行和归一化（单纯形约束）"""
    df = df.copy()
    return df.div(df.sum(axis=1), axis=0)

def build_X(mix):
    """特征矩阵：13有loss域 + other聚合，drop参考域(uspto_backgrounds)消共线性"""
    m = normalize(mix)
    X = m[LOSS_DOMS].copy()
    X['other'] = m[NO_LOSS].sum(axis=1)   # 4个无loss域聚合
    X = X.drop(columns=['uspto_backgrounds'])  # 参考域
    return X

REF = 'uspto_backgrounds'
X_train = build_X(train_mix)
X_test = {s: build_X(test_mix[s]) for s in test_mix}
X_est = {s: build_X(est_mix[s]) for s in est_mix}
FEATS = list(X_train.columns)
print('特征(rank=%d): %s' % (len(FEATS), ', '.join(FEATS)))

def ols_fit(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    pred = Xc @ beta
    ss_res = np.sum((y.values - pred) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, pred, r2

def evaluate(model, X, y, scale=''):
    """返回预测并计算指标（在原始loss尺度）"""
    beta = model['beta']
    Xc = np.column_stack([np.ones(len(X)), X.values])
    if model['form'] == 'log':
        pred = np.exp(Xc @ beta)
    else:
        pred = Xc @ beta
    mae = np.mean(np.abs(pred - y.values))
    rmse = np.sqrt(np.mean((pred - y.values) ** 2))
    rho = spearmanr(pred.ravel(), y.values.ravel()).statistic
    return {'mae': mae, 'rmse': rmse, 'rho': rho, 'pred': pred}

# ---------- 模型族对比（仅用1M训练集，域独立OLS） ----------
results = {}
for form in ['linear', 'log']:
    models = {}
    for j in LOSS_COLS:
        y = train_loss[j]
        if form == 'log':
            beta, pred, r2 = ols_fit(X_train, np.log(y))
        else:
            beta, pred, r2 = ols_fit(X_train, y)
        models[j] = {'beta': beta, 'form': form, 'r2': r2, 'pred': pred}
    results[form] = models

    print(f'\n========== 模型: {form} ==========')
    print('训练集各域R²: ' + ', '.join(f'{j}={models[j]["r2"]:.3f}' for j in LOSS_COLS))
    print('训练R²均值=%.4f' % np.mean([models[j]['r2'] for j in LOSS_COLS]))
    for s in ['1m', '60m', '1B']:
        evals = [evaluate(models[j], X_test[s], test_loss[s][j]) for j in LOSS_COLS]
        mae = np.mean([e['mae'] for e in evals]); rmse = np.mean([e['rmse'] for e in evals])
        rho = np.mean([e['rho'] for e in evals])
        print(f'检验[{s}]: MAE={mae:.4f} RMSE={rmse:.4f} 域均Spearman={rho:.4f}')

# ---------- log-linear模型：检验集尺度校正（利用同配比配对） ----------
# 估计各尺度相对1M的域级log偏移，再看校正后是否跨尺度一致
print('\n========== 尺度配对分析（同一配比不同尺度） ==========')
for s in ['60m', '1B']:
    common = sorted(set(test_loss['1m'].index) & set(test_loss[s].index))
    logdiff = pd.DataFrame({
        j: np.log(test_loss[s].loc[common, j]) - np.log(test_loss['1m'].loc[common, j])
        for j in LOSS_COLS}, index=common)
    print(f'{s} vs 1M: 域级log偏移均值= ' + ', '.join(f'{j}={logdiff[j].mean():+.3f}' for j in LOSS_COLS))
    print(f'  偏移的域间差异(极差)={logdiff.mean().max()-logdiff.mean().min():.3f}, 配比内偏移std均值={logdiff.std(axis=1).mean():.3f}')

# ---------- 尺度因子校正后的跨尺度检验 ----------
print('\n========== 跨尺度检验（log-linear + 尺度校正） ==========')
for s in ['60m', '1B']:
    common = sorted(set(test_loss['1m'].index) & set(test_loss[s].index))
    logoff = pd.Series({j: np.log(test_loss[s].loc[common, j]).mean() - np.log(test_loss['1m'].loc[common, j]).mean()
                        for j in LOSS_COLS})
    errs, rhos = [], []
    for j in LOSS_COLS:
        m = results['log'][j]
        Xc = np.column_stack([np.ones(len(X_test[s])), X_test[s].values])
        pred = np.exp(Xc @ m['beta'] + logoff[j])
        errs.append(np.mean(np.abs(pred - test_loss[s][j].values)))
        rhos.append(spearmanr(pred, test_loss[s][j].values).statistic)
    print(f'检验[{s}] 尺度校正后: MAE={np.mean(errs):.4f} 域均Spearman={np.mean(rhos):.4f}')

# ---------- 外推稳健性：10B/70B ----------
print('\n========== 外推表（10B/70B，幂律外推loss） ==========')
for s in ['10b', '70b']:
    common = sorted(set(est_loss[s].index) & set(train_loss.index))
    logoff = pd.Series({j: np.log(est_loss[s].loc[common, j]).mean() - np.log(train_loss.loc[common, j]).mean()
                        for j in LOSS_COLS})
    errs, rhos = [], []
    for j in LOSS_COLS:
        m = results['log'][j]
        Xc = np.column_stack([np.ones(len(X_est[s])), X_est[s].values])
        pred = np.exp(Xc @ m['beta'] + logoff[j])
        errs.append(np.mean(np.abs(pred - est_loss[s][j].values)))
        rhos.append(spearmanr(pred, est_loss[s][j].values).statistic)
    print(f'外推[{s}]: 尺度校正后MAE={np.mean(errs):.4f} 域均Spearman={np.mean(rhos):.4f}')

# ---------- 领域影响分析：系数解释 ----------
print('\n========== 领域影响（log-linear系数，相对参考域uspto_backgrounds） ==========')
coef_df = pd.DataFrame({j: results['log'][j]['beta'] for j in LOSS_COLS})
coef_df.index = ['intercept'] + FEATS
print(coef_df.round(3).to_string())
coef_df.to_csv(os.path.join(OUT, 'mix_loglinear_coef.csv'), encoding='utf-8-sig')

# 各域提升相对效果：把某域配比从0.1增加到0.3（其他等比例缩减），预测13域loss的几何均值变化
print('\n========== 配比敏感性：单域 +0.2 对13域平均log-loss的影响 ==========')
base_p = pd.Series(1.0 / len(LOSS_DOMS), index=LOSS_DOMS)  # 均匀配比
sens = {}
for dom in LOSS_DOMS:
    new_p = base_p.copy()
    new_p[dom] += 0.2
    new_p = new_p / new_p.sum()
    d_logL = 0.0
    for j in LOSS_COLS:
        m = results['log'][j]
        x0 = base_p[LOSS_DOMS].copy(); x0['other'] = 0.0; x0 = x0.drop(REF)
        x1 = new_p[LOSS_DOMS].copy(); x1['other'] = 0.0; x1 = x1.drop(REF)
        d_logL += (x1.values @ m['beta'][1:] - x0.values @ m['beta'][1:]) / len(LOSS_COLS)
    sens[dom] = d_logL
sens = pd.Series(sens).sort_values()
print('（负=整体Loss下降，性能提升）')
print(sens.round(4).to_string())
sens.to_csv(os.path.join(OUT, 'mix_domain_sensitivity.csv'), encoding='utf-8-sig')
print('\n配比建模完成')
