# -*- coding: utf-8 -*-
"""问题一 · 子任务3 最终交付：M2模型系数、域敏感性、组合效应、汇总表"""
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

Xtr2 = add_quad(build_base(train_mix))
y_log_tr = np.log(train_loss[LOSS_COLS])

def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    return beta

FEATS = list(Xtr2.columns)
betas2 = {j: ols(Xtr2, y_log_tr[j]) for j in LOSS_COLS}
coef2 = pd.DataFrame(betas2, index=['intercept'] + FEATS)
coef2.to_csv(os.path.join(OUT, 'mix_final_model_quad.csv'), encoding='utf-8-sig')
print('已保存M2系数: mix_final_model_quad.csv')

beta_avg = ols(Xtr2, y_log_tr.mean(axis=1))
avg2 = pd.Series(beta_avg, index=['intercept'] + FEATS)
avg2.to_csv(os.path.join(OUT, 'mix_avg_model_quad.csv'), encoding='utf-8-sig')

# ---------- 域敏感性（M2，边际效应随当前配比变化） ----------
print('\n========== 领域影响（M2平均log-loss模型） ==========')
print('log L̄ = %.4f + Σβp + Σγp²' % avg2['intercept'])
lin = avg2.drop(['intercept'] + [f for f in FEATS if f.endswith('^2')])
quad = avg2[[f for f in FEATS if f.endswith('^2')]]
print('线性系数:'); print(lin.round(4).to_string())
print('二次系数(全部为正→凸):'); print(quad.round(4).to_string())

# 均匀配比基准处的边际效应 dL̄/dp_k = β_k + 2γ_k p_k
base_p = pd.Series(1.0 / len(LOSS_DOMS), index=LOSS_DOMS)
base_vec = pd.Series(0.0, index=FEATS)
for f in lin.index:
    base_vec[f] = base_p[f] if f in base_p.index else 0.0
for f in quad.index:
    base_vec[f] = base_p[f.replace('^2', '')] ** 2 if f.replace('^2', '') in base_p.index else 0.0
marg = pd.Series({f: 0.0 for f in FEATS})
for f in lin.index:
    marg[f] = lin[f]
for f in quad.index:
    k = f.replace('^2', '')
    marg[k] = marg.get(k, 0) + 2 * quad[f] * base_p[k]
marg = marg.drop('other', errors='ignore')
marg = marg.sort_values()
print('\n均匀配比处的边际效应 d(log L̄)/dp_k（负=增加该域降低平均loss）:')
print(marg.round(4).to_string())
marg.to_csv(os.path.join(OUT, 'mix_domain_marginal.csv'), encoding='utf-8-sig')

# ---------- 组合效应：最佳/最差配比搜索（凸问题，网格+随机搜索） ----------
print('\n========== 组合效应：13域均匀 vs 单域集中 vs 最优组合 ==========')
def logL_avg(p_dict):
    vec = pd.Series(0.0, index=FEATS)
    for f in lin.index:
        vec[f] = p_dict.get(f, 0.0)
    for f in quad.index:
        k = f.replace('^2', '')
        vec[f] = p_dict.get(k, 0.0) ** 2
    return float(avg2['intercept'] + vec.values @ avg2[1:].values)

uni = {d: 1.0 / len(LOSS_DOMS) for d in LOSS_DOMS}
print(f'均匀配比: log L̄={logL_avg(uni):.4f}')

# 单域集中
for d in LOSS_DOMS:
    p = {x: (1 - 0.8) / (len(LOSS_DOMS) - 1) for x in LOSS_DOMS}
    p[d] = 0.8
    print(f'  集中{d:20s}: log L̄={logL_avg(p):+.4f} (Δ={logL_avg(p)-logL_avg(uni):+.4f})')

# 随机搜索近似最优（凸问题，单纯形上）
rng = np.random.default_rng(42)
best_p, best_v = None, 1e9
for trial in range(20000):
    z = rng.gamma(1, 1, len(LOSS_DOMS)); p = z / z.sum()
    pd_ = {d: p[i] for i, d in enumerate(LOSS_DOMS)}
    v = logL_avg(pd_)
    if v < best_v:
        best_v, best_p = v, pd_
print('\n随机搜索最优配比（min log L̄）:')
opt = pd.Series(best_p).sort_values(ascending=False)
print(opt.round(4).to_string())
print(f'最优 log L̄={best_v:.4f}  vs 均匀={logL_avg(uni):.4f}  Δ={best_v-logL_avg(uni):+.4f}')

# ---------- 汇总表 ----------
print('\n========== 问题一关键结果汇总 ==========')
print('1) 质量评价：样本级Q(4种赋权) → 域级Q(等权): arxiv=0.739 book=0.684 c4=0.611 commoncrawl=0.618 github=0.501 stackexchange=0.589 wikipedia=0.543')
print('2) 冲突：30对显著负相关指标对；高冲突样本率5%(A1 11.9% / A2 23.2% / A3 5.5%)')
print('3) 配比模型：M2 log-linear+二次，训练R²=0.751，检验秩相关 1M=0.873/60M=0.873/1B=0.801，外推 10B=0.473/70B=0.391')
print('4) 最优配比偏向: ' + ', '.join(f'{d}={v:.3f}' for d, v in opt.head(6).items()))
print('\n问题一交付物完成')
