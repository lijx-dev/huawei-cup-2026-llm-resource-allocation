# -*- coding: utf-8 -*-
"""
问题一 · 子任务3 补充：
(1) 质量Q引入消融实验（6个可映射域，A16映射 + domain_Q）
(2) 领域组合交互效应（两两配比的协同/拮抗）
(3) 外推稳健性诊断（10B/70B秩相关为何下降：配比效应在大尺度是否减弱）
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

# 域级Q（等权方案，来自domain_Q.csv；A1抽样+扩展集混合加权）
Q_map = {'arxiv': 0.739, 'github': 0.501, 'stackexchange': 0.589,
         'wikipedia_en': 0.543, 'gutenberg_pg_19': 0.684, 'pile_cc': 0.618}

def normalize(df):
    return df.div(df.sum(axis=1), axis=0)

def build_X(mix, with_q=False):
    m = normalize(mix)
    X = m[LOSS_DOMS].copy()
    X['other'] = m[NO_LOSS].sum(axis=1)
    if with_q:
        mapped = list(Q_map)
        denom = m[mapped].sum(axis=1)
        # 质量加权配比均值 Z = Σ Q_k p_k / Σ p_k（无可映射域时取全域Q均值）
        Z = (m[mapped].values * np.array([Q_map[k] for k in mapped])).sum(axis=1) / denom.replace(0, np.nan)
        X['q_weighted'] = Z.fillna(np.mean(list(Q_map.values())))
    X = X.drop(columns=[REF])
    return X

FEATS = list(build_X(train_mix).columns)
FEATS_Q = list(build_X(train_mix, with_q=True).columns)

def ols_fit(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    pred = Xc @ beta
    r2 = 1 - np.sum((y.values - pred) ** 2) / np.sum((y.values - y.values.mean()) ** 2)
    return beta, r2

# ---------- (1) 质量Q消融 ----------
print('========== 质量Q消融：基线 vs +质量加权特征 ==========')
for with_q in [False, True]:
    Xtr = build_X(train_mix, with_q=with_q)
    Xte = {s: build_X(test_mix[s], with_q=with_q) for s in test_mix}
    r2s, maes, rhos = [], [], []
    for j in LOSS_COLS:
        y = np.log(train_loss[j])
        beta, r2 = ols_fit(Xtr, y)
        r2s.append(r2)
        Xc_te = np.column_stack([np.ones(len(Xte['1m'])), Xte['1m'].values])
        pred = np.exp(Xc_te @ beta)
        maes.append(np.mean(np.abs(pred - test_loss['1m'][j].values)))
        rhos.append(spearmanr(pred, test_loss['1m'][j].values).statistic)
    tag = '基线(无Q)' if not with_q else '含Q(质量加权)'
    print(f'{tag}: 训练R²均值={np.mean(r2s):.4f}  检验1M MAE={np.mean(maes):.4f}  秩相关={np.mean(rhos):.4f}')

# 质量与域难度的关系：可映射域的基准log-loss与Q的相关
print('\n========== 域级Q与该域loss基准的关系 ==========')
Xtr0 = build_X(train_mix)
alphas = {}
for j in LOSS_COLS:
    beta, _ = ols_fit(Xtr0, np.log(train_loss[j]))
    alphas[j] = beta[0]
alpha_ser = pd.Series(alphas)
mapped_alpha = alpha_ser[[k for k in Q_map if k in alpha_ser.index]]
q_ser = pd.Series(Q_map)[mapped_alpha.index]
r = np.corrcoef(mapped_alpha.values, q_ser.values)[0, 1]
print('可映射6域: log-loss截距α vs 域级Q, Pearson r=%.3f' % r)
print(pd.DataFrame({'alpha_logL': mapped_alpha.round(3), 'Q': q_ser.round(3)}).to_string())

# ---------- (2) 组合交互效应 ----------
print('\n========== 两两组合交互：对13域平均log-loss的影响 ==========')
base_p = pd.Series(1.0 / len(LOSS_DOMS), index=LOSS_DOMS)
beta_models = {}
for j in LOSS_COLS:
    beta, _ = ols_fit(Xtr0, np.log(train_loss[j]))
    beta_models[j] = beta

def dlogL(p_new, p_old):
    d = 0.0
    for j in LOSS_COLS:
        x1 = p_new[LOSS_DOMS].copy(); x1['other'] = 0.0; x1 = x1.drop(REF)
        x0 = p_old[LOSS_DOMS].copy(); x0['other'] = 0.0; x0 = x0.drop(REF)
        d += (x1.values @ beta_models[j][1:] - x0.values @ beta_models[j][1:]) / len(LOSS_COLS)
    return d

sens1 = {}
for dom in LOSS_DOMS:
    new_p = base_p.copy(); new_p[dom] += 0.2; new_p = new_p / new_p.sum()
    sens1[dom] = dlogL(new_p, base_p)

# 检验线性可加性：A+B同时+0.1（各）的效果 vs 单独+0.2的效果之和
print('单域+0.2 与 双域各+0.1 的偏离（非线性/交互证据，非0则存在交互）')
viol = []
for a, b in [('dm_mathematics','hackernews'), ('github','freelaw'), ('arxiv','pile_cc'),
             ('dm_mathematics','github'), ('stackexchange','wikipedia_en')]:
    p_ab = base_p.copy(); p_ab[a] += 0.1; p_ab[b] += 0.1; p_ab = p_ab / p_ab.sum()
    eff_ab = dlogL(p_ab, base_p)
    half = (sens1[a] + sens1[b]) / 2   # 线性外推期望（0.1+0.1 vs 0.2）
    print(f'  {a}+{b}: 实际双域效果={eff_ab:+.4f}  线性外推={half:+.4f}  偏离={eff_ab-half:+.4f}')
    viol.append(eff_ab - half)
print('偏离幅度 vs 单域效果量级：max|偏离|=%.4f' % max(abs(v) for v in viol))

# ---------- (3) 外推稳健性诊断 ----------
print('\n========== 外推稳健性诊断 ==========')
for s in ['10b', '70b']:
    comm = sorted(set(est_loss[s].index) & set(train_loss.index))
    # 域固定效应解释率：只用域均值预测 vs log-linear模型
    yhat_dom = np.log(train_loss.loc[comm]).mean().values  # 域均值log-loss（对配比不敏感）
    y_true = np.log(est_loss[s].loc[comm]).values
    r2_dom = 1 - np.sum((y_true - yhat_dom[None, :]) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    # 配比解释率：配比方差占loss方差比例（训练集上）
    Xc_tr = np.column_stack([np.ones(len(Xtr0)), Xtr0.values])
    res_ratios = []
    for j in LOSS_COLS:
        ytr = np.log(train_loss[j])
        sst = np.sum((ytr - ytr.mean()) ** 2)
        pred_tr = Xc_tr @ beta_models[j]
        res_ratios.append(np.sum((ytr - pred_tr) ** 2) / sst)
    print(f'外推[{s}]: 域均值基准可解释R²={r2_dom:.3f}（配比不敏感模型的表现）')
    print(f'  训练集log-linear解释率=1-残差比: 各域={np.round(1-np.array(res_ratios),3)}')
    # 外推集各配比间log-loss的离散度
    spread = y_true.std(axis=0).mean()
    print(f'  外推集域内配比间log-loss std(均值)={spread:.4f}（配比效应强度代理）')

# 尺度越大配比效应越弱的直接证据：同一批配比(common)在各尺度下的域内log-loss std
print('\n========== 同批配比跨尺度：配比效应强度随尺度的变化 ==========')
comm_base = sorted(set(test_loss['1m'].index) & set(test_loss['60m'].index) & set(test_loss['1B'].index))
for tag, ldf in [('1M', test_loss['1m']), ('60M', test_loss['60m']), ('1B', test_loss['1B']),
                 ('10B(est)', est_loss['10b']), ('70B(est)', est_loss['70b'])]:
    comm = [i for i in comm_base if i in ldf.index]
    if not comm:
        continue
    y = np.log(ldf.loc[comm].values)
    within = y.std(axis=0).mean()   # 同一配比下13域log-loss的离散度（域间）
    across = y.std(axis=1).mean()   # 同一域下不同配比log-loss的离散度（配比间）
    print(f'{tag:10s}: 域间log-loss std={within:.4f}  配比间log-loss std={across:.4f}  配比效应占比={across/(within+across)*100:.1f}%')
print('\n补充分析完成')
