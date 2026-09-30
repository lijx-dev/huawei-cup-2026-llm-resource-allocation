# -*- coding: utf-8 -*-
"""
问题一 · 子任务2：质量冲突消解
- 冲突定义：样本级（22指标评分分歧度）+ 指标级（成对负相关）
- 成因分析：三族指标间的结构性对立
- 稳健聚合：均值 vs 中位数 vs 去极值截尾均值 对比
- 扩展集检验：A1抽样集 vs A2/A3扩展集结论一致性
"""
import os
import numpy as np
import pandas as pd

OUT = r'd:\F题\q1_quality_results'
BASE = r'd:\F题\F题\real_attachments\A_data_value'

# 复用q1_quality_pipeline的预处理：直接重建归一化特征
import lzma, json

def softmax_expect(logits):
    logits = np.asarray(logits, dtype=float)
    logits = logits[np.isfinite(logits)]
    if logits.size == 0: return np.nan
    z = logits - logits.max()
    p = np.exp(z); p = p / p.sum()
    return float(np.dot(np.arange(p.size), p))

def load(path):
    rows = []
    with lzma.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            r.pop('content', None)
            rows.append(r)
    return pd.DataFrame(rows)

SCALAR_KEYS = ['dsir_books','rps_lines_ending_with_terminal_punctution_mark','rps_doc_num_sentences',
               'rps_doc_word_count','rps_doc_frac_no_alph_words','rps_doc_frac_chars_top_2gram',
               'rps_lines_uppercase_letter_fraction','rps_doc_frac_unique_words','rps_lines_numerical_chars_fraction',
               'dsir_math','rps_doc_mean_word_length','dsir_wiki','rps_doc_frac_chars_top_3gram','rps_doc_unigram_entropy']
LIST_KEYS = ['fineweb_edu','modernbert_cleanliness','qurater','ad_en','modernbert_reasoning',
             'fluency_en','modernbert_professionalism','modernbert_readability']
NEGATIVE = ['rps_doc_frac_no_alph_words','rps_lines_numerical_chars_fraction',
            'rps_lines_uppercase_letter_fraction','rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram']

def preprocess(df, domain_tag):
    df = df.copy()
    df['_source_domain'] = domain_tag
    comp = pd.DataFrame(index=df.index)
    comp['fineweb_edu'] = df['fineweb_edu'].apply(lambda x: x[0] if isinstance(x, list) and len(x) else x)
    comp['ad_nonad'] = df['ad_en'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x)
    comp['fluency_en'] = df['fluency_en'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x)
    comp['qurater_mean'] = df['qurater'].apply(lambda x: np.nanmean(x) if isinstance(x, list) and len(x) else x)
    for k in ['modernbert_cleanliness','modernbert_readability','modernbert_reasoning','modernbert_professionalism']:
        comp[k + '_level'] = df[k].apply(lambda x: softmax_expect(x) if isinstance(x, list) else x)
    full = pd.concat([df[SCALAR_KEYS].reset_index(drop=True), comp.reset_index(drop=True)], axis=1)
    full['_source_domain'] = df['_source_domain'].values
    wc = full['rps_doc_word_count'].replace(0, np.nan)
    for f in ['dsir_books','dsir_wiki','dsir_math']:
        full[f] = full[f] / wc
    for f in NEGATIVE:
        full[f] = -full[f]
    return full

print('加载并预处理 ...')
d1 = preprocess(load(os.path.join(BASE, 'slimpajama_quality_signal_sample.jsonl.xz')), 'A1')
d2 = preprocess(load(os.path.join(BASE, 'slimpajama_quality_extended', 'arxiv_part-6777d8857c6e-000486.jsonl.xz')), 'A2')
d3 = preprocess(load(os.path.join(BASE, 'slimpajama_quality_extended', 'github_part-6777d8857c6e-000275.jsonl.xz')), 'A3')

ALL = pd.concat([d1, d2, d3], ignore_index=True)
FEATURES = [c for c in ALL.columns if c != '_source_domain']
medians = ALL[FEATURES].median()
ALL[FEATURES] = ALL[FEATURES].fillna(medians)

# 稳健归一化
params = {f: (np.quantile(ALL[f], 0.01), np.quantile(ALL[f], 0.99)) for f in FEATURES}
for f in FEATURES:
    lo, hi = params[f]
    x = ALL[f].clip(lo, hi)
    ALL[f] = (x - lo) / (hi - lo) if hi > lo else 0.5

# ---------- A. 指标级冲突：成对负相关 ----------
R = ALL[FEATURES].corr()
neg_pairs = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = R.iloc[i, j]
        if r < -0.15:
            neg_pairs.append((FEATURES[i], FEATURES[j], round(r, 3)))
neg_pairs.sort(key=lambda x: x[2])
print('\n=== 显著负相关指标对 (r < -0.15, 共%d对) ===' % len(neg_pairs))
for a, b, r in neg_pairs:
    print(f'  {a:42s} vs {b:42s}  r={r:+.3f}')
R.to_csv(os.path.join(OUT, 'indicator_corr.csv'), encoding='utf-8-sig')

# 族间平均相关系数矩阵（结构性对立诊断）
FAMILIES = {
    'G1_统计指标': [f for f in FEATURES if f.startswith('rps_')],
    'G2_DSIR': [f for f in FEATURES if f.startswith('dsir_')],
    'G3_模型评分': [f for f in FEATURES if not f.startswith('rps_') and not f.startswith('dsir_')],
}
print('\n=== 族间平均|相关系数| ===')
names = list(FAMILIES)
fb = pd.DataFrame(index=names, columns=names)
for a in names:
    for b in names:
        sub = R.loc[FAMILIES[a], FAMILIES[b]]
        fb.loc[a, b] = round(sub.abs().mean().mean(), 3)
print(fb.to_string())

# ---------- B. 样本级冲突 ----------
# 冲突指数：样本在22指标上的评分离散度（mad）——评分分歧越大，冲突越强
score_mat = ALL[FEATURES].values
conflict_idx = np.nanmedian(np.abs(score_mat - score_mat.mean(axis=1, keepdims=True)), axis=1)
ALL['conflict'] = conflict_idx
thr = np.quantile(conflict_idx, 0.95)
ALL['high_conflict'] = conflict_idx >= thr
print('\n=== 样本级冲突指数 ===')
print('均值=%.4f  95%%分位=%.4f  高冲突样本占比=%.2f%%' % (conflict_idx.mean(), thr, ALL['high_conflict'].mean()*100))

# 高冲突样本的模式：哪些指标分歧最大
hc = ALL[ALL['high_conflict']]
lc = ALL[~ALL['high_conflict']]
divergence = (hc[FEATURES].mean() - lc[FEATURES].mean()).abs().sort_values(ascending=False)
print('\n=== 高冲突 vs 低冲突样本：指标均值差(绝对值) Top8（分歧最大的指标） ===')
for f, v in divergence.head(8).items():
    print(f'  {f:42s} 差={v:.4f}  高冲突均值={hc[f].mean():.3f}  低冲突均值={lc[f].mean():.3f}')

# 高冲突样本的域分布
print('\n=== 高冲突样本域分布（与总体对比） ===')
hc_dom = ALL[ALL['high_conflict']].groupby('_source_domain').size() / ALL[ALL['high_conflict']].groupby('_source_domain').size().sum()
all_dom = ALL.groupby('_source_domain').size() / len(ALL)
print(pd.DataFrame({'高冲突占比': hc_dom.round(4), '总体占比': all_dom.round(4)}).round(4).to_string())

# ---------- C. 稳健聚合对比 ----------
# 用去极值截尾均值(trimmed mean, 剔除每样本极端10%指标) 与 中位数 对比
trimmed = np.array([np.mean(np.sort(row)[2:-2]) for row in score_mat])
median_q = np.median(score_mat, axis=1)
mean_q = score_mat.mean(axis=1)
AGG = pd.DataFrame({'mean': mean_q, 'trimmed': trimmed, 'median': median_q})
AGG['domain'] = ALL['_source_domain'].values
print('\n=== 三种聚合方式的相关性 ===')
print(AGG[['mean','trimmed','median']].corr().round(4).to_string())
print('\n=== 三种聚合下的域排名 ===')
rank_df = AGG.groupby('domain')[['mean','trimmed','median']].mean().round(4)
rank_df['mean_rank'] = rank_df['mean'].rank(ascending=False)
rank_df['trimmed_rank'] = rank_df['trimmed'].rank(ascending=False)
rank_df['median_rank'] = rank_df['median'].rank(ascending=False)
print(rank_df.sort_values('mean', ascending=False).to_string())
rank_df.to_csv(os.path.join(OUT, 'domain_Q_agg_compare.csv'), encoding='utf-8-sig')

# ---------- D. 扩展集一致性检验 ----------
print('\n=== 扩展集一致性：各数据源内高冲突率 ===')
for tag, sub in [('A1抽样集', d1), ('A2-arxiv扩展', d2), ('A3-github扩展', d3)]:
    ids = sub.index
    print(f'{tag}: 样本数={len(ids)}, 高冲突率={ALL.loc[ids, "high_conflict"].mean()*100:.2f}%')
print('\n=== 扩展集一致性：各数据源内指标负相关对数量 ===')
for tag, sub in [('A1抽样集', d1), ('A2-arxiv扩展', d2), ('A3-github扩展', d3)]:
    ids = sub.index
    Rsub = ALL.loc[ids, FEATURES].corr()
    npairs = sum(1 for i in range(len(FEATURES)) for j in range(i+1, len(FEATURES)) if Rsub.iloc[i, j] < -0.15)
    print(f'{tag}: r<-0.15 的指标对数量 = {npairs}')
print('\n冲突分析完成')
