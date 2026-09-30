# -*- coding: utf-8 -*-
"""
问题一 · 子任务1：数据质量评价（修正版）
- 22维指标预处理：列表压缩(NaN鲁棒)、方向统一、稳健归一化
- 主方案：三族等权(族内等权)；对比：全局熵权 / 全局CRITIC
- 样本级Q 与 域级Q（A1抽样集 vs A2/A3扩展集对照）
输出：q1_quality_results/
"""
import lzma, json, os
import numpy as np
import pandas as pd

OUT = r'd:\F题\q1_quality_results'
os.makedirs(OUT, exist_ok=True)
BASE = r'd:\F题\F题\real_attachments\A_data_value'

# ---------- 1. 加载 ----------
def load_jsonl_xz(path, drop_content=True):
    rows = []
    with lzma.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if drop_content:
                r.pop('content', None)
            rows.append(r)
    return rows

print('加载数据 ...')
a1 = load_jsonl_xz(os.path.join(BASE, 'slimpajama_quality_signal_sample.jsonl.xz'))
a2 = load_jsonl_xz(os.path.join(BASE, 'slimpajama_quality_extended', 'arxiv_part-6777d8857c6e-000486.jsonl.xz'))
a3 = load_jsonl_xz(os.path.join(BASE, 'slimpajama_quality_extended', 'github_part-6777d8857c6e-000275.jsonl.xz'))
print(f'A1={len(a1)}  A2={len(a2)}  A3={len(a3)}')
df1 = pd.DataFrame(a1); df2 = pd.DataFrame(a2); df3 = pd.DataFrame(a3)
df2['_source_domain'] = 'arxiv'; df3['_source_domain'] = 'github'
del a1, a2, a3

SCALAR_KEYS = ['dsir_books','rps_lines_ending_with_terminal_punctution_mark','rps_doc_num_sentences',
               'rps_doc_word_count','rps_doc_frac_no_alph_words','rps_doc_frac_chars_top_2gram',
               'rps_lines_uppercase_letter_fraction','rps_doc_frac_unique_words','rps_lines_numerical_chars_fraction',
               'dsir_math','rps_doc_mean_word_length','dsir_wiki','rps_doc_frac_chars_top_3gram','rps_doc_unigram_entropy']

def softmax_expect(logits):
    logits = np.asarray(logits, dtype=float)
    logits = logits[np.isfinite(logits)]
    if logits.size == 0:
        return np.nan
    z = logits - logits.max()
    p = np.exp(z); p = p / p.sum()
    return float(np.dot(np.arange(p.size), p))

def compress_lists(df):
    out = pd.DataFrame(index=df.index)
    out['fineweb_edu'] = df['fineweb_edu'].apply(lambda x: x[0] if isinstance(x, list) and len(x) else x)
    out['ad_nonad'] = df['ad_en'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x)
    out['fluency_en'] = df['fluency_en'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else x)
    out['qurater_mean'] = df['qurater'].apply(lambda x: np.nanmean(x) if isinstance(x, list) and len(x) else x)
    for k in ['modernbert_cleanliness','modernbert_readability','modernbert_reasoning','modernbert_professionalism']:
        out[k + '_level'] = df[k].apply(lambda x: softmax_expect(x) if isinstance(x, list) else x)
    return out

comp1, comp2, comp3 = compress_lists(df1), compress_lists(df2), compress_lists(df3)
FEATURES = SCALAR_KEYS + list(comp1.columns)

full1 = pd.concat([df1[['id'] + SCALAR_KEYS + ['_source_domain']].reset_index(drop=True), comp1.reset_index(drop=True)], axis=1)
full2 = pd.concat([df2[['id'] + SCALAR_KEYS + ['_source_domain']].reset_index(drop=True), comp2.reset_index(drop=True)], axis=1)
full3 = pd.concat([df3[['id'] + SCALAR_KEYS + ['_source_domain']].reset_index(drop=True), comp3.reset_index(drop=True)], axis=1)
print('特征数:', len(FEATURES), ' 总样本:', len(full1)+len(full2)+len(full3))

# ---------- 2. DSIR长度混淆修正：原始值为逐文档累计log比值，按词数归一为每token重要性 ----------
for df in [full1, full2, full3]:
    wc = df['rps_doc_word_count'].replace(0, np.nan)
    for f in ['dsir_books', 'dsir_wiki', 'dsir_math']:
        df[f] = df[f] / wc
    df['rps_doc_word_count'] = df['rps_doc_word_count'].fillna(0)
print('DSIR已按词数归一为每token重要性（消除长度混淆）')

# ---------- 3. 方向统一 ----------
NEGATIVE = ['rps_doc_frac_no_alph_words','rps_lines_numerical_chars_fraction',
            'rps_lines_uppercase_letter_fraction','rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram']
for f in NEGATIVE:
    full1[f] = -full1[f]; full2[f] = -full2[f]; full3[f] = -full3[f]

# ---------- 3. NaN处理（中位数填充，按全量拟合）----------
ALL = pd.concat([full1[FEATURES], full2[FEATURES], full3[FEATURES]], axis=0)
nan_cnt = ALL[FEATURES].isna().sum()
print('各特征NaN数(>0的列出):', nan_cnt[nan_cnt > 0].to_dict() if (nan_cnt > 0).any() else '无')
medians = ALL[FEATURES].median()
for df in [full1, full2, full3]:
    df[FEATURES] = df[FEATURES].fillna(medians)

# ---------- 4. 稳健归一化（1%/99%裁剪 + Min-Max）----------
def fit_winsor(X, lo=0.01, hi=0.99):
    return {f: (np.quantile(X[f], lo), np.quantile(X[f], hi)) for f in X.columns}
params = fit_winsor(pd.concat([full1[FEATURES], full2[FEATURES], full3[FEATURES]], axis=0))

def normalize(df):
    out = pd.DataFrame(index=df.index)
    for f in FEATURES:
        lo, hi = params[f]
        x = df[f].clip(lo, hi)
        rng = hi - lo
        out[f] = (x - lo) / rng if rng > 0 else 0.5
    return out

n1, n2, n3 = normalize(full1), normalize(full2), normalize(full3)

# ---------- 5. 三族结构 ----------
GROUP1 = [f for f in FEATURES if f.startswith('rps_')]            # 文本自然度统计 (11)
GROUP2 = [f for f in FEATURES if f.startswith('dsir_')]           # DSIR重要性 (3)
GROUP3 = [f for f in FEATURES if f not in GROUP1 + GROUP2]        # 模型评分 (8)
FAMILIES = {'G1_统计指标': GROUP1, 'G2_DSIR重要性': GROUP2, 'G3_模型评分': GROUP3}
print('族划分:', {k: len(v) for k, v in FAMILIES.items()})

w_group = {}
for f in FEATURES:
    for gname, members in FAMILIES.items():
        if f in members:
            w_group[f] = (1/3) / len(members)
w_group = pd.Series(w_group, index=FEATURES)

def entropy_weights(X):
    m, n = X.shape
    p = X.clip(lower=1e-12)
    p = p / p.sum(axis=0)
    e = - (p * np.log(p)).sum(axis=0) / np.log(m)
    d = 1 - e
    return d / d.sum()

def critic_weights(X):
    std = X.std(axis=0)
    R = X.corr()
    conflict = 1 - R.abs()
    c = std.values * conflict.sum(axis=0).values
    return pd.Series(c / c.sum(), index=X.columns)

X_all = pd.concat([n1, n2, n3], axis=0)
w_ent = entropy_weights(X_all)
w_cri = critic_weights(X_all)
w_eq = pd.Series(1.0/len(FEATURES), index=FEATURES)

w_df = pd.DataFrame({'group_equal': w_group, 'entropy': w_ent, 'critic': w_cri, 'equal': w_eq})
w_df['family'] = [next((g for g, m in FAMILIES.items() if f in m), '') for f in FEATURES]
w_df.to_csv(os.path.join(OUT, 'weights.csv'), encoding='utf-8-sig')
print('\n=== 权重对比 ===')
print(w_df.round(4).to_string())

# ---------- 6. 综合分 ----------
def composite(df, w):
    return (df[FEATURES] * w).sum(axis=1)

for df in [n1, n2, n3]:
    df['Q_group'] = composite(df, w_group)
    df['Q_entropy'] = composite(df, w_ent)
    df['Q_critic'] = composite(df, w_cri)
    df['Q_equal'] = composite(df, w_eq)

for df, full in [(n1, full1), (n2, full2), (n3, full3)]:
    df['id'] = full['id']; df['domain'] = full['_source_domain']

# 样本级Q输出
allQ = pd.concat([n1, n2, n3], ignore_index=True)[['id','domain','Q_group','Q_entropy','Q_critic','Q_equal']]
allQ.to_csv(os.path.join(OUT, 'sample_Q.csv'), index=False, encoding='utf-8-sig')
print('\n=== 四种赋权Q的相关性 ===')
print(allQ[['Q_group','Q_entropy','Q_critic','Q_equal']].corr().round(4).to_string())

# ---------- 7. 域级聚合与对照 ----------
agg_rows = []
for df, tag in [(n1,'A1抽样'), (n2,'A2-arxiv扩展'), (n3,'A3-github扩展')]:
    g = df.groupby('domain')[['Q_group','Q_entropy','Q_critic','Q_equal']].agg(['mean','std','count'])
    g.columns = ['_'.join(c) for c in g.columns]
    g['source'] = tag
    agg_rows.append(g)
agg = pd.concat(agg_rows)
agg.to_csv(os.path.join(OUT, 'domain_Q.csv'), encoding='utf-8-sig')
print('\n=== 域级质量分Q（主方案 group_equal） ===')
print(agg[['Q_group_mean','Q_group_std','Q_group_count','source']].round(4).to_string())

# A1抽样 vs 全量对照
print('\n=== A1抽样集 vs 全量(含扩展) 域级Q对照（主方案） ===')
a1g = n1.groupby('domain')['Q_group'].mean()
fullg = pd.concat([n1, n2, n3]).groupby('domain')['Q_group'].mean()
cmpdf = pd.DataFrame({'A1抽样': a1g, '全量': fullg, '绝对差异': (fullg - a1g).abs()})
print(cmpdf.round(4).to_string())
cmpdf.to_csv(os.path.join(OUT, 'A1_vs_full_domain_Q.csv'), encoding='utf-8-sig')
print('\n完成。输出:', OUT)
