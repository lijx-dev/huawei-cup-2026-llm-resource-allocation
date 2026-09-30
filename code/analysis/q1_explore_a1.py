# -*- coding: utf-8 -*-
"""问题一：A1质量信号数据探查"""
import lzma, json, statistics
from collections import Counter

path = r'd:\F题\F题\real_attachments\A_data_value\slimpajama_quality_signal_sample.jsonl.xz'

records = []
with lzma.open(path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3000:
            break
        records.append(json.loads(line))
print('抽样探查记录数:', len(records))

# 字段类型判断
keys = list(records[0].keys())
scalar_keys, list_keys = [], []
for k in keys:
    v = records[0][k]
    if isinstance(v, list):
        list_keys.append(k)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        scalar_keys.append(k)
print('标量字段(%d):' % len(scalar_keys), scalar_keys)
print('列表字段(%d):' % len(list_keys), list_keys)

# 每个标量指标：范围、均值、缺失
print('\n===== 标量指标分布 (3000条样本) =====')
for k in scalar_keys:
    vals = [r[k] for r in records if r.get(k) is not None]
    miss = len(records) - len(vals)
    if vals:
        lo, hi = min(vals), max(vals)
        mean = sum(vals)/len(vals)
        print(f'{k:45s} n={len(vals):5d} miss={miss:4d} min={lo:.4f} max={hi:.4f} mean={mean:.4f}')

# 列表指标维度
print('\n===== 列表指标维度结构 =====')
for k in list_keys:
    lens = Counter(len(r[k]) for r in records if r.get(k) is not None)
    print(f'{k:45s} 维度分布: {dict(lens)}')
    # 展示一个样例
    for r in records:
        if r.get(k) is not None:
            print(f'    样例: {[round(x,3) for x in r[k]][:8]}')
            break

# 域分布
print('\n===== _source_domain 分布 =====')
cnt = Counter(r['_source_domain'] for r in records)
print(dict(cnt))

# content 字段概况
lens = [len(r.get('content') or '') for r in records]
print('\ncontent长度: min=%d max=%d mean=%.1f' % (min(lens), max(lens), sum(lens)/len(lens)))
