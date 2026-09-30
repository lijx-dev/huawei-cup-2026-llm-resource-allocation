# -*- coding: utf-8 -*-
"""A18 regmix_domain_sample 探索：逐域文本统计"""
import lzma, json, numpy as np

domain_stats = {}
with lzma.open(r'D:\F题\F题\real_attachments\A_data_value\regmix_domain_sample.jsonl.xz', 'rt') as f:
    for line in f:
        obj = json.loads(line)
        d = obj['_source_domain']
        t = obj.get('text', '')
        if d not in domain_stats:
            domain_stats[d] = {'n': 0, 'lens': [], 'empty': 0}
        domain_stats[d]['n'] += 1
        domain_stats[d]['lens'].append(len(t))
        if len(t.strip()) == 0:
            domain_stats[d]['empty'] += 1

print(f"{'域':25s} {'样本量':>8s} {'空文本':>6s} {'平均长度':>10s} {'中位长度':>10s} {'最大长度':>10s}")
for d in sorted(domain_stats, key=lambda x: -domain_stats[x]['n']):
    s = domain_stats[d]
    lens = np.array(s['lens'])
    print(f"{d:25s} {s['n']:8d} {s['empty']:6d} {lens.mean():10.0f} {np.median(lens):10.0f} {lens.max():10d}")

total = sum(s['n'] for s in domain_stats.values())
print(f"\n总计: {total} 条, 17 域")
print(f"文本总量: {sum(sum(s['lens']) for s in domain_stats.values())/1e6:.1f}M 字符")

# 对比A17域摘要
import os
import pandas as pd
a17 = pd.read_csv(r'D:\F题\F题\real_attachments\A_data_value\regmix_domain_summary.csv')
print("\nA17 regmix_domain_summary.csv 列:", list(a17.columns))
print(a17.head(20).to_string())
