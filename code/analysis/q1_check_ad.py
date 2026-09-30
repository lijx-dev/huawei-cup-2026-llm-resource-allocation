# -*- coding: utf-8 -*-
"""验证ad_en/fluency_en二分类logits的类别方向 + 全量A1域分布"""
import lzma, json
from collections import defaultdict

path = r'd:\F题\F题\real_attachments\A_data_value\slimpajama_quality_signal_sample.jsonl.xz'

ad = defaultdict(list)
fl = defaultdict(list)
dom_cnt = defaultdict(int)
n = 0
with lzma.open(path, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        d = r['_source_domain']
        dom_cnt[d] += 1
        ad[d].append(r['ad_en'])
        fl[d].append(r['fluency_en'])
        n += 1
        if n >= 20000:
            break

print('记录数(前2万):', n)
print('域分布:', dict(dom_cnt))
print('\n=== ad_en 各域两维均值 (label0, label1) ===')
for d in sorted(ad):
    v0 = sum(x[0] for x in ad[d])/len(ad[d])
    v1 = sum(x[1] for x in ad[d])/len(ad[d])
    print(f'{d:14s} n={len(ad[d]):6d}  label0={v0:8.3f}  label1={v1:8.3f}')
print('\n=== fluency_en 各域两维均值 ===')
for d in sorted(fl):
    v0 = sum(x[0] for x in fl[d])/len(fl[d])
    v1 = sum(x[1] for x in fl[d])/len(fl[d])
    print(f'{d:14s} n={len(fl[d]):6d}  label0={v0:8.3f}  label1={v1:8.3f}')
