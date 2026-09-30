# -*- coding: utf-8 -*-
"""逐任务结果(detailed_results)结构核验 + 上下文长度字段排查
用户要求：逐任务结果别忽略；上下文长度不是自由寻优变量。
"""
import os, json, glob
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\C_efficiency_evolution'
DR = os.path.join(BASE, 'detailed_results')

print('=' * 90)
print('一、detailed_results 目录规模')
print('=' * 90)
models = sorted([d for d in os.listdir(DR) if os.path.isdir(os.path.join(DR, d))])
print(f'模型目录数 = {len(models)}')
print('示例：', models[:6])

# 取一个模型，看其内部文件
m0 = models[0]
files0 = sorted(os.listdir(os.path.join(DR, m0)))
print(f'\n[{m0}] 内文件数 = {len(files0)}')
print('  ', files0[:6])

print()
print('=' * 90)
print('二、单个 results JSON 的键结构')
print('=' * 90)
p0 = os.path.join(DR, m0, files0[0])
with open(p0, 'r', encoding='utf-8') as f:
    d0 = json.load(f)
print(f'文件: {files0[0]}')
print(f'顶层键 ({len(d0)} 个): {list(d0.keys())}')
print(f'顶层键数量较多，判断为 {{任务名: 任务结果}} 的映射')

# 打印一个任务的结构
k = list(d0.keys())[0]
print(f'\n--- 示例任务 [{k}] 的字段 ---')
v = d0[k]
if isinstance(v, dict):
    for kk, vv in v.items():
        s = str(vv)
        print(f'  {kk:<24} : {type(vv).__name__:<8} {s[:80]}')
else:
    print(f'  {type(v).__name__}: {str(v)[:200]}')

print()
print('=' * 90)
print('三、全部任务名清单（逐任务维度）')
print('=' * 90)
tasks = list(d0.keys())
print(f'任务数 = {len(tasks)}')
for t in tasks:
    print('  -', t)

print()
print('=' * 90)
print('四、逐任务指标提取：acc / acc_norm 等数值字段的覆盖')
print('=' * 90)
rows = []
for t in tasks:
    v = d0[t]
    if isinstance(v, dict):
        num = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
        rows.append({'task': t, 'numeric_fields': ','.join(num.keys()),
                     'n_num': len(num),
                     'aliases': str(v.get('alias', ''))[:40],
                     'n_samples': v.get('samples', v.get('num_samples', np.nan))})
df_t = pd.DataFrame(rows)
print(df_t.to_string(index=False))

print()
print('=' * 90)
print('五、覆盖度：模型 × 任务 的矩阵是否完整')
print('=' * 90)
task_cnt = {}
for m in models:
    fs = sorted(os.listdir(os.path.join(DR, m)))
    if not fs:
        continue
    try:
        with open(os.path.join(DR, m, fs[0]), 'r', encoding='utf-8') as f:
            dd = json.load(f)
        for t in dd.keys():
            task_cnt[t] = task_cnt.get(t, 0) + 1
    except Exception as e:
        print(f'  [跳过] {m}: {e}')
print(f'解析成功模型数 = {len(models)}')
print('逐任务被覆盖的模型数：')
for t, c in sorted(task_cnt.items(), key=lambda x: -x[1]):
    flag = '完整' if c == len(models) else f'缺 {len(models)-c}'
    print(f'  {t:<40} {c:>4}/{len(models)}  [{flag}]')

print()
print('=' * 90)
print('六、上下文长度 / 序列长度 字段排查（用户：不是自由寻优变量）')
print('=' * 90)
lb = os.path.join(BASE, 'leaderboard_extended_timeseries.csv')
if os.path.exists(lb):
    L = pd.read_csv(lb)
    print(f'leaderboard_extended_timeseries.csv: n={len(L)}')
    print('列名：', list(L.columns))
    cand = [c for c in L.columns if any(s in c.lower() for s in
            ['context', 'ctx', 'seq', 'window', 'max_len', 'length', 'token'])]
    print(f'疑似上下文/长度字段: {cand if cand else "无"}')
    # 模型名里带 32K/128K 的变体
    namecol = [c for c in L.columns if 'model' in c.lower() or 'name' in c.lower()]
    if namecol:
        nc = namecol[0]
        import re
        suff = L[nc].astype(str).str.extract(r'[-_](\d{2,3}K)$')[0]
        print(f'模型名带 NNK 后缀（上下文变体）的记录数: {suff.notna().sum()} / {len(L)}')
        if suff.notna().sum() > 0:
            print('  取值分布:', suff.value_counts().to_dict())
            sub = L[suff.notna()]
            yr = sub.groupby(suff.dropna()).size().to_dict()
            print('  这些变体按上下文后缀计数:', yr)
else:
    print('未找到 leaderboard_extended_timeseries.csv')
