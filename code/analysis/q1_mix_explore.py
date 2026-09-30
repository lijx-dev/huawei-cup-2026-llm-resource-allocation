# -*- coding: utf-8 -*-
"""问题一 · 子任务3 探索：A4-A15 配比与Loss数据结构诊断"""
import os
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'

def load_mix(name):
    df = pd.read_csv(os.path.join(BASE, name))
    df = df.set_index('index')
    return df

def load_loss(name):
    df = pd.read_csv(os.path.join(BASE, name))
    df = df.set_index('index')
    df.columns = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in df.columns]
    return df

train_mix = load_mix('train_mixture_1m.csv')      # 512 x 17
train_loss = load_loss('train_pile_loss_1m.csv')  # 512 x 13
test_mix_1m = load_mix('test_mixture_1m.csv')     # 256 x 17
test_loss_1m = load_loss('test_pile_loss_1m.csv')
test_mix_60m = load_mix('test_mixture_60m.csv')
test_loss_60m = load_loss('test_pile_loss_60m.csv')
test_mix_1B = load_mix('test_mixture_1B.csv')
test_loss_1B = load_loss('test_pile_loss_1B.csv')
est_mix_10b = load_mix('est_mixture_10b.csv')
est_loss_10b = load_loss('est_pile_loss_10b.csv')
est_mix_70b = load_mix('est_mixture_70b.csv')
est_loss_70b = load_loss('est_pile_loss_70b.csv')

MIX_COLS = list(train_mix.columns)
LOSS_COLS = list(train_loss.columns)

print('=== 配比行和（应为1） ===')
for name, df in [('train', train_mix), ('test_1m', test_mix_1m), ('test_60m', test_mix_60m),
                 ('test_1B', test_mix_1B), ('est_10b', est_mix_10b), ('est_70b', est_mix_70b)]:
    s = df.sum(axis=1)
    print(f'{name}: n={len(df)}, 行和min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}')

print('\n=== 17域配比非零占比（稀疏性） ===')
sp = pd.DataFrame({'train非零%': (train_mix > 1e-6).mean()*100,
                   'test非零%': (pd.concat([test_mix_1m, test_mix_60m, test_mix_1B]) > 1e-6).mean()*100,
                   'train均值': train_mix.mean(),
                   'train中位数': train_mix.median(),
                   'train最大': train_mix.max()})
print(sp.round(4).to_string())

print('\n=== Loss 统计（各尺度） ===')
for name, df in [('train_1M', train_loss), ('test_1M', test_loss_1m), ('test_60M', test_loss_60m),
                 ('test_1B', test_loss_1B), ('est_10B', est_loss_10b), ('est_70B', est_loss_70b)]:
    print(f'{name}: 全局mean={df.values.mean():.4f} 各域mean范围=[{df.mean().min():.4f},{df.mean().max():.4f}]')

# 各尺度间同一配比的loss对比：检验集1m/60m/1B共享配比index吗？
print('\n=== 检验集配比index重合 ===')
print('test_1m ∩ test_60m index:', len(set(test_mix_1m.index) & set(test_mix_60m.index)))
print('test_1m ∩ test_1B index:', len(set(test_mix_1m.index) & set(test_mix_1B.index)))

# 外推集：10b/70b配比相同吗
print('\n=== 外推集配比index重合 ===')
print('est_10b ∩ est_70b index:', len(set(est_mix_10b.index) & set(est_mix_70b.index)))
print('est_10b与train配比重合:', len(set(est_mix_10b.index) & set(train_mix.index)))

# 训练集loss的域间相关性
print('\n=== 训练集13域loss相关性 ===')
print(train_loss.corr().round(2).to_string())

# 配对loss差异：同index的 test_1m vs test_60m vs test_1B（如果index重叠）
common = set(test_loss_1m.index) & set(test_loss_60m.index)
if len(common) > 10:
    sub1m, sub60 = test_loss_1m.loc[list(common)], test_loss_60m.loc[list(common)]
    diff = sub60 - sub1m
    print('\n=== 60M vs 1M 同配比loss差（13域） ===')
    print(diff.mean().round(4).to_string())
    print('差的范围:', diff.values.min().round(4), diff.values.max().round(4))

# 单域主导配比的loss：找配比>0.8的配方，看其各域loss
dom = train_mix[train_mix.max(axis=1) > 0.8]
print(f'\n=== 单域主导(>0.8)配方 {len(dom)} 个，主导域分布 ===')
print(dom.idxmax(axis=1).value_counts().to_string())
