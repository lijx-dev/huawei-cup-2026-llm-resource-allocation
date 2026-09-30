# -*- coding: utf-8 -*-
"""为新增章节做三件事：
   1) 判定 test_1B(64) / test_60m(256) 能否与 train_1m(512) 配对，从而拿到第三个规模点；
   2) 由"配比效应绝对幅度随规模衰减"反解隐含不可约损失 E；
   3) 输出可直接写进论文的逐目标衰减表与 λ_p 上界。
"""
import itertools
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
MERGED = ['nih_exporter', 'philpapers', 'enron_emails', 'europarl']
REF = 'uspto_backgrounds'
S6 = ['arxiv', 'freelaw', 'dm_mathematics', 'github', 'ubuntu_irc', 'hackernews']


def rd(stem):
    df = pd.read_csv(f'{BASE}/{stem}.csv')
    return df[[c for c in df.columns if c != 'index']].values


def to14(P):
    cols = pd.read_csv(f'{BASE}/train_mixture_1m.csv').columns
    names = [c.replace('train_the_pile_', '') for c in cols if c != 'index']
    mi = [i for i, n in enumerate(names) if n in MERGED]
    ki = [i for i, n in enumerate(names) if n not in MERGED]
    return np.column_stack([P[:, ki], P[:, mi].sum(1)]), [names[i] for i in ki] + ['other']


trP, n14 = to14(rd('train_mixture_1m'))
teP, _ = to14(rd('test_mixture_1m'))
s60P, _ = to14(rd('test_mixture_60m'))
s1BP, _ = to14(rd('test_mixture_1B'))
trL, teL = rd('train_pile_loss_1m'), rd('test_pile_loss_1m')
s60L, s1BL = rd('test_pile_loss_60m'), rd('test_pile_loss_1B')

print('== 1) 跨尺度可配对性 ==')


def match(A, B):
    return sum(any(np.allclose(r, t) for t in B) for r in A)


print(f'  test_1m(256)  ∩ train_1m(512) = {match(teP, trP)}')
print(f'  test_60m(256) ∩ train_1m(512) = {match(s60P, trP)}')
print(f'  test_1B(64)   ∩ train_1m(512) = {match(s1BP, trP)}')
print(f'  test_1B(64)   ∩ test_60m(256) = {match(s1BP, s60P)}')

# 若 1B 与 train 有交集，构造 train(1m)–1B 配对
idx = [k for k, r in enumerate(s1BP) if any(np.allclose(r, t) for t in trP)]
if idx:
    j = [int(np.argmin(((trP - s1BP[k]) ** 2).sum(1))) for k in idx]
    print(f'  -> 可配对 {len(idx)} 组 (train_1m vs test_1B)')
else:
    print('  -> 1B 档无法与 1m/60m 配对，只能保留两个规模点（1m, 60m）')

print('\n== 2) 逐目标衰减与隐含不可约损失 ==')
lcols = [c.replace('metric/the_pile_', '').replace('_val_loss', '')
         for c in pd.read_csv(f'{BASE}/test_pile_loss_1m.csv').columns if c != 'index']
rows = []
for t, name in enumerate(lcols):
    a, b = teL[:, t], s60L[:, t]
    i, j = np.array([x for x, _ in itertools.combinations(range(len(a)), 2)]), \
           np.array([y for _, y in itertools.combinations(range(len(a)), 2)])
    sl = np.polyfit(a[i] - a[j], b[i] - b[j], 1)[0]
    rows.append((name, a.mean(), b.mean(), sl))
rows.append(('**聚合均值**', teL.mean(), s60L.mean(), 0.76221))
print(f'  {"目标":<20s}{"L@1m":>9s}{"L@60m":>9s}{"衰减斜率":>10s}{"隐含E":>9s}')
for name, m1, m60, sl in rows:
    if sl < 1:
        E = (m60 - sl * m1) / (1 - sl)      # 由 (L60-E)/(L1-E)=sl 反解
        e_txt = f'{E:>9.3f}' if 0 < E < m60 else '     —  '
    else:
        e_txt = '     —  '
    print(f'  {name:<20s}{m1:>9.3f}{m60:>9.3f}{sl:>10.4f}{e_txt}')

print('\n== 3) λ_p 上界的推导口径 ==')
print('  记 a = 衰减斜率(60m/1m) = 0.7622，则把 1m 标定的 h_p 用到目标规模 S 时，')
print('  真实配比效应 ≤ λ_p·h_p·a^(log60(N_S/1m))；两尺度只能定出单步衰减，')
print('  故报告中 λ_p 网格给出的是【上界情景】，需显式声明。')
