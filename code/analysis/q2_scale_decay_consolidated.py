# -*- coding: utf-8 -*-
"""新增章节（配比效应的跨规模衰减）的统一复现脚本。
   数据：d:\\F题\\F题\\real_attachments\\A_data_value\\regmix_tables\\
     test_mixture_1m.csv / test_pile_loss_1m.csv        (256 配比, 1M 参数)
     test_mixture_60m.csv / test_pile_loss_60m.csv      (256 配比, 60M 参数，配比与 1m 完全相同)
     test_mixture_1B.csv / test_pile_loss_1B.csv        (64 配比, 1B 参数，配比与 1m/60m 不同)
     train_mixture_1m.csv / train_pile_loss_1m.csv      (512 配比, 1M 参数；M2 拟合用)
   输出：可配对性、逐目标衰减斜率与稳健幅度比、极端点定位、E 反解诊断、λ_p 上界系数。
"""
import itertools
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
MERGED = ['nih_exporter', 'philpapers', 'enron_emails', 'europarl']


def rd(stem):
    df = pd.read_csv(f'{BASE}/{stem}.csv')
    return df[[c for c in df.columns if c != 'index']].values


def to14(P):
    cols = pd.read_csv(f'{BASE}/train_mixture_1m.csv').columns
    names = [c.replace('train_the_pile_', '') for c in cols if c != 'index']
    mi = [i for i, n in enumerate(names) if n in MERGED]
    ki = [i for i, n in enumerate(names) if n not in MERGED]
    return np.column_stack([P[:, ki], P[:, mi].sum(1)])


trP, teP = to14(rd('train_mixture_1m')), to14(rd('test_mixture_1m'))
s60P, s1BP = to14(rd('test_mixture_60m')), to14(rd('test_mixture_1B'))
L1, L60, L1B = rd('test_pile_loss_1m'), rd('test_pile_loss_60m'), rd('test_pile_loss_1B')
l1, l60, l1b = L1.mean(1), L60.mean(1), L1B.mean(1)
cols = [c for c in pd.read_csv(f'{BASE}/test_pile_loss_1m.csv').columns if c != 'index']
NAMES = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in cols]

print('=' * 74)
print('【0】数据规模与可配对性')
print('=' * 74)
print(f'  train_1m   : {rd("train_mixture_1m").shape[0]:>4d} 配比 × {rd("train_mixture_1m").shape[1]:>2d} 域')
print(f'  test_1m    : {teP.shape[0]:>4d} 配比   test_60m : {s60P.shape[0]:>4d} 配比   test_1B : {s1BP.shape[0]:>3d} 配比')
print(f'  test_1m 与 test_60m 的配比是否逐行相同: {np.allclose(teP, s60P)}  (最大差 {np.abs(teP - s60P).max():.1e})')
mt = lambda A, B: sum(any(np.allclose(r, t) for t in B) for r in A)
print(f'  交集 test_1m ∩ train_1m = {mt(teP, trP)}/256    test_60m ∩ train_1m = {mt(s60P, trP)}/256')
print(f'  交集 test_1B ∩ train_1m = {mt(s1BP, trP)}/64     test_1B ∩ test_60m = {mt(s1BP, s60P)}/64')
print(f'  => 1m 与 60m 构成 256 组"同配比、跨规模"配对；1B 档配比互不相同，无法配对')
print(f'  聚合损失均值: 1m={l1.mean():.4f}  60m={l60.mean():.4f}  1B={l1b.mean():.4f}')

print('\n' + '=' * 74)
print('【1】配比效应的绝对幅度是否随规模衰减（只给量级，不判别 Form A/B）')
print('=' * 74)
idx = list(itertools.combinations(range(len(l1)), 2))
i = np.array([a for a, _ in idx]); j = np.array([b for _, b in idx])
d1, d60 = l1[i] - l1[j], l60[i] - l60[j]
s, c = np.polyfit(d1, d60, 1)
res = d60 - (s * d1 + c)
se = np.sqrt((res ** 2).sum() / (len(d1) - 2) / ((d1 - d1.mean()) ** 2).sum())
print(f'  聚合回归: dL_60m = {s:.5f} · dL_1m {c:+.6f}')
print(f'    斜率 se={se:.5f}   t(斜率=1)={(s - 1) / se:+.2f}   R²={1 - (res ** 2).sum() / ((d60 - d60.mean()) ** 2).sum():.5f}')
print(f'    Form A 预测斜率=1（配比效应与规模无关）；Form B 预测斜率<1（配比乘在随规模收缩的可约损失上）')
print(f'    => 实测 {s:.4f}，t(斜率=1)={(s - 1) / se:+.1f}')
print(f'    【不可据此判别 Form A/B】本表无 N、D 字段，1M→60M 的 N 与 D 同步变化，')
print(f'    该斜率同时含"可约损失随规模收缩"与"token 预算变化"，两者无法归因到单条通道。')
print(f'    Form A/B 的判别依据见 4.6(c) 的截距符号与反解可行性；本节只提供量级。')
print(f'  稳健性（剔除 |dL_1m| 超过阈值的极端配比对）：')
for thr in [0.5, 1.0, 1.2]:
    m = np.abs(d1) <= thr
    s2, c2 = np.polyfit(d1[m], d60[m], 1)
    print(f'    |dL_1m|≤{thr}: 保留 {m.sum():>6d}/{len(m)}  斜率={s2:.4f}  截距={c2:+.6f}')

print('\n' + '=' * 74)
print('【2】稳健幅度比 vs 极差比（为何不能只报极差）')
print('=' * 74)
print(f'  差分标准差比 sd(dL60)/sd(dL1) = {d60.std() / d1.std():.4f}')
for lo, hi in [(1, 99), (5, 95), (10, 90), (25, 75)]:
    r1 = np.percentile(d1, hi) - np.percentile(d1, lo)
    r60 = np.percentile(d60, hi) - np.percentile(d60, lo)
    print(f'  {lo:>2d}–{hi:<3d} 分位跨度比 = {r60 / r1:.4f}')
print(f'  0–100 极差比 = {(l60.max() - l60.min()) / (l1.max() - l1.min()):.4f}  <- 由单一配比主导，不可单独引用')
k1, k60 = int(np.argmax(d1)), int(np.argmax(d60))
print(f'  使 dL_1m 最大的配比对 (idx {i[k1]},{j[k1]}): dL_1m={d1[k1]:.4f} -> dL_60m={d60[k1]:.4f}')
print(f'  使 dL_60m 最大的配比对 (idx {i[k60]},{j[k60]}): dL_1m={d1[k60]:.4f} -> dL_60m={d60[k60]:.4f}')
print(f'  两个极值对是否同一个: {k1 == k60}；是否都含配比 idx 86: {86 in (i[k1], j[k1]) and 86 in (i[k60], j[k60])}')
r = np.argsort(l1)[::-1][:3]
print(f'  1m 聚合损失最高的 3 个配比: ' + ', '.join(f'idx{int(x)}(L1={l1[x]:.3f},L60={l60[x]:.3f})' for x in r))

print('\n' + '=' * 74)
print('【3】逐目标衰减斜率与稳健幅度比')
print('=' * 74)
print(f'  {"目标":<19s}{"L@1m":>8s}{"L@60m":>8s}{"斜率":>9s}{"sd比":>8s}{"5-95比":>8s}')
rows = []
for t, nm in enumerate(NAMES):
    a, b = L1[:, t], L60[:, t]
    aa, bb = a[i] - a[j], b[i] - b[j]
    sl = np.polyfit(aa, bb, 1)[0]
    sdr = bb.std() / aa.std()
    qr = (np.percentile(bb, 95) - np.percentile(bb, 5)) / (np.percentile(aa, 95) - np.percentile(aa, 5))
    rows.append((nm, a.mean(), b.mean(), sl, sdr, qr))
    print(f'  {nm:<19s}{a.mean():>8.3f}{b.mean():>8.3f}{sl:>9.4f}{sdr:>8.4f}{qr:>8.4f}')
R = np.array([[r[3], r[4], r[5]] for r in rows])
print(f'  {"均值":<19s}{l1.mean():>8.3f}{l60.mean():>8.3f}{R[:, 0].mean():>9.4f}{R[:, 1].mean():>8.4f}{R[:, 2].mean():>8.4f}')
print(f'  {"中位数":<19s}{"":>8s}{"":>8s}{np.median(R[:, 0]):>9.4f}{np.median(R[:, 1]):>8.4f}{np.median(R[:, 2]):>8.4f}')
print(f'  斜率<1 的目标: {(R[:, 0] < 1).sum()}/13    sd比<1: {(R[:, 1] < 1).sum()}/13    5-95比<1: {(R[:, 2] < 1).sum()}/13')

print('\n' + '=' * 74)
print('【4】h_p 的对数尺度不变性（与衰减结论互证）')
print('=' * 74)
pref = teP.mean(0)
ref = int(np.argmin(((teP - pref) ** 2).sum(1)))
h1, h60 = np.log(l1 / l1[ref]), np.log(l60 / l60[ref])
k, cc = np.polyfit(h1, h60, 1)
rr = h60 - (k * h1 + cc)
sek = np.sqrt((rr ** 2).sum() / (len(h1) - 2) / ((h1 - h1.mean()) ** 2).sum())
print(f'  参考配比 idx={ref}；corr(h_1m, h_60m) = {np.corrcoef(h1, h60)[0, 1]:+.4f}')
print(f'  回归 h_60m = {k:.4f}·h_1m {cc:+.5f}  (se_k={sek:.4f}, t(k=1)={(k - 1) / sek:+.2f})')
print(f'  => 相对配比效应（对数尺度）近似可迁移；绝对幅度随规模收缩（与【1】一致）')

print('\n' + '=' * 74)
print('【5】不可约损失 E 反解：一致性诊断（非结论）')
print('=' * 74)
print('  设 L(N,p)=E+R(N)·g(p) 且 R 按比例收缩，则 (L60-E)/(L1-E)=斜率，可解 E。')
print(f'  {"目标":<19s}{"L@1m":>8s}{"L@60m":>8s}{"斜率":>9s}{"解出E":>10s}{"判定":>7s}')
ok = 0
for nm, m1, m60, sl, _, _ in rows:
    if sl < 1:
        E = (m60 - sl * m1) / (1 - sl)
        tag = '合理' if 0 < E < m60 else '越界'
        ok += tag == '合理'
        print(f'  {nm:<19s}{m1:>8.3f}{m60:>8.3f}{sl:>9.4f}{E:>10.3f}{tag:>7s}')
    else:
        print(f'  {nm:<19s}{m1:>8.3f}{m60:>8.3f}{sl:>9.4f}{"不可识别":>10s}')
E_agg = (l60.mean() - s * l1.mean()) / (1 - s)
print(f'  聚合口径: 斜率={s:.4f}, 截距={c:+.6f} -> E={E_agg:.3f} (越界)')
print(f'  => 13 个目标中仅 {ok} 个给出物理可行 E；比例收缩假设不成立，E 不能由两个规模点识别。')

print('\n' + '=' * 74)
print('【6】衰减分解：绝对幅度衰减是否等于"配比效应减弱"')
print('=' * 74)
print('  对同一对配方 (i,j)：dL = l_i - l_j，dlogL = log l_i - log l_j，且 dL ≈ l_bar · dlogL。')
print('  因此 dL 的跨规模斜率 = (损失水平比) × (对数离散度比)：')
lvl = l60.mean() / l1.mean()
dl1, dl60 = np.log(l1)[i] - np.log(l1)[j], np.log(l60)[i] - np.log(l60)[j]
slog = np.polyfit(dl1, dl60, 1)[0]
print(f'    损失水平比 l60/l1                = {lvl:.5f}')
print(f'    对数差分的跨规模回归斜率           = {slog:.5f}  (≈1 表示 log 尺度无 p×N 交互)')
print(f'    逐目标 log L 离散度: 1m={np.std(np.log(L1), axis=0).mean():.5f}, '
      f'60m={np.std(np.log(L60), axis=0).mean():.5f}, '
      f'比={np.std(np.log(L60), axis=0).mean() / np.std(np.log(L1), axis=0).mean():.5f}')
print(f'    乘积 损失水平比 × 对数离散度比      = {lvl * (np.std(np.log(L60), axis=0).mean() / np.std(np.log(L1), axis=0).mean()):.5f}')
print(f'    实测 dL 斜率（见【1】）             = {s:.5f}')
print('  => 绝对幅度的衰减几乎完全由"损失水平随规模下降"解释；log 尺度的配比离散度基本不变。')
print('     因此不能用 dL 斜率去折减 λ_p：h_p 本身是对数量，跨规模可迁移。')

print('\n' + '=' * 74)
print('【7】结论与负面清单')
print('=' * 74)
print('  可报告：')
print('    (1) 1m 与 60m 有 256 组同配方配对观测，是 A 端唯一无设计混淆的跨规模对照；')
print('    (2) 配比效应绝对幅度随规模收缩，斜率 0.7622（13/13 目标同向）；')
print('    (3) 该收缩由损失水平下降解释（水平比 %.5f × 对数离散度比 %.5f ≈ %.5f）；' % (
    lvl, np.std(np.log(L60), axis=0).mean() / np.std(np.log(L1), axis=0).mean(),
    lvl * (np.std(np.log(L60), axis=0).mean() / np.std(np.log(L1), axis=0).mean())))
print('    (4) log 尺度相对响应近似尺度不变（k=1.0415，corr 0.9687），故以 h_p 作对数乘子结构合理。')
print('  不可报告（负面清单）：')
print('    (a) 不得用"极差比 0.9950"声称配比效应不衰减（单一配比 idx86 主导，见【2】）；')
print('    (b) 不得由本节判别 Form A / Form B：配比表无 N、D 字段，且 A 端切片内 N 与 D 共线，')
print('        两个规模点上的差异是 N 与 D 的联合变化，无法归因；')
print('    (c) 不得把本节当作 A→B 迁移强度 λ_p 的估计（本节是 A 端内部证据）；')
print('    (d) 不得用 E 反解值（【5】11/13 为负）作任何结论。')
