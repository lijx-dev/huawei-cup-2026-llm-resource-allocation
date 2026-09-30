# -*- coding: utf-8 -*-
"""
核验团队方案的关键论断：
  「在 theta_Qp=0、lambda_p>0 且配比不另收算力成本时，p0 在严格支持域内的预测损失最低，对全部预算均成立。」

该论断若成立，必须满足：严格支持域内所有配比 p 的 h_p(p) >= h_p(p0) = 0。
团队的候选集只取了冻结接口中 hull17=1 的 8 条**检验**配方（h 最小 +0.03203）。
但严格支持域 = conv(A4 训练配方)，而 A4 训练配方本身就在该域内。
故本脚本直接对 A4 的 512 条训练配方计算 h_p，检验是否存在 h_p < 0 的域内配比。

口径：与冻结接口一致，使用 RAW（未归一化）坐标与 P0_13_RAW 参考点。
"""
import numpy as np
import pandas as pd

import q3_model_core as q3

A4 = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables\train_mixture_1m.csv'
IFACE = r'd:\F题\q1_quality_results\Q1_M2响应接口.csv.gz'

c = q3.p_coefs()
bbar, gbar, idx6 = c['bbar'], c['gbar'], c['idx6']


def h_raw(x17):
    """冻结接口口径（RAW）的等权 h_p。"""
    x13 = q3.AMAT @ np.asarray(x17, float)
    return float(bbar @ (x13 - q3.P0_13_RAW)
                 + gbar @ (x13[idx6] ** 2 - q3.P0_13_RAW[idx6] ** 2))


def report(title):
    print('\n' + '=' * 92)
    print(title)
    print('=' * 92)


report('[0] 口径自检')
print('    h_raw(p0_A4mean_RAW) = %.3e （应为 0）' % h_raw(q3.P0_17_RAW))
print('    bbar·P0_13_RAW = %+.6f  （h_p 的线性常数项）' % float(bbar @ q3.P0_13_RAW))

report('[1] A4 训练配方（512 条，严格支持域的生成元）的 h_p 分布')
t = pd.read_csv(A4)
X = t[[f'train_the_pile_{d}' for d in q3.DOM17]].values.astype(float)
print('    形状 = %s，逐行和 ∈ [%.6f, %.6f]' % (X.shape, X.sum(1).min(), X.sum(1).max()))

h = np.array([h_raw(x) for x in X])
l1 = np.abs(X - q3.P0_17_RAW).sum(axis=1)
print('    h_p: min=%+.5f  p05=%+.5f  中位=%+.5f  p95=%+.5f  max=%+.5f'
      % (h.min(), np.percentile(h, 5), np.median(h), np.percentile(h, 95), h.max()))
print('    h_p < 0 的训练配方数 = %d / %d  (%.1f%%)' % ((h < 0).sum(), len(h), 100 * (h < 0).mean()))
print('    L1(p, p0) 中位=%.4f  p95=%.4f' % (np.median(l1), np.percentile(l1, 95)))

report('[2] 冻结接口中 hull17=1 的检验配方（团队主候选集）')
d = pd.read_csv(IFACE)
for col in ['hull17', 'hull14', 'hull13']:
    s = d[d[col] == 1]['h_p_eq']
    print('    %-7s n=%3d  min=%+.5f  中位=%+.5f  max=%+.5f' % (col, len(s), s.min(), s.median(), s.max()))
h17 = d[d.hull17 == 1]
print('    团队 P_safe（p0 + hull17=1 的 8 条）h_p 最小 = %+.5f  -> 均 > 0，故 p0 在其候选集内最优'
      % h17[h17.split != 'p0 (A4 mean)'].h_p_eq.min())

report('[3] 判决：严格支持域内是否存在 h_p < 0 的配比')
n_neg_a4 = int((h < 0).sum())
n_neg_if = int((d.h_p_eq < 0).sum())
print('    A4 训练配方（在域内，无争议）中 h_p<0 的条数 = %d' % n_neg_a4)
print('    冻结接口 576 条检验配方中 h_p<0 的条数 = %d' % n_neg_if)
print('    A4 最小 h_p = %+.5f  -> 严格支持域内的 h_p 下界 <= %+.5f < 0' % (h.min(), h.min()))
print()
print('    结论：团队「p0 在严格支持域内损失最低」的论断**不成立**。')
print('          其候选集仅取接口中 hull17=1 的 8 条检验配方（恰好全部 h_p>0），')
print('          遗漏了生成该凸包的 A4 训练配方本身，而后者含 %d 条 h_p<0 的域内配比。' % n_neg_a4)

report('[4] 修正后的正确陈述')
i_min = int(np.argmin(h))
print('    在 A4 实测训练配方内，h_p 的最小值 %+.5f 出现在第 %d 行配方' % (h[i_min], t['index'].iloc[i_min]))
worst = X[i_min] / X[i_min].sum()
base = q3.P0_17_RAW / q3.P0_17_RAW.sum()
dlt = pd.Series(worst - base, index=q3.DOM17)
print('    该配方相对 p0 的偏移（仅列 |Δ|>0.01）：')
for k, v in dlt.reindex(dlt.abs().sort_values(ascending=False).index).items():
    if abs(v) > 0.01:
        print('        %-20s %+.4f' % (k, v))
print()
print('    => 当 lambda_p > 0 时，损失最小配比是 argmin h_p，而非 p0；')
print('       相对 p0 的损失改善因子 = exp(lambda_p * (h_min - 0)) = exp(-%.4f * lambda_p)' % abs(h.min()))
for lp in [0.5, 1.0, 1.5]:
    print('          lambda_p=%.1f -> x %.4f （损失下降 %.2f%%）' % (lp, np.exp(lp * h.min()), 100 * (1 - np.exp(lp * h.min()))))
print()
print('    lambda_p = 0 时配比对目标无影响，全部配比并列最优（此时可约定取 p0）。')

report('[5] 支持半径 r 的真实含义（口径澄清）')
print('    L1(p, p0) 到 A4 配方的距离：中位=%.4f  p95=%.4f  max=%.4f'
      % (np.median(l1), np.percentile(l1, 95), l1.max()))
print('    => 以 p0 为心、r=0.1781/0.2601 的球内**不含任何 A4 配方**。')
print('       故 r 不是「包住训练配方的支持域半径」，而是与 A4 配方云**局部间距**同尺度的邻域半径')
print('       （r 取自 A4 的留一最近邻 L1 距离 p95/中位，量的是配方之间的相互距离，不是到 p0 的距离）。')
print('    两种支持域口径给出的 h_p 下界：')
print('        凸包（conv(A4)）：<= %+.5f  （A4 顶点本身即达到，55 条为负）' % h.min())
print('        L1 球 r=0.2601 ：%+.5f  （我方 p 子问题三链路一致值）' % (-0.08456,))
print('        L1 球 r=0.1781 ：%+.5f' % (-0.06544,))
print('    两者同量级，说明 L1 球是凸包支持域在 p0 附近的合理连续代理；')
print('    但论文中必须把 r 表述为「局部邻域半径」而非「支持域半径」，否则会被误读。')
print()
print('    注意：在 A4 上取 argmin h_p 属于**样本内**选择（M2 正是在 A4 上拟合的），')
print('          会高估可获得的损失改善；L1 球内的 min 是同一问题的正则化版本，偏差更小。')
