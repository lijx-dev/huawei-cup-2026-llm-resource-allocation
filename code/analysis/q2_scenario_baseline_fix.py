# -*- coding: utf-8 -*-
"""核验 §2.6.4 情景敏感性表的基线 L 口径

问题：脚本 q2_p_scenario_rebuild.py 的 loss() 用 E - E1*Q，
      而拟合器参数化是 E + E1*(1-Q) = E + E1 - E1*Q，
      即被识别的常数项是 E~ = E + E1 = 1.7473。
      若用 E=1.631681 且漏掉 +E1，基线会低 0.1157（约 4.6%）。

本脚本：1) 取 B6 中位工作点附近实测行；
        2) 用两种口径分别算模型值；
        3) 输出修正后的情景表。
"""
import os
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
b67 = pd.concat([b6, b7], ignore_index=True).drop_duplicates(
    subset=['N_params_B', 'D_tokens_B', 'Q_score'])

Nt, Dt, Qt = 1.0, 150.0, 0.5
d = np.sqrt((np.log(b67.N_params_B / Nt)) ** 2
            + (np.log(b67.D_tokens_B / Dt)) ** 2
            + ((b67.Q_score - Qt) / 0.1) ** 2)
near = b67.loc[d.sort_values().index[:6],
               ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss']]
print('B6∪B7 中距工作点 (N=1.0B, D=150B, Q=0.5) 最近的 6 行实测：')
print(near.to_string(index=False))
print(f'\n实测 L 中位 = {near.val_loss.median():.5f}')

# ---------- 拟合参数（q2_recommended_fit.py 全样本） ----------
P = dict(E=1.631681, A=0.639571, a=0.282713,
         B=1.426335, b=0.299794, rN=0.349709, rD=0.130121, E1=0.115652)
Etilde = P['E'] + P['E1']
print(f"\nE={P['E']:.6f}  E1={P['E1']:.6f}  E~=E+E1={Etilde:.4f}")


def parts(N, D, Q):
    AN = P['A'] * N ** -P['a'] * np.exp(-P['rN'] * Q)
    BD = P['B'] * D ** -P['b'] * np.exp(-P['rD'] * Q)
    return AN, BD


AN, BD = parts(Nt, Dt, Qt)
print(f'通道项：A·N^-a·exp(-rN·Q) = {AN:.5f}，B·D^-b·exp(-rD·Q) = {BD:.5f}')
print(f'加性项：-E1·Q = {-P["E1"] * Qt:.5f}')

L_wrong = P['E'] + AN + BD - P['E1'] * Qt          # 旧口径（漏 +E1）
L_right = Etilde + AN + BD - P['E1'] * Qt          # 正确口径
print(f'\n旧口径基线（漏 +E1）：{L_wrong:.5f}   ← 文档原表 2.40839')
print(f'正确基线（E~=1.7473）：{L_right:.5f}')
print(f'差额 = {L_right - L_wrong:.5f} = E1 = {P["E1"]:.6f}')

# ---------- 重建情景表 ----------
print('\n' + '=' * 78)
print(f'修正后配比情景敏感性（N={Nt}B, D={Dt}B, Q={Qt}；基线 L={L_right:.5f}）')
print('=' * 78)


def loss(phi, form, base_ref):
    if form == 'A':
        L = Etilde + AN + BD * np.exp(phi) - P['E1'] * Qt
    else:
        L = Etilde + (AN + BD - P['E1'] * Qt) * np.exp(phi)
    return L, (L / base_ref - 1) * 100


print(f"{'phi':>7}{'FormA L':>11}{'FormB L':>11}{'A-B':>10}{'A变化%':>10}{'B变化%':>10}")
for phi in [0.0, 0.05, 0.10, 0.15, 0.20]:
    la, pa_ = loss(phi, 'A', L_right)
    lb, pb_ = loss(phi, 'B', L_right)
    print(f'{phi:>7.2f}{la:>11.5f}{lb:>11.5f}{la - lb:>+10.5f}{pa_:>9.3f}%{pb_:>9.3f}%')

# ---------- 与旧口径的百分比差异（相对变化对基线敏感） ----------
print('\n相对变化百分比：旧口径 vs 新口径')
print(f"{'phi':>7}{'A_old%':>10}{'A_new%':>10}{'B_old%':>10}{'B_new%':>10}")
for phi in [0.0, 0.05, 0.10, 0.15, 0.20]:
    la_o, pa_o = loss(phi, 'A', L_wrong)
    lb_o, pb_o = loss(phi, 'B', L_wrong)
    la_n, pa_n = loss(phi, 'A', L_right)
    lb_n, pb_n = loss(phi, 'B', L_right)
    print(f'{phi:>7.2f}{pa_o:>9.3f}%{pa_n:>9.3f}%{pb_o:>9.3f}%{pb_n:>9.3f}%')

# ---------- 映射到 lambda_p 情景 ----------
HP13_P95 = 0.12439
HPV_P95 = 0.21606
print('\n映射到 lambda_p 情景（Form A，正确基线）：')
print(f"{'lambda_p':>9}{'phi(13等权)':>14}{'A变化%':>10}{'phi(逐目标)':>14}{'A变化%':>10}")
for lp in [0.5, 1.0, 1.5]:
    p1, q1 = loss(lp * HP13_P95, 'A', L_right)
    p2, q2 = loss(lp * HPV_P95, 'A', L_right)
    print(f'{lp:>9.1f}{lp * HP13_P95:>14.4f}{q1:>9.3f}%{lp * HPV_P95:>14.4f}{q2:>9.3f}%')

lo_h, hi_h = -0.06114, 0.22662
print('\nlambda_p=1 遍历实测 h_p 全范围（正确基线）：')
for form in ['A', 'B']:
    a_, pa_ = loss(lo_h, form, L_right)
    b_, pb_ = loss(hi_h, form, L_right)
    print(f'  Form {form}: [{pa_:+.3f}%, {pb_:+.3f}%]')
