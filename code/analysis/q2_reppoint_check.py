# -*- coding: utf-8 -*-
"""核验 §2.7 代表点 (N=1.3, D=50, Q=0.7) 的 L=2.5343 / R=0.9026 是否可复现"""
import os
import numpy as np
import pandas as pd

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
d = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment.csv'))
e = pd.read_csv(os.path.join(B, 'supplementary_NQ_experiment_expanded.csv'))

for nm, df in [('B6', d), ('B7', e)]:
    m = np.isclose(df.val_loss, 2.5343, atol=5e-4)
    print(f'{nm} 中 val_loss≈2.5343 的行数: {m.sum()}')
    if m.sum():
        print(df[m].to_string())

P = dict(E=1.631681, A=0.639571, a=0.282713, B=1.426335, b=0.299794,
         rN=0.349709, rD=0.130121, E1=0.115652)


def L(N, D, Q):
    return (P['E'] + P['A'] * N ** -P['a'] * np.exp(-P['rN'] * Q)
            + P['B'] * D ** -P['b'] * np.exp(-P['rD'] * Q) - P['E1'] * Q)


print('\n用自由拟合参数（B6∪B7）：')
for (N, D, Q) in [(1.3, 50, 0.7), (1.0, 50, 0.7), (2.8, 50, 0.7), (1.3, 150, 0.7)]:
    v = L(N, D, Q)
    print(f'  N={N:<5}D={D:<5}Q={Q}: L={v:.4f}   R=L-Etilde={v - P["E"]:.4f}')

print('\n网格上 L 最接近 2.5343 的点：')
best = []
for N in sorted(e.N_params_B.unique()):
    for D in sorted(e.D_tokens_B.unique()):
        for Q in sorted(e.Q_score.unique()):
            v = L(N, D, Q)
            best.append((abs(v - 2.5343), N, D, Q, v))
best.sort()
for err, N, D, Q, v in best[:6]:
    print(f'  N={N:<5}D={D:<5}Q={Q}: L={v:.4f} (差 {err:.4f})')

print('\n反解：若 R=0.9026 在 (D=50,Q=0.7) 处成立，隐含的 N 为')
BD = P['B'] * 50 ** -P['b'] * np.exp(-P['rD'] * 0.7)
target = 0.9026 + P['E1'] * 0.7 - BD
Nimp = (target / (P['A'] * np.exp(-P['rN'] * 0.7))) ** (-1 / P['a'])
print(f'  BD={BD:.4f}, 目标 AN^-a e^-rNQ={target:.4f}, 隐含 N={Nimp:.4f} B')
