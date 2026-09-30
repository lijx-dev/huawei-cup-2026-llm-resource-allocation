# -*- coding: utf-8 -*-
"""从论文表5.7 反演其 B6 质量模型的隐含参数，检验单通道假设的参数塌缩

论文 5.2.2 的质量模型（单通道，质量只挂 D 项）：
    M_Q:  L = E + A*N^-a + B*D^-b*exp{gamma_Q*(Q - Q_0)}，  Q_0 = 0.5
在 Q = Q_0 处：
    M_N = -dL/dN = a*A*N^(-a-1)          （只依赖 N）
    M_D = -dL/dD = b*B*D^(-b-1)          （只依赖 D）
    M_Q = -dL/dQ = gamma_Q*B*D^-b        （只依赖 D，与 N 无关）
    eps_N = dlogR/dlogN = -M_N*N/R
    dN/dQ = -(dL/dQ)/(dL/dN) = -M_Q/M_N
"""
import numpy as np

# ---- 论文表5.7 的三行（N, D, M_N, M_D, M_Q, eps_N, eps_D, dN/dQ） ----
rows = [
    (0.7,  10.0, 0.2379,  0.01421, 0.4536, -0.08229, -0.07023, -1.9067),
    (0.41, 300.0, 0.4715, 0.000338, 0.3232, -0.11308, -0.05924, -0.6855),
    (6.9,  300.0, 0.01274, 0.000338, 0.3232, -0.06602, -0.07604, -25.3648),
]
GQ = 0.317893          # 论文 5.4.2 报告的全量 B6 拟合值
L_BASE = 2.61862       # 论文表5.6，lambda_p=0，B6 中位工作点
N_MED, D_MED = 1.0, 150.0

print('=' * 84)
print('1) 检验"质量只挂 D 项"的可检验后果：M_Q 是否与 N 无关')
print('=' * 84)
print(f'  (0.41, 300)  M_Q = {rows[1][4]:.4f}')
print(f'  (6.90, 300)  M_Q = {rows[2][4]:.4f}   <- N 相差 16.8 倍，M_Q 完全相同')
print(f'  (0.70,  10)  M_Q = {rows[0][4]:.4f}')
print('  => 单通道结构下 M_Q = gamma_Q*B*D^-b 只依赖 D，此恒等是模型强制的。')

# ---- 反演 beta ----
b = np.log(rows[0][4] / rows[1][4]) / np.log(D_MED / 10.0 * 10.0 / 10.0 * (300.0 / 10.0))
b = np.log(rows[0][4] / rows[1][4]) / np.log(300.0 / 10.0)
print(f'\n  由 M_Q(10)/M_Q(300) = (300/10)^beta 反演：')
print(f'    beta = ln({rows[0][4]}/{rows[1][4]}) / ln(30) = {b:.5f}')
print(f'    对照 B1 的 beta = 0.27988  ->  塌缩 {(b / 0.27988 - 1) * 100:+.1f}%')

B_imp = rows[1][4] / GQ / 300.0 ** (-b)
print(f'    由 M_Q(300)=gamma_Q*B*300^-beta 反演：B = {B_imp:.4f}   （B1 的 B=1.24031）')

# ---- 反演 alpha ----
a = np.log(rows[1][2] / rows[0][2]) / np.log(0.7 / 0.41) - 1.0
print(f'\n  由 M_N(0.41)/M_N(0.7) = (0.41/0.7)^(-alpha-1) 反演：')
print(f'    alpha = {a:.5f}   （B1 的 alpha=0.33998；本稿双挂模型 alpha=0.28271）')
A_imp = rows[0][2] / a / 0.7 ** (-a - 1)
print(f'    由 M_N(0.7)=alpha*A*0.7^(-alpha-1) 反演：A = {A_imp:.4f}   （B1 的 A=0.35398）')

# ---- 用反演参数重构表5.7，检验自洽 ----
print('\n' + '=' * 84)
print('2) 用反演参数 (alpha, A, beta, B, gamma_Q) 重构表5.7')
print('=' * 84)


def pred(N, D):
    MN = a * A_imp * N ** (-a - 1)
    MD = b * B_imp * D ** (-b - 1)
    MQ = GQ * B_imp * D ** (-b)
    return MN, MD, MQ


print(f"{'工作点':<14}{'M_N':>10}{'M_D':>11}{'M_Q':>10}{'dN/dQ':>11}")
for N, D, MN0, MD0, MQ0, eN0, eD0, d0 in rows:
    MN, MD, MQ = pred(N, D)
    print(f'({N},{D})'.ljust(14) + f'{MN:>10.5f}{MD:>11.6f}{MQ:>10.4f}{-MQ / MN:>11.4f}')
    print(f'{"  论文值":<14}{MN0:>10.5f}{MD0:>11.6f}{MQ0:>10.4f}{d0:>11.4f}')

# ---- 基线一致性 ----
print('\n' + '=' * 84)
print('3) 同一工作点 (N=1B, D=150B, Q_B=0.5) 的基线对照')
print('=' * 84)
L_paper = L_BASE
L_ours = 2.52405
L_obs = 2.5142
print(f'  论文单通道模型      L = {L_paper:.5f}   偏差 {L_paper - L_obs:+.5f} ({(L_paper / L_obs - 1) * 100:+.2f}%)')
print(f'  本稿双通道+加性     L = {L_ours:.5f}   偏差 {L_ours - L_obs:+.5f} ({(L_ours / L_obs - 1) * 100:+.2f}%)')
print(f'  B7 实测             L = {L_obs:.5f}')
E_imp = L_paper - A_imp * N_MED ** -a - B_imp * D_MED ** (-b)
print(f'\n  由基线反推论文 B6 模型的 E ≈ {E_imp:.4f}（B1 的 E=1.68980，本稿 E~=1.74733）')
print(f'  注：E 的反演对 M_N/M_Q 的舍入较敏感，仅供参考；beta 与 alpha 的反演只用比值，稳健。')
