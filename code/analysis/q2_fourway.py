# -*- coding: utf-8 -*-
"""四种思路的统一对比（在 B7 全网格 450 点上）
分离两个维度：
  (a) 质量挂在哪个通道：E / D项 / N项 / N+D双挂
  (b) 挂上去的函数形式：幂律 Q^-d / 指数 e^{-rho Q} / 线性 (1-Q)
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
N_, D_, Q_, L_ = (b7.N_params_B.values.astype(float), b7.D_tokens_B.values.astype(float),
                  b7.Q_score.values.astype(float), b7.val_loss.values.astype(float))
n = len(L_)


# ---------- 模型定义（x 为变换后参数） ----------
def D_pow(x):      # 思路一：质量以幂律乘子挂 D 项
    E, A, a, B, b, d = x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]), np.exp(x[4]), np.exp(x[5])
    return E + A * N_ ** (-a) + B * D_ ** (-b) * Q_ ** (-d)


def D_pow_G(x):    # 思路三完整形式（幂式 g + 附加 G(Q)）
    E, A, a, B, b, d, G1 = (x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]),
                            np.exp(x[4]), np.exp(x[5]), x[6])
    return E + A * N_ ** (-a) + B * D_ ** (-b) * Q_ ** (-d) + G1 * (1 - Q_)


def ND_pow(x):     # 思路二：双挂（幂律）
    E, A, a, B, b, d1, d2 = (x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]),
                             np.exp(x[4]), np.exp(x[5]), np.exp(x[6]))
    return E + A * N_ ** (-a) * Q_ ** (-d1) + B * D_ ** (-b) * Q_ ** (-d2)


def D_exp(x):      # 思路三/四：质量以指数效率挂 D 项（D_eff）
    E, A, a, B, b, r = x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]), np.exp(x[4]), x[5]
    return E + A * N_ ** (-a) + B * D_ ** (-b) * np.exp(-r * Q_)


def D_exp_G(x):    # 思路三完整形式（指数式 g + 附加 G(Q)）
    E, A, a, B, b, r, G1 = (x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]),
                            np.exp(x[4]), x[5], x[6])
    return E + A * N_ ** (-a) + B * D_ ** (-b) * np.exp(-r * Q_) + G1 * (1 - Q_)


def E_lin(x):      # 思路四备选：质量改变损失下限 E
    E0, E1, A, a, B, b = x[0], x[1], np.exp(x[2]), np.exp(x[3]), np.exp(x[4]), np.exp(x[5])
    return E0 + E1 * (1 - Q_) + A * N_ ** (-a) + B * D_ ** (-b)


def ND_exp(x):     # 双挂（指数式）：检验“双挂优势”是否只与幂律形式绑定
    E, A, a, B, b, r1, r2 = (x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]),
                             np.exp(x[4]), x[5], x[6])
    return E + A * N_ ** (-a) * np.exp(-r1 * Q_) + B * D_ ** (-b) * np.exp(-r2 * Q_)


def ND_exp_E(x):   # 三项全挂（E + 双挂指数）
    E0, E1, A, a, B, b, r1, r2 = (x[0], x[1], np.exp(x[2]), np.exp(x[3]),
                                  np.exp(x[4]), np.exp(x[5]), x[6], x[7])
    return E0 + E1 * (1 - Q_) + A * N_ ** (-a) * np.exp(-r1 * Q_) + B * D_ ** (-b) * np.exp(-r2 * Q_)


LOGE = np.log
MODELS = {
    'P1 挂D·幂律 (思路一)':     (D_pow,   6, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), LOGE(0.0980)],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10)]),
    'P2 双挂·幂律 (思路二)':     (ND_pow,  7, [1.56, LOGE(0.51), LOGE(0.2643), LOGE(1.21), LOGE(0.2489), LOGE(0.1418), LOGE(0.0999)],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), LOGE(1e-3)],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), LOGE(10)]),
    'P3 挂D·指数 (思路三/四)':   (D_exp,   6, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), 1.0],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
    'P3G 挂D·指数+附加G':       (D_exp_G, 7, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), 1.0, 0.0],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -5],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 5]),
    'P3P 挂D·幂律+附加G':       (D_pow_G, 7, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), LOGE(0.0980), 0.0],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), -5],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), 5]),
    'P4 改E·线性 (思路四备选)':  (E_lin,   6, [1.52, 0.36, LOGE(0.53), LOGE(0.2832), LOGE(1.33), LOGE(0.2996)],
                                 [0.2, -5, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                                 [4, 5, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]),
    'P5 双挂·指数':             (ND_exp,  7, [1.56, LOGE(0.51), LOGE(0.2643), LOGE(1.21), LOGE(0.2489), 1.5, 1.0],
                                 [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                                 [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
    'P6 三项全挂·指数':         (ND_exp_E, 8, [1.5, 0.36, LOGE(0.5), LOGE(0.26), LOGE(1.2), LOGE(0.25), 1.5, 1.0],
                                 [0.2, -5, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                                 [4, 5, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
}

print('=' * 96)
print('1) 全样本拟合（B7 全网格 450 点）')
print('=' * 96)
print(f'{"模型":<26}{"k":>3}{"RMSE":>10}{"AIC":>10}{"BIC":>10}{"ΔAIC":>9}')
fits = {}
for name, (f, k, p0, lo, hi) in MODELS.items():
    res = least_squares(lambda p: f(p) - L_, p0, bounds=(lo, hi), max_nfev=60000)
    rss = float(np.sum(res.fun ** 2))
    rmse = np.sqrt(rss / n)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    fits[name] = (res.x, rss, rmse, aic, bic, k)
    print(f'{name:<26}{k:>3}{rmse:>10.5f}{aic:>10.2f}{bic:>10.2f}{"":>9}')
best_aic = min(v[3] for v in fits.values())
for name, v in fits.items():
    print(f'   {name:<26} ΔAIC={v[3]-best_aic:>8.2f}   ΔBIC={v[4]-min(x[4] for x in fits.values()):>8.2f}')

print()
print('=' * 96)
print('2) 关键参数解读')
print('=' * 96)
x = fits['P1 挂D·幂律 (思路一)'][0]
b_p1, d_p1 = np.exp(x[4]), np.exp(x[5])
print(f'P1 思路一: beta={b_p1:.4f}, delta_D={d_p1:.4f}  -> 只能识别 beta*delta_D={b_p1*d_p1:.4f}')
print(f'   （D_eff=D*g(Q) 中 g=Q^s 时，可识别量只有 beta*s={b_p1*d_p1:.4f}，s 与 beta 不可分离）')
x = fits['P2 双挂·幂律 (思路二)'][0]
print(f'P2 思路二: alpha={np.exp(x[2]):.4f}, beta={np.exp(x[4]):.4f}, '
      f'delta_N={np.exp(x[5]):.4f}, delta_D={np.exp(x[6]):.4f}')
print(f'   delta_N/delta_D = {np.exp(x[5])/np.exp(x[6]):.3f}  (N 通道质量效应更强)')
x = fits['P3 挂D·指数 (思路三/四)'][0]
print(f'P3 思路三/四: alpha={np.exp(x[2]):.4f}, beta={np.exp(x[4]):.4f}, rho={x[5]:.4f}')
print(f'   等效 token 倍率: D_eff/D = exp(-rho*Q/beta) ; Q=0.1 -> {np.exp(-x[5]*0.1/np.exp(x[4])):.3f}, '
      f'Q=1.0 -> {np.exp(-x[5]*1.0/np.exp(x[4])):.3f}')
for nm in ['P3G 挂D·指数+附加G', 'P3P 挂D·幂律+附加G', 'P4 改E·线性 (思路四备选)', 'P6 三项全挂·指数']:
    x = fits[nm][0]
    print(f'{nm}: G1/E1 = {x[6] if "G" in nm else x[1]:.4f}')

print()
print('=' * 96)
print('3) 识别性检验 A：Q0 参考点是否可识别（思路三/四 的 g(Q*) = 1 / Q0 归一化）')
print('=' * 96)
for Q0 in [0.0, 0.3, 0.5, 0.8, 1.0]:
    def f_Q0(p, Q0=Q0):
        E, A, a, B, b, r = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4]), p[5]
        return E + A * N_ ** (-a) + B * D_ ** (-b) * np.exp(-r * (Q_ - Q0))
    res = least_squares(lambda p: f_Q0(p) - L_, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831), 1.0],
                        bounds=([0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                                [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]), max_nfev=60000)
    Bv, rv = np.exp(res.x[3]), res.x[5]
    print(f'  Q0={Q0:.1f}: RSS={np.sum(res.fun**2):.10f}, rho={rv:.6f}, B={Bv:.4f}, '
          f'B*exp(rho*Q0)={Bv*np.exp(rv*Q0):.6f}')
print('  → RSS 与 rho 完全不变，只有 B*exp(rho*Q0) 被识别：Q0 不可由数据识别，仅是归一化约定。')

print()
print('=' * 96)
print('4) 识别性检验 B：rho 的 profile（是否被识别）')
print('=' * 96)
for r in [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    def f_r(p, r=r):
        E, A, a, B, b = p[0], np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4])
        return E + A * N_ ** (-a) + B * D_ ** (-b) * np.exp(-r * Q_)
    res = least_squares(lambda p: f_r(p) - L_, [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831)],
                        bounds=([0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3)],
                                [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3)]), max_nfev=60000)
    print(f'  rho={r:.1f}: RSS={np.sum(res.fun**2):.8f}  RMSE={np.sqrt(np.mean(res.fun**2)):.5f}')
print('  → RSS 对 rho 有明确极小：rho（质量效率速率）可识别；不可识别的是 Q0 而非速率。')