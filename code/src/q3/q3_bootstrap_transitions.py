# -*- coding: utf-8 -*-
"""
问题三·融合方案：结构性转移阈值的 B6 Bootstrap 传播
（对齐团队颗粒度：团队已做 500 组区块 Bootstrap，本脚本给出我方口径下的同量级结果）

做法
----
1. 用 B6（supplementary_NQ_experiment.csv）拟合 8 参数广义标度律，
   与问题二冻结 SET_A 对账；B7 保持留出，不参与此处重拟合；
2. 对 (N,D) 分组做 400 次 Bootstrap 重抽（seed=20260924，与问题二一致），每次重新拟合；
3. 对每个 Bootstrap 参数集，在**盒约束**下重识别质量通道的
       C_act  = 质量通道激活阈值（纯规模分支上 MB(Q0) > MC(Q0) 的最小预算）
       C_sat  = 质量触顶 + 规模双触顶同时发生的预算（闭式）
   并对 15 个 (g, L_ctx) 组合、3 档预算处的盒约束最优损失 L* 重算；
4. 报 2.5% / 50% / 97.5% 分位数。

口径说明
--------
- C_act 用「纯规模盒最优分支上的 MB-MC 变号」定义，与 q3_model_core.act_threshold 同口径，
  只是把无约束纯规模解换成**盒约束**纯规模解；C_act < C_min 时记为「最小可行预算即激活」。
- η 不可由 C7 识别，本脚本固定 η=2e-4；η 的影响由 q3_fused_grid.py 的压力情景覆盖。

运行： python q3_bootstrap_transitions.py
输出： q3_bootstrap_transitions.json
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import q3_model_core as q3
from q3_paths import A_DIR, B_DIR, INPUT_DIR, OUTPUT_DIR
from q3_fused_grid import (solve_box, c_min, c_sat, c_act,
                           pure_box_analytic, mc_envelope,
                           NMIN, NMAX, DMIN, DMAX)

BASE = B_DIR
Q0 = 0.50
ETA = 2e-4
GS = ['exp', 'pow', 'log']
LCTX = [2048, 4096, 8192, 32768, 131072]
NBOOT = 400
SEED = 20260924

NAMES = ['E', 'A', 'alpha', 'B', 'beta', 'rho_N', 'rho_D', 'E1']
LO = [0.2, 1e-8, 1e-3, 1e-8, 1e-3, 1e-3, 1e-3, 1e-3]
HI = [4.0, 1e8, 3.0, 1e8, 3.0, 10.0, 10.0, 10.0]


def rec(p, N, D, Q):
    E, A, a, B, b, rN, rD, E1 = p
    # E 是冻结模型中的 E_tilde；与 q3.loss 保持同一截距口径。
    return E + A * N ** (-a) * np.exp(-rN * Q) + B * D ** (-b) * np.exp(-rD * Q) - E1 * Q


def main():
    b6 = pd.read_csv(BASE / 'supplementary_NQ_experiment.csv')
    N_ = b6.N_params_B.values.astype(float)
    D_ = b6.D_tokens_B.values.astype(float)
    Q_ = b6.Q_score.values.astype(float)
    L_ = b6.val_loss.values.astype(float)
    n = len(L_)

    print('=' * 118)
    print('问题三·融合方案：转移阈值 B6 Bootstrap（B6 n=%d，B=%d，seed=%d）' % (n, NBOOT, SEED))
    print('=' * 118)

    res = least_squares(lambda p: rec(p, N_, D_, Q_) - L_, q3.PAR_DEF, bounds=(LO, HI), max_nfev=200000)
    x = res.x
    rss = float(np.sum(res.fun ** 2))
    print('\n[0] SET_A 点估计（用于自检，应与问题二冻结值一致）')
    for nm, v, ref in zip(NAMES, x, q3.PAR_DEF):
        print('    %-6s = %+.6f   (冻结值 %+.6f, 相对差 %.2e)' % (nm, v, ref, abs(v - ref) / abs(ref)))
    print('    RMSE = %.6f   AIC = %.2f' % (np.sqrt(rss / n), n * np.log(rss / n) + 2 * 8))

    # ---- 点估计阈值（自检：应与团队 32768 档 1.265e19 / 1.643e19 / 8.788e18 对齐）----
    print('\n[1] 点估计阈值自检（η=2e-4, Q0=0.50）')
    print('    %-7s %-5s %-13s %-13s %-13s %s' % ('Lctx', 'g', 'C_min', 'C_act', 'C_sat', 'C_act 性质'))
    print('    ' + '-' * 100)
    point = []
    for Lctx in LCTX:
        for gname in GS:
            ca, note = c_act(gname, Q0, Lctx, ETA, par=tuple(x))
            cm = c_min(Lctx, ETA)
            cs = c_sat(gname, Q0, Lctx, ETA)
            point.append(dict(Lctx=Lctx, g=gname, C_min=cm, C_act=ca, C_sat=cs, note=note))
            print('    %-7d %-5s %-13.5e %-13s %-13.5e %s'
                  % (Lctx, gname, cm, ('%.5e' % ca) if ca else '—', cs, note))

    # ---- Bootstrap ----
    print('\n[2] Bootstrap：重抽 (N,D) 分组（共 %d 组）' % b6.groupby(['N_params_B', 'D_tokens_B']).ngroups)
    groups = [g for _, g in b6.groupby(['N_params_B', 'D_tokens_B'])]
    ng = len(groups)
    rng = np.random.default_rng(SEED)
    boot = []
    for b in range(NBOOT):
        idx = rng.integers(0, ng, ng)
        sub = pd.concat([groups[i] for i in idx], ignore_index=True)
        try:
            rb = least_squares(lambda p: rec(p, sub.N_params_B.values.astype(float),
                                             sub.D_tokens_B.values.astype(float),
                                             sub.Q_score.values.astype(float))
                               - sub.val_loss.values.astype(float),
                               x, bounds=(LO, HI), max_nfev=20000)
            if not rb.success or not np.all(np.isfinite(rb.x)):
                raise RuntimeError('Bootstrap 拟合未收敛')
            boot.append(rb.x)
        except Exception:
            pass
        if (b + 1) % 100 == 0:
            print('    已完成 %d/%d（成功 %d）' % (b + 1, NBOOT, len(boot)))
    boot = np.array(boot)
    print('    Bootstrap 成功 %d/%d' % (len(boot), NBOOT))
    if len(boot) != NBOOT:
        raise RuntimeError('Bootstrap 成功次数不足：%d/%d' % (len(boot), NBOOT))
    print('\n    参数 95% 百分位区间：')
    for i, nm in enumerate(NAMES):
        lo_, hi_ = np.percentile(boot[:, i], [2.5, 97.5])
        print('      %-6s = %+.6f  [%+.6f, %+.6f]  %s'
              % (nm, x[i], lo_, hi_, '显著非零' if (lo_ > 0 or hi_ < 0) else '含 0'))

    # ---- 阈值分位数 ----
    print('\n[3] 转移阈值 Bootstrap 分位数（2.5% / 50% / 97.5%）')
    print('    %-7s %-5s | %-34s | %-34s' % ('Lctx', 'g', 'C_act', 'C_sat'))
    print('    ' + '-' * 100)
    th = []
    for Lctx in LCTX:
        for gname in GS:
            ca_l, cs_l, boundary = [], [], 0
            cm = c_min(Lctx, ETA)
            for p in boot:
                ca, note = c_act(gname, Q0, Lctx, ETA, par=tuple(p))
                if ca is not None:
                    ca_l.append(ca)
                    if note.startswith('最小可行'):
                        boundary += 1
                cs_l.append(c_sat(gname, Q0, Lctx, ETA))
            ca_q = np.percentile(ca_l, [2.5, 50, 97.5]) if ca_l else [np.nan] * 3
            cs_q = np.percentile(cs_l, [2.5, 50, 97.5])
            th.append(dict(Lctx=Lctx, g=gname, C_min=cm,
                           C_act_q=[float(v) for v in ca_q],
                           C_sat_q=[float(v) for v in cs_q],
                           n_act_found=len(ca_l), n_boundary_act=boundary,
                           frac_boundary_act=boundary / max(len(ca_l), 1)))
            print('    %-7d %-5s | %.4e %.4e %.4e | %.4e %.4e %.4e'
                  % (Lctx, gname, ca_q[0], ca_q[1], ca_q[2], cs_q[0], cs_q[1], cs_q[2]))

    # ---- L* 的 Bootstrap 带（盒约束主表关键格点）----
    print('\n[4] 盒约束最优损失 L* 的 Bootstrap 带（Lctx=32768, η=2e-4, Q0=0.50）')
    print('    %-5s %-8s | %-34s' % ('g', 'C', 'L*  [2.5%, 50%, 97.5%]'))
    print('    ' + '-' * 100)
    lbands = []
    for gname in GS:
        for C in [1e19, 1e22, 1e24]:
            vals = []
            for p in boot:
                r = solve_box(C, gname, Q0, 32768, eta=ETA, nd=220, nq=160, refine=1, par=tuple(p))
                if r is not None:
                    vals.append(r['L'])
            q = np.percentile(vals, [2.5, 50, 97.5]) if vals else [np.nan] * 3
            r0 = solve_box(C, gname, Q0, 32768, eta=ETA, nd=600, nq=400, refine=3)
            lbands.append(dict(g=gname, C=C, L_point=(r0['L'] if r0 else None),
                               L_q=[float(v) for v in q], n_ok=len(vals)))
            print('    %-5s %-8.0e | %.5f  %.5f  %.5f   (点估计 %s)'
                  % (gname, C, q[0], q[1], q[2], ('%.5f' % r0['L']) if r0 else '不可行'))

    out = dict(n_data=n, n_boot=len(boot), seed=SEED, eta=ETA, Q0=Q0,
               point_estimate=dict(zip(NAMES, [float(v) for v in x])),
               param_ci={nm: [float(v) for v in np.percentile(boot[:, i], [2.5, 97.5])]
                         for i, nm in enumerate(NAMES)},
               point_thresholds=point, threshold_bands=th, loss_bands=lbands)
    with open(OUTPUT_DIR / 'q3_bootstrap_transitions.json', 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    print('\n已写出 q3_bootstrap_transitions.json')


if __name__ == '__main__':
    main()
