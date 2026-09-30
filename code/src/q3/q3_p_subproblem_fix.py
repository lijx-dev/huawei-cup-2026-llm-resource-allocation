# -*- coding: utf-8 -*-
"""
问题三配比子问题（v3）：支持半径约束下 h_p 的精确极值 + 独立核验。

为什么要重做（v1/v2 的两处口径错误）
------------------------------------
1) **p0 不在单纯形上**。附件 A 的配方以 3 位小数存储，逐行和 ∈ [0.996, 1.003]
   （A4 均值行和 = 0.99988086）。旧版直接拿这个原始均值当 p0，于是
   F = {x>=0, Σx=1, ||x-p0||_1<=r} 自相矛盾 —— 顶点构造给出 Σx = 0.99988 ≠ 1，
   枚举出的「顶点」根本不满足题面的 Σp=1。v3 统一按行归一化（题面口径），
   RAW 口径只用于与问题一冻结接口对账。
2) **max 用 SLSQP 且未校验可行性**。旧版主脚本在 r=0.2601 的 max 分支返回了
   L1=0.7537 > r 的违约点，把损失包络上端放大到 h=+0.6593（虚高 9 倍）。
   h_p 在 17 维配方上是**凸二次型**（六个二次系数全为正），凸函数在凸多面体上的
   最大值必在顶点取得 —— 故 max 可精确枚举，不需要迭代求解器。

数学结构（决定求解路线）
------------------------
    h17(x) = bbar·(A x - p0) + Σ_{j∈I6} gbar_j ((A x)_j² - p0_j²),  A = AMAT(13×17)
    gbar 全正 => h17 凸。
      (i)  max：F 的顶点处 |Z|（压到 0 的坐标）+ |W|（L1 的 kink 坐标）= 15，
           自由坐标恰 2 个（一个「搬出余量」k、一个「搬入目的」i）=> 枚举 (Z,k,i) 覆盖全部顶点；
      (ii) min：凸规划，局部即全局 => 多初值 SLSQP 可靠，收敛初值离散度即全局性自证。

核验设计（三条互不共享代码路径的链路）
--------------------------------------
    L1  朴素循环枚举：显式构造 x、直接求值 h17(x)，不用任何广播/预计算；
    L2  严格可行 SLSQP（增广变量 t 把 L1 写成线性约束），只保留 |L1-r|<=1e-9 的点；
    L3  可行域随机采样（L1 球面投影），不假设任何顶点结构。
三条链路若都打不过枚举值，则枚举值即约束最优。
"""
import json
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

import q3_model_core as q3
from q3_paths import A_DIR, B_DIR, INPUT_DIR, OUTPUT_DIR

P0_17, P0_17_RAW, P0_13, P0_13_RAW = q3.P0_17, q3.P0_17_RAW, q3.P0_13, q3.P0_13_RAW
AMAT, DOM17 = q3.AMAT, q3.DOM17
h17 = q3.h17


# ---------------- L1：朴素循环顶点枚举（独立实现）----------------
def naive_vertex_enum(r, tol=1e-12):
    m = r / 2.0
    best, bx, nsub, nfeas = -np.inf, None, 0, 0
    for size in range(0, 18):
        for Z in combinations(range(17), size):
            if sum(P0_17[j] for j in Z) > m + tol:
                continue
            nsub += 1
            rest = m - sum(P0_17[j] for j in Z)
            for k in range(17):
                if k in Z or rest > P0_17[k] + tol:
                    continue
                for i in range(17):
                    if i in Z or i == k:
                        continue
                    x = P0_17.copy()
                    for j in Z:
                        x[j] = 0.0
                    x[k] -= rest
                    x[i] += m
                    if x.min() < -1e-12 or abs(x.sum() - 1.0) > 1e-12:
                        continue
                    if np.abs(x - P0_17).sum() > r + 1e-9:
                        continue
                    nfeas += 1
                    v = h17(x)
                    if v > best:
                        best, bx = v, x
    return best, bx, nsub, nfeas


# ---------------- L2：严格可行性 SLSQP（增广变量，无 clip/renormalize）----------------
def strict_slsqp_max(r, nstart=200, seed=3):
    """逐坐标引入 t_j>=|x_j-p0_j|，并要求 sum(t_j)<=r。"""
    n = 17

    def ineq(z):
        x, t = z[:n], z[n:]
        return np.concatenate([P0_17 - x + t, x - P0_17 + t, [r - t.sum()]])

    def jac_ineq(z):
        J = np.zeros((2 * n + 1, 2 * n))
        for j in range(n):
            J[j, j], J[j, n + j] = -1.0, 1.0
            J[n + j, j], J[n + j, n + j] = 1.0, 1.0
        J[-1, n:] = -1.0
        return J

    cons = [dict(type='eq', fun=lambda z: z[:n].sum() - 1.0,
                 jac=lambda z: np.concatenate([np.ones(n), np.zeros(n)])),
            dict(type='ineq', fun=ineq, jac=jac_ineq)]
    bnds = [(0.0, 1.0)] * n + [(0.0, r)] * n
    f = lambda z: -h17(z[:n])
    rng = np.random.default_rng(seed)
    starts = [np.concatenate([P0_17, np.zeros(n)])]
    for _ in range(nstart):
        x0 = rng.dirichlet(np.ones(n) * 0.25)
        x0 = P0_17 + (x0 - P0_17) * min(1.0, 0.9 * r / np.abs(x0 - P0_17).sum())
        starts.append(np.concatenate([x0, np.abs(x0 - P0_17)]))
    best, bx, nfeas, ninfeas = -np.inf, None, 0, 0
    for s in starts:
        try:
            rr = minimize(f, s, method='SLSQP', bounds=bnds, constraints=cons,
                          options=dict(maxiter=800, ftol=1e-12))
        except Exception:
            continue
        x = rr.x[:n]
        if x.min() < -1e-9 or abs(x.sum() - 1.0) > 1e-9:
            continue
        if np.abs(x - P0_17).sum() > r + 1e-9:
            ninfeas += 1
            continue
        nfeas += 1
        v = h17(x)
        if v > best:
            best, bx = v, x
    return best, bx, nfeas, ninfeas


# ---------------- L3：可行域随机采样（不假设顶点结构）----------------
def random_ascent(r, n=40000, seed=5):
    rng = np.random.default_rng(seed)
    best, bx = -np.inf, None
    for _ in range(n):
        d = rng.dirichlet(np.ones(17) * 0.3)
        v = d - P0_17
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            x = np.clip(P0_17 + mid * v, 0, None)
            x = x / x.sum()
            if np.abs(x - P0_17).sum() < r:
                lo = mid
            else:
                hi = mid
        x = np.clip(P0_17 + hi * v, 0, None)
        if x.sum() <= 0:
            continue
        x = x / x.sum()
        if np.abs(x - P0_17).sum() > r + 1e-9:
            continue
        val = h17(x)
        if val > best:
            best, bx = val, x
    return best, bx


def empirical_range():
    """实测范围：A4 训练配方（512）+ A6–A11 检验配方（576）。按题面 Σp=1 行归一化。"""
    import pandas as pd
    base = A_DIR
    cols = [f'train_the_pile_{d}' for d in DOM17]
    out = {}
    for name, lab in [('train_mixture_1m', 'A4 训练配方'), ('test_mixture_1m', 'A6–A7 (1M)'),
                      ('test_mixture_60m', 'A8–A9 (60M)'), ('test_mixture_1B', 'A10–A11 (1B)')]:
        d = pd.read_csv(base / (name + '.csv'))
        X = d[cols].values.astype(float)
        X = X / X.sum(axis=1, keepdims=True)          # 行归一化到单纯形
        hv = np.array([h17(x) for x in X])
        l1 = np.abs(X - P0_17).sum(axis=1)
        out[name] = dict(label=lab, n=len(X), h_min=float(hv.min()), h_max=float(hv.max()),
                         h_p05=float(np.percentile(hv, 5)), h_p95=float(np.percentile(hv, 95)),
                         l1_med=float(np.median(l1)), l1_p95=float(np.percentile(l1, 95)),
                         rowsum_min=float(d[cols].values.sum(axis=1).min()),
                         rowsum_max=float(d[cols].values.sum(axis=1).max()))
    return out


def main():
    print('=' * 106)
    print('问题三配比子问题（v3）：严格单纯形口径 + 支持半径约束下 h_p 的精确极值')
    print('=' * 106)

    print('\n[0] 结构核对与口径自检')
    c = q3.p_coefs()
    print('    二次项域 %s' % list(c['qt']))
    print('    gbar = %s  -> 全部为正，h_p 为凸二次型' % np.round(c['gbar'], 6))
    print('    Σ P0_17_RAW = %.10f（附件 A 原始 3 位小数数据）' % P0_17_RAW.sum())
    print('    Σ P0_17     = %.12f（按题面 Σp=1 归一化后）' % P0_17.sum())
    print('    h17(p0) = %.3e （应为 0）' % h17(P0_17))
    print('    AMAT@p0_17 vs P0_13 最大偏差 = %.3e （应为 0）'
          % np.max(np.abs(AMAT @ P0_17 - P0_13)))
    print('    bbar·p0 = %+.6f （h_p 的线性常数项，v1 曾漏减）' % float(c['bbar'] @ P0_13))

    out = dict(gbar=[float(v) for v in c['gbar']], quad_terms=list(c['qt']),
               p0_17=P0_17.tolist(), p0_17_raw=P0_17_RAW.tolist(),
               p0_rowsum_raw=float(P0_17_RAW.sum()),
               h_at_p0=float(h17(P0_17)),
               bbar_p0=float(c['bbar'] @ P0_13))

    # ---------- 口径差异量化 ----------
    import pandas as pd
    rt = pd.read_csv(A_DIR / 'test_mixture_1m.csv')
    cols17 = [f'train_the_pile_{d}' for d in DOM17]
    Xr = rt[cols17].values.astype(float)
    gap = np.array([abs(h17(x / x.sum()) - h17(x, p0=P0_13_RAW)) for x in Xr])
    print('\n[0b] 口径 A（RAW，与 Q1 冻结接口一致）vs 口径 B（归一化，本子问题采用）')
    print('    %d 个 1M 检验配方上 |Δh_p| ：max = %.3e，中位 = %.3e' % (len(Xr), gap.max(), np.median(gap)))
    print('    参照：h_p 支持半径 r=0.2601、实测极差约 0.31 -> 口径差异相对量级 < 1e-4')
    out['convention_gap'] = dict(max=float(gap.max()), median=float(np.median(gap)))

    # ---------- 无支持约束的退化解 ----------
    vals = np.array([h17(np.eye(17)[i]) for i in range(17)])
    j = int(np.argmax(vals))
    print('\n[0c] 无支持约束时的 max（凸性 => 单纯形顶点，纯外推）')
    print('    max h_p = %+.4f 于顶点 %s=1.0 （L1=%.4f，远离训练配方域）'
          % (vals[j], DOM17[j], np.abs(np.eye(17)[j] - P0_17).sum()))
    out['full_simplex_max'] = dict(h=float(vals[j]), domain=DOM17[j],
                                   l1=float(np.abs(np.eye(17)[j] - P0_17).sum()))

    # ---------- 实测范围 ----------
    print('\n[1] 实测范围（真实配方，行归一化后；无外推）')
    emp = empirical_range()
    print('    %-16s %6s %10s %10s %10s %10s %9s' %
          ('数据集', 'n', 'h_min', 'h_p05', 'h_p95', 'h_max', 'L1中位'))
    for k, v in emp.items():
        print('    %-16s %6d %+10.4f %+10.4f %+10.4f %+10.4f %9.4f'
              % (v['label'], v['n'], v['h_min'], v['h_p05'], v['h_p95'], v['h_max'], v['l1_med']))
    out['empirical'] = emp

    # ---------- 支持半径下的极值 + 三链路核验 ----------
    print('\n[2] 支持半径约束下的极值（顶点枚举精确 + 三链路独立核验）')
    print('    %-8s %8s %12s %12s %12s %12s %11s %11s %s'
          % ('r (L1)', 'n顶点子集', 'max(枚举)', 'max(朴素)', 'max(严格SLSQP)',
             'max(随机上升)', 'min(SLSQP)', 'min(罚函数)', '一致性'))
    rows = []
    for r in [0.1781, 0.2601]:
        vmax, vx, nsub = q3.vertex_enum_max(r)
        nmax, nx, nsub2, nfeas = naive_vertex_enum(r)
        s2, x2, nf2, ni2 = strict_slsqp_max(r, nstart=200)
        s3, x3 = random_ascent(r)
        smin, sx, nok_min, spread_min = q3.slsqp_extreme_p(+1, r, nstart=80)
        pmin, _ = q3.penalty_extreme_p(+1, r, nstart=30)
        pmin_ok = (pmin is None) or (smin <= pmin + 1e-6)
        ok = bool(abs(vmax - nmax) < 1e-9 and nf2 > 0 and np.isfinite(s2)
                  and s2 <= vmax + 1e-6 and s3 <= vmax + 1e-6
                  and np.isfinite(smin) and spread_min < 1e-9 and pmin_ok)
        rows.append(dict(r=r, max_vertex=float(vmax), max_naive=float(nmax),
                         max_strict_slsqp=float(s2), max_random=float(s3),
                         min_slsqp=float(smin),
                         min_penalty=float(pmin) if pmin is not None else None,
                         n_vertex_subsets=int(nsub), n_naive_subsets=int(nsub2),
                         n_naive_feasible=int(nfeas), n_strict_feasible=int(nf2),
                         n_strict_rejected=int(ni2), n_ok_min=int(nok_min),
                         min_spread=float(spread_min), l1_vertex=float(np.abs(vx - P0_17).sum()),
                         consistent=ok, x_max=vx.tolist(), x_min=sx.tolist()))
        print('    %-8.4f %8d %+12.6f %+12.6f %+12.6f %+12.6f %+11.6f %+11.6f %s'
              % (r, nsub, vmax, nmax, s2, s3, smin,
                 pmin if pmin is not None else np.nan, 'OK' if ok else '需检查'))
        print('             顶点 L1=%.6f（约束 r=%.4f）  min 收敛初值=%d 离散度=%.2e  '
              '严格SLSQP 可行/被拒=%d/%d'
              % (np.abs(vx - P0_17).sum(), r, nok_min, spread_min, nf2, ni2))
    print('    核验逻辑：L1 朴素枚举与向量化枚举须逐位一致；L2/L3 是「不受顶点结构约束」的')
    print('              独立搜索，若均不超过枚举值，则枚举值即约束最优。')
    out['extremes'] = rows

    # ---------- 与旧口径对照 ----------
    print('\n[3] 与旧口径对照（暴露修正幅度）')
    old = {0.2601: (-0.0846, +0.6593), 0.1781: (-0.0655, +0.0642)}
    for r, (omn, omx) in old.items():
        new = [x for x in rows if abs(x['r'] - r) < 1e-9][0]
        print('    r=%.4f  旧 max=%+.4f -> 新 max=%+.4f ；极差 %.4f -> %.4f'
              % (r, omx, new['max_vertex'], omx - omn, new['max_vertex'] - new['min_slsqp']))
        out.setdefault('compare_old', []).append(
            dict(r=r, old_min=omn, old_max=omx, new_min=new['min_slsqp'],
                 new_max=new['max_vertex']))
    print('    旧 r=0.2601 的 max 对应迭代点 L1=0.7537 > r，是**违反支持约束**的外推点。')

    # ---------- 极值配方 ----------
    row = [x for x in rows if abs(x['r'] - 0.2601) < 1e-9][0]
    xm, xmn = np.array(row['x_max']), np.array(row['x_min'])
    print('\n[4] r=0.2601 的极值配方（相对 p0 的偏移，仅列 |Δ|>0.005 的域）')
    print('    max h_p = %+.4f 的配方：' % row['max_vertex'])
    for jj in np.argsort(-np.abs(xm - P0_17)):
        if abs(xm[jj] - P0_17[jj]) > 0.005:
            print('      %-20s p0=%.4f -> %.4f  (Δ=%+.4f)'
                  % (DOM17[jj], P0_17[jj], xm[jj], xm[jj] - P0_17[jj]))
    print('    min h_p = %+.4f 的配方：' % row['min_slsqp'])
    for jj in np.argsort(-np.abs(xmn - P0_17)):
        if abs(xmn[jj] - P0_17[jj]) > 0.005:
            print('      %-20s p0=%.4f -> %.4f  (Δ=%+.4f)'
                  % (DOM17[jj], P0_17[jj], xmn[jj], xmn[jj] - P0_17[jj]))
    out['extreme_recipes'] = dict(r=0.2601, dom=DOM17, p0=P0_17.tolist(),
                                  x_max=xm.tolist(), x_min=xmn.tolist())

    # ---------- 损失包络 ----------
    print('\n[5] 配比损失包络（工作点 C=1e22, g=exp, Q0=0.5, Lctx=4096）')
    base = q3.solve_NDQ(1e22, 'exp', 0.5, 4096, nq=200)
    hmin, hmax = row['min_slsqp'], row['max_vertex']
    print('    h_p 支持区间（r=0.2601）：[%+.4f, %+.4f]；基线 L*=%.5f（Q*=%.4f）'
          % (hmin, hmax, base['L'], base['Q']))
    print('    %-7s %11s %13s %13s %11s %10s %13s'
          % ('lam_p', 'B~_eff', 'L(h_min)', 'L(h_max)', '极差', '占L*', '重优化后 L*'))
    env = []
    for lam in [0.0, 0.5, 1.0, 1.5]:
        Bf = q3.B_T * np.exp(lam * hmin)          # λ_p>0 时取最优配比方向 h_min
        r2 = q3.solve_NDQ(1e22, 'exp', 0.5, 4096, nq=200,
                          par=(q3.E_T, q3.A_T, q3.ALPHA, Bf, q3.BETA,
                               q3.RHO_N, q3.RHO_D, q3.E1))
        Llo = q3.loss(base['N_B'], base['D_B'], base['Q'], lam, hmin)
        Lhi = q3.loss(base['N_B'], base['D_B'], base['Q'], lam, hmax)
        env.append(dict(lam=lam, h_min=float(hmin), h_max=float(hmax),
                        L_lo=float(Llo), L_hi=float(Lhi), range=float(abs(Lhi - Llo)),
                        B_eff=float(Bf), N_reopt=float(r2['N_B']), D_reopt=float(r2['D_B']),
                        L_reopt=float(r2['L']), dL_reopt=float(r2['L'] - base['L'])))
        print('    %-7.1f %11.5f %13.5f %13.5f %11.5f %9.2f%% %13.5f'
              % (lam, Bf, Llo, Lhi, abs(Lhi - Llo),
                 100 * abs(Lhi - Llo) / base['L'], r2['L']))
    out['envelope_fixed'] = env
    out['envelope_workpoint'] = dict(C=1e22, g='exp', Q0=0.5, Lctx=4096,
                                     L_base=float(base['L']), N=base['N_B'],
                                     D=base['D_B'], Q=base['Q'], r=0.2601)

    with open(OUTPUT_DIR / 'q3_p_subproblem_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n已写出 q3_p_subproblem_results.json')


if __name__ == '__main__':
    main()
