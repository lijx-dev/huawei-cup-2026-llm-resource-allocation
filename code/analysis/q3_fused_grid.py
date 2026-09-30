# -*- coding: utf-8 -*-
"""
问题三·融合方案主表生成器（盒约束下的条件最优 + 结构性转移阈值）

融合口径
--------
求解层 = 我方解析结构（固定 (d,Q) 时 L 关于 n 严格递减 => 取最大可行 n；
         D 由预算精确消去；Q 维网格 + 逐级收窄精修）
约束层 = 团队支撑域纪律（n∈[0.07,11.97] B、d∈[10,600] B、Q∈[Q0,1]）
输出层 = 团队交付网格（3 预算 × 5 上下文 × 3 质量成本，另加 η 三档压力情景）
转移层 = 团队七指示量 R(C) + C_act/C_sat 闭式核对

不可行判定
----------
C_min(Lctx,η) = (6+ηLctx)·1e18·n_min·d_min
C < C_min 时盒内无可行点，记 infeasible，不填补任何假设解。

运行： python q3_fused_grid.py
输出： q3_fused_grid.json / q3_fused_grid_eta2e-4.csv / q3_fused_grid_all.csv
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import q3_model_core as q3

Q0_MAIN = 0.50
BUDGETS = [1e19, 1e22, 1e24]
GS = ['exp', 'pow', 'log']
LCTX = [2048, 4096, 8192, 32768, 131072]
ETAS = [1e-4, 2e-4, 4e-4]

NMIN, NMAX = q3.BOX['N']
DMIN, DMAX = q3.BOX['D']


def c_min(Lctx, eta):
    return (6.0 + eta * Lctx) * 1e18 * NMIN * DMIN


def c_sat(gname, Q0, Lctx, eta):
    """质量触顶 + 规模双触顶同时发生的预算（团队闭式）。"""
    g, _ = q3.G[gname]
    return ((6.0 + eta * Lctx) * 1e18 * NMAX * DMAX
            + 1e9 * DMAX * max(g(1.0) - g(Q0), 0.0))


def solve_box(C, gname, Q0, Lctx, eta=2e-4, nd=600, nq=400, refine=3, par=None):
    """盒约束下的条件最优。返回 dict 或 None（不可行）。"""
    g, _ = q3.G[gname]
    K = 6.0 + eta * Lctx
    Ctil = C / 1e18
    if Ctil < K * NMIN * DMIN:
        return None

    def scan(dlo, dhi, Qlo, Qhi, nd_, nq_):
        ds = np.linspace(dlo, dhi, nd_)
        best = None
        for Q in np.linspace(Qlo, Qhi, nq_):
            Psi = max(g(Q) - g(Q0), 0.0) / 1e9
            n = np.minimum(NMAX, (Ctil / ds - Psi) / K)
            m = n >= NMIN
            if not m.any():
                continue
            L = q3.loss(n[m], ds[m], Q, par=par)
            i = int(np.argmin(L))
            if best is None or L[i] < best[0]:
                best = (float(L[i]), float(n[m][i]), float(ds[m][i]), float(Q))
        return best

    b = scan(DMIN, DMAX, Q0, 1.0, nd, nq)
    if b is None:
        return None
    for it in range(refine):
        span_d = (DMAX - DMIN) / (nd ** (it + 1)) * 6
        span_q = (1.0 - Q0) / (nq ** (it + 1)) * 6
        nb = scan(max(DMIN, b[2] - span_d), min(DMAX, b[2] + span_d),
                  max(Q0, b[3] - span_q), min(1.0, b[3] + span_q), 400, 400)
        if nb is None or nb[0] >= b[0]:
            break
        b = nb

    L, n, d, Q = b
    Psi = max(g(Q) - g(Q0), 0.0) / 1e9
    idle = Ctil - (K * n + Psi) * d
    s_tr = 6.0 * n * d / Ctil
    s_at = eta * Lctx * n * d / Ctil
    s_Q = d * Psi / Ctil
    state = {
        'B_Q_gt0': int(Q > Q0 + 1e-9),
        'Q_eq1': int(Q >= 1.0 - 1e-9),
        'd_eq_dmin': int(d <= DMIN + 1e-6),
        'n_eq_nmin': int(n <= NMIN + 1e-6),
        'n_eq_nmax': int(n >= NMAX - 1e-6),
        'd_eq_dmax': int(d >= DMAX - 1e-6),
        'B_idle_gt0': int(idle > 1e-9),
    }
    dom = max([('train', s_tr), ('attn', s_at), ('Q', s_Q)], key=lambda t: t[1])[0]
    return dict(L=L, N=n, D=d, Q=Q, idle_C18=float(idle), idle_frac=float(idle / Ctil),
                s_train=s_tr, s_attn=s_at, s_Q=s_Q, ratio_attn_train=(s_at / s_tr if s_tr > 0 else None),
                state=state, dominant=dom,
                n_at_bound=('min' if state['n_eq_nmin'] else 'max' if state['n_eq_nmax'] else 'interior'),
                d_at_bound=('min' if state['d_eq_dmin'] else 'max' if state['d_eq_dmax'] else 'interior'))


def pure_box(C, gname, Q0, Lctx, eta=2e-4, par=None):
    """盒约束下 Q≡Q0 的纯规模基准损失（用于 gain_vs_pure）。"""
    pd_ = pure_box_analytic(C, gname, Q0, Lctx, eta, par)
    if pd_ is None:
        return None
    n, d = pd_
    return dict(L=float(q3.loss(n, d, Q0, par=par)), N=n, D=d, Q=Q0)


def pure_box_analytic(C, gname, Q0, Lctx, eta, par=None):
    """盒约束下 Q≡Q0 的纯规模最优（解析解，无需网格）。返回 (n, d) 或 None。

    预算松弛（两个上界都够得着）时直接返回 (n_max, d_max)；
    否则在预算曲线 K·n·d = C 上取 L 最小的可行点。
    """
    E, A, a, B, b, rN, rD, e1 = q3.par_of(par)
    K = 6.0 + eta * Lctx
    Ctil = C / 1e18
    if K * NMIN * DMIN > Ctil:
        return None
    if K * NMAX * DMAX <= Ctil:
        return float(NMAX), float(DMAX)
    n0 = ((a * A * np.exp(-rN * Q0) * Ctil ** b)
          / (b * B * np.exp(-rD * Q0) * K ** b)) ** (1.0 / (a + b))
    d0 = Ctil / (K * n0)
    if n0 > NMAX:
        n, d = NMAX, Ctil / (K * NMAX)
    elif d0 > DMAX:
        n, d = Ctil / (K * DMAX), DMAX
    elif n0 < NMIN:
        n, d = NMIN, Ctil / (K * NMIN)
    elif d0 < DMIN:
        n, d = Ctil / (K * DMIN), DMIN
    else:
        n, d = n0, d0
    if n < NMIN - 1e-9 or d < DMIN - 1e-9 or n > NMAX + 1e-9 or d > DMAX + 1e-9:
        return None
    return float(n), float(d)


def mc_envelope(n, d, C, gname, Q0, Lctx, eta, par=None):
    """给定预算 C 与纯规模点 (n,d)，返回 (MB, MC)。MC=inf 表示 Q 不可融资。"""
    E, A, a, B, b, rN, rD, e1 = q3.par_of(par)
    g, gp = q3.G[gname]
    K = 6.0 + eta * Lctx
    Ctil = C / 1e18
    A_N = A * n ** (-a) * np.exp(-rN * Q0)
    A_D = B * d ** (-b) * np.exp(-rD * Q0)
    MB = e1 + rN * A_N + rD * A_D
    if K * n * d < Ctil * (1 - 1e-12):          # 预算松弛，Q 由闲置预算承担
        return float(MB), 0.0
    if n > NMIN * (1 + 1e-9):                   # n 仍自由：走 n 通道
        mu = (a * A_N / n) / (d * K)
    elif d > DMIN * (1 + 1e-9):                 # d 仍自由：走 d 通道
        mu = (b * A_D / d) / (K * n)
    else:                                       # 两个变量都钉在下界，Q 不可融资
        return float(MB), float('inf')
    return float(MB), float(mu * d * gp(Q0) / 1e9)


def c_act(gname, Q0, Lctx, eta=2e-4, par=None):
    """质量通道激活阈值：盒约束纯规模分支上 MB(Q0)-MC(Q0) 由负变正的最小预算。

    返回 (C 或 None, 性质说明)。C 等于 C_min 时说明「最小可行预算处即已激活」。
    """
    lo, hi = c_min(Lctx, eta), c_sat(gname, Q0, Lctx, eta)
    if hi <= lo:
        return None, 'C_sat<=C_min'

    def f(C):
        pd_ = pure_box_analytic(C, gname, Q0, Lctx, eta, par)
        if pd_ is None:
            return -1.0
        MB, MC = mc_envelope(pd_[0], pd_[1], C, gname, Q0, Lctx, eta, par)
        return -1.0 if not np.isfinite(MC) else MB - MC

    lo2 = lo * (1 + 1e-6)
    if f(lo2) > 0:
        return float(lo), '最小可行预算处即激活'
    if f(hi) < 0:
        return None, '始终不激活'
    return float(brentq(f, lo2, hi, xtol=1e-8 * lo, maxiter=400)), '内部可达'


def main():
    print('=' * 120)
    print('问题三·融合方案：盒约束主表（我方解析结构 + 团队支撑域纪律）')
    print('盒约束 n∈[%.2f,%.2f] B 参数  d∈[%.0f,%.0f] B tokens  Q∈[Q0,1]；参数 SET_A；p=p0'
          % (NMIN, NMAX, DMIN, DMAX))
    print('=' * 120)

    rows = []
    for eta in ETAS:
        for Lctx in LCTX:
            cm = c_min(Lctx, eta)
            for gname in GS:
                cs = c_sat(gname, Q0_MAIN, Lctx, eta)
                for C in BUDGETS:
                    r = solve_box(C, gname, Q0_MAIN, Lctx, eta=eta)
                    rec = dict(eta=eta, Lctx=Lctx, g=gname, C=C,
                               C_min=cm, C_sat=cs, feasible=r is not None)
                    if r is None:
                        rec['note'] = 'C < C_min，盒内无可行点'
                    else:
                        pp = pure_box(C, gname, Q0_MAIN, Lctx, eta=eta)
                        rec.update(r)
                        rec['L_pure'] = pp['L'] if pp else None
                        rec['gain_vs_pure'] = (pp['L'] - r['L']) if pp else None
                        rec['in_box'] = True
                    rows.append(rec)

    df = pd.DataFrame(rows)
    for k in ['B_Q_gt0', 'Q_eq1', 'd_eq_dmin', 'n_eq_nmin', 'n_eq_nmax', 'd_eq_dmax', 'B_idle_gt0']:
        df[k] = [r['state'][k] if isinstance(r.get('state'), dict) else np.nan for r in rows]
    keep = ['eta', 'Lctx', 'g', 'C', 'feasible', 'N', 'D', 'Q', 'L',
            's_train', 's_attn', 's_Q', 'ratio_attn_train', 'idle_frac',
            'n_at_bound', 'd_at_bound', 'dominant', 'gain_vs_pure',
            'B_Q_gt0', 'Q_eq1', 'd_eq_dmin', 'n_eq_nmin', 'n_eq_nmax', 'd_eq_dmax', 'B_idle_gt0',
            'C_min', 'C_sat']
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep]
    df.to_csv(r'd:\F题\q3_fused_grid_all.csv', index=False, encoding='utf-8-sig')
    df[np.isclose(df['eta'], 2e-4)].to_csv(
        r'd:\F题\q3_fused_grid_eta2e-4.csv', index=False, encoding='utf-8-sig')

    print('\n[1] 主表切片：η=2e-4, Q0=0.50（45 情景，其中 %d 个不可行）'
          % int((~df[np.isclose(df['eta'], 2e-4)]['feasible']).sum()))
    sub = df[np.isclose(df['eta'], 2e-4)].reset_index(drop=True)
    print('%-7s %-5s %-8s | %-9s %-9s %-8s %-10s | %-7s %-7s %s'
          % ('Lctx', 'g', 'C', 'N(B)', 'D(B)', 'Q*', 'L*', 'idle%', 'gain', 'R(C) 活跃指示量'))
    print('-' * 130)
    for _, r in sub.iterrows():
        if not r['feasible']:
            print('%-7d %-5s %-8.0e | %-9s %-9s %-8s %-10s | %-7s %-7s %s'
                  % (r['Lctx'], r['g'], r['C'], '—', '—', '—', '—', '—', '—', '不可行(C<C_min)'))
            continue
        flags = ''.join([n for n, k in [('Q', 'B_Q_gt0'), ('1', 'Q_eq1'), ('d-', 'd_eq_dmin'),
                                        ('n-', 'n_eq_nmin'), ('n+', 'n_eq_nmax'), ('d+', 'd_eq_dmax'),
                                        ('I', 'B_idle_gt0')] if r[k] == 1]) or '(纯内点)'
        print('%-7d %-5s %-8.0e | %9.4f %9.3f %8.4f %10.5f | %7.2f %7.4f %s'
              % (r['Lctx'], r['g'], r['C'], r['N'], r['D'], r['Q'], r['L'],
                 100 * r['idle_frac'], r['gain_vs_pure'], flags))

    print('\n[2] 结构性转移阈值（盒约束口径；团队七指示量判据）')
    print('    C_min = 最小可行预算；C_act = 质量通道激活阈值；C_sat = 规模与质量双触顶预算')
    th = []
    print('%-7s %-5s %-13s %-13s %-13s %s'
          % ('Lctx', 'g', 'C_min', 'C_act', 'C_sat', 'C_act 性质'))
    print('-' * 130)
    for eta in ETAS:
        for Lctx in LCTX:
            for gname in GS:
                cm = c_min(Lctx, eta)
                cs = c_sat(gname, Q0_MAIN, Lctx, eta)
                ca, note = c_act(gname, Q0_MAIN, Lctx, eta=eta)
                th.append(dict(eta=eta, Lctx=Lctx, g=gname, C_min=cm, C_act=ca,
                               C_sat=cs, C_act_note=note,
                               boundary_activation=note.startswith('最小可行')))
                if np.isclose(eta, 2e-4):
                    print('%-7d %-5s %-13.5e %-13s %-13.5e %s'
                          % (Lctx, gname, cm, ('%.5e' % ca) if ca else '—', cs, note))

    print('\n[3] C_act 口径说明')
    bnd = [t for t in th if np.isclose(t['eta'], 2e-4) and t['boundary_activation']]
    print('    η=2e-4 的 15 个 (Lctx,g) 组合中，%d 个在最小可行预算处即已激活（C_act = C_min）：'
          % len(bnd))
    for t in bnd:
        print('      Lctx=%-7d g=%-4s C_act = C_min = %.5e' % (t['Lctx'], t['g'], t['C_min']))
    print('    其余 %d 个为内部可达阈值。' % (15 - len(bnd)))
    print('    注：旧口径（无盒约束纯规模分支）曾给出 6 个 C_act 中 4 个低于 C_min 的「不可达」结论，')
    print('        该现象是**外推分支**的产物；在盒约束分支上重算后不再出现，已在本脚本中修正。')

    out = dict(rows=rows, thresholds=th,
               config=dict(Q0=Q0_MAIN, budgets=BUDGETS, Lctx=LCTX, gs=GS, etas=ETAS,
                           box=dict(N=list(q3.BOX['N']), D=list(q3.BOX['D'])),
                           params='SET_A', p='p0'))
    with open(r'd:\F题\q3_fused_grid.json', 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1, default=float)
    print('\n已写出 q3_fused_grid.json / q3_fused_grid_eta2e-4.csv / q3_fused_grid_all.csv')


if __name__ == '__main__':
    main()
