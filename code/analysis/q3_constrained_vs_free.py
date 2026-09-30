# -*- coding: utf-8 -*-
"""
量化「是否施加 B6 标定支撑域约束」对问题三结论的影响。

我方主脚本的联合最优是在**无盒约束**下求得的（只要求预算等式成立），
团队方案则硬性限定 n∈[0.07,11.97]、d∈[10,600]。
本脚本在同一模型、同一参数下同时求两种解，量化差异与代价。

盒约束版本（团队口径）：
  固定 (d,Q) 时 L 关于 n 严格递减 => 取最大可行 n
      n = min(n_max, (C/1e18 / d - Psi(Q)) / K)
  再在 (d,Q) 上有界网格寻优；n < n_min 视为该点不可行。
  若 n 由 n_max 封顶，则预算未用尽，记录 idle。
"""
import json
import numpy as np

import q3_model_core as q3

Q0 = 0.50
BUDGETS = [1e19, 1e22, 1e24]
GS = ['exp', 'pow', 'log']
LCTX = [4096, 32768]

NMIN, NMAX = q3.BOX['N']
DMIN, DMAX = q3.BOX['D']


def constrained(C, gname, Q0, Lctx, par=None, nd=900, nq=500, refine=3):
    """盒约束下的最优配置。返回 (L, n, d, Q, idle_frac)。"""
    g, _ = q3.G[gname]
    K = q3.k_of(Lctx)
    Ctil = C / 1e18

    def scan(dlo, dhi, Qlo, Qhi, nd_, nq_):
        ds = np.linspace(dlo, dhi, nd_)
        Qs = np.linspace(Qlo, Qhi, nq_)
        best = None
        for Q in Qs:
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
    for it in range(refine):                      # 逐级收窄网格
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
    return dict(L=L, N=n, D=d, Q=Q, idle_C18=float(idle), idle_frac=float(idle / Ctil))


def free(C, gname, Q0, Lctx, par=None):
    r = q3.solve_NDQ(C, gname, Q0, Lctx, par=par, nq=800)
    return dict(L=r['L'], N=r['N_B'], D=r['D_B'], Q=r['Q'])


def in_box(x):
    return (NMIN - 1e-9 <= x['N'] <= NMAX + 1e-9
            and DMIN - 1e-9 <= x['D'] <= DMAX + 1e-9)


print('=' * 118)
print('问题三：无盒约束（我方主脚本） vs  B6 标定支撑域盒约束（团队口径）')
print('=' * 118)
print('盒约束 n∈[%.2f,%.2f] B 参数，d∈[%.0f,%.0f] B tokens，Q∈[%.1f,1.0]；参数 SET_A；p=p0'
      % (NMIN, NMAX, DMIN, DMAX, Q0))

rows = []
for Lctx in LCTX:
    for gname in GS:
        for C in BUDGETS:
            f = free(C, gname, Q0, Lctx)
            c = constrained(C, gname, Q0, Lctx)
            rec = dict(Lctx=Lctx, g=gname, C=C, free=f, cons=c)
            if c is not None:
                rec['dL'] = c['L'] - f['L']
                rec['dL_pct'] = 100.0 * (c['L'] - f['L']) / f['L']
            rows.append(rec)

hdr = ('%-7s %-5s %-7s | %-32s | %-32s | %s'
       % ('Lctx', 'g', 'C', '无约束最优 (N,D,Q,L)', '盒约束最优 (N,D,Q,L)', 'ΔL'))
print()
print(hdr)
print('-' * 118)
for r in rows:
    f, c = r['free'], r['cons']
    fs = 'N=%.4f D=%.3f Q=%.4f L=%.5f' % (f['N'], f['D'], f['Q'], f['L'])
    if c is None:
        print('%-7d %-5s %-7.0e | %-32s | %-32s | %s'
              % (r['Lctx'], r['g'], r['C'], fs, '盒约束下不可行', '—'))
        continue
    cs = 'N=%.4f D=%.3f Q=%.4f L=%.5f' % (c['N'], c['D'], c['Q'], c['L'])
    print('%-7d %-5s %-7.0e | %-32s | %-32s | %+.5f (%+.2f%%)'
          % (r['Lctx'], r['g'], r['C'], fs, cs, r['dL'], r['dL_pct']))

print()
print('=' * 118)
print('要点')
print('=' * 118)
print('[a] 我方无约束解越界情况（盒约束会被触发吗）')
for r in rows:
    f = r['free']
    oob = []
    if f['N'] < NMIN - 1e-9: oob.append('N<%.2f' % NMIN)
    if f['N'] > NMAX + 1e-9: oob.append('N>%.2f' % NMAX)
    if f['D'] < DMIN - 1e-9: oob.append('D<%.0f' % DMIN)
    if f['D'] > DMAX + 1e-9: oob.append('D>%.0f' % DMAX)
    if oob:
        print('    Lctx=%-6d g=%-4s C=%.0e : 越界 %s' % (r['Lctx'], r['g'], r['C'], ', '.join(oob)))

print()
print('[b] 盒约束解是否出现未用预算（idle>0）')
any_idle = False
for r in rows:
    c = r['cons']
    if c and c['idle_frac'] > 1e-6:
        any_idle = True
        print('    Lctx=%-6d g=%-4s C=%.0e : idle=%.2f%%  (N 触顶 n_max=%.2f)'
              % (r['Lctx'], r['g'], r['C'], 100 * c['idle_frac'], NMAX))
if not any_idle:
    print('    无（本例预算区间内盒约束未导致预算闲置）')

print()
print('[c] 两解差异汇总')
dl = [r['dL_pct'] for r in rows if r.get('cons')]
dl = [x for x in dl if np.isfinite(x)]
if dl:
    print('    ΔL 相对偏差：中位=%.2f%%  最大=%.2f%%  最小=%.2f%%' % (np.median(dl), max(dl), min(dl)))
    print('    盒约束总是使损失不降（ΔL>=0），差值即「诚实报告支撑域」的代价。')

with open(r'd:\F题\q3_constrained_vs_free.json', 'w', encoding='utf-8') as fp:
    json.dump(rows, fp, ensure_ascii=False, indent=1, default=float)
print('\n已写出 q3_constrained_vs_free.json')
