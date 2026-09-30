# -*- coding: utf-8 -*-
"""
问题三补充核验：KKT 条件逐项校验 + 独立求解器交叉验证 + 包络定理自洽性。

为什么需要这个脚本
------------------
题面要求"给出结构性转移的明确数学定义与识别方法"。主脚本用数值扫描 + 二分定位阈值，
这属于"识别"，但"定义"必须落在 KKT 条件上才算严格。本脚本做三件事：

[1] KKT 逐项校验：在 9 个（预算 × 成本函数）算例上验证
      - 预算约束取等（互补松弛）
      - 乘子符号 λ>=0、μ0>=0、μ1>=0
      - 平稳性残差（N 通道 / D 通道 / Q 通道）
    并把乘子按"哪一支约束活跃"分类，验证结构性转移的定义与 KKT 活跃集一一对应。

[2] 独立求解器交叉验证：用 SLSQP（对数参数空间 + 多初值）独立复算最优配置，
    与主脚本"解析消去 D + 一维求根"的链路对比，确认最优点不是数值假象。

[3] 包络定理自洽性：最优损失对预算的对数弹性应等于
      d ln(L*-E_eff)/d ln C = -λ / (L*-E_eff) * (C/1e18)^{-1} * ...
    即由 KKT 乘子 λ 直接给出，与主脚本 [3] 的数值斜率对照。

单位约定（与 q3_model_core 完全一致）
------------------------------------
    N 以 B(1e9) 计、D 以 B tokens(1e9) 计、成本以 FLOPs 计。
    缩放成本  c = (K*N + Psi(Q)) * D ，真实成本 = 1e18 * c ，Ctil = C/1e18。
"""
import json
import numpy as np
from scipy.optimize import minimize

import q3_model_core as q3
from q3_paths import A_DIR, B_DIR, INPUT_DIR, OUTPUT_DIR


def kkt_check(C, gname, Q0, Lctx, par=None, lam_p=0.0, hp=0.0, nq=200, tol=1e-9):
    """在给定算例的最优解处逐项校验 KKT 条件。"""
    r = q3.solve_NDQ(C, gname, Q0, Lctx, lam=lam_p, hp=hp, par=par, nq=nq)
    E, A, a, B, b, rN, rD, e1 = q3.par_of(par)
    g, gp = q3.G[gname]
    K = q3.k_of(Lctx)
    Ctil = C / 1e18
    N, D, Q = r['N_B'], r['D_B'], r['Q']

    Psi = max(g(Q) - g(Q0), 0.0) / 1e9
    Psi_p = gp(Q) / 1e9 if Q > Q0 + 1e-12 else 0.0     # 右导数（Q 的下界是拐点）

    # 目标函数偏导（缩放坐标）
    dLdN = -a * A * N ** (-a - 1.0) * np.exp(-rN * Q)
    dLdD = -b * B * D ** (-b - 1.0) * np.exp(-rD * Q) * np.exp(lam_p * hp)
    dLdQ = (-rN * A * N ** (-a) * np.exp(-rN * Q)
            - rD * B * D ** (-b) * np.exp(-rD * Q) * np.exp(lam_p * hp) - e1)

    # 约束梯度（缩放坐标）
    dcdN = K * D
    dcdD = K * N + Psi
    dcdQ = D * Psi_p

    # 乘子：N/D 通道各解一个 λ，两者应一致
    lam_N = -dLdN / dcdN
    lam_D = -dLdD / dcdD

    # 预算互补松弛：c 应恰好等于 Ctil
    c_act = (K * N + Psi) * D
    comp = abs(c_act - Ctil) / Ctil

    # Q 通道：dLdQ + lam*dcdQ - mu0 + mu1 = 0
    red = dLdQ + lam_D * dcdQ                     # 约化目标对 Q 的（单侧）导数
    on_lo = Q <= Q0 + 1e-9
    on_hi = Q >= 1.0 - 1e-9
    if on_lo:
        mu0, mu1 = red, 0.0
    elif on_hi:
        mu0, mu1 = 0.0, -red
    else:
        mu0 = mu1 = 0.0
    q_res = red - mu0 + mu1                        # 平稳性残差

    # 包络定理：dL*/dC = -λ * dc/dC = -λ / 1e18
    #
    # 口径提示：弹性必须对"模型自身的渐近地板" E_eff(Q) = E - E1*Q 取，而不是对固定的
    # E - E1*Q0 取。Q* 饱和到 1 时两者相差 E1*(1-Q0)，用错地板会把弹性系统性放大
    # （主脚本 [3] 即用了固定地板，故报出 -0.166 而非 -0.146）。
    E_eff = E - e1 * Q
    E_eff0 = E - e1 * Q0
    elas = (C / (r['L'] - E_eff)) * (-lam_D / 1e18)
    elas_wrongfloor = (C / (r['L'] - E_eff0)) * (-lam_D / 1e18)

    state_q = 'boundary' if on_lo else ('saturated' if on_hi else 'interior')
    sh = dict(train=r['s_train'], attn=r['s_attn'], Q=r['s_Q'])
    dom = max(sh, key=sh.get)

    return dict(C=C, g=gname, Q0=Q0, Lctx=Lctx, N=N, D=D, Q=Q, L=r['L'],
                state_q=state_q, dominant=dom,
                lam=float(lam_D), lam_mismatch=float(abs(lam_N - lam_D)),
                mu0=float(mu0), mu1=float(mu1), q_res=float(abs(q_res)),
                comp_res=float(comp), elastic_kkt=float(elas),
                elastic_wrongfloor=float(elas_wrongfloor),
                shares=sh)


def xcheck_slsqp(C, gname, Q0, Lctx, par=None, lam_p=0.0, hp=0.0, nstart=24, seed=0):
    """独立求解器（SLSQP，对数参数空间，多初值）复算最优配置。"""
    E, A, a, B, b, rN, rD, e1 = q3.par_of(par)
    g, _ = q3.G[gname]
    K = q3.k_of(Lctx)
    Ctil = C / 1e18
    rng = np.random.default_rng(seed)

    def obj(x):
        N, D, Q = 10.0 ** x[0], 10.0 ** x[1], x[2]
        return q3.loss(N, D, Q, lam_p, hp, par)

    cons = [dict(type='ineq',
                 fun=lambda x: Ctil - (K * 10.0 ** x[0]
                                       + max(g(x[2]) - g(Q0), 0.0) / 1e9) * 10.0 ** x[1])]
    bnds = [(-3.0, 4.0), (-3.0, 5.0), (Q0, 1.0)]
    starts = [np.array([np.log10(0.3), np.log10(3.0), 0.6]),
              np.array([np.log10(1.0), np.log10(20.0), 0.8]),
              np.array([np.log10(10.0), np.log10(200.0), 0.95])]
    starts += [np.array([rng.uniform(-2, 2), rng.uniform(-1, 4), rng.uniform(Q0, 1.0)])
               for _ in range(nstart)]
    best, ok, vals = None, 0, []
    for s in starts:
        try:
            rr = minimize(obj, s, method='SLSQP', bounds=bnds, constraints=cons,
                          options=dict(maxiter=600, ftol=1e-14))
        except Exception:
            continue
        if not np.isfinite(rr.fun):
            continue
        # 可行性复核
        N, D, Q = 10.0 ** rr.x[0], 10.0 ** rr.x[1], rr.x[2]
        if (K * N + max(g(Q) - g(Q0), 0.0) / 1e9) * D > Ctil * (1 + 1e-8):
            continue
        ok += 1
        vals.append(rr.fun)
        if best is None or rr.fun < best[0]:
            best = (float(rr.fun), N, D, Q)
    if best is None:
        return None
    return dict(L=best[0], N=best[1], D=best[2], Q=best[3], n_ok=ok,
                spread=float(max(vals) - min(vals)))


def main():
    out = {}
    Q0, Lctx = 0.5, 4096
    budgets = [1e19, 1e22, 1e24]
    gs = ['exp', 'pow', 'log']
    LAB = q3.G_LABEL

    print('=' * 100)
    print('问题三补充核验：KKT 逐项校验 / 独立求解器交叉验证 / 包络定理自洽性')
    print('=' * 100)

    # ---------- [1] KKT 逐项校验 ----------
    print('\n[1] KKT 逐项校验（9 算例）')
    print('    %-6s %-6s %-10s %-9s %10s %10s %9s %11s %10s'
          % ('C', 'g(Q)', 'Q 状态', '主导成本', 'lambda', 'mu0', 'mu1',
             '平稳残差', '预算残差'))
    rows = []
    for C in budgets:
        for gn in gs:
            k = kkt_check(C, gn, Q0, Lctx)
            rows.append(k)
            print('    %-6s %-6s %-10s %-9s %10.4e %10.4e %9.4e %11.2e %10.2e'
                  % ('1e%d' % round(np.log10(C)), LAB[gn], k['state_q'], k['dominant'],
                     k['lam'], k['mu0'], k['mu1'], max(k['q_res'], k['lam_mismatch']),
                     k['comp_res']))
    out['kkt'] = rows

    # 乘子符号总检
    bad = [r for r in rows if (r['lam'] < -1e-12 or r['mu0'] < -1e-10 or r['mu1'] < -1e-10
                               or r['q_res'] > 1e-8 or r['lam_mismatch'] > 1e-8
                               or r['comp_res'] > 1e-10)]
    print('    符号/残差校验：%d/%d 通过' % (len(rows) - len(bad), len(rows)))
    if bad:
        for r in bad:
            print('      [不通过] C=%.0e g=%s lam=%.3e mu0=%.3e mu1=%.3e q_res=%.2e comp=%.2e'
                  % (r['C'], r['g'], r['lam'], r['mu0'], r['mu1'], r['q_res'], r['comp_res']))
    out['kkt_pass'] = dict(n=len(rows), n_bad=len(bad))

    # ---------- [2] 独立求解器交叉验证 ----------
    print('\n[2] 独立求解器交叉验证（SLSQP 多初值 vs 解析消去 D + 一维求根）')
    print('    %-6s %-6s %12s %12s %9s %12s %12s %12s %9s'
          % ('C', 'g(Q)', 'N*解析(B)', 'N*SLSQP(B)', 'ΔN%', 'D*解析(B)',
             'D*SLSQP(B)', 'ΔL', '收敛数'))
    xs = []
    for C in budgets:
        for gn in gs:
            a = q3.solve_NDQ(C, gn, Q0, Lctx, nq=200)
            b_ = xcheck_slsqp(C, gn, Q0, Lctx)
            dN = 100 * abs(b_['N'] - a['N_B']) / a['N_B']
            xs.append(dict(C=C, g=gn, N_ana=a['N_B'], N_sls=float(b_['N']),
                           D_ana=a['D_B'], D_sls=float(b_['D']), Q_ana=a['Q'],
                           Q_sls=float(b_['Q']), dN_pct=float(dN),
                           dL=float(abs(b_['L'] - a['L'])), n_ok=b_['n_ok'],
                           spread=b_['spread']))
            print('    %-6s %-6s %12.4f %12.4f %8.3f%% %12.3f %12.3f %12.2e %9d'
                  % ('1e%d' % round(np.log10(C)), LAB[gn], a['N_B'], b_['N'], dN,
                     a['D_B'], b_['D'], xs[-1]['dL'], b_['n_ok']))
    out['xcheck'] = xs
    mx = max(x['dN_pct'] for x in xs)
    mdl = max(x['dL'] for x in xs)
    print('    最大偏差：ΔN = %.3f%% ，ΔL = %.2e ；多初值成功收敛数 %d–%d'
          % (mx, mdl, min(x['n_ok'] for x in xs), max(x['n_ok'] for x in xs)))

    # ---------- [3] 包络定理自洽性 ----------
    print('\n[3] 包络定理自洽性：d ln(L*-E_eff(Q*))/d ln C = -lambda * C / (1e18 * (L*-E_eff(Q*)))')
    print('    地板口径必须自洽取 E_eff(Q) = E - E1*Q（Q*=1 时 E1*(1-Q0) 的差会把弹性放大）')
    print('    %-6s %-6s %10s %14s %14s %12s %14s'
          % ('C', 'g(Q)', 'lambda', '弹性(KKT自洽)', '弹性(数值自洽)', '相对差', '弹性(错地板)'))
    el = []
    for C in budgets:
        for gn in gs:
            k = kkt_check(C, gn, Q0, Lctx)

            def excess(c):
                rr = q3.solve_NDQ(c, gn, Q0, Lctx, nq=200)
                return rr['L'] - (q3.E_T - q3.E1 * rr['Q'])

            eps = 0.01
            num = ((np.log(excess(C * (1 + eps))) - np.log(excess(C * (1 - eps))))
                   / (np.log(C * (1 + eps)) - np.log(C * (1 - eps))))
            rel = abs(k['elastic_kkt'] - num) / abs(num)
            el.append(dict(C=C, g=gn, lam=k['lam'], elas_kkt=k['elastic_kkt'],
                           elas_num=float(num), rel=float(rel),
                           elas_wrongfloor=k['elastic_wrongfloor']))
            print('    %-6s %-6s %10.5f %14.6f %14.6f %11.2e %14.6f'
                  % ('1e%d' % round(np.log10(C)), LAB[gn], k['lam'],
                     k['elastic_kkt'], num, rel, k['elastic_wrongfloor']))
    out['envelope'] = el
    print('    最大相对差 = %.2e （KKT 乘子与数值预算弹性自洽）'
          % max(x['rel'] for x in el))
    print('    纯规模解析值 -αβ/(α+β) = %.5f ；质量饱和段的 KKT 弹性与之同量级，'
          % (-q3.ALPHA * q3.BETA / (q3.ALPHA + q3.BETA)))
    print('    说明：Q* 饱和后 Q 通道冻结，弹性应回到经典值附近（差距来自剩余 E1 项）。')

    with open(OUTPUT_DIR / 'q3_kkt_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n已写出 q3_kkt_results.json')


if __name__ == '__main__':
    main()
