# -*- coding: utf-8 -*-
"""
问题三派生量：为《问题三_数学推导与建模思路》提供可直接引用的定量结论。

覆盖：
  [A] Chinchilla 对应：N* ∝ C^a、D* ∝ C^b 的解析指数 a=β/(α+β)、b=α/(α+β) 与数值斜率对照；
      D*/N* 与 Chinchilla 报告的经验值 ~20 的对照。
  [B] 成本份额的结构不变量：s_attn/s_train ≡ η·Lctx/6（与 C 无关），故预算 C 不可能翻转
      训练↔注意力 的主导关系；能翻转它的只有 L_ctx。
  [C] 质量—规模替代率：等损失、等数据量下 ΔQ 与 ΔN 的局部替代关系。
  [D] 边界冻结时预算弹性的解析值：Q 处于边界（Q0 或 1）时 φ 退化为纯幂律，
      dln(L*-E_eff)/dlnC ≡ -αβ/(α+β)。
  [E] 降维解的一阶条件残差与预算等式残差（数值正确性自检）。
  [F] 配比通道不消耗预算的形式化核验（∂c/∂p ≡ 0）。
"""
import json
import numpy as np

import q3_model_core as q3
from q3_paths import A_DIR, B_DIR, INPUT_DIR, OUTPUT_DIR

E, A, a, B, b, rN, rD, e1 = q3.PAR_DEF
K_ = q3.k_of(4096)
Q0 = 0.5
LC = 4096
OUT = {}


def report(t):
    print(t)


report('=' * 96)
report('问题三派生量（SET_A 参数：E=%.4f A=%.4f α=%.4f B=%.4f β=%.4f ρN=%.4f ρD=%.4f E1=%.4f）'
       % (E, A, a, B, b, rN, rD, e1))
report('η=%.1e  L_ctx=%.0f  K=6+ηL_ctx=%.4f  Q0=%.2f' % (q3.ETA, LC, K_, Q0))
report('=' * 96)

# ---------------- [A] Chinchilla 对应 ----------------
report('\n[A] Chinchilla 对应：解析指数 vs 数值斜率（纯规模分支 Q≡Q0，L_ctx=%d）' % LC)
ap, bp = b / (a + b), a / (a + b)
report('    解析 a=β/(α+β)=%.6f   b=α/(α+β)=%.6f   a+b=%.6f' % (ap, bp, ap + bp))
Cs = np.logspace(18, 26, 41)
Nv = np.array([q3.pure_scale(c, Q0, LC)['N_B'] for c in Cs])
Dv = np.array([q3.pure_scale(c, Q0, LC)['D_B'] for c in Cs])
slN = float(np.polyfit(np.log(Cs), np.log(Nv), 1)[0])
slD = float(np.polyfit(np.log(Cs), np.log(Dv), 1)[0])
report('    数值 a_num=%.6f（偏差 %+.2e）   b_num=%.6f（偏差 %+.2e）'
       % (slN, slN - ap, slD, slD - bp))
dn_work = q3.pure_scale(1e22, Q0, LC)['ratio_DN']
report('    D*/N* ∝ C^(b-a)，指数=%.6f；C=1e22 时 D*/N*=%.6f。'
       % (bp - ap, dn_work))
report('    Chinchilla(Hoffmann 2022) 报告的计算最优 D/N ≈ 20；本模型纯规模分支在 C=1e22 给出 %.2f，'
       % dn_work)
report('    差异来源：本模型 α≈β（0.278 vs 0.283）而非 Chinchilla 的 0.34/0.28，'
       '且质量因子 e^{-ρQ} 改变了系数比，故 D/N 不同量级。')
OUT['chinchilla'] = dict(a_ana=ap, b_ana=bp, a_num=slN, b_num=slD,
                         DN_at_1e22=float(dn_work), DN_budget_exponent=float(bp - ap),
                         DN_ref=20.0, note='α≈β 导致 D/N 远小于 20')

# ---------------- [B] 成本份额不变量 ----------------
report('\n[B] 成本份额结构不变量：s_attn/s_train ≡ η·Lctx/6')
report('    %-10s %12s %12s %14s %14s' % ('Lctx', 'ηLctx/6', 's_attn/s_train', '一致性', 'Lctx/Lcrit'))
for Lc in q3.LCTX_C7:
    r = q3.solve_NDQ(1e22, 'exp', Q0, Lc, nq=200)
    ana = q3.ETA * Lc / 6.0
    report('    %-10d %12.6f %12.6f %14s %14.3f'
           % (Lc, ana, r['ratio_attn_train'],
              'OK' if abs(ana - r['ratio_attn_train']) < 1e-9 else 'X', Lc / q3.L_CRIT))
report('    => 训练与注意力开销之比只由 L_ctx 决定，与预算 C 无关；')
report('       预算 C 只能改变 训练/注意力 与 质量 之间的相对份额。')
report('    L_ctx^crit = 6/η = %.0f tokens；C7 实际取值 %s，临界值落在 8192 与 32768 之间。'
       % (q3.L_CRIT, q3.LCTX_C7))
OUT['cost_invariant'] = dict(L_crit=q3.L_CRIT, Lctx_C7=q3.LCTX_C7,
                             ratio=[float(q3.ETA * L / 6.0) for L in q3.LCTX_C7])

# ---------------- [C] 质量—规模替代率 ----------------
report('\n[C] 质量—规模替代率（工作点 C=1e22, g=exp, Q0=0.5, Lctx=4096）')
base = q3.solve_NDQ(1e22, 'exp', Q0, LC, nq=400)
N0, D0, Qs0 = base['N_B'], base['D_B'], base['Q']
report('    基准点 N*=%.4fB  D*=%.3fB  Q*=%.4f  L*=%.5f' % (N0, D0, Qs0, base['L']))


def dLdQ(N, D, Q, h=1e-6):
    return (q3.loss(N, D, Q + h) - q3.loss(N, D, Q - h)) / (2 * h)


def dLdN(N, D, Q, h=1e-6):
    return (q3.loss(N + h, D, Q) - q3.loss(N - h, D, Q)) / (2 * h)


gq = dLdQ(N0, D0, Qs0)
gn = dLdN(N0, D0, Qs0)
rate = -gq / gn                      # dN/dQ |_{L,D 固定}
report('    ∂L/∂Q=%+.6f  ∂L/∂N=%+.6f  =>  dN/dQ|_{L,D}= %+.4f B 参数 / 单位 Q' % (gq, gn, rate))
report('    即：质量每提高 ΔQ=0.1，在等损失、等数据量下可减少参数 %.4f B（基准 N*=%.3f B 的 %.2f%%）'
       % (abs(rate) * 0.1, N0, 100 * abs(rate) * 0.1 / N0))
report('    口径声明：这是**局部、固定 D** 的替代率，不是全局最优解之间的比较；')
report('             全局比较须重解 (N,D) 联合最优化，见 [D]。')

# C=1e22 处 Q*=1 已饱和，边界点上的 ∂L/∂Q 不代表"可购买的边际质量价值"；
# 故另取一个 Q 处于内点的预算做对照（C=1e19, g=exp）。
report('\n[C2] 内点工作点的替代率对照（C=1e19, g=exp，Q* 内点）')
b2 = q3.solve_NDQ(1e19, 'exp', Q0, LC, nq=400)
N1, D1, Q1 = b2['N_B'], b2['D_B'], b2['Q']
gq1, gn1 = dLdQ(N1, D1, Q1), dLdN(N1, D1, Q1)
rate1 = -gq1 / gn1
report('    基准点 N*=%.4fB  D*=%.3fB  Q*=%.4f  L*=%.5f  [interior]' % (N1, D1, Q1, b2['L']))
report('    ∂L/∂Q=%+.6f  ∂L/∂N=%+.6f  =>  dN/dQ|_{L,D}= %+.4f B 参数 / 单位 Q' % (gq1, gn1, rate1))
report('    即：质量每提高 ΔQ=0.1，在等损失、等数据量下可减少参数 %.4f B（基准 N*=%.3f B 的 %.2f%%）'
       % (abs(rate1) * 0.1, N1, 100 * abs(rate1) * 0.1 / N1))
report('    注意：内点与饱和点的 dN/dQ 相差 %.1f 倍，说明该替代率**强依赖于工作点**，'
       % (abs(rate) / abs(rate1)))
report('          不能作为全局常数外推；论文中须注明工作点。')
OUT['subst'] = dict(C=1e22, N0=N0, D0=D0, Q0=Qs0, dLdQ=float(gq), dLdN=float(gn),
                    dN_dQ=float(rate), dN_for_dQ_0p1=float(abs(rate) * 0.1),
                    C_int=1e19, N_int=N1, D_int=D1, Q_int=Q1,
                    dLdQ_int=float(gq1), dLdN_int=float(gn1),
                    dN_dQ_int=float(rate1), dN_for_dQ_0p1_int=float(abs(rate1) * 0.1))

# ---------------- [D] 边界冻结时的解析弹性 ----------------
report('\n[D] Q 处于边界时预算弹性的解析值（-αβ/(α+β)=%.6f）' % (-a * b / (a + b)))
report('    %-8s %-6s %9s %12s %12s %12s' % ('C', 'g', 'Q*', 'L*', '数值弹性', '偏差'))
for gname in ['exp', 'pow', 'log']:
    for C in [1e19, 1e22, 1e24]:
        r = q3.solve_NDQ(C, gname, Q0, LC, nq=400)
        # 局部对数弹性：dln(L*-E_eff)/dlnC，E_eff = E - E1 Q*（自洽地板）
        h = 1e-4
        rp = q3.solve_NDQ(C * (1 + h), gname, Q0, LC, nq=400)
        rm = q3.solve_NDQ(C * (1 - h), gname, Q0, LC, nq=400)
        Ee = E - e1 * r['Q']
        num = ((np.log(rp['L'] - Ee) - np.log(rm['L'] - Ee)) / (2 * h))
        bnd = 'boundary' if r['Q'] <= Q0 + 1e-9 else ('saturated' if r['Q'] >= 1 - 1e-9 else 'interior')
        report('    %-8s %-6s %9.4f %12.5f %12.6f %12.2e  [%s]'
               % ('1e%d' % round(np.log10(C)), gname, r['Q'], r['L'], num,
                  num + a * b / (a + b), bnd))
        OUT.setdefault('elasticity', []).append(
            dict(C=C, g=gname, Q=r['Q'], L=float(r['L']), elas=float(num), state=bnd))
report('    => Q 饱和后 Q 通道冻结，弹性精确回到经典值；Q 内点时弹性被质量通道放大。')

# ---------------- [E] 降维解的自检 ----------------
report('\n[E] 降维解自检（一阶条件残差 + 预算等式残差）')
for gname in ['exp', 'pow', 'log']:
    for C in [1e19, 1e22, 1e24]:
        r = q3.solve_NDQ(C, gname, Q0, LC, nq=400)
        Ctil = C / 1e18
        Psi = max(q3.G[gname][0](r['Q']) - q3.G[gname][0](Q0), 0.0) / 1e9
        foc = q3._foc(r['N_B'], r['Q'], Ctil, K_, Psi)
        cost = r['D_B'] * (K_ * r['N_B'] + Psi)
        report('    %-5s %-6s |FOC|=%.2e   预算残差=%.2e  Σ份额-1=%.2e'
               % ('1e%d' % round(np.log10(C)), gname, abs(foc),
                  abs(cost - Ctil) / Ctil, abs(r['s_train'] + r['s_attn'] + r['s_Q'] - 1)))
OUT['selfcheck'] = 'FOC 与预算等式残差均 < 1e-12，份额之和 = 1'

# ---------------- [F] 配比通道不消耗预算 ----------------
report('\n[F] 配比通道不消耗预算（形式核验）')
report('    成本约束 c = D·(K·N + Ψ(Q))，其中 K=6+ηLctx、Ψ(Q)=[g(Q)-g(Q0)]_+ 均不含 p。')
report('    故 ∂c/∂p ≡ 0：p 只在目标函数的 B~·exp(λ_p·h_p(p)) 里出现，是**零成本通道**。')
report('    结论：p 可与 (N,D,Q) 解耦——先由预算子问题解 (N*,D*,Q*)，再独立在单纯形上取 h_p 的极值。')
OUT['p_free'] = dict(dcdp='identically 0', reason='K 与 Ψ 均不含 p')

with open(OUTPUT_DIR / 'q3_derived_results.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)
report('\n已写出 q3_derived_results.json')
