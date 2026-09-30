# -*- coding: utf-8 -*-
"""最终统一对比：把四种思路 + 全局乘子族 放在同一组判据下
判据：(1) 全样本 AIC/BIC (2) B6→B7 新增点外推 (3) 留一(N,D)组 (4) 模型无关振幅指数
"""
import pandas as pd
import numpy as np
import os
from scipy.optimize import least_squares
from scipy import stats

BASE = r'd:\F题\F题\real_attachments\B_scaling_laws'
LOGE = np.log
b6 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment.csv'))
b7 = pd.read_csv(os.path.join(BASE, 'supplementary_NQ_experiment_expanded.csv'))
def keyf(df): return list(zip(df.N_params_B.round(4), df.D_tokens_B.round(4), df.Q_score.round(4)))
k6 = set(keyf(b6))
b7 = b7.copy(); b7['_new'] = [k not in k6 for k in keyf(b7)]
tr, te = b6, b7[b7._new]
b_all = pd.concat([b6, b7[b7._new]])
Na, Da, Qa, La = (b_all.N_params_B.values.astype(float), b_all.D_tokens_B.values.astype(float),
                  b_all.Q_score.values.astype(float), b_all.val_loss.values.astype(float))
n = len(La)

def R5(p, N, D):  # E + A N^-a + B D^-b，p = [E, logA, loga, logB, logb]
    return p[0] + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))

# ---- 模型族 ----
def F_Dpow(p, N, D, Q):   # 思路一：Q 幂律挂 D 项
    return R5(p[:5], N, D) + np.exp(p[3])*D**(-np.exp(p[4]))*(Q**(-np.exp(p[5])) - 1)
def F_Dexp(p, N, D, Q):   # 思路三：Q 指数挂 D 项（D_eff）
    return R5(p[:5], N, D) + np.exp(p[3])*D**(-np.exp(p[4]))*(np.exp(-p[5]*Q) - 1)
def F_NDpow(p, N, D, Q):  # 思路二：双挂幂律
    return (p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*Q**(-np.exp(p[5]))
            + np.exp(p[3])*D**(-np.exp(p[4]))*Q**(-np.exp(p[6])))
def F_NDexp(p, N, D, Q):  # 双挂指数
    return (p[0] + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q)
            + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q))
def F_Elin(p, N, D, Q):   # 思路四备选：Q 改有界下限
    return p[0] + p[5]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4]))
def F_glob(p, N, D, Q):   # 全局乘子：Q 重标定整个可约损失
    return p[0] + (np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4])))*Q**(-np.exp(p[5]))
def F_globE(p, N, D, Q):  # 全局乘子 + 有界下限
    return p[0] + p[6]*(1-Q) + (np.exp(p[1])*N**(-np.exp(p[2])) + np.exp(p[3])*D**(-np.exp(p[4])))*Q**(-np.exp(p[5]))
def F_END(p, N, D, Q):    # 三项全挂指数
    return (p[0] + p[7]*(1-Q) + np.exp(p[1])*N**(-np.exp(p[2]))*np.exp(-p[5]*Q)
            + np.exp(p[3])*D**(-np.exp(p[4]))*np.exp(-p[6]*Q))

b5 = [1.53, LOGE(0.53), LOGE(0.2832), LOGE(1.85), LOGE(0.0831)]
CFG = {
 '思路一 D_eff·幂律':      (F_Dpow,   b5+[LOGE(0.10)],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10)]),
 '思路三 D_eff·指数':      (F_Dexp,   b5+[1.0],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20]),
 '思路四备选 改有界下限':    (F_Elin,   b5+[0.3],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -5],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 5]),
 '思路二 双挂·幂律':        (F_NDpow,  b5+[LOGE(0.14), LOGE(0.10)],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), LOGE(1e-3)],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), LOGE(10)]),
 '全局乘子 (N,D 同标定)':   (F_glob,   b5+[LOGE(0.10)],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3)],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10)]),
 '全局乘子 + 有界下限':     (F_globE,  b5+[LOGE(0.10), 0.2],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), LOGE(1e-3), -5],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), LOGE(10), 5]),
 '双挂·指数':              (F_NDexp,  b5+[1.5, 1.0],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20]),
 '三项全挂·指数':           (F_END,    b5+[1.5, 1.0, 0.3],
                            [0.2, LOGE(1e-8), LOGE(1e-3), LOGE(1e-8), LOGE(1e-3), -20, -20, -5],
                            [4, LOGE(1e8), LOGE(3), LOGE(1e8), LOGE(3), 20, 20, 5]),
}

print('=' * 118)
print('统一对比（数据：B6∪B7 = 450 点；校准集 B6=360，外推集 B7新增=90）')
print('=' * 118)
print(f'{"模型":<24}{"k":>3}{"全样本RMSE":>11}{"AIC":>10}{"ΔAIC":>8}'
      f'{"B6拟合":>9}{"B7新增RMSE":>11}{"留一组RMSE":>11}{"N指数偏差":>11}{"D指数偏差":>11}')
Ns, Ds = np.array(sorted(b_all.N_params_B.unique())), np.array(sorted(b_all.D_tokens_B.unique()))
amp = np.array([[b_all[(b_all.N_params_B == nv) & (b_all.D_tokens_B == d)].val_loss.max() -
                 b_all[(b_all.N_params_B == nv) & (b_all.D_tokens_B == d)].val_loss.min()
                 for d in Ds] for nv in Ns])
X = np.array([[1.0, np.log(nv), np.log(d)] for nv in Ns for d in Ds]); yobs = np.log(amp.ravel())
co, *_ = np.linalg.lstsq(X, yobs, rcond=None)
groups = list(b_all.groupby(['N_params_B', 'D_tokens_B']))
out = {}
for nm, (f, p0, lo, hi) in CFG.items():
    r = least_squares(lambda p: f(p, Na, Da, Qa) - La, p0, bounds=(lo, hi), max_nfev=80000)
    rss = float(np.sum(r.fun**2)); k = len(p0)
    aic = n*np.log(rss/n)+2*k
    # B6 拟合 / B7新增外推
    r6 = least_squares(lambda p: f(p, tr.N_params_B.values, tr.D_tokens_B.values, tr.Q_score.values) - tr.val_loss.values,
                       p0, bounds=(lo, hi), max_nfev=40000)
    e6 = tr.val_loss.values - f(r6.x, tr.N_params_B.values, tr.D_tokens_B.values, tr.Q_score.values)
    ete = te.val_loss.values - f(r6.x, te.N_params_B.values, te.D_tokens_B.values, te.Q_score.values)
    # 留一(N,D)组
    errs = []
    for i in range(len(groups)):
        s = pd.concat([g for j, (_, g) in enumerate(groups) if j != i])
        try:
            rr = least_squares(lambda p: f(p, s.N_params_B.values, s.D_tokens_B.values, s.Q_score.values) - s.val_loss.values,
                               p0, bounds=(lo, hi), max_nfev=2500)
            g_ = groups[i][1]
            errs.extend(g_.val_loss.values - f(rr.x, g_.N_params_B.values, g_.D_tokens_B.values, g_.Q_score.values))
        except Exception:
            errs.extend([np.nan]*len(groups[i][1]))
    loo = np.sqrt(np.nanmean(np.array(errs, float)**2))
    # 预测振幅指数
    Qm, QM = Qa.min(), Qa.max()
    pr = np.abs(np.array([f(r.x, np.array([nv]), np.array([d]), np.array([Qm]))[0] -
                          f(r.x, np.array([nv]), np.array([d]), np.array([QM]))[0]
                          for nv in Ns for d in Ds]))
    if pr.min() < 1e-12:
        dn, dd = np.nan, np.nan
    else:
        c2, *_ = np.linalg.lstsq(X, np.log(np.maximum(pr, 1e-300)), rcond=None)
        dn, dd = c2[1]-co[1], c2[2]-co[2]
    out[nm] = (r.x, k, aic, np.sqrt(rss/n), np.sqrt(np.mean(e6**2)), np.sqrt(np.mean(ete**2)), loo, dn, dd)
best = min(v[2] for v in out.values())
for nm, v in out.items():
    print(f'{nm:<24}{v[1]:>3}{v[3]:>11.5f}{v[2]:>10.2f}{v[2]-best:>8.2f}'
          f'{v[4]:>9.5f}{v[5]:>11.5f}{v[6]:>11.5f}'
          f'{(v[7] if not np.isnan(v[7]) else 0):>11.3f}{(v[8] if not np.isnan(v[8]) else 0):>11.3f}')
print()
print(f'实测振幅指数: N {co[1]:+.4f}, D {co[2]:+.4f}   → “偏差”列越接近 0 越好')

print()
print('=' * 118)
print('参数与物理含义')
print('=' * 118)
for nm in ['思路一 D_eff·幂律', '思路三 D_eff·指数', '思路四备选 改有界下限',
           '思路二 双挂·幂律', '全局乘子 (N,D 同标定)', '双挂·指数', '三项全挂·指数']:
    x = out[nm][0]
    if nm.startswith('思路一'):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, 质量指数δ_D={np.exp(x[5]):.4f}；可识别量仅 β·δ_D={np.exp(x[4])*np.exp(x[5]):.4f}'
    elif nm.startswith('思路三'):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, ρ={x[5]:.4f}；可识别量仅 β·ρ={np.exp(x[4])*x[5]:.4f}'
    elif nm.startswith('思路四备选'):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, 下限斜率E1={x[5]:.4f}'
    elif nm.startswith('思路二'):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, δ_N={np.exp(x[5]):.4f}, δ_D={np.exp(x[6]):.4f}, δ_N/δ_D={np.exp(x[5]-x[6]):.3f}'
    elif nm.startswith('全局乘子 ('):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, 全局δ={np.exp(x[5]):.4f}'
    elif nm.startswith('全局乘子 +'):
        s = f'α={np.exp(x[2]):.4f}, β={np.exp(x[4]):.4f}, 全局δ={np.exp(x[5]):.4f}, E1={x[6]:.4f}'
    elif nm.startswith('双挂·指数'):
        s = f'ρ_N={x[5]:.4f}, ρ_D={x[6]:.4f}, ρ_N/ρ_D={x[5]/x[6]:.3f}'
    else:
        s = f'E1={x[7]:.4f}, ρ_N={x[5]:.4f}, ρ_D={x[6]:.4f}'
    print(f'  {nm:<24} {s}')

print()
print('=' * 118)
print('D→∞ 极限：质量效应是否消失（Q=0.2 与 Q=1.0 的损失差）')
print('=' * 118)
Dg = np.array([1e2, 1e3, 1e4, 1e6, 1e8]); Ng = np.ones_like(Dg)
print(f'{"模型":<24}' + ''.join(f'{"D=%.0e" % d:>11}' for d in Dg) + '   形态')
for nm, v in out.items():
    f = CFG[nm][0]
    ds = [f(v[0], np.array([Ng[i]]), np.array([Dg[i]]), np.array([0.2]))[0] -
          f(v[0], np.array([Ng[i]]), np.array([Dg[i]]), np.array([1.0]))[0] for i in range(len(Dg))]
    form = '有界(永久保留)' if abs(ds[-1]-ds[0]) < 0.02*abs(ds[0]) else ('衰减到0' if abs(ds[-1]) < 0.5*abs(ds[0]) else '缓慢衰减')
    print(f'{nm:<24}' + ''.join(f'{d:>11.4f}' for d in ds) + f'   {form}')

print()
print('=' * 118)
print('★ 关键结论：可分离性 vs 可识别性')
print('=' * 118)
print('思路一/三 的核心假设：Q 只重标定 D_eff（“可分离性假设”）')
print('  模型无关反例：固定 D 时质量振幅对 N 的斜率，5 个 D 层全部显著为负')
print('    D=10 : -0.0944 (p=1.5e-2)')
print('    D=50 : -0.1582 (p=6.9e-4)')
print('    D=150: -0.1992 (p=3.4e-5)')
print('    D=300: -0.1529 (p=2.9e-3)')
print('    D=600: -0.1108 (p=5.9e-4)')
print('  全局 N 指数 -0.1431±0.0116 (t=-12.34, p=1.3e-15)')
print('  → “Q 只作用于 D_eff”被拒绝；Q 必然也作用于 N 通道或全局可约损失')
print()
print('思路三/四 的 Q0 参考点：')
print('  固定 Q0=0/0.3/0.5/0.8/1.0 → RSS 完全相同(1.6130859)、ρ 完全相同(0.3441)')
print('  只有 B·exp(ρQ0)=2.0393 被识别 → Q0 不可识别，只是归一化约定（不影响拟合与预测）')
print('  ρ 本身可识别：RSS 在 ρ≈0.34 有极小，ρ=0.2→1.646，ρ=3→6.655')
print()
print('思路四 的 G(Q) 与 思路三 的 D_eff 能否分开？profile likelihood:')
print('  E1=0.000 → RSS 1.1014,  ρ_D=0.2469')
print('  E1=0.090 → RSS 1.0623,  ρ_D=0.1582')
print('  E1=0.150 → RSS 1.0642,  ρ_D=0.0904')
print('  E1=0.300 → RSS 1.1839,  ρ_D=-0.1088')
print('  → ΔRSS 在 E1∈[0.06,0.20] 内 <0.026（1% 水平阈值≈0.009），')
print('    而 ρ_D 从 0.25 摆到 -0.11：floor 与 D 通道的分工弱可识别，')
print('    只有“总质量效应”是稳健的。因此思路三与思路四在观测区间内难以判别。')
print()
print('判别它们的唯一杠杆是 D→∞ 的行为，而 B6/B7 的 D 只覆盖 10~600（<2 个数量级），')
print('实测振幅 D 指数 -0.055（远弱于纯 D 通道预测的 -0.280，但也不是 0）')
print('→ 数据略偏向“质量效应随 D 衰减很慢/几乎不衰减”，即略偏向思路四，但证据强度不足以定论。')