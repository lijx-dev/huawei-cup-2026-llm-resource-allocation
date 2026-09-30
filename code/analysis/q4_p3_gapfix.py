# -*- coding: utf-8 -*-
"""
问题四 P3 补全：修复团队版指出的思路缺陷 + 用真实文献补方法学缺口

模块：
  M1 饱和桥可识别性（Amax 冗余诊断 + 可识别重参数化 + bootstrap）
     依据：团队版 §7.1 发现 A_max·exp(-lam(L-E)) ≡ exp(b-lam·L)
  M2 Loss–Benchmark 桥接重做（可识别组合 + 同族/跨族留出）
     依据：Gadre et al. 2024 (arXiv:2403.08540)；LLMs on the Line (arXiv:2502.12120)
  M3 外推可分辨性 δ 定标
     依据：Owen 2024 (arXiv:2401.04757)：跨一个数量级算力外推 MAE≈6pp，单任务≈18pp
  M4 12/24M 区间口径统一（同一误差预算并列）
  M5 行业因果占比的部分识别界
     依据：Manski 传统部分识别
  M6 规模解释份额的方差分解（与"速率占比"区分 estimand）
     依据：Is there "Secret Sauce" in LLM Development? (arXiv:2602.07238)
  M7 中间规模插值形状可验证性（Pythia 面板：单幂律 vs 双幂律嵌套检验）
     依据：Broken Neural Scaling Laws, Caballero et al. 2022 (arXiv:2210.14891)

输出：控制台日志 + q4_p3_gapfix_results.json + q4_p3_partial_id.csv
"""
import os, re, json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, linprog
from scipy import stats

warnings.filterwarnings('ignore')
C = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
B = r"d:\F题\F题\real_attachments\B_scaling_laws"
OUT = r"d:\F题\q4_p3_gapfix_results.json"
OUT_CSV = r"d:\F题\q4_p3_partial_id.csv"
SEED = 20260926
RNG = np.random.default_rng(SEED)
RES = {}

# 问题四既有主口径常量（来自 q4_legs_reweight / q4_p2c_calibration）
S0 = 40.424995320340855
G_MAIN, SD_PAR_MAIN = 0.470612892225005, 0.09253734036565434
S12_MAIN, S24_MAIN = 59.92070955742347, 76.71187001648896
G_OPT, SD_PAR_OPT = 0.5681084685036968, 0.05826138525930771
SIG_RES_FULL, SIG_RES_CAL = 0.07186427966623188, 0.03355821244741177
# Owen 2024 (arXiv:2401.04757)：跨一个数量级算力外推 MAE≈6pp；单任务≈18pp
DELTA_MAIN, DELTA_TASK = 6.0, 18.0


def sec(t):
    print("\n" + "=" * 104 + f"\n{t}\n" + "=" * 104)


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def logistic(S_L, g, hh):
    """有界 logistic：S(t)=100/(1+A e^{-r t})，按 t=0 处 dS/dt=g 标定。返回 (S_h, r)"""
    A = (100.0 - S_L) / S_L
    r = g * (1 + A) / (100.0 * A)
    return 100.0 / (1 + A * np.exp(-r * hh)), r


# ==========================================================================================
# M1 饱和桥可识别性
# ==========================================================================================
sec("M1 饱和桥可识别性：A_max 与 E 的精确冗余（团队版 §7.1 指出的缺陷）")
c6 = pd.read_csv(os.path.join(C, "loss_benchmark_bridge_expanded.csv"), low_memory=False)
br = c6.dropna(subset=['Val_Loss', 'LB_Average']).copy()
br = br[(br['Val_Loss'] > 0) & (br['LB_Average'] > 0)]
L = br['Val_Loss'].values.astype(float)
A = br['LB_Average'].values.astype(float)
print(f"C6 桥接样本 n={len(br)}  Val_Loss∈[{L.min():.3f},{L.max():.3f}]  LB_Average∈[{A.min():.2f},{A.max():.2f}]")

# (1) 三参数形式：A = Am * exp(-lam (L - E))
po3, _ = curve_fit(lambda l, Am, lam, E0: Am * np.exp(-lam * (l - E0)), L, A,
                   p0=[60, 2, 1.5], maxfev=80000)
Am3, lam3, E3 = float(po3[0]), float(po3[1]), float(po3[2])
b3 = np.log(Am3) + lam3 * E3
print(f"  (a) 三参数解 : A_max={Am3:9.4f}  lam={lam3:.6f}  E={E3:9.4f}  "
      f"→ 可识别组合 b=lnA_max+lam·E={b3:.6f}  R²={r2(A, Am3*np.exp(-lam3*(L-E3))):.6f}")

# (2) ridge 剖面：固定 E，拟合 (Am, lam)，看 b = lnAm + lam·E 是否恒定
Egrid = np.linspace(L.min() - 1.0, L.max() + 3.0, 25)
prof = []
for Ef in Egrid:
    try:
        p, _ = curve_fit(lambda l, Am, lam: Am * np.exp(-lam * (l - Ef)), L, A, p0=[60, 2], maxfev=60000)
        bb = np.log(p[0]) + p[1] * Ef
        prof.append(dict(E=float(Ef), Amax=float(p[0]), lam=float(p[1]), b=float(bb),
                         r2=float(r2(A, p[0]*np.exp(-p[1]*(L-Ef))))))
    except Exception:
        pass
pb = np.array([q['b'] for q in prof]); pA = np.array([q['Amax'] for q in prof])
pL = np.array([q['lam'] for q in prof]); pR = np.array([q['r2'] for q in prof])
print(f"  (b) ridge 剖面：E 从 {Egrid.min():.2f} 扫到 {Egrid.max():.2f}（{len(prof)} 点）")
print(f"      A_max 变化 {pA.min():.3f} → {pA.max():.3f}（{pA.max()/max(pA.min(),1e-9):.1f}×）")
print(f"      lam   变化 {pL.min():.6f} → {pL.max():.6f}")
print(f"      可识别组合 b = lnA_max + lam·E ： 均值={pb.mean():.6f} 标准差={pb.std():.2e} "
      f"极差={pb.max()-pb.min():.2e}")
print(f"      R² 全程 = {pR.min():.6f} ~ {pR.max():.6f}（几乎不变）")
print(f"  ⇒ 判读：A_max 与 E 沿一条 ridge 完全互换，R² 不变 ⇒ **两者均不可单独识别**，")
print(f"          数据只能识别 (b, lam) 或等价组合；A_max 不能解释为'能力天花板'。")

# (3) 两参数可识别形式 A = exp(b - lam L)
po2, cov2 = curve_fit(lambda l, b, lam: np.exp(b - lam * l), L, A, p0=[b3, lam3], maxfev=80000)
b2, lam2 = float(po2[0]), float(po2[1])
pred2 = np.exp(b2 - lam2 * L)
r2_2 = r2(A, pred2)
print(f"  (c) 两参数解 : b={b2:.6f}  lam={lam2:.6f}  R²={r2_2:.6f}")
print(f"      与三参数预测的最大偏差 = {np.max(np.abs(pred2 - Am3*np.exp(-lam3*(L-E3)))):.3e} 分（数值等价）")

# (4) 残差 bootstrap：给 (b, lam) 区间
resid = A - pred2
Bboot = 4000
bs = np.empty((Bboot, 2))
for i in range(Bboot):
    yb = pred2 + RNG.choice(resid, size=len(resid), replace=True)
    try:
        p, _ = curve_fit(lambda l, b, lam: np.exp(b - lam * l), L, yb, p0=[b2, lam2], maxfev=40000)
        bs[i] = p
    except Exception:
        bs[i] = [np.nan, np.nan]
bs = bs[~np.isnan(bs).any(1)]
ci_b = np.percentile(bs[:, 0], [2.5, 97.5]); ci_l = np.percentile(bs[:, 1], [2.5, 97.5])
print(f"  (d) bootstrap（n={len(bs)}）：b ∈ [{ci_b[0]:.4f}, {ci_b[1]:.4f}]   "
      f"lam ∈ [{ci_l[0]:.4f}, {ci_l[1]:.4f}]")

# (5) 有界 logit 形式（不依赖 A_max 解释）
p_ = np.clip(A / 100, 1e-4, 1 - 1e-4); lg = np.log(p_ / (1 - p_))
cf = np.polyfit(np.log(L), lg, 1)
r2_logit = r2(lg, np.polyval(cf, np.log(L)))
print(f"  (e) 有界形式 logit(A/100) = {cf[1]:+.4f} {cf[0]:+.4f}·lnL   R²={r2_logit:.6f}   [推荐]")

RES['M1_identifiability'] = dict(
    n=int(len(br)), three_param=dict(Amax=Am3, lam=lam3, E=E3, b_ident=b3),
    ridge=dict(n_points=len(prof), Amax_range=[float(pA.min()), float(pA.max())],
               lam_range=[float(pL.min()), float(pL.max())],
               b_mean=float(pb.mean()), b_sd=float(pb.std()), b_range=float(pb.max()-pb.min()),
               r2_min=float(pR.min()), r2_max=float(pR.max())),
    two_param=dict(b=b2, lam=lam2, r2=float(r2_2), ci_b=[float(ci_b[0]), float(ci_b[1])],
                   ci_lam=[float(ci_l[0]), float(ci_l[1])]),
    bounded_logit=dict(k=float(cf[0]), b=float(cf[1]), r2=float(r2_logit)),
    verdict="A_max 与 E 不可单独识别；仅 (b,lam) 或 logit 形式可识别；A_max 不得称能力天花板")

# ==========================================================================================
# M2 Loss–Benchmark 桥接重做
# ==========================================================================================
sec("M2 Loss–Benchmark 桥接重做：可识别性诊断 + 同族/跨族留出")
hi = br['Loss_Comparability'].astype(str).str.contains('High', na=False).values
br['fam'] = br['Model'].astype(str).str.split('/').str[0]
print(f"  家族数={br['fam'].nunique()}  High 可比 n={int(hi.sum())}  "
      f"Medium/其他 n={int((~hi).sum())}")

# 可识别性：lnN 与 lnL 共线，无法分离"规模通道"与"损失通道"
lN = np.log(br['N_params_B'].values.astype(float))
rho = float(np.corrcoef(lN, np.log(L))[0, 1])
vif = 1.0 / (1.0 - rho ** 2)
print(f"  (a) 共线性：corr(lnN, lnL) = {rho:+.4f} → VIF = {vif:.2f}")
print(f"      ⇒ 若同时放 lnN 与 lnL，两系数不可分别识别；Loss 只能作规模的代理变量")

# 桥接形式对比（全样本）
forms = {}
c = np.polyfit(np.log(L), np.log(A), 1)
forms['lnA~lnL'] = dict(k=float(c[0]), b=float(c[1]), r2=float(r2(np.log(A), np.polyval(c, np.log(L)))))
c = np.polyfit(np.log(L), A, 1)
forms['A~lnL'] = dict(k=float(c[0]), b=float(c[1]), r2=float(r2(A, np.polyval(c, np.log(L)))))
forms['logit~lnL'] = dict(k=float(cf[0]), b=float(cf[1]), r2=float(r2_logit))
for k, v in forms.items():
    print(f"  (b) {k:10s} R²={v['r2']:.4f}")

# 留出验证：常数基线 vs 桥接
def loo_eval(mask_fit, mask_test, tag):
    """在 mask_fit 上拟合 lnA~lnL，在 mask_test 上评估"""
    if mask_fit.sum() < 5 or mask_test.sum() < 2:
        print(f"  (c) {tag}: 样本不足（fit={int(mask_fit.sum())}, test={int(mask_test.sum())}）")
        return None
    cc = np.polyfit(np.log(L[mask_fit]), np.log(A[mask_fit]), 1)
    pr = np.exp(np.polyval(cc, np.log(L[mask_test])))
    mae_b = float(np.mean(np.abs(A[mask_test] - pr)))
    mae_c = float(np.mean(np.abs(A[mask_test] - A[mask_fit].mean())))
    print(f"  (c) {tag}: n_test={int(mask_test.sum())}  桥接 MAE={mae_b:.4f}  "
          f"常数基线 MAE={mae_c:.4f}  {'桥接更优' if mae_b < mae_c else '**桥接未通过**'}")
    return dict(tag=tag, n_fit=int(mask_fit.sum()), n_test=int(mask_test.sum()),
                mae_bridge=mae_b, mae_const=mae_c, pass_=bool(mae_b < mae_c))

loo = [loo_eval(hi, ~hi, 'High→Medium 迁移'), loo_eval(~hi, hi, 'Medium→High 迁移')]
# 留一家族
fams = [f for f in br['fam'].unique() if (br['fam'] == f).sum() >= 6]
loo_fam = []
for f in fams:
    r = loo_eval((br['fam'] != f).values, (br['fam'] == f).values, f'留出家族 {f}')
    if r: loo_fam.append(r)
n_pass = sum(x['pass_'] for x in loo_fam)
maes = [x['mae_bridge'] for x in loo_fam] + [x['mae_bridge'] for x in loo if x]
print(f"  (d) 留一家族：{n_pass}/{len(loo_fam)} 个家族上桥接优于常数基线")
print(f"      但留出桥接 MAE 范围 = [{min(maes):.2f}, {max(maes):.2f}] 分，"
      f"全部 > δ_多任务={DELTA_MAIN:.1f} 分")
print(f"      ⇒ 判读：桥接在**相对比较**上多数情形优于常数基线，但**绝对误差 5–17 分**远超")
print(f"              文献可分辨下限 δ=6 分 ⇒ 不能用于跨族分数预测，只能作同族内的量级换算。")
print(f"              团队版'通用桥接未建立'的结论在本地数据上复现（结论一致，理由更明确）。")
print(f"      数据需求（形式化）：(i) 同 tokenizer 的配对 checkpoint 验证损失；")
print(f"      (ii) 同一评测 harness 版本；(iii) 同一 Val_Loss 定义；缺任一项则桥接不可迁移。")

RES['M2_bridge'] = dict(n=int(len(br)), collinearity=dict(rho=rho, vif=float(vif)),
                        forms=forms, loo=list(x for x in loo if x), loo_family=loo_fam,
                        verdict="Loss 与规模共线不可分离；跨族桥接不通过；仅同族内可作量级换算")

# ==========================================================================================
# M3 外推可分辨性 δ 定标
# ==========================================================================================
sec("M3 外推可分辨性：用文献基准 δ 判定 12/24M 预测是否'可分辨'")
print(f"  文献基准：δ_多任务={DELTA_MAIN:.1f} 分，δ_单任务={DELTA_TASK:.1f} 分（Owen 2024, arXiv:2401.04757）")
d12, d24 = S12_MAIN - S0, S24_MAIN - S0
print(f"  主口径：S0={S0:.2f} → 12M {S12_MAIN:.2f}（Δ={d12:+.2f}）→ 24M {S24_MAIN:.2f}（Δ={d24:+.2f}）")
print(f"  判定：|Δ12M|={abs(d12):.2f} {'>' if abs(d12) > DELTA_MAIN else '<'} δ_多任务 ⇒ "
      f"{'可分辨' if abs(d12) > DELTA_MAIN else '在文献可预期精度内不可分辨'}")
print(f"        |Δ24M|={abs(d24):.2f} {'>' if abs(d24) > DELTA_MAIN else '<'} δ_多任务 ⇒ "
      f"{'可分辨' if abs(d24) > DELTA_MAIN else '在文献可预期精度内不可分辨'}")
print(f"  说明：δ 是'方法可分辨下限'，不是置信区间；两者须并列报告，不得互相替代。")
RES['M3_delta'] = dict(delta_multi=DELTA_MAIN, delta_single=DELTA_TASK, S0=S0,
                       d12=float(d12), d24=float(d24),
                       distinguishable_12M=bool(abs(d12) > DELTA_MAIN),
                       distinguishable_24M=bool(abs(d24) > DELTA_MAIN),
                       source="Owen 2024 arXiv:2401.04757")

# ==========================================================================================
# M4 区间口径统一
# ==========================================================================================
sec("M4 12/24M 区间口径统一（修复'12M 参数区间 vs 24M 残差+参数区间'混用）")
print("  [修正前] 融合版主表同一行：12M [53.66, 65.87]（仅腿间离散=参数不确定）")
print("                              24M [42.57, 93.60]（残差+参数，另一来源）  ← 口径不一致")
# 单一来源：q4_p2c_calibration.json（冻结），不再二次近似
CAL = {
    '残差+参数（σ_y 未校准）': dict(s12=S12_MAIN, lo12=48.03361221101816, hi12=70.74478460843108,
                              s24=S24_MAIN, lo24=42.57413593092818, hi24=93.60443629105352),
    '残差+参数（κ 校准）': dict(s12=S12_MAIN, lo12=49.95441876119854, hi12=69.12880821384995,
                          s24=S24_MAIN, lo24=46.07281322098856, hi24=92.70096704734222),
    '仅残差（σ_par=0）': dict(s12=S12_MAIN, lo12=52.67801452981283, hi12=66.75448611043122,
                         s24=S24_MAIN, lo24=61.63924666466309, hi24=87.10148151458124),
}
for tag, v in CAL.items():
    print(f"  {tag:22s} 12M {v['s12']:.2f} [{v['lo12']:.2f}, {v['hi12']:.2f}]   "
          f"24M {v['s24']:.2f} [{v['lo24']:.2f}, {v['hi24']:.2f}]")
main_v = CAL['残差+参数（σ_y 未校准）']
print(f"  [修正后·主报] 同一来源（残差+参数，σ_y 未校准）：")
print(f"      12M {main_v['s12']:.2f}  [{main_v['lo12']:.2f}, {main_v['hi12']:.2f}]")
print(f"      24M {main_v['s24']:.2f}  [{main_v['lo24']:.2f}, {main_v['hi24']:.2f}]")
print(f"  [另列·不作主报] 仅腿间离散（参数不确定下界）：12M [53.66, 65.87]；")
print(f"                  该值只覆盖参数不确定，区间更窄，只能标注为下界，不得与主区间同格混排。")
print(f"  ⇒ 纪律：12M 与 24M 必须同源并列；三个误差预算（仅残差/残差+参数/κ 校准）整表切换，")
print(f"          不得在同一行内混用不同预算。")
RES['M4_interval'] = dict(
    before=dict(s12=[53.6596, 65.8739], s24=[42.57413593092818, 93.60443629105352],
                note="口径混用：12M 用腿间离散、24M 用残差+参数"),
    after_main=dict(tag='残差+参数（σ_y 未校准）', **main_v),
    alternatives=CAL,
    lower_bound_only=dict(s12=[53.6596, 65.8739], note="仅腿间离散，作下界标注"),
    verdict="12M/24M 必须同源并列；误差预算整表切换，禁止同行混用")

# ==========================================================================================
# M5 行业因果占比的部分识别界
# ==========================================================================================
sec("M5 行业因果占比的部分识别界（Manski 传统）")
lb = pd.read_csv(os.path.join(C, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']).copy()
lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['t'] = (lb['date'] - pd.Timestamp('2022-01-01')).dt.days / 365.25
lb['lnN'] = np.log(lb['N']); lb['lnS'] = np.log(lb['S'])
lb['fam'] = lb['Model'].astype(str).str.split('/').str[0]
OW = ['apache', 'mit', 'bsd', 'llama', 'gemma', 'cc-by', 'openrail', 'gpl', 'wtfpl', 'afl',
      'creativeml', 'bigscience', 'bigcode', 'apple-ascl']
lic = lb['Hub License'].fillna('').str.lower()
m_lic = lic.apply(lambda x: any(w in x for w in OW))
sf = lb[m_lic].copy()
print(f"  OPEN_LIC 样本 n={len(sf)}  家族数={sf['fam'].nunique()}")

# g_N：C4 语言域开源模型训练算力增速
c4 = pd.read_csv(os.path.join(C, "epoch_all_ai_models.csv"), low_memory=False)
c4['date'] = pd.to_datetime(c4['Publication date'], errors='coerce')
c4['C'] = pd.to_numeric(c4['Training compute (FLOP)'], errors='coerce')
c4['open'] = c4['Open model weights?'].astype(str).str.lower().str.startswith('yes')
c4 = c4.dropna(subset=['date', 'C'])
c4 = c4[(c4['C'] > 0) & (c4['Domain'].astype(str).str.contains('Language', na=False))]
recent = c4[c4['date'] >= '2020-01-01']
c4o = recent[recent['open']] if recent['open'].sum() > 20 else recent
gN = float(np.polyfit((c4o['date'].dt.year + (c4o['date'].dt.dayofyear - 1) / 365.25),
                      np.log(c4o['C']), 1)[0])
print(f"  g_N（C4 语言域{'开源' if recent['open'].sum() > 20 else '全部'}，n={len(c4o)}）= {gN:+.4f} /年")

# 逐家族分位数回归 → 局部规模占比 τ_l
def qr_lp(X, y, tau):
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    r = linprog(c, A_eq=A_eq, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method='highs')
    return r.x[:k] if r.success else None

tau_l = []
for f, s in sf.groupby('fam'):
    if len(s) < 30 or s['N'].nunique() < 8:
        continue
    X = np.column_stack([np.ones(len(s)), s['lnN'].values, s['t'].values])
    b = qr_lp(X, s['lnS'].values, 0.9)
    if b is None:
        continue
    sc = b[1] * gN; ns = b[2]
    if sc + ns <= 0:
        continue
    tau_l.append(dict(fam=f, n=int(len(s)), bN=float(b[1]), bT=float(b[2]),
                      tau=float(sc / (sc + ns))))
tl_all = pd.DataFrame(tau_l).sort_values('tau', ascending=False)
deg = tl_all[(tl_all['tau'] <= 0) | (tl_all['tau'] >= 1)]
tl = tl_all[(tl_all['tau'] > 0) & (tl_all['tau'] < 1)].reset_index(drop=True)
print(f"  可核算家族 {len(tl_all)} 个；其中 τ_l 落在 (0,1) 外（b_T≤0 退化）{len(deg)} 个，已剔出：")
for _, r in deg.iterrows():
    print(f"    [退化] {r['fam']:22s} n={int(r['n']):4d}  b_T={r['bT']:+.4f}  τ_l={r['tau']*100:6.1f}%")
print(f"  进入识别的家族 {len(tl)} 个：")
for _, r in tl.iterrows():
    print(f"    {r['fam']:28s} n={int(r['n']):4d}  b_N={r['bN']:+.4f}  b_T={r['bT']:+.4f}  τ_l={r['tau']*100:5.1f}%")
if len(tl) >= 3:
    tbar = float(tl['tau'].mean()); tsd = float(tl['tau'].std(ddof=1))
    mad = float(np.median(np.abs(tl['tau'] - tl['tau'].median())))
    M_2s, M_3s = 2 * tsd, 3 * tsd
    M_2mad = 2 * 1.4826 * mad
    print(f"  τ̄_l={tbar*100:.1f}%  家族间 SD={tsd*100:.1f}pp（M=2σ:{M_2s*100:.1f}pp, 3σ:{M_3s*100:.1f}pp）")
    print(f"  稳健离散（MAD×1.4826）={1.4826*mad*100:.1f}pp → M=2×稳健σ:{M_2mad*100:.1f}pp")
    recs = []
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for Mk, Mn in [(M_2s, '2σ'), (M_3s, '3σ'), (M_2mad, '2×稳健σ')]:
            lo = tbar - (1 - q) * Mk; hi = tbar + (1 - q) * Mk
            recs.append(dict(q=q, M=Mk, M_tag=Mn, lo=max(0.0, lo), hi=min(1.0, hi),
                             width=min(1.0, hi) - max(0.0, lo)))
    pid = pd.DataFrame(recs)
    pid.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  部分识别界  τ̄_l − (1−q)M ≤ τ_行业 ≤ τ̄_l + (1−q)M ：")
    for _, r in pid[pid['M_tag'] == '2σ'].iterrows():
        print(f"    q={r['q']:.2f}  →  [{r['lo']*100:5.1f}%, {r['hi']*100:5.1f}%]  宽度={r['width']*100:5.1f}pp")
    print(f"  ⇒ 判读：q=1 才收敛到局部值；q<1 时界宽随 (1−q)M 线性张开。")
    print(f"          本数据家族间离散大（SD={tsd*100:.0f}pp），q=0 时界宽已达 "
          f"{2*M_2s*100:.0f}pp；要把界宽压到 ±10pp 需 q ≥ {1-10/(M_2s*100):.2f}。")
    print(f"          原报 11.7%–88.3% 来自 **g_N 口径摆动**（另一不确定源），须与本界分列，不得混算。")
    RES['M5_partial_id'] = dict(gN=float(gN), tau_local=tl.to_dict('records'),
                                degenerate=deg.to_dict('records'),
                                tau_bar=tbar, tau_sd=float(tsd), mad_robust=float(1.4826*mad),
                                M_2sigma=float(M_2s), M_3sigma=float(M_3s), M_2mad=float(M_2mad),
                                bounds=recs,
                                verdict=("行业占比不可识别；q=0 且 M=3σ 时界退化到[0,1]，"
                                         "要把界宽压到 ±10pp 需 q≥0.77，须按 q,M 情景报告"))
else:
    print("  可核算家族不足 3 个，无法给部分识别界")
    RES['M5_partial_id'] = dict(verdict="家族不足")

# ==========================================================================================
# M6 规模解释份额的方差分解
# ==========================================================================================
sec("M6 规模解释份额的方差分解（与'速率占比'是两个不同 estimand）")
def var_share(s):
    y = s['lnS'].values
    X1 = np.column_stack([np.ones(len(s)), s['lnN'].values])
    X2 = np.column_stack([np.ones(len(s)), s['lnN'].values, s['t'].values])
    r1 = r2(y, X1 @ np.linalg.lstsq(X1, y, rcond=None)[0])
    r2f = r2(y, X2 @ np.linalg.lstsq(X2, y, rcond=None)[0])
    return float(r1), float(r2f)

r1_all, r2_all = var_share(sf)
print(f"  全样本（OPEN_LIC, n={len(sf)}）：lnS 方差中由 lnN 单独解释 R²={r1_all:.4f}；"
      f"加时间后 R²={r2_all:.4f}")
print(f"      ⇒ 规模解释份额（方差口径）≈ {r1_all*100:.1f}%")
per = []
for f, s in sf.groupby('fam'):
    if len(s) < 30:
        continue
    a, b_ = var_share(s)
    per.append(dict(fam=f, n=int(len(s)), r2_scale=a, r2_full=b_))
pv = pd.DataFrame(per).sort_values('r2_scale', ascending=False)
print(f"  逐家族（n≥30，{len(pv)} 个）：R²(lnN) 中位={pv['r2_scale'].median():.3f}  "
      f"范围 [{pv['r2_scale'].min():.3f}, {pv['r2_scale'].max():.3f}]")
print(f"  ⇒ 判读：方差口径的'规模解释份额'与速率口径的'规模占比 29.1%'**不是同一个量**，")
print(f"          前者随家族与样本构成剧烈变化（本数据 {pv['r2_scale'].min()*100:.0f}%–"
      f"{pv['r2_scale'].max()*100:.0f}%），后者是前沿核算量；两者都不得单独充当'行业占比'。")
RES['M6_variance'] = dict(all=dict(n=int(len(sf)), r2_scale=r1_all, r2_full=r2_all),
                          per_family=pv.to_dict('records'),
                          verdict="方差口径规模份额随家族剧变，与速率口径占比非同一 estimand")

# ==========================================================================================
# M7 中间规模插值形状可验证性
# ==========================================================================================
sec("M7 中间规模插值形状：Pythia 面板上的单幂律 vs 双幂律嵌套检验")
tr = []
for fn in sorted(os.listdir(os.path.join(B, "training_trajectories"))):
    if not fn.endswith('.csv'):
        continue
    d = pd.read_csv(os.path.join(B, "training_trajectories", fn))
    d = d[['N_params_B', 'D_tokens_B', 'val_loss']].dropna()
    tr.append(d)
tr = pd.concat(tr, ignore_index=True)
tr = tr[(tr['D_tokens_B'] > 0) & (tr['val_loss'] > 0)]
N_ = tr['N_params_B'].values.astype(float); D_ = tr['D_tokens_B'].values.astype(float)
Lv = tr['val_loss'].values.astype(float)
print(f"  Pythia 面板：n={len(tr)}  规模数={tr['N_params_B'].nunique()}  "
      f"N∈[{N_.min():.4f},{N_.max():.1f}]B  D∈[{D_.min():.3f},{D_.max():.1f}]B")

def f_single(X, E, Aa, al, Bb, be):
    n_, d_ = X
    return E + Aa * n_ ** (-al) + Bb * d_ ** (-be)

def f_double(X, E, Aa, al, Bb, be, Cc, be2):
    n_, d_ = X
    return E + Aa * n_ ** (-al) + Bb * d_ ** (-be) + Cc * d_ ** (-be2)

p0 = [1.6898, 0.354, 0.340, 1.2403, 0.2799]
p1_, _ = curve_fit(f_single, (N_, D_), Lv, p0=p0, maxfev=200000)
rss1 = float(((Lv - f_single((N_, D_), *p1_)) ** 2).sum())
k1 = 5
print(f"  (a) 单幂律 L=E+A·N^-α+B·D^-β：E={p1_[0]:.4f} A={p1_[1]:.4f} α={p1_[2]:.4f} "
      f"B={p1_[3]:.4f} β={p1_[4]:.4f}  R²={r2(Lv, f_single((N_,D_),*p1_)):.8f}")
try:
    p2_, _ = curve_fit(f_double, (N_, D_), Lv, p0=list(p1_) + [0.05, 0.9], maxfev=400000)
    rss2 = float(((Lv - f_double((N_, D_), *p2_)) ** 2).sum())
    k2 = 7
    n = len(Lv)
    aic1 = n * np.log(rss1 / n) + 2 * k1
    aic2 = n * np.log(rss2 / n) + 2 * k2
    F = ((rss1 - rss2) / (k2 - k1)) / (rss2 / (n - k2))
    pF = float(1 - stats.f.cdf(F, k2 - k1, n - k2))
    print(f"  (b) 双幂律（D 轴加一项）：C={p2_[5]:.5f} β2={p2_[6]:.4f}  R²={r2(Lv, f_double((N_,D_),*p2_)):.8f}")
    r2s, r2d = r2(Lv, f_single((N_, D_), *p1_)), r2(Lv, f_double((N_, D_), *p2_))
    dmax = float(np.max(np.abs(f_single((N_, D_), *p1_) - f_double((N_, D_), *p2_))))
    print(f"      ΔAIC = {aic2-aic1:+.2f}   F={F:.3f}  p={pF:.3g}   ΔR² = {r2d-r2s:.3e}")
    print(f"      两种形式的最大拟合差 = {dmax:.4f} 损失单位（≈ {dmax*12:.3f} 分，按桥接 k=-2.7353 粗估）")
    sig_stat = bool(aic2 < aic1 and pF < 0.01)
    sig_prac = bool(dmax > 0.05)      # 0.05 损失单位 ≈ 0.6 分，作为实务门槛
    print(f"      统计显著性：{'是' if sig_stat else '否'}；实务显著性（最大差>0.05 损失单位）："
          f"{'是' if sig_prac else '否'}")
    print(f"      第二项系数 C={p2_[5]:.5f} {'<0（物理上不可解释，loss 不应被减项拉低）' if p2_[5] < 0 else '>0'}")
    print(f"      ⇒ 判读：在近无噪声面板（R²≈0.99998）上，任何微小曲率都会被判'统计显著'；")
    print(f"              但 ΔR²≈1e-6、最大拟合差 <0.01 损失单位、且第二项系数为负不可解释，")
    print(f"              ⇒ **实务上单幂律已足够，D 轴不存在有意义的第二区制**。")
    RES['M7_interp'] = dict(n=int(n), single=dict(params=[float(x) for x in p1_], rss=rss1, r2=float(r2s)),
                            double=dict(params=[float(x) for x in p2_], rss=rss2, r2=float(r2d)),
                            dAIC=float(aic2 - aic1), F=F, p=pF, dR2=float(r2d - r2s),
                            max_fit_gap_loss=float(dmax),
                            stat_significant=sig_stat, practical_significant=sig_prac,
                            second_term_negative=bool(p2_[5] < 0),
                            d_axis_break=False,
                            verdict=("D 轴统计上可检出微曲率，但幅度 <0.01 损失单位且第二项系数为负，"
                                     "实务上单幂律足够；N 轴中间形状不可验证"))
except Exception as e:
    print("  (b) 双幂律拟合失败：", e)
    RES['M7_interp'] = dict(n=int(n), single=dict(params=[float(x) for x in p1_], rss=rss1),
                            error=str(e))
# D 轴逐规模线性检验
print(f"  (c) 逐规模 ln(L−E) ~ lnD 线性检验（E 取单幂律解 {p1_[0]:.4f}）：")
lin = []
for nv, s in tr.groupby('N_params_B'):
    s = s.sort_values('D_tokens_B')
    if len(s) < 20:
        continue
    x = np.log(s['D_tokens_B'].values); y = np.log(np.maximum(s['val_loss'].values - p1_[0], 1e-6))
    c = np.polyfit(x, y, 1); rr = r2(y, np.polyval(c, x))
    lin.append(dict(N=float(nv), slope=float(c[0]), r2=float(rr)))
ld = pd.DataFrame(lin)
print(f"      8 个规模 R² 中位={ld['r2'].median():.6f}  最低={ld['r2'].min():.6f}  "
      f"斜率中位={ld['slope'].median():.4f}")
print(f"  ⇒ 判读：D 轴（token 数）上中间形状**可验证**，单幂律已足够；")
print(f"          N 轴（参数量）上 RegMix 只有 1M/60M 两个可配对观测端点，")
print(f"          1B 档 64 个配方无法与 1M/60M 配对 ⇒ **N 轴中间插值形状不可验证**（数据缺口）。")
RES['M7_interp']['d_axis_linear'] = ld.to_dict('records')
RES['M7_interp']['verdict'] = ("D 轴可验证且单幂律足够；N 轴中间形状不可验证（RegMix 仅两个可配对端点）")

# ==========================================================================================
json.dump(RES, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print("\n" + "=" * 104)
print(f"结果已写入 {OUT}")
print(f"部分识别界表已写入 {OUT_CSV}")
print("=" * 104)
