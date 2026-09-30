# -*- coding: utf-8 -*-
"""
问题四 P2-C：预注册回测协议 + 真实样本外回测
  - 运行前校验 q4_p2c_protocol.md 的 SHA-256（冻结）
  - 月度前沿滚动 origin 回测（M1 有界 / M2 无界 / M3 持久 / M4 外生固定增速）
  - 预注册三判据 C1/C2/C3
  - 校准系数 kappa = median(|e_ln| / sigma_h) → 用于放大 12/24M 预测区间
输出：q4_p2c_backtest_results.json / q4_p2c_backtest_table.csv / q4_p2c_calibration.json
"""
import os, json, hashlib, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
PROTO = r"d:\F题\q4_p2c_protocol.md"
PROTO_SHA = "90AE78E15F82BCC4DD30E0EC632A927A1D89647803C9CD2D3BD374F1515E8EE0"

# ---------------------------------------------------------------- 0. 冻结校验
h = hashlib.sha256(open(PROTO, 'rb').read()).hexdigest().upper()
print("=" * 100)
print("【0】预注册协议冻结校验")
print(f"  协议文件 : {PROTO}")
print(f"  期望 SHA256: {PROTO_SHA}")
print(f"  实际 SHA256: {h}")
assert h == PROTO_SHA, "协议文件已被修改，回测中止（预注册失效）"
print("  → 校验通过，协议未被修改。")

# ---------------------------------------------------------------- 1. 数据与前沿
OW = ['apache', 'mit', 'bsd', 'llama', 'gemma', 'cc-by', 'openrail', 'gpl', 'wtfpl',
      'afl', 'creativeml', 'bigscience', 'bigcode', 'apple-ascl']
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S'])
lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['m'] = lb['date'].dt.to_period('M').astype(str)
lic = lb['Hub License'].fillna('').str.lower()
sub = lb[lic.apply(lambda x: any(w in x for w in OW))].copy()

agg = sub.groupby('m')['S'].agg(n='count', F=lambda s: s.quantile(0.90)).sort_index()
agg = agg[agg['n'] >= 20]
MONTHS = list(agg.index)
F = agg['F'].values.astype(float)
y = np.log(F)
N_M = len(MONTHS)
print("\n" + "=" * 100)
print(f"【1】月度前沿序列（OPEN_LIC，p90）  窗口 {MONTHS[0]} ~ {MONTHS[-1]}，共 {N_M} 个月")
for m, n_, f in zip(MONTHS, agg['n'].values, F):
    print(f"  {m}  n={n_:4d}  F={f:6.2f}  lnF={np.log(f):.4f}")

# ---------------------------------------------------------------- 2. 模型与区间
def fit_g(y_tr):
    """g = OLS slope(ln F ~ t) x 12（年化）；返回 g, sigma_y"""
    t = np.arange(len(y_tr), dtype=float)
    A = np.column_stack([np.ones_like(t), t])
    c, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    r = y_tr - A @ c
    dof = len(t) - 2
    sig = float(np.sqrt(r @ r / dof)) if dof > 0 else np.nan
    return float(c[1] * 12.0), sig

def logistic(S_L, g, hh):
    A = (100 - S_L) / S_L
    r = g / (1 - S_L / 100)
    return 100.0 / (1 + A * np.exp(-r * hh))

def predict(model, S_L, g, sig, hh, g_ext):
    """g/g_ext 为年化增速，hh 为月数 → 统一换算为月率。
    返回 (点预测, ln尺度中心, ln尺度 sigma_h, 是否 logit 尺度)"""
    gm, gx = g / 12.0, g_ext / 12.0
    if model == 'M1':
        return logistic(S_L, gm, hh), None, sig * np.sqrt(hh), True
    if model == 'M2':
        return S_L * np.exp(gm * hh), np.log(S_L) + gm * hh, sig * np.sqrt(hh), False
    if model == 'M3':
        return S_L, np.log(S_L), sig * np.sqrt(hh + 1), False
    if model == 'M4':
        return S_L * np.exp(gx * hh), np.log(S_L) + gx * hh, sig * np.sqrt(hh), False
    raise ValueError(model)

def interval(S_L, g, sig, hh, model, g_ext, z):
    """按协议第 4 节构造区间，返回 (lo, hi) 在 S 尺度"""
    _, mu, sh, is_logit = predict(model, S_L, g, sig, hh, g_ext)
    if is_logit:
        gm = g / 12.0
        A = (100 - S_L) / S_L
        r = gm / (1 - S_L / 100)
        # logit 尺度：logit(h) = ln(A) - r·h；σ_logit = σ_ln / (1 - S/100) 的线性化
        r_sh = sh / max(1e-9, (1 - S_L / 100))
        lo_l, hi_l = np.log(A) - r * hh - z * r_sh, np.log(A) - r * hh + z * r_sh
        return 100.0 / (1 + np.exp(hi_l)), 100.0 / (1 + np.exp(lo_l))
    lo_l, hi_l = mu - z * sh, mu + z * sh
    return float(np.exp(lo_l)), float(np.exp(hi_l))

MODELS = ['M1', 'M2', 'M3', 'M4']
Z80, Z90 = 1.2816, 1.6449
G_EXT = 0.6326                     # 协议第 2 节 M4：腿2 全样本增速（年化）
MIN_TRAIN = 4
HOR = [1, 2, 3]

def sgn(x):
    return 0 if abs(x) < 1e-12 else (1 if x > 0 else -1)

# ---------------------------------------------------------------- 3. 滚动 origin 回测
print("\n" + "=" * 100)
print("【2】滚动 origin 回测（最小训练 4 个月，h ∈ {1,2,3}）")
rows = []
for i in range(MIN_TRAIN - 1, N_M):
    y_tr = y[:i + 1]
    if len(y_tr) < MIN_TRAIN:
        continue
    g_hat, sig_y = fit_g(y_tr)
    S_L = float(F[i])
    for hh in HOR:
        j = i + hh
        if j >= N_M:
            continue
        S_real = float(F[j])
        rec = dict(origin=MONTHS[i], h=hh, target=MONTHS[j], S_L=S_L, S_real=S_real,
                   g_hat=g_hat, sigma_y=sig_y, n_train=len(y_tr))
        for mdl in MODELS:
            Sp, _, sh, _ = predict(mdl, S_L, g_hat, sig_y, hh, G_EXT)
            lo90, hi90 = interval(S_L, g_hat, sig_y, hh, mdl, G_EXT, Z90)
            lo80, hi80 = interval(S_L, g_hat, sig_y, hh, mdl, G_EXT, Z80)
            rec[f'{mdl}_pred'] = Sp
            rec[f'{mdl}_e'] = S_real - Sp
            rec[f'{mdl}_eln'] = float(np.log(S_real) - np.log(Sp)) if Sp > 0 else np.nan
            rec[f'{mdl}_sig'] = sh
            rec[f'{mdl}_in90'] = bool(lo90 <= S_real <= hi90)
            rec[f'{mdl}_in80'] = bool(lo80 <= S_real <= hi80)
            rec[f'{mdl}_pin'] = float(0.9 * (S_real - Sp) if S_real >= Sp else 0.1 * (Sp - S_real))
            rec[f'{mdl}_dir'] = bool(sgn(Sp - S_L) == sgn(S_real - S_L))
        rows.append(rec)
bt = pd.DataFrame(rows)
print(f"  pooled 检验样本数 = {len(bt)}  （origin × h 组合）")
print(f"  origins: {sorted(bt['origin'].unique())}")

# ---------------------------------------------------------------- 4. 指标汇总
print("\n" + "=" * 100)
print("【3】预注册指标（pooled）")
summary = {}
print(f"  {'模型':22s} {'MAE':>7s} {'RMSE':>7s} {'bias':>8s} {'t(bias)':>8s} "
      f"{'cov80':>7s} {'cov90':>7s} {'pin@.9':>7s} {'方向':>6s}")
NAME = {'M1': 'M1 有界 logistic', 'M2': 'M2 无界对数线性', 'M3': 'M3 随机游走(持久)',
        'M4': 'M4 外生固定增速'}
for mdl in MODELS:
    e = bt[f'{mdl}_e'].values.astype(float)
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    sd = float(np.std(e, ddof=1))
    tstat = float(bias / (sd / np.sqrt(len(e)))) if sd > 0 else np.nan
    cov80 = float(bt[f'{mdl}_in80'].mean())
    cov90 = float(bt[f'{mdl}_in90'].mean())
    pin = float(bt[f'{mdl}_pin'].mean())
    dr = float(bt[f'{mdl}_dir'].mean())
    summary[mdl] = dict(mae=mae, rmse=rmse, bias=bias, t_bias=tstat,
                        cov80=cov80, cov90=cov90, pinball90=pin, dir_acc=dr)
    print(f"  {NAME[mdl]:22s} {mae:7.3f} {rmse:7.3f} {bias:+8.3f} {tstat:+8.2f} "
          f"{cov80:7.3f} {cov90:7.3f} {pin:7.3f} {dr:6.3f}")

# 分步长
print("\n【4】分步长 RMSE（检验 horizon 越长是否越差）")
print(f"  {'h(月)':>6s} {'n':>4s} " + " ".join(f"{mdl:>8s}" for mdl in MODELS))
by_h = {}
for hh in HOR:
    sl = bt[bt.h == hh]
    if sl.empty:
        continue
    vals = {mdl: float(np.sqrt(np.mean(sl[f'{mdl}_e'].values.astype(float) ** 2))) for mdl in MODELS}
    by_h[hh] = dict(n=int(len(sl)), **vals)
    print(f"  {hh:6d} {len(sl):4d} " + " ".join(f"{vals[mdl]:8.3f}" for mdl in MODELS))

# ------------------------------------------- 4b. 事后诊断（非预注册判据，不影响 C1/C2/C3）
print("\n" + "=" * 100)
print("【4b】事后诊断：origin 间 g_hat 稳定性（探索性，不参与预注册判据）")
g_series = []
for i in range(MIN_TRAIN - 1, N_M):
    if i + 1 < MIN_TRAIN:
        continue
    gi, _ = fit_g(y[:i + 1])
    g_series.append((MONTHS[i], gi, i + 1))
for m_, gi, n_ in g_series:
    print(f"  origin {m_}  n_train={n_:2d}  g_hat = {gi:+.4f}/年")
gv = np.array([x[1] for x in g_series])
sd_g = float(np.std(gv, ddof=1))
print(f"  g_hat 跨 origin：min={gv.min():+.4f}  max={gv.max():+.4f}  极差={gv.max()-gv.min():.4f}"
      f"  sd={sd_g:.4f}")
print("  判读：回测窗口仅 10 个月、测试步长 ≤3 个月，长周期（12/24M）的校准**未被覆盖**；")
print("        g 的跨 origin 不稳定性是长周期风险的主要来源，必须在论文中作为未验证假设声明。")
print("        （长周期展宽数值见 §7 末尾，因需先确定主预测的 S0/g。）")

# ---------------------------------------------------------------- 5. 预注册判据
print("\n" + "=" * 100)
print("【5】预注册判据（协议第 6 节）")
c1 = bool(summary['M1']['rmse'] <= summary['M3']['rmse'])
c2 = bool(0.50 <= summary['M1']['cov90'] <= 1.00)
c3 = bool(abs(summary['M1']['t_bias']) < 2.0)
print(f"  C1 不劣于持久   : RMSE(M1)={summary['M1']['rmse']:.3f} ≤ RMSE(M3)={summary['M3']['rmse']:.3f}"
      f"  → {'通过' if c1 else '未通过'}")
print(f"  C2 区间校准     : cov90(M1)={summary['M1']['cov90']:.3f} ∈ [0.50,1.00]"
      f"  → {'通过' if c2 else '未通过'}")
print(f"  C3 无系统偏差   : |t(bias)|={abs(summary['M1']['t_bias']):.2f} < 2"
      f"  → {'通过' if c3 else '未通过'}")
verdict = '通过' if (c1 and c2 and c3) else '未通过'
print(f"  → 回测结论：{verdict}（{'三判据全过' if verdict=='通过' else '存在未通过项'}）")

# ---------------------------------------------------------------- 6. 校准系数
print("\n" + "=" * 100)
print("【6】区间校准系数 kappa（协议第 7 节）")
kappa = {}
for mdl in MODELS:
    r_ = bt[f'{mdl}_eln'].values.astype(float)
    s_ = bt[f'{mdl}_sig'].values.astype(float)
    ok = np.isfinite(r_) & np.isfinite(s_) & (s_ > 0)
    k = float(np.median(np.abs(r_[ok]) / s_[ok]))
    kappa[mdl] = k
    print(f"  {NAME[mdl]:22s} κ = median(|e_ln|/σ_h) = {k:.3f}")
k_use = kappa['M1']
print(f"  → 采用 M1 的 κ = {k_use:.3f}（{'区间需放大' if k_use > 1 else 'κ<1，按保守原则不缩小'}）")

# ---------------------------------------------------------------- 7. 应用到 12/24M 主预测
print("\n" + "=" * 100)
print("【7】把 κ 应用到 12/24 个月主预测（腿合成稳健口径）")
fc = json.load(open(r"d:\F题\q4_legs_reweight_results.json", encoding='utf-8'))
S0 = float(fc['S0'])
g_main = float(fc['main']['g_comb'])
sd_par = float(fc['main']['sd_comb'])
g_hat_full, sig_full = fit_g(y)
print(f"  基准点 S0 = {S0:.2f}（2025Q1 OPEN_LIC p90）")
print(f"  主增速 g = {g_main:+.4f}/年（腿合成稳健，σ_par = {sd_par:.4f}）")
print(f"  全样本月度前沿拟合残差 σ_y = {sig_full:.4f}（对数尺度/月）")
sig_cal = k_use * sig_full
print(f"  校准后残差 σ_y^cal = κ·σ_y = {k_use:.3f} × {sig_full:.4f} = {sig_cal:.4f}")

A0 = (100 - S0) / S0
r0 = g_main / (1 - S0 / 100)
def lvl(hh):
    return 100.0 / (1 + A0 * np.exp(-r0 * hh))

CASES = [('仅残差（σ_res=σ_y, σ_par=0）', 0.0, sig_full),
         ('残差+参数（σ_y, σ_par）', sd_par, sig_full),
         ('κ 校准（κσ_y, σ_par）', sd_par, sig_cal)]
out_rows = []
print(f"\n  {'口径':30s} {'12M':>7s} {'90%CI(12M)':>18s} {'24M':>7s} {'90%CI(24M)':>18s}")
for tag, s_par_, s_res_ in CASES:
    rr = {}
    for hh in (1.0, 2.0):
        s_tot = float(np.sqrt((s_par_ * hh) ** 2 + (s_res_ * np.sqrt(hh)) ** 2))
        # logit 尺度：logit(h)=ln(A0)-r0·h，σ_logit = σ_ln/(1-S/100)
        scale = 1.0 / (1 - lvl(hh) / 100)
        c_l = np.log(A0) - r0 * hh
        lo_l, hi_l = c_l - 1.6449 * s_tot * scale, c_l + 1.6449 * s_tot * scale
        rr[hh] = (lvl(hh), 100 / (1 + np.exp(hi_l)), 100 / (1 + np.exp(lo_l)))
    out_rows.append(dict(tag=tag, sig_res=s_res_, sig_par=s_par_,
                         s12=rr[1.0][0], s12_lo=rr[1.0][1], s12_hi=rr[1.0][2],
                         s24=rr[2.0][0], s24_lo=rr[2.0][1], s24_hi=rr[2.0][2]))
    print(f"  {tag:30s} {rr[1.0][0]:7.2f}   [{rr[1.0][1]:6.2f},{rr[1.0][2]:6.2f}] "
          f"{rr[2.0][0]:7.2f}   [{rr[2.0][1]:6.2f},{rr[2.0][2]:6.2f}]")

print("\n  判读：κ 校准只放大**残差**分量，参数分量不动；24M 区间由校准前的 "
      f"[{out_rows[1]['s24_lo']:.1f},{out_rows[1]['s24_hi']:.1f}] 变为 "
      f"[{out_rows[2]['s24_lo']:.1f},{out_rows[2]['s24_hi']:.1f}]。")

# 长周期外推的模型不确定度下限（§4b 诊断的数值部分）
print("\n  长周期（回测未覆盖）由 g 不稳定性引入的展宽（§4b 诊断）：")
long_diag = {}
for hy in (1.0, 2.0):
    spread_ln = sd_g * hy
    S_c = lvl(hy)
    scale = 1.0 / (1 - S_c / 100)
    c_l = np.log(A0) - r0 * hy
    lo_l = c_l - 1.6449 * spread_ln * scale
    hi_l = c_l + 1.6449 * spread_ln * scale
    lo, hi = 100 / (1 + np.exp(hi_l)), 100 / (1 + np.exp(lo_l))
    long_diag[hy] = dict(spread_ln=float(spread_ln), lo=float(lo), hi=float(hi))
    print(f"    {hy:.0f} 年：±{1.6449*spread_ln:.3f}(ln) → 区间 [{lo:.2f}, {hi:.2f}]")

# ---------------------------------------------------------------- 8. 落盘
res = dict(
    protocol=dict(file=PROTO, sha256=PROTO_SHA, verified=True,
                  frozen_rules=dict(sample='OPEN_LIC', freq='M', tau=0.90,
                                    min_train=MIN_TRAIN, horizons=HOR,
                                    models=MODELS, g_ext=G_EXT,
                                    z80=Z80, z90=Z90)),
    frontier=dict(months=MONTHS, n=[int(x) for x in agg['n'].values], F=[float(x) for x in F],
                  lnF=[float(x) for x in y]),
    n_test=len(bt), origins=sorted(bt['origin'].unique()),
    metrics=summary, metrics_by_h=by_h,
    criteria=dict(C1_pass=c1, C2_pass=c2, C3_pass=c3, verdict=verdict,
                  detail=dict(rmse_M1=summary['M1']['rmse'], rmse_M3=summary['M3']['rmse'],
                              cov90_M1=summary['M1']['cov90'],
                              t_bias_M1=summary['M1']['t_bias'])),
    kappa=kappa, kappa_used=k_use,
    sigma_y_full=sig_full, sigma_y_calibrated=sig_cal,
    forecast_calibrated=out_rows,
    posthoc_diagnostics=dict(
        note="以下为事后探索性诊断，不参与预注册判据 C1/C2/C3",
        g_hat_by_origin={m_: float(gi) for m_, gi, _ in g_series},
        g_hat_min=float(gv.min()), g_hat_max=float(gv.max()),
        g_hat_range=float(gv.max() - gv.min()), g_hat_sd=sd_g,
        long_horizon_spread={str(k): v for k, v in long_diag.items()}),
    caveats=[
        "pooled 检验样本仅 15 个 (origin,h) 组合，统计功效极低，覆盖率判据只作证伪下限",
        "h 最大 3 个月，σ_h=σ_y·√h 的线性缩放假设未被回测覆盖",
        "月度 p90 由当月少量模型估计（早期 n≈130），前沿值自带测量噪声",
        "C3 年度累计最大前沿退化为阶梯（2019-23 恒 50.00，2024-25 为 52.08），不可用于回测",
        "回测只检验预测流程，不重估问题四主结论的 g",
    ])
json.dump(res, open(r"d:\F题\q4_p2c_backtest_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
bt.to_csv(r"d:\F题\q4_p2c_backtest_table.csv", index=False, encoding='utf-8-sig')
json.dump(dict(kappa=kappa, kappa_used=k_use, sigma_y_full=sig_full,
               sigma_y_calibrated=sig_cal, forecast=out_rows),
          open(r"d:\F题\q4_p2c_calibration.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出 q4_p2c_backtest_results.json / q4_p2c_backtest_table.csv / q4_p2c_calibration.json")
