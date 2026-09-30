# -*- coding: utf-8 -*-
"""
P1-A 层一：Chinchilla 算力当量效率分解（定稿版）

文献依据：Hoffmann et al. 2022（Chinchilla）。外部前沿 L*_Chin(C) 由发表常数给出，
**不从本数据拟合**，用作两件事：
  (i)  效率水平尺 rho_i = C*_Chin(L_i) / (6 N_i D_i)  —— 量化"离算力最优配置有多远"
  (ii) 标度指数的外部对照：本数据自拟合 (alpha_hat, beta_hat) vs Chinchilla (0.34, 0.28)

重要更正（相对方案初稿 §5.1）：
  * 初稿把 dln(L-E)/dt 拆成 b_ext*dlnC/dt + (-dlnrho/dt)。该式把"N 指数"与"算力指数"混用：
    对固定 D 的族内数据，lnC 与 lnN 只差常数，回归出的 b 是 -alpha 而不是 -1/s。故不能直接对照。
  * 更关键：本数据**没有真实的 (N, D, Loss, 时间) 面板**（唯一真实配对 C6∩Epoch 的 75 条里
    只有 7 条 Pythia 带 D，且年份全为 2023，无时间变异）。因此"规模/非规模的时间分解"
    在本数据上**不可识别**，只能报水平量 rho 与横截面指数，时间分解交由 §5.2/§5.3。

数据分级：
  Tier0 真实无日期：B1 Pythia 终末 + B4 跨族收敛      —— 用于 rho 水平与指数拟合
  Tier1 真实+日期 ：C6 桥接 ∩ C4 Epoch（仅 7 条带 D） —— 时间变异不足，只作说明
  Tier2 真实+粗年份：B5 文献（年份取自文献标签）        —— 跨论文验证集不可比，仅作参照
  Tier3 估算+日期 ：B10 × B9（数据说明第371行标注"估算，非观测"）—— 只作回声参照
  gC：C4 Epoch Language 域 Training compute（真实公开报告值）
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = r"d:\F题\F题\real_attachments"
B = os.path.join(R, "B_scaling_laws")
C = os.path.join(R, "C_efficiency_evolution")
OUT = r"d:\F题\q4_p1_chinchilla_rho_results.json"

E_CH, A_CH, ALPHA_CH, B_CH, BETA_CH = 1.69, 406.4, 0.34, 410.7, 0.28
k_ = ALPHA_CH * A_CH / (BETA_CH * B_CH)
s_ = 1 / ALPHA_CH + 1 / BETA_CH
K_ = (A_CH + B_CH * k_) * (k_ ** (1 / BETA_CH) / 6) ** (-1 / s_)
def Lchin(Cflop):
    return E_CH + K_ * np.asarray(Cflop, float) ** (-1 / s_)
def Cstar(L):
    L = np.asarray(L, float); out = np.full(L.shape, np.nan)
    ok = L > E_CH + 1e-9
    out[ok] = ((L[ok] - E_CH) / K_) ** (-s_)
    return out

print("=" * 100)
print("### 0. 外部 Chinchilla 前沿标定（Hoffmann 2022, Approach-3）")
print(f"  E={E_CH}, A={A_CH}, alpha={ALPHA_CH}, B={B_CH}, beta={BETA_CH}")
print(f"  => L*_Chin(C) = {E_CH} + {K_:.2f} * C^(-{1/s_:.5f})")
print(f"  自检 C=5.04e23 FLOPs（Gopher 280B×300B tokens）-> L*={float(Lchin(5.04e23)):.4f}（论文图约 1.93）")

# ---------------- 数据 ----------------
def mk(df, nm, dm, lm, ycol, src, tier):
    d = df.rename(columns={nm: "N", dm: "D", lm: "L"}).copy()
    d["src"], d["tier"] = src, tier
    d["year"] = pd.to_datetime(d[ycol], errors="coerce").dt.year if ycol else np.nan
    return d[["N", "D", "L", "year", "src", "tier"]]

pl = pd.read_csv(os.path.join(B, "pythia_training_log_existing.csv"))
p0 = pl.sort_values("D_tokens_B").groupby("N_params_B").tail(1)
T0a = mk(p0, "N_params_B", "D_tokens_B", "val_loss", None, "B1 Pythia终末(真实)", 0)
sb = pd.read_csv(os.path.join(B, "scaling_baseline.csv"))
T0b = mk(sb[sb.is_converged == 1], "N_params_B", "D_tokens_B", "val_loss", None, "B4 跨族收敛(真实)", 0)
ps = pd.read_csv(os.path.join(B, "published_scaling_data.csv"))
ps["yr"] = ps.source.astype(str).str.split().str[-1]
T2 = mk(ps, "N_params_B", "D_tokens_B", "val_loss", "yr", "B5 文献(真实)", 2)

br = pd.read_csv(os.path.join(C, "loss_benchmark_bridge_expanded.csv"))
br["tail"] = br.Model.astype(str).str.split("/").str[-1].str.lower()
ea = pd.read_csv(os.path.join(C, "epoch_all_ai_models.csv"), low_memory=False)
ea = ea[ea.Domain == "Language"].copy()
ea["tail"] = ea.Model.astype(str).str.strip().str.lower()
m1 = br.merge(ea.drop_duplicates("tail")[["tail", "Publication date"]], on="tail", how="left")
T1 = mk(m1, "N_params_B", "D_tokens_B", "Val_Loss", "Publication date", "C6∩Epoch(真实)", 1)

lb = pd.read_csv(os.path.join(B, "supplementary_large_baseline.csv"))
lm = pd.read_csv(os.path.join(B, "supplementary_large_models.csv"))
lb["family"] = lb.family.astype(str).str.strip(); lm["model_name"] = lm.model_name.astype(str).str.strip()
m3 = lb.merge(lm, left_on="family", right_on="model_name", how="left", suffixes=("", "_y"))
T3 = mk(m3, "N_params_B", "D_tokens_B", "val_loss", "publication_date", "B10 估算(非观测)", 3)

def add_rho(d):
    d = d.copy()
    d["C_1e18"] = 6.0 * d.N * d.D
    d["rho"] = Cstar(d.L) / 1e18 / d.C_1e18
    d["ln_rho"] = np.log(d.rho)
    return d
T0a, T0b, T1, T2, T3 = map(add_rho, (T0a, T0b, T1, T2, T3))
T0 = pd.concat([T0a, T0b], ignore_index=True)

print("\n" + "=" * 100)
print("### 1. 效率水平 rho（描述性；rho<1 表示比算力最优配置更费算力）")
print("  注：L -> E=1.69 时 C*_Chin 呈 6.5 次幂爆炸（病态），只报中位与 IQR，不报极值。")
for tag, d in [("Tier0a B1 Pythia(真实)", T0a), ("Tier0b B4 跨族(真实)", T0b),
               ("Tier1 C6∩Epoch(真实)", T1), ("Tier2 B5文献(真实)", T2), ("Tier3 B10估算", T3)]:
    q = d.rho.quantile([0.25, 0.5, 0.75])
    print(f"  {tag:22s} n={len(d):3d}  rho 中位={q[0.5]:6.3f}  IQR=[{q[0.25]:.3f},{q[0.75]:.3f}]  "
          f"对数 sd={d.ln_rho.std():.2f}  算力冗余 1/rho 中位={1/q[0.5]:6.2f}x")
q1 = T1.rho.quantile([0.25, 0.5, 0.75])
print(f"\n  【关键读数】Tier1（现代良好配置模型：Pythia + Qwen2/2.5 + Llama 等）"
      f"rho 中位 = {q1[0.5]:.3f} → 比 Chinchilla 最优多用约 {(1/q1[0.5]-1)*100:.0f}% 算力。")

# ---------------- 2. 标度指数：本数据 vs 外部 ----------------
print("\n" + "=" * 100)
print("### 2. 标度指数的外部对照（Chinchilla 0.34/0.28 vs 本数据自拟合）")
pool = pd.concat([T0a, T0b, T2], ignore_index=True).dropna(subset=["N", "D", "L"])
pool = pool[pool.L > E_CH + 1e-6]
def resid(p, N, D, L):
    E, A, al, Bt, be = p
    return (E + A * N ** (-al) + Bt * D ** (-be)) - L
for tag, Efix in [("E 自由", None), ("E 固定 1.69", E_CH)]:
    if Efix is None:
        r = least_squares(resid, [1.7, 400, 0.34, 400, 0.28],
                          args=(pool.N.values, pool.D.values, pool.L.values),
                          bounds=([1.0, 1, 0.01, 1, 0.01], [2.5, 1e6, 2, 1e6, 2]), max_nfev=40000)
        E_, A_, al_, B_, be_ = r.x
    else:
        def r2(p, N, D, L): return (Efix + p[0] * N ** (-p[1]) + p[2] * D ** (-p[3])) - L
        r = least_squares(r2, [400, 0.34, 400, 0.28], args=(pool.N.values, pool.D.values, pool.L.values),
                          bounds=([1, 0.01, 1, 0.01], [1e6, 2, 1e6, 2]), max_nfev=40000)
        E_, (A_, al_, B_, be_) = Efix, r.x
    rmse = float(np.sqrt(np.mean(r.fun ** 2)))
    print(f"  [{tag}] n={len(pool)}  E={E_:.4f}  A={A_:.1f}  alpha={al_:.4f}  B={B_:.1f}  beta={be_:.4f}  RMSE={rmse:.4f}")
    print(f"        vs Chinchilla: alpha {ALPHA_CH} (差 {al_-ALPHA_CH:+.3f})  beta {BETA_CH} (差 {be_-BETA_CH:+.3f})")
    if Efix is None:
        fit_free = dict(E=float(E_), A=float(A_), alpha=float(al_), B=float(B_), beta=float(be_), rmse=rmse)
    else:
        fit_fix = dict(E=float(E_), A=float(A_), alpha=float(al_), B=float(B_), beta=float(be_), rmse=rmse)
print("  判读：本数据的 N 指数与 Chinchilla 量级一致但偏小；D 指数偏小说明数据里的"
      "数据量回报更弱（受族内 D 覆盖窄所限）。外部前沿不能直接强加于本数据。")

# ---------------- 3. 时间分解的可识别性 ----------------
print("\n" + "=" * 100)
print("### 3. 时间分解可识别性检查")
print(f"  Tier1（唯一真实配对）n={len(T1)}，其中带 D 的 = {T1.D.notna().sum()}，"
      f"年份取值 = {sorted(T1.dropna(subset=['D']).year.unique().tolist())}")
print("  => 真实配对全部同年（2023），**无时间变异** → c（固定算力下的技术进步率）不可识别。")
t2f = T2.dropna(subset=["year", "N", "D", "L"]); t2f = t2f[t2f.L > E_CH + 1e-6]
t2f = t2f.assign(lnC=np.log(6 * t2f.N * t2f.D), y=np.log(t2f.L - E_CH))
X = np.column_stack([np.ones(len(t2f)), t2f.lnC, t2f.year.astype(float)])
cf, *_ = np.linalg.lstsq(X, t2f.y.values, rcond=None)
rr = t2f.y.values - X @ cf
se = np.sqrt(np.diag((rr @ rr / (len(t2f) - 3)) * np.linalg.pinv(X.T @ X)))
print(f"  Tier2（B5 文献，跨论文验证集不可比）n={len(t2f)}: "
      f"b_hat={cf[1]:+.4f}(se {se[1]:.4f})  c_hat={cf[2]:+.4f}/年(se {se[2]:.4f})")
print("  => c_hat 虽显著，但'年份'与'论文/验证集'完全混淆（Kaplan/Hoffmann/Touvron 各用不同 val set），"
      "不能解释为技术进步率。")

# ---------------- 4. d lnC/dt ----------------
lg = ea.copy()
lg["Cflop"] = pd.to_numeric(lg["Training compute (FLOP)"], errors="coerce")
lg["t"] = pd.to_datetime(lg["Publication date"], errors="coerce")
lg = lg[(lg.Cflop > 0) & lg.t.notna()].copy()
lg["year"] = lg.t.dt.year + (lg.t.dt.dayofyear - 1) / 365.25
lg = lg[lg.year >= 2020]
gC_all = float(np.polyfit(lg.year, np.log(lg.Cflop), 1)[0])
qq = lg.groupby(lg.year.astype(int)).Cflop.quantile(0.9)
gC_p90 = float(np.polyfit(qq.index.values, np.log(qq.values), 1)[0])
print("\n" + "=" * 100)
print("### 4. d lnC/dt（C4 Epoch Language 域，真实公开报告值）")
print(f"  全样本 OLS   dlnC/dt = {gC_all:+.4f}/年 (n={len(lg)})")
print(f"  年度 p90 前沿 dlnC/dt = {gC_p90:+.4f}/年 (n={len(qq)} 年)")
print(f"  对照：本方案 §9.2 机制腿所用 g_C = 1.2875/年（同源口径的另一种切法）")

# ---------------- 5. 结论 ----------------
print("\n" + "=" * 100)
print("### 5. 层一的结论（修订）")
print(f"  1) 外部 Chinchilla 前沿可用且自检通过（Gopher 算力处 L*={float(Lchin(5.04e23)):.3f}）。")
print(f"  2) 效率水平：Tier1 现代良好配置模型 rho 中位 {q1[0.5]:.3f}（约多用 {(1/q1[0.5]-1)*100:.0f}% 算力）；"
      f"Tier0b 宽口径跨族 rho 中位 {T0b.rho.median():.3f} → **配置低效是普遍且量级很大的**。")
print(f"  3) 标度指数：本数据自拟合 alpha={fit_free['alpha']:.3f}/beta={fit_free['beta']:.3f}"
      f"（外部 0.34/0.28）→ 量级一致、外部前沿不能强加。")
print(f"  4) **时间分解不可识别**：本数据无真实 (N,D,L,时间) 面板，规模/非规模的时间份额"
      f"只能由 §5.2（排行榜得分口径 30.5%–45.0%）与 §5.3（Qwen Shapley 79%–82%）给出，"
      f"二者不一致 → 该份额应报**区间 30%–82%** 并声明识别不足。")

json.dump(dict(chinchilla=dict(E=E_CH, A=A_CH, alpha=ALPHA_CH, B=B_CH, beta=BETA_CH,
                               Lchin=f"L = {E_CH} + {K_:.4f} * C^(-{1/s_:.6f})",
                               selfcheck_L_at_5p04e23=float(Lchin(5.04e23))),
               own_fit=dict(free=fit_free, E_fixed=fit_fix, n_pooled=int(len(pool))),
               rho=dict(T0a=dict(n=int(len(T0a)), median=float(T0a.rho.median())),
                        T0b=dict(n=int(len(T0b)), median=float(T0b.rho.median())),
                        T1=dict(n=int(len(T1)), median=float(q1[0.5]),
                                iqr=[float(q1[0.25]), float(q1[0.75])]),
                        T2=dict(n=int(len(T2)), median=float(T2.rho.median())),
                        T3=dict(n=int(len(T3)), median=float(T3.rho.median()))),
               gC=dict(all=gC_all, p90=gC_p90),
               tier2_fit=dict(n=int(len(t2f)), b_hat=float(cf[1]), c_hat=float(cf[2]), c_hat_se=float(se[2])),
               identifiable=dict(time_decomposition=False,
                                 reason="无真实 (N,D,L,t) 面板；唯一真实配对同年(2023)无时间变异",
                                 scale_share_envelope=[0.305, 0.82]),
               caveat="B10 的 Loss 为估算值（数据说明第371行），不作独立证据；rho 在 L->E 时病态，只报中位/IQR。"),
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pd.concat([T0a, T0b, T1, T2, T3], ignore_index=True).to_csv(
    r"d:\F题\q4_p1_chinchilla_rho_all.csv", index=False, encoding="utf-8-sig")
print("\n已写出 q4_p1_chinchilla_rho_results.json / q4_p1_chinchilla_rho_all.csv")
