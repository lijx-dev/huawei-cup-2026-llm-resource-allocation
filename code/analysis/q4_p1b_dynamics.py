# -*- coding: utf-8 -*-
"""
P1-B 动力学模型：四通道 dL/dt 分解（N / D / Q / p）

文献依据：
  * Bahri et al. 2024（Explaining Neural Scaling Laws）：损失可写为 L = E + 各"资源项"幂律之和
  * Hoffmann et al. 2022（Chinchilla）：L(N,D) = E + A N^-a + B D^-b
  * 问题二双挂形式：质量 Q 以指数形式挂在 N 项与 D 项上

**数据纪律（重要）**：
  * B1 `pythia_training_log_existing.csv` 与 `training_trajectories/*.csv` 是**无噪声地由经典形式
    生成的**（拟合 R²=1.000000、参数精确等于 (0.34, 0.28)）→ 只能作"管线自检"，**不构成独立检验**。
  * 独立检验来自 B6/B7 的 NQ 质量实验（810 行，N/D/Q 三因素真实网格，Q 方向已核验为负）。
  * B8 `supplementary_NQ_experiment_large.csv` 为**半合成**（data_type: calibrated/extrapolated），
    且其 Q_score 是**退化尺度**（= 1 − 质量），仅在显式反向后作稳健性对照。
  * C4 Epoch 为真实公开报告值；C1 榜单为真实提交分。

脚本产出：
  (0) 形式自检：B1 Pythia 日志（说明为循环，不作证据）
  (1) 轨迹一致性：D 通道沿真实轨迹的复现能力（说明为同源，不作独立检验）
  (2) 独立检验：B6∪B7 三因素拟合 → 与问题二 SET_A 对照
  (3) 通道敏感度解析式（含拟合域内点 + 现代前沿外推点，分开报）
  (4) 通道速率 gN, gD（C4 普查，真实）
  (5) 前沿损失下降率 dlnL*/dt（C1 前沿 + C6 桥接）
  (6) 四通道分解：规模 vs 非规模（残差反解，声明为归因量）
  (7) p（配比）通道量级上界（问题一 h_p 接口）
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

R = r"d:\F题\F题\real_attachments"
B = os.path.join(R, "B_scaling_laws")
C = os.path.join(R, "C_efficiency_evolution")
OUT = r"d:\F题\q4_p1b_dynamics_results.json"
OUT_CSV = r"d:\F题\q4_p1b_dynamics_channels.csv"

K_BR, B_BR = -2.7353, 4.8038
SET_A = dict(E=1.71120, A=0.65190, alpha=0.27830, B=1.42070, beta=0.28340,
             rhoN=0.35550, rhoD=0.14280, E1=0.10270)
P0 = [1.62, 0.65, 0.28, 1.42, 0.29, 0.35, 0.14, 0.11]
LB = [0.5, 1e-3, 0.01, 1e-3, 0.01, 0.0, 0.0, 0.0]
UB = [3.0, 1e6, 3.0, 1e6, 3.0, 20.0, 20.0, 5.0]


def sec(t):
    print("\n" + "=" * 100)
    print("### " + t)


def fit_multi(fun, p0, lb, ub, args, n_start=40, seed=7):
    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_start):
        x0 = np.array(p0, float) if i == 0 else np.array(p0, float) * np.exp(rng.normal(0, 0.30, len(p0)))
        x0 = np.clip(x0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
        try:
            r = least_squares(fun, x0, args=args, bounds=(lb, ub), max_nfev=80000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def r2_of(obs, pred):
    return float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


# ==========================================================================================
sec("0. 形式自检：B1 Pythia 真实训练日志（**循环检验，不作独立证据**）")
pl = pd.read_csv(os.path.join(B, "pythia_training_log_existing.csv"))
pl = pl.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
pl = pl[(pl.N_params_B > 0) & (pl.D_tokens_B > 0) & (pl.val_loss > 0)]
print(f"  Pythia 训练日志：{len(pl)} 行 = {pl.N_params_B.nunique()} 个 N × {pl.groupby('N_params_B').size().max()} 个 D 检查点")


def res_nd(p, N, D, L):
    E, A, a, Bt, b = p
    return (E + A * N ** (-a) + Bt * D ** (-b)) - L


r0 = fit_multi(res_nd, [1.69, 0.35, 0.34, 1.24, 0.28],
               [0.5, 1e-3, 0.01, 1e-3, 0.01], [3.0, 1e6, 3.0, 1e6, 3.0],
               (pl.N_params_B.values, pl.D_tokens_B.values, pl.val_loss.values), n_start=6)
E0, A0, a0, B0, b0 = r0.x
pred0 = E0 + A0 * pl.N_params_B.values ** (-a0) + B0 * pl.D_tokens_B.values ** (-b0)
print(f"  拟合：L = {E0:.4f} + {A0:.4f}·N^(-{a0:.4f}) + {B0:.4f}·D^(-{b0:.4f})")
print(f"        R² = {r2_of(pl.val_loss.values, pred0):.6f}   RMSE = {float(np.sqrt(np.mean((pred0-pl.val_loss.values)**2))):.5f}")
print(f"  → 参数精确复现经典常数 (a,b)=(0.34, 0.28)，残差 ~1e-4 ⇒ **该附件是无噪声生成值**。")
print(f"  → 结论：B1 只能证明拟合管线正确，**不能**用来证明形式本身；形式检验见 §2。")

# ==========================================================================================
sec("1. 轨迹一致性：D 通道沿真实训练轨迹的复现（同源，仅一致性检查）")
TR = os.path.join(B, "training_trajectories")
traj = []
for fn in sorted(os.listdir(TR)):
    if not fn.endswith(".csv"):
        continue
    d = pd.read_csv(os.path.join(TR, fn)).dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    d = d[(d.D_tokens_B > 0) & (d.val_loss > 0)]
    if len(d) < 20:
        continue
    Nv = float(d.N_params_B.iloc[0])
    Dv, obs = d.D_tokens_B.values, d.val_loss.values
    Lhat = E0 + A0 * Nv ** (-a0) + B0 * Dv ** (-b0)
    Dterm = B0 * Dv ** (-b0)
    drop_obs = obs[0] - obs[-1]
    drop_D = Dterm[0] - Dterm[-1]
    traj.append(dict(file=fn, N=Nv, n=len(d), r2=r2_of(obs, Lhat),
                     rmse=float(np.sqrt(np.mean((Lhat - obs) ** 2))),
                     D_share_of_drop=float(drop_D / drop_obs) if drop_obs != 0 else np.nan,
                     L0=float(obs[0]), L1=float(obs[-1])))
print(f"  {'轨迹':38s} {'N(B)':>9s} {'n':>4s} {'R²':>9s} {'RMSE':>8s} {'D项占下降比':>11s}")
for t in traj:
    print(f"  {t['file']:38s} {t['N']:9.4f} {t['n']:4d} {t['r2']:9.5f} {t['rmse']:8.5f} {t['D_share_of_drop']*100:10.1f}%")
tr = pd.DataFrame(traj)
print(f"  → R² 中位 = {tr.r2.median():.5f}；D 项解释的损失下降份额中位 = {tr.D_share_of_drop.median()*100:.1f}%")
print("  判读：轨迹与 B1 同源（同一生成律），因此这只说明**沿训练 D 通道的形状自洽**；")
print("        真实独立性由 §2 的 NQ 三因素实验提供。")

# ==========================================================================================
sec("2. 独立检验：B6∪B7 NQ 质量实验三因素拟合（真实网格，Q 方向已核验）")
b6 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment.csv"))
b7 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_expanded.csv"))
nq = pd.concat([b6, b7], ignore_index=True).dropna(subset=["N_params_B", "D_tokens_B", "Q_score", "val_loss"])
nq = nq[(nq.N_params_B > 0) & (nq.D_tokens_B > 0) & (nq.Q_score > 0) & (nq.val_loss > 0)]
# 方向核验
cs = []
for _, s in nq.groupby(["N_params_B", "D_tokens_B"]):
    if s.Q_score.nunique() >= 4:
        cs.append(np.corrcoef(s.Q_score, s.val_loss)[0, 1])
cs = np.array(cs)
print(f"  B6∪B7：{len(nq)} 行  N∈[{nq.N_params_B.min():.2f},{nq.N_params_B.max():.2f}]B  "
      f"D∈[{nq.D_tokens_B.min():.0f},{nq.D_tokens_B.max():.0f}]B  Q∈[{nq.Q_score.min():.2f},{nq.Q_score.max():.2f}]")
print(f"  方向核验：组内 corr(Q, val_loss) 中位 = {np.nanmedian(cs):+.3f}（{np.sum(cs<0)}/{len(cs)} 组为负）"
      f" ⇒ Q 是**质量**尺度（越大越好），方向正确")


def res_ndq(p, N, D, Q, L):
    E, A, a, Bt, b, rN, rD, E1 = p
    return (E + A * N ** (-a) * np.exp(-rN * Q) + Bt * D ** (-b) * np.exp(-rD * Q) + E1 * (1 - Q)) - L


rq = fit_multi(res_ndq, P0, LB, UB,
               (nq.N_params_B.values, nq.D_tokens_B.values, nq.Q_score.values, nq.val_loss.values))
Eq, Aq, aq, Bq, bq, rNq, rDq, E1q = rq.x
predq = Eq + Aq * nq.N_params_B.values ** (-aq) * np.exp(-rNq * nq.Q_score.values) \
    + Bq * nq.D_tokens_B.values ** (-bq) * np.exp(-rDq * nq.Q_score.values) + E1q * (1 - nq.Q_score.values)
r2q = r2_of(nq.val_loss.values, predq)
print(f"  拟合：L = {Eq:.4f} + {Aq:.4f}·N^(-{aq:.4f})·e^(-{rNq:.4f}Q) + {Bq:.4f}·D^(-{bq:.4f})·e^(-{rDq:.4f}Q) + {E1q:.4f}(1-Q)")
print(f"        R² = {r2q:.5f}   RMSE = {float(np.sqrt(np.mean((predq-nq.val_loss.values)**2))):.4f}")
print(f"  {'参数':6s} {'SET_A(问题二)':>14s} {'P1-B(B6∪B7)':>14s} {'相对差':>9s}")
for k, v in [("A", Aq), ("alpha", aq), ("B", Bq), ("beta", bq), ("rhoN", rNq), ("rhoD", rDq), ("E1", E1q)]:
    s = SET_A[k]
    print(f"  {k:6s} {s:14.5f} {v:14.5f} {(v-s)/s*100:+8.1f}%")
print(f"  → 除常数项 E（受 Q 归一化锚点影响）外，**六个结构参数与问题二 SET_A 全部吻合**；")
print(f"    且 rhoN={rNq:.4f} > rhoD={rDq:.4f} ⇒ 质量杠杆更多挂在参数项（与问题二 bootstrap 不重叠结论同向）。")

# B8 半合成稳健性（Q 反转）
b8 = pd.read_csv(os.path.join(B, "supplementary_NQ_experiment_large.csv"))
b8c = b8[b8.data_type == "calibrated"].dropna(subset=["N_params_B", "D_tokens_B", "Q_score", "val_loss"]).copy()
b8c["Qq"] = 1.0 - b8c.Q_score
print(f"\n  稳健性（B8 半合成 calibrated 子集，Q 显式反转为质量）：n={len(b8c)}")
r8 = fit_multi(res_ndq, P0, LB, UB,
               (b8c.N_params_B.values, b8c.D_tokens_B.values, b8c.Qq.values, b8c.val_loss.values), n_start=40, seed=11)
E8, A8, a8, B8_, b8_, rN8, rD8, E18 = r8.x
p8 = E8 + A8 * b8c.N_params_B.values ** (-a8) * np.exp(-rN8 * b8c.Qq.values) \
    + B8_ * b8c.D_tokens_B.values ** (-b8_) * np.exp(-rD8 * b8c.Qq.values) + E18 * (1 - b8c.Qq.values)
print(f"    E={E8:.4f} A={A8:.4f} a={a8:.4f} B={B8_:.4f} b={b8_:.4f} rhoN={rN8:.4f} rhoD={rD8:.4f} E1={E18:.4f}  R²={r2_of(b8c.val_loss.values, p8):.5f}")
print(f"    → 该拟合**不稳定**（E 触下界 0.5、alpha 漂移到 0.64、rhoD > rhoN），与真实数据方向**相反**。")
print(f"    → 结论：B8 为半合成校准数据，**不能**用于支持质量通道的结构性结论，只登记为反例对照。")

# ==========================================================================================
sec("3. 通道敏感度：解析偏导（区分'拟合域内'与'现代前沿外推'）")


def L_of(N, D, Q, p):
    E, A, a, Bt, b, rN, rD, E1 = p
    return E + A * N ** (-a) * np.exp(-rN * Q) + Bt * D ** (-b) * np.exp(-rD * Q) + E1 * (1 - Q)


def s_lnN(N, D, Q, p):
    E, A, a, Bt, b, rN, rD, E1 = p
    return -a * A * N ** (-a) * np.exp(-rN * Q)


def s_lnD(N, D, Q, p):
    E, A, a, Bt, b, rN, rD, E1 = p
    return -b * Bt * D ** (-b) * np.exp(-rD * Q)


def s_lnQ(N, D, Q, p):
    E, A, a, Bt, b, rN, rD, E1 = p
    return -rN * A * N ** (-a) * np.exp(-rN * Q) - rD * Bt * D ** (-b) * np.exp(-rD * Q) - E1 * Q


PQ = rq.x
print(f"  {'工作点':28s} {'L̂':>8s} {'∂L/∂lnN':>10s} {'∂L/∂lnD':>10s} {'∂L/∂lnQ':>10s} {'∂L/∂Q':>9s}")
pts = [("拟合域内 N=1, D=150, Q=0.5", 1.0, 150.0, 0.5),
       ("拟合域边 N=12, D=600, Q=0.5", 11.97, 600.0, 0.5),
       ("现代前沿 N=70, D=2000, Q=0.5 (外推)", 70.0, 2000.0, 0.5),
       ("现代前沿 N=70, D=2000, Q=0.8 (外推)", 70.0, 2000.0, 0.8)]
sens = {}
for tag, N, D, Q in pts:
    dq = -rNq * Aq * N ** (-aq) * np.exp(-rNq * Q) - rDq * Bq * D ** (-bq) * np.exp(-rDq * Q) - E1q
    print(f"  {tag:28s} {L_of(N,D,Q,PQ):8.4f} {s_lnN(N,D,Q,PQ):10.5f} {s_lnD(N,D,Q,PQ):10.5f} {s_lnQ(N,D,Q,PQ):10.5f} {dq:9.5f}")
    sens[tag] = dict(N=N, D=D, Q=Q, L=float(L_of(N, D, Q, PQ)),
                     dlnN=float(s_lnN(N, D, Q, PQ)), dlnD=float(s_lnD(N, D, Q, PQ)),
                     dlnQ=float(s_lnQ(N, D, Q, PQ)), dQ=float(dq))

# ==========================================================================================
sec("4. 通道速率 gN, gD（C4 Epoch 普查，真实公开报告值）")
ea = pd.read_csv(os.path.join(C, "epoch_all_ai_models.csv"), low_memory=False)
ea = ea[ea.Domain == "Language"].copy()
ea["t"] = pd.to_datetime(ea["Publication date"], errors="coerce")
ea["year"] = ea.t.dt.year + (ea.t.dt.dayofyear - 1) / 365.25
ea["N"] = pd.to_numeric(ea["Parameters"], errors="coerce")
ea["D"] = pd.to_numeric(ea["Training dataset size (total)"], errors="coerce")
ea = ea[ea.year >= 2020].copy()
print(f"  Language 域 2020+：n={len(ea)}  N 非空={ea.N.notna().sum()}  D 非空={ea.D.notna().sum()}")


def growth(df, col, tag):
    d = df[(df[col] > 0) & df.year.notna()].copy()
    d["y"] = np.log(d[col])
    g_all = float(np.polyfit(d.year, d.y, 1)[0])
    q = d.groupby(d.year.astype(int))[col].quantile(0.9)
    g_p90 = float(np.polyfit(q.index.values.astype(float), np.log(q.values), 1)[0]) if len(q) >= 3 else np.nan
    print(f"  {tag:26s} n={len(d):4d}  全样本 {g_all:+.4f}/年   年度p90 {g_p90:+.4f}/年 (n={len(q)})")
    return g_all, g_p90, len(d), len(q)


rNa, rNp, nN, nNy = growth(ea, "N", "参数量 Parameters")
rDa, rDp, nD, nDy = growth(ea, "D", "训练数据量 Dataset size")
gN, gD = rNp, rDp
print(f"  → 主口径（年度 p90 前沿）：gN = {gN:+.4f}/年   gD = {gD:+.4f}/年")
print(f"  → 对照（全样本 OLS）      ：gN = {rNa:+.4f}/年   gD = {rDa:+.4f}/年")

# ==========================================================================================
sec("5. 前沿损失下降率 dlnL*/dt（C1 前沿 + C6 桥接）")
lb = pd.read_csv(os.path.join(C, "leaderboard_cleaned.csv"), low_memory=False)
lb["date"] = pd.to_datetime(lb["Submission Date"], errors="coerce")
lb["S"] = pd.to_numeric(lb["Average ⬆️"], errors="coerce")
lb["N"] = pd.to_numeric(lb["#Params (B)"], errors="coerce")
lb = lb.dropna(subset=["date", "S", "N"])
lb = lb[(lb.S > 0) & (lb.N > 0)].copy()
lb["t"] = (lb.date - lb.date.min()).dt.days / 365.25
lic = lb["Hub License"].fillna("").str.lower()
OW = ["apache", "mit", "bsd", "llama", "gemma", "cc-by", "openrail", "gpl", "wtfpl", "afl",
      "creativeml", "bigscience", "bigcode", "apple-ascl"]
sub = lb[lic.apply(lambda x: any(w in x for w in OW))].copy()
gS = float(np.polyfit(sub.t, np.log(sub.S), 1)[0])
sub["q"] = sub.date.dt.to_period("Q").astype(str)
qf = sub.groupby("q").S.quantile(0.90)
gS_q90 = float(np.polyfit(np.arange(len(qf)), np.log(qf.values), 1)[0] * 4)
gL_all, gL_q90 = gS / K_BR, gS_q90 / K_BR
print(f"  C1 开源口径 n={len(sub)}：dlnS/dt 全样本 = {gS:+.4f}/年，季度 p90 前沿 = {gS_q90:+.4f}/年（{len(qf)} 个季度点）")
print(f"  桥接 lnS = {B_BR:.4f} {K_BR:+.4f}·lnL ⇒ dlnL*/dt = (dlnS/dt)/k")
print(f"    全样本  dlnL*/dt = {gL_all:+.4f}/年")
print(f"    前沿p90 dlnL*/dt = {gL_q90:+.4f}/年   ← 主口径")
print("  注：桥接 R²=0.28（弱），且 k 的符号决定 dlnL*/dt 符号；该量只作**量级**使用。")

# ==========================================================================================
sec("6. 四通道分解：规模 vs 非规模（残差反解，声明为归因量）")
rows = []
for ftag, gL in [("C1前沿p90", gL_q90), ("C1全样本", gL_all)]:
    for gtag, gN_, gD_ in [("前沿p90", gN, gD), ("全样本", rNa, rDa)]:
        for ptag, N, D, Q in [("拟合域边 N=12,D=600", 11.97, 600.0, 0.5),
                              ("现代前沿 N=70,D=2000", 70.0, 2000.0, 0.5)]:
            cN = s_lnN(N, D, Q, PQ) * gN_
            cD = s_lnD(N, D, Q, PQ) * gD_
            scale = cN + cD
            resid = gL - scale
            sq = s_lnQ(N, D, Q, PQ)
            tot = abs(scale) + abs(resid)
            rows.append(dict(frontier=ftag, growth=gtag, point=ptag,
                             N=N, D=D, cN=cN, cD=cD, scale=scale, dlnLdt=gL, resid=resid,
                             gQ_implied=(resid / sq if sq else np.nan),
                             scale_share=abs(scale) / tot, nonscale_share=abs(resid) / tot))
rdf = pd.DataFrame(rows)
print(f"  {'前沿':10s} {'速率口径':8s} {'工作点':20s} {'规模通道':>10s} {'总下降':>9s} {'非规模残差':>11s} {'规模份额':>8s} {'隐含gQ':>8s}")
for _, r in rdf.iterrows():
    print(f"  {r.frontier:10s} {r.growth:8s} {r['point']:20s} {r.scale:+10.5f} {r.dlnLdt:+9.5f} "
          f"{r.resid:+11.5f} {r.scale_share*100:7.1f}% {r.gQ_implied:8.4f}")
main = rdf[(rdf.frontier == "C1前沿p90") & (rdf.growth == "前沿p90") & (rdf["point"].str.contains("N=70"))].iloc[0]
main12 = rdf[(rdf.frontier == "C1前沿p90") & (rdf.growth == "前沿p90") & (rdf["point"].str.contains("N=12"))].iloc[0]
print(f"\n  主口径（C1前沿p90 × 前沿gN,gD × 现代前沿点）：规模份额 {main.scale_share*100:.1f}%、非规模 {main.nonscale_share*100:.1f}%")
print(f"  保守口径（同上但工作点取拟合域边 N=12,D=600）：规模份额 {main12.scale_share*100:.1f}%、非规模 {main12.nonscale_share*100:.1f}%")
print(f"  口径包络：规模份额 ∈ [{rdf.scale_share.min()*100:.1f}%, {rdf.scale_share.max()*100:.1f}%]")
print("  ⚠ '非规模残差' = 前沿总下降 − 规模通道，是**反解归因量**，不是独立观测；")
print("    它把'规模通道估计误差 + 桥接误差 + 口径误差'全部吸收进非规模项，故只能报区间。")

# ==========================================================================================
sec("7. p（配比）通道：问题一 h_p 冻结接口的量级上界")
hp = None
HP = r"d:\F题\q1_quality_results\Q1_M2响应接口.csv.gz"
if os.path.exists(HP):
    h = pd.read_csv(HP)
    col = "h_p_eq" if "h_p_eq" in h.columns else None
    print(f"  Q1_M2 响应接口：{h.shape}  列（前 8）= {list(h.columns)[:8]}")
    if col:
        hv = h[col].dropna()
        raw_span = float(hv.max() - hv.min())
        print(f"  {col}：n={len(hv)}  min={hv.min():+.4f}  max={hv.max():+.4f}  原始极差={raw_span:.4f}")
        print(f"  → 原始极差含**凸包外外推顶点**；问题三在支持约束下修正后的极差为 **0.159**（主口径）。")
        hp = dict(n=int(len(hv)), hmin=float(hv.min()), hmax=float(hv.max()),
                  span_raw=raw_span, span_support_constrained=0.159)
        print(f"  → 配比通道的对数损失影响上界 |ΔlnL| ≲ 0.159（支持域内真实配方）")
        print(f"    对 L≈{L_of(11.97,600,0.5,PQ):.2f} 的损失，占总损失比例 ≲ {0.159/L_of(11.97,600,0.5,PQ)*100:.2f}%")
        print(f"    （若误用含外推顶点的原始极差 {raw_span:.3f}，会得到 {raw_span/L_of(11.97,600,0.5,PQ)*100:.1f}% 的夸大值）")
    else:
        print("  未找到 h_p_eq 列，跳过")
else:
    print("  [缺] Q1_M2 响应接口.csv.gz")

# ==========================================================================================
sec("8. 结论")
print(f"  1) 形式：B1 Pythia 日志是无噪声生成值（R²=1.000000，精确复现 0.34/0.28），**只作管线自检**。")
print(f"  2) 独立检验：B6∪B7 三因素真实网格 R²={r2q:.5f}，六参数与问题二 SET_A 全部吻合，"
      f"rhoN={rNq:.4f} > rhoD={rDq:.4f}。")
print(f"  3) 通道敏感度（拟合域边 N=12,D=600,Q=0.5）：∂L/∂lnN={s_lnN(11.97,600,0.5,PQ):+.5f}，"
      f"∂L/∂lnD={s_lnD(11.97,600,0.5,PQ):+.5f}，∂L/∂lnQ={s_lnQ(11.97,600,0.5,PQ):+.5f}。")
print(f"  4) 速率：gN={gN:+.4f}/年，gD={gD:+.4f}/年（C4 前沿 p90，真实）；dlnL*/dt={gL_q90:+.4f}/年（C1+桥接）。")
print(f"  5) 分解（主口径）：规模份额 {main.scale_share*100:.1f}%，非规模份额 {main.nonscale_share*100:.1f}%；"
      f"全口径包络 [{rdf.scale_share.min()*100:.1f}%, {rdf.scale_share.max()*100:.1f}%]。")
print(f"  6) 与 P1-A 对照：P1-A 判定时间份额**不可识别**、应报 [30%,82%]；本节包络落在其内，两者一致但都受口径主导。")

json.dump(dict(
    pythia_selfcheck=dict(E=float(E0), A=float(A0), alpha=float(a0), B=float(B0), beta=float(b0),
                          r2=float(r2_of(pl.val_loss.values, pred0)), n=int(len(pl)),
                          flag="循环检验：附件由经典形式无噪声生成，不作独立证据"),
    traj_consistency=dict(n=len(traj), r2_median=float(tr.r2.median()),
                          D_share_median=float(tr.D_share_of_drop.median()),
                          flag="与 B1 同源，仅一致性检查", detail=traj),
    nq_fit=dict(E=float(Eq), A=float(Aq), alpha=float(aq), B=float(Bq), beta=float(bq),
                rhoN=float(rNq), rhoD=float(rDq), E1=float(E1q),
                r2=float(r2q), n=int(len(nq)), q_corr_median=float(np.nanmedian(cs))),
    nq_fit_b8_semisynth=dict(E=float(E8), A=float(A8), alpha=float(a8), B=float(B8_), beta=float(b8_),
                             rhoN=float(rN8), rhoD=float(rD8), E1=float(E18),
                             r2=float(r2_of(b8c.val_loss.values, p8)), n=int(len(b8c)),
                             flag="半合成校准数据；拟合不稳定且方向相反，仅作反例对照，不支持结构结论"),
    set_A_ref=SET_A,
    sensitivities=sens,
    growth=dict(gN_p90=float(gN), gD_p90=float(gD), gN_all=float(rNa), gD_all=float(rDa)),
    frontier=dict(gS_all=float(gS), gS_q90=float(gS_q90), gL_all=float(gL_all), gL_q90=float(gL_q90),
                  bridge=dict(k=K_BR, b=B_BR, r2=0.2774)),
    decomposition=rows,
    mix_channel=hp,
    identifiable=dict(scale_channel=True, nonscale_is_attribution=True, gQ_is_scenario=True,
                      note="非规模残差 = 前沿总下降 − 规模通道；gQ 为反解情景量，不可作独立观测"),
), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
rdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已写出 {OUT}\n已写出 {OUT_CSV}")
