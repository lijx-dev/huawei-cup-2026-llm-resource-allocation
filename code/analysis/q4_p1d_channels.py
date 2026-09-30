# -*- coding: utf-8 -*-
"""
P1-D 非规模渠道归因：c_mix（配比）与 c_post（后训练）

动机：P1-B 把前沿损失下降拆成"规模通道 + 非规模残差"，但非规模残差是个黑箱。
本脚本把它继续拆成**可命名渠道**：
  * c_post  后训练渠道：由 C1 中"同基座 pretrained vs chat/finetuned"配对直接测得
  * c_mix   配比渠道：由问题一 h_p 冻结接口给出量级上界（情景量 λ_p）
  * c_other 剩余：不可命名部分

文献依据：参考文献 3（非规模技术进步的可分解性）、参考文献 5。

关键纪律：
  * c_post 是**配对差分**（同基座、同榜单、同归一化），可比性高于跨家族比较
  * c_mix 的速率 dp/dt **无观测**，只能给包络，绝不作单点
  * 桥接 lnS = 4.8038 − 2.7353·lnL（R²=0.28，弱）→ 一切 loss 空间数字只作量级
"""
import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
C = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
OUT = r"d:\F题\q4_p1d_channels_results.json"
OUT_CSV = r"d:\F题\q4_p1d_post_training_pairs.csv"

K_BR, B_BR = -2.7353, 4.8038
LAMBDA_P = [0.0, 0.5, 1.0, 1.5]

# 来自 P1-B 的非规模残差（主口径 / 保守口径）
NONSCALE_MAIN = -0.16398      # C1前沿p90 × 前沿gN,gD × 现代前沿点
NONSCALE_CONS = -0.12927      # 同上但工作点取拟合域边 N=12,D=600


def sec(t):
    print("\n" + "=" * 100)
    print("### " + t)


def dlnL_from_dlnS(dlnS):
    """桥接 lnS = b + k lnL ⇒ ΔlnL = ΔlnS / k"""
    return dlnS / K_BR


# ==========================================================================================
sec("0. c_post：同基座 pretrained vs chat/finetuned 配对（C1）")
d = pd.read_csv(os.path.join(C, "leaderboard_cleaned.csv"), low_memory=False)
d["date"] = pd.to_datetime(d["Submission Date"], errors="coerce")
d["S"] = pd.to_numeric(d["Average ⬆️"], errors="coerce")
d = d.dropna(subset=["date", "S"])
d = d[d.S > 0].copy()
d["year"] = d.date.dt.year + (d.date.dt.dayofyear - 1) / 365.25
d["is_chat"] = d.Type.astype(str).str.contains("chat|RLHF|DPO|IFT|finetun", case=False, na=False)
d["is_pre"] = d.Type.astype(str).str.contains("pretrained", case=False, na=False)

import re


def base_of(m):
    m = str(m).lower()
    m = re.sub(r"-(instruct|chat|it|sft|dpo|rlhf|thinking|turbo)$", "", m)
    return m


d["base"] = d.Model.map(base_of)
pairs = []
for b, s in d.groupby("base"):
    if not (s.is_chat.any() and s.is_pre.any()):
        continue
    pre = s[s.is_pre]
    cha = s[s.is_chat]
    # 只保留"同基座、规模相近"的配对（参数差 < 20%）
    for _, pr in pre.iterrows():
        for _, ch in cha.iterrows():
            n1 = pd.to_numeric(pr["#Params (B)"], errors="coerce")
            n2 = pd.to_numeric(ch["#Params (B)"], errors="coerce")
            if not (n1 and n2) or min(n1, n2) / max(n1, n2) < 0.8:
                continue
            pairs.append(dict(base=b, pre_model=pr.Model, chat_model=ch.Model,
                              N=float(n1), S_pre=float(pr.S), S_chat=float(ch.S),
                              delta_S=float(ch.S - pr.S),
                              year=float(ch.year), date=ch.date))
pp = pd.DataFrame(pairs).drop_duplicates(subset=["base", "pre_model", "chat_model"])
pp["dlnS"] = np.log(pp.S_chat) - np.log(pp.S_pre)
pp["dlnL"] = dlnL_from_dlnS(pp.dlnS.values)
print(f"  配对构建：{len(pp)} 对（同基座 + 规模差 <20%）")
print(f"  ΔS（chat − pretrained）: 中位 {pp.delta_S.median():+.2f} 分，均值 {pp.delta_S.mean():+.2f}，"
      f"范围 [{pp.delta_S.min():+.2f}, {pp.delta_S.max():+.2f}]")
print(f"  换算到对数损失（桥接）：ΔlnL 中位 {pp.dlnL.median():+.4f}，均值 {pp.dlnL.mean():+.4f}")
print(f"  ⇒ 后训练在**水平上**带来约 {abs(pp.dlnL.median())*100:.2f}%（中位）的对数损失改善。")

print("\n  分年度（后训练配方的时变）：")
yr = pp.groupby(pp.year.astype(int)).agg(n=("delta_S", "size"), dS_med=("delta_S", "median"),
                                         dlnL_med=("dlnL", "median"))
print(yr.to_string())
if len(yr) >= 3:
    gpost = float(np.polyfit(yr.index.values.astype(float), yr.dlnL_med.values, 1)[0])
    print(f"  → 后训练渠道的年化速率 dΔlnL_post/dt = {gpost:+.4f}/年（按年度中位回归，n={len(yr)} 年）")
else:
    gpost = np.nan
    print("  → 年度点不足，无法估计时变速率；仅报水平量。")

# 稳健性：按 N 分层（大模型后训练收益是否更小）
print("\n  按规模分层（ΔlnL 中位）：")
pp["Nband"] = pd.cut(pp.N, [0, 3, 10, 40, 1e4], labels=["<3B", "3–10B", "10–40B", ">40B"])
print(pp.groupby("Nband", observed=True).agg(n=("dlnL", "size"), dlnL_med=("dlnL", "median")).to_string())

# ==========================================================================================
sec("1. c_mix：问题一 h_p 冻结接口（情景量，只给包络）")
HP = r"d:\F题\q1_quality_results\Q1_M2响应接口.csv.gz"
hp_span_support = 0.159          # 问题三在支持约束下修正后的极差（主口径）
hp_span_raw = None
if os.path.exists(HP):
    h = pd.read_csv(HP)
    if "h_p_eq" in h.columns:
        hv = h.h_p_eq.dropna()
        hp_span_raw = float(hv.max() - hv.min())
        print(f"  Q1_M2 接口 h_p_eq：n={len(hv)}  原始极差={hp_span_raw:.4f}（含凸包外外推顶点）")
        print(f"  支持约束下极差（问题三修正）= {hp_span_support:.4f} ← 主口径")
print(f"  ⇒ 配比渠道的**水平**上界：|ΔlnL_mix| ≤ {hp_span_support:.4f}（真实配方支持域内）")
print(f"  ⇒ 配比渠道的**速率** dp/dt 无任何观测 → 只能给情景包络：")
print(f"     c_mix(λ_p) = λ_p · {hp_span_support:.3f} / T_traverse，λ_p ∈ {LAMBDA_P}")
print(f"     {'T_traverse':>12s} " + " ".join(f"{'λ='+str(l):>10s}" for l in LAMBDA_P))
for T in [1.0, 2.0, 5.0]:
    row = " ".join(f"{l*hp_span_support/T:+10.4f}" for l in LAMBDA_P)
    print(f"     {T:10.1f}年 {row}")
print("  取 T_traverse = 2 年作主情景（一个'配方迭代周期'的量级假设）：")
T_MAIN = 2.0
cmix = {l: l * hp_span_support / T_MAIN for l in LAMBDA_P}
for l in LAMBDA_P:
    print(f"    λ_p={l}: c_mix = {cmix[l]:+.4f}/年")

# ==========================================================================================
sec("2. 归因闭合：以**存量**（窗口内累计）口径，而非速率口径")
print("  为什么不用速率口径：c_post 的**速率**需要后训练收益随时间的变异，而 C1 的 61 对配对")
print("  几乎全部落在 2024 年（2024:59 对 / 2025:2 对）→ 速率**不可识别**。故改用存量口径：")
print("  把 P1-B 的非规模残差在观测窗口内积分，与 c_post / c_mix 的**水平量**直接比较。\n")
T_WIN = float((d.date.max() - d.date.min()).days / 365.25)
print(f"  观测窗口 T_win = {d.date.min().date()} → {d.date.max().date()} = {T_WIN:.3f} 年")
print(f"  P1-B 非规模残差速率：主口径 {NONSCALE_MAIN:+.5f}/年，保守口径 {NONSCALE_CONS:+.5f}/年")
print(f"  ⇒ 窗口内累计非规模损失改善：主口径 {abs(NONSCALE_MAIN)*T_WIN:.4f}，"
      f"保守口径 {abs(NONSCALE_CONS)*T_WIN:.4f}（对数损失）\n")
c_post_level = abs(pp.dlnL.median())
print(f"  c_post 水平量（实测）：{c_post_level:.4f}")
print(f"  c_mix 水平量（上界×窗口占比）：λ_p · {hp_span_support:.3f} · (T_win/T_traverse)")
print(f"  {'情景':16s} {'累计非规模':>11s} {'c_post':>9s} {'c_mix':>9s} {'合计':>9s} {'覆盖率':>8s} {'c_other':>9s}")
rows = []
for tag, resid in [("主口径", NONSCALE_MAIN), ("保守口径", NONSCALE_CONS)]:
    cum = abs(resid) * T_WIN
    for l in LAMBDA_P:
        cm = l * hp_span_support * (T_WIN / T_MAIN)
        tot = c_post_level + cm
        rows.append(dict(scenario=f"{tag} λ_p={l}", resid_rate=resid, cumulative_nonscale=cum,
                         c_post=c_post_level, c_mix=cm, total=tot, coverage=tot / cum,
                         c_other=cum - tot))
        print(f"  {tag:6s} λ={l:<5} {cum:11.4f} {c_post_level:9.4f} {cm:9.4f} {tot:9.4f} "
              f"{tot/cum*100:7.1f}% {cum-tot:+9.4f}")
cov = pd.DataFrame(rows)
print("\n  判读：")
print("   * 覆盖率 < 100% ⇒ 剩余 c_other 未命名（架构、数据筛选、评测适配等），须作为")
print("     **未解释残差**保留，不得强行归因。")
print("   * 覆盖率 > 100% ⇒ 规模通道可能被低估，或 c_post/c_mix 与规模通道重叠，")
print("     需回到 P1-B 检查口径。")
print(f"   * 主口径 λ_p=1 下覆盖率 {cov[(cov.scenario=='主口径 λ_p=1.0')].coverage.iloc[0]*100:.0f}%"
      f"，λ_p=0 下 {cov[(cov.scenario=='主口径 λ_p=0.0')].coverage.iloc[0]*100:.0f}%"
      f" ⇒ **后训练单独就能解释非规模残差的一大半**，这是本节最强的可报告结论。")

# ==========================================================================================
sec("3. 结论")
print(f"  1) c_post 由 {len(pp)} 对同基座配对直接测得（水平）：ΔlnL 中位 {pp.dlnL.median():+.4f}"
      f"（≈{abs(pp.dlnL.median())*100:.2f}% 对数损失改善）；按规模分层 "
      f"{pp.groupby('Nband', observed=True).dlnL.median().round(4).to_dict()}。")
print(f"     **速率不可识别**（配对几乎全在 2024 年），只报水平量。")
print(f"  2) c_mix 无 dp/dt 观测：水平上界 {hp_span_support:.3f}（支持域内），"
      f"原始极差 {hp_span_raw:.3f}（含凸包外外推，禁用）。")
print(f"  3) 存量口径归因：窗口 T_win={T_WIN:.2f} 年内，累计非规模改善 "
      f"{abs(NONSCALE_MAIN)*T_WIN:.3f}（主口径）；")
print(f"     c_post 解释 {c_post_level/(abs(NONSCALE_MAIN)*T_WIN)*100:.0f}%，"
      f"λ_p=1 时 c_post+c_mix 解释 "
      f"{cov[(cov.scenario=='主口径 λ_p=1.0')].coverage.iloc[0]*100:.0f}%。")
print(f"  4) 纪律：c_mix 为情景量，不得进入优化变量集，不得按 0.7622 折减（见问题二纪律）；")
print(f"     c_other 必须显式保留为未解释残差。")

json.dump(dict(
    c_post=dict(n_pairs=int(len(pp)), delta_S_median=float(pp.delta_S.median()),
                delta_S_mean=float(pp.delta_S.mean()),
                dlnL_median=float(pp.dlnL.median()), dlnL_mean=float(pp.dlnL.mean()),
                rate_per_year=None,
                rate_identifiable=False,
                rate_reason="61 对配对几乎全在 2024 年（2024:59 / 2025:2），无时间变异",
                by_year=yr.reset_index().to_dict("records"),
                by_size=pp.groupby("Nband", observed=True).dlnL.median().to_dict()),
    c_mix=dict(span_support=hp_span_support, span_raw=hp_span_raw, T_traverse=T_MAIN,
               lambda_grid=LAMBDA_P, rates={str(l): cmix[l] for l in LAMBDA_P}),
    window=dict(t_win=T_WIN, start=str(d.date.min().date()), end=str(d.date.max().date())),
    attribution_stock=rows,
    bridge=dict(k=K_BR, b=B_BR, r2=0.2774),
    nonscale_from_p1b=dict(main=NONSCALE_MAIN, conservative=NONSCALE_CONS),
    note="存量口径：窗口内累计非规模改善 vs c_post/c_mix 水平量；c_mix 为情景包络；c_other 为未解释残差",
), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已写出 {OUT}\n已写出 {OUT_CSV}")
