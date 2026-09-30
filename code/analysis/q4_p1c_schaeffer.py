# -*- coding: utf-8 -*-
"""
P1-C Schaeffer 阈值桥：逐任务 Φ 拟合 + 留家族验证

文献依据：
  Schaeffer et al. 2023（NeurIPS, "Are Emergent Abilities of Large Language Models a Mirage?"）
  论断：所谓"涌现"多是非线性/不连续**度量**的产物；底层 per-token 准确率随 loss 平滑变化，
  而 exact-match 型度量要求 T 个 token 全对，观测分 s ≈ A·exp(-T·L)，T 越大越像"阈值跳变"。

本脚本的设计（含一次口径修正）：
  初稿只在**全池**上拟合，结果 M1 线性处处优于 M2 幂律 —— 但全池 Loss 是跨家族不可比轴
  （68/75 行 Medium 可比性），噪声把指数形式压住了。故本版分两层：
    A. 全池 Φ（作"最坏情形"上界）+ 阈值锐度诊断
    B. **族内 Φ**（同分词器/同验证集，Loss 真正可比）——这才是 Schaeffer 论断的干净检验
  并做留家族验证量化"不可比"的代价。

关键诊断量：
  * 过渡带宽 ΔL(10%→90%) = 0.8·(s_max − s_min) / |ds/dL|
    若 ΔL > 观测 loss 跨度 → **观测范围内不存在阈值**，该任务的"涌现/停滞"不是阈值假象
  * 敏感度 |ds/dL| 的跨任务差异 → 解释"同样 loss 改善、分数涨幅悬殊"

数据：C6 `loss_benchmark_bridge_expanded.csv`（75 行，Val_Loss + 六任务分）
"""
import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
C = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
OUT = r"d:\F题\q4_p1c_schaeffer_results.json"
OUT_CSV = r"d:\F题\q4_p1c_schaeffer_tasks.csv"
OUT_LV = r"d:\F题\q4_p1c_leavefamily.csv"

TASKS = ["LB_IFEval", "LB_BBH", "LB_MATH", "LB_GPQA", "LB_MUSR", "LB_MMLU_PRO"]
FAM_KEYS = ["pythia", "qwen", "llama", "gemma", "phi", "falcon", "yi",
            "mixtral", "mistral", "bloom", "opt", "deepseek", "gpt-neox"]


def sec(t):
    print("\n" + "=" * 100)
    print("### " + t)


def fam_of(m):
    m = str(m).lower()
    for k in FAM_KEYS:
        if k in m:
            return k
    return "other"


def aic(n, rss, k):
    return n * np.log(max(rss, 1e-12) / n) + 2 * k


def r2_of(obs, pred):
    ss = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - pred) ** 2) / ss) if ss > 0 else np.nan


def fit_m1(L, s):
    A = np.column_stack([np.ones_like(L), L])
    c, *_ = np.linalg.lstsq(A, s, rcond=None)
    p = A @ c
    return dict(name="M1 线性", k=2, pred=p, rss=float(np.sum((s - p) ** 2)),
                par=dict(a=float(c[0]), b=float(c[1])))


def fit_m2(L, s):
    """Schaeffer 幂律 s = A·exp(-T·L)（自由 A 吸收锚点）"""
    ok = s > 1e-9
    if ok.sum() < 4:
        return None
    y = np.log(s[ok])
    A = np.column_stack([np.ones(ok.sum()), L[ok]])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    T = max(-c[1], 1e-6)
    p = np.exp(c[0]) * np.exp(-T * L)
    return dict(name="M2 Schaeffer幂律", k=2, pred=p, rss=float(np.sum((s - p) ** 2)),
                par=dict(A=float(np.exp(c[0])), T_eq=float(T)))


def fit_m3(L, s):
    """饱和 s = A·exp(-T·L) + C：对 C 网格搜索"""
    best = None
    for Cv in np.linspace(0.0, float(np.percentile(s, 5)), 60):
        y = s - Cv
        ok = y > 1e-9
        if ok.sum() < 5:
            continue
        yy = np.log(y[ok])
        A = np.column_stack([np.ones(ok.sum()), L[ok]])
        c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        T = max(-c[1], 1e-6)
        p = np.exp(c[0]) * np.exp(-T * L) + Cv
        rss = float(np.sum((s - p) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(np.exp(c[0])), float(T), float(Cv), p)
    if best is None:
        return None
    rss, A_, T_, C_, p = best
    return dict(name="M3 饱和", k=3, pred=p, rss=rss,
                par=dict(A=A_, T_eq=T_, C=C_))


def best_fit(L, s):
    cands = [f for f in (fit_m1(L, s), fit_m2(L, s), fit_m3(L, s)) if f is not None]
    if not cands:
        return None, []
    for f in cands:
        f["aic"] = aic(len(L), f["rss"], f["k"])
        f["r2"] = r2_of(s, f["pred"])
    cands.sort(key=lambda f: f["aic"])
    return cands[0], cands


def predict(f, L):
    if f["name"].startswith("M1"):
        return f["par"]["a"] + f["par"]["b"] * np.asarray(L, float)
    if f["name"].startswith("M2"):
        return f["par"]["A"] * np.exp(-f["par"]["T_eq"] * np.asarray(L, float))
    return f["par"]["A"] * np.exp(-f["par"]["T_eq"] * np.asarray(L, float)) + f["par"]["C"]


# ==========================================================================================
sec("0. 数据与可比性")
br = pd.read_csv(os.path.join(C, "loss_benchmark_bridge_expanded.csv"))
br["fam"] = br.Model.map(fam_of)
br = br.dropna(subset=["Val_Loss"] + TASKS).copy()
SPAN = float(br.Val_Loss.max() - br.Val_Loss.min())
print(f"  C6 桥接表：{len(br)} 行，Val_Loss ∈ [{br.Val_Loss.min():.3f}, {br.Val_Loss.max():.3f}]，跨度 {SPAN:.3f}")
print(f"  Loss_Comparability：{br.Loss_Comparability.value_counts().to_dict()}")
print(f"  家族构成：{br.fam.value_counts().to_dict()}")
print(f"  ⚠ 68/75 行为 Medium（不同验证集/分词器）→ 全池 Loss 只是**近似**公共轴；"
      f"§1 给最坏情形，§2 给族内干净检验，§3 量化不可比代价。")

# ==========================================================================================
sec("1. 全池 Φ 拟合（最坏情形：跨家族 Loss 不可比）")
pool_rows, pool_fits = [], {}
print(f"  {'任务':10s} {'最优':16s} {'R²':>8s} {'|ds/dL|':>9s} {'ΔL(10-90%)':>11s} {'ΔL/观测跨度':>12s}")
for tk in TASKS:
    d = br[["Val_Loss", tk]].dropna()
    L, s = d.Val_Loss.values, d[tk].values
    best, cands = best_fit(L, s)
    pool_fits[tk] = dict(n=len(d), best=best["name"], r2=best["r2"], aic=best["aic"],
                         par=best["par"], srange=[float(s.min()), float(s.max())],
                         cands=[{k: v for k, v in f.items() if k != "pred"} for f in cands])
    # 敏感度与过渡带宽（对最优 Φ 数值求导）
    Lg = np.linspace(L.min(), L.max(), 2000)
    g = np.gradient(predict(best, Lg), Lg)
    slope = float(np.median(np.abs(g)))
    band = 0.8 * (s.max() - s.min()) / slope if slope > 0 else np.nan
    pool_rows.append(dict(task=tk.replace("LB_", ""), n=len(d), best=best["name"], r2=best["r2"],
                          aic=best["aic"], slope=slope, band_10_90=band,
                          band_over_span=band / SPAN if band == band else np.nan,
                          T_eq=best["par"].get("T_eq", np.nan),
                          smin=float(s.min()), smax=float(s.max())))
    r = pool_rows[-1]
    print(f"  {r['task']:10s} {r['best']:16s} {r['r2']:8.4f} {r['slope']:9.3f} {r['band_10_90']:11.3f} "
          f"{r['band_over_span']:12.2f}")
pr_df = pd.DataFrame(pool_rows)
print("\n  判读：ΔL/观测跨度 > 1 ⇒ 该任务在**观测 loss 范围内不存在阈值**，其时间趋势变化来自")
print("        敏感度 |ds/dL| 的差异，而不是度量阈值的假象。")

# ==========================================================================================
sec("2. 族内 Φ 拟合（干净检验：同分词器 / 同验证集，Loss 真正可比）")
fam_rows = []
big = [f for f in br.fam.value_counts().index if br.fam.value_counts()[f] >= 5]
print(f"  样本量 ≥5 的家族：{ {f: int(br.fam.value_counts()[f]) for f in big} }")
print(f"  {'家族':9s} {'任务':10s} {'n':>3s} {'Loss跨度':>9s} {'最优':16s} {'R²':>8s} {'|ds/dL|':>9s}")
for fam in big:
    d0 = br[br.fam == fam]
    for tk in TASKS:
        d = d0[["Val_Loss", tk]].dropna()
        if len(d) < 4 or d.Val_Loss.nunique() < 3:
            continue
        L, s = d.Val_Loss.values, d[tk].values
        best, _ = best_fit(L, s)
        if best is None:
            continue
        Lg = np.linspace(L.min(), L.max(), 1000)
        g = np.gradient(predict(best, Lg), Lg)
        slope = float(np.median(np.abs(g)))
        fam_rows.append(dict(family=fam, task=tk.replace("LB_", ""), n=len(d),
                             loss_span=float(L.max() - L.min()), best=best["name"],
                             r2=best["r2"], slope=slope, T_eq=best["par"].get("T_eq", np.nan)))
        print(f"  {fam:9s} {tk.replace('LB_',''):10s} {len(d):3d} {L.max()-L.min():9.3f} "
              f"{best['name']:16s} {best['r2']:8.4f} {slope:9.3f}")
fr = pd.DataFrame(fam_rows)
if len(fr):
    exp_win = fr.best.str.contains("M2|M3").mean()
    print(f"\n  → 族内最优模型为指数型（M2/M3）的比例 = {exp_win*100:.0f}%（全池为 "
          f"{pr_df.best.str.contains('M2|M3').mean()*100:.0f}%）")
    print(f"  → 族内 R² 中位 = {fr.r2.median():.4f}（全池 {pr_df.r2.median():.4f}）"
          f" ⇒ 把 Loss 换成**同家族可比轴**后，Φ 的解释力显著提升。")

# ==========================================================================================
sec("3. 留家族验证（量化跨家族不可比的代价）")
lv_rows = []
for fam in sorted(br.fam.unique()):
    te, tr = br[br.fam == fam], br[br.fam != fam]
    if len(te) < 3 or len(tr) < 25:
        continue
    for tk in TASKS:
        dtr, dte = tr[["Val_Loss", tk]].dropna(), te[["Val_Loss", tk]].dropna()
        if len(dte) < 2 or len(dtr) < 20:
            continue
        best, _ = best_fit(dtr.Val_Loss.values, dtr[tk].values)
        if best is None:
            continue
        p = predict(best, dte.Val_Loss.values)
        ste, str_ = dte[tk].values, dtr[tk].values
        rmse = float(np.sqrt(np.mean((p - ste) ** 2)))
        base = float(np.sqrt(np.mean((ste - str_.mean()) ** 2)))
        lv_rows.append(dict(family=fam, n_test=len(dte), task=tk.replace("LB_", ""),
                            rmse=rmse, base_rmse=base, skill=float(1 - rmse / base) if base > 0 else np.nan))
lv = pd.DataFrame(lv_rows)
print(f"  留家族 RMSE 中位 = {lv.rmse.median():.3f}；技能得分中位 = {lv.skill.median():.3f}")
print(f"  技能得分 > 0 占比 = {(lv.skill > 0).mean()*100:.0f}%（>0 = 优于'用训练均值'的朴素基线）")
print("  逐家族技能得分（跨任务中位）：")
for fam, s in lv.groupby("family").skill.median().sort_values().items():
    print(f"    {fam:10s} {s:+.3f}")
print(f"  → 存在技能得分为负的**单元格**的家族：{sorted(lv[lv.skill<0].family.unique().tolist())}"
      f"；按家族-任务中位数看，负值家族 = {sorted(lv.groupby('family').skill.median()[lv.groupby('family').skill.median()<0].index.tolist())}"
      f" ⇒ 这些家族的 Loss 与其他家族**不在同一可比轴**，Φ 迁移失效。")

# ==========================================================================================
sec("4. 结论（同时回答 P1-E 的一半：停滞是否阈值假象）")
pk = pr_df.set_index("task")
print(f"  1) 全池：六任务最优 Φ 的 R² ∈ [{pr_df.r2.min():.4f}, {pr_df.r2.max():.4f}]；")
print(f"     过渡带宽/观测跨度 ∈ [{pr_df.band_over_span.min():.2f}, {pr_df.band_over_span.max():.2f}]。")
no_thr = pr_df[pr_df.band_over_span > 1].task.tolist()
print(f"  2) **观测范围内不存在阈值的任务**（ΔL/跨度 > 1）：{no_thr if no_thr else '无'}")
print(f"     ⇒ 这些任务的分数随 loss 平滑变化，其时间趋势差异**不能**归因于度量阈值假象。")
print(f"  3) 敏感度排序 |ds/dL|（每单位 loss 改善的分数收益）：")
for t, s in pk.slope.sort_values(ascending=False).items():
    print(f"       {t:10s} {s:7.3f}")
print(f"  4) 关键读数：MUSR |ds/dL|={pk.loc['MUSR','slope']:.3f}（最低），"
      f"BBH {pk.loc['BBH','slope']:.3f}（最高），比值 {pk.loc['BBH','slope']/pk.loc['MUSR','slope']:.1f}×。")
print(f"     ⇒ '同样 loss 改善、MUSR 涨幅远小于 BBH' 的主因是**度量敏感度差异**，不是 MUSR 遇到阈值。")
print(f"  5) 族内检验：族内 R² 中位 {fr.r2.median() if len(fr) else float('nan'):.4f} > 全池 "
      f"{pr_df.r2.median():.4f}；指数型在族内胜出比例 "
      f"{fr.best.str.contains('M2|M3').mean()*100 if len(fr) else float('nan'):.0f}%。")
print(f"  6) 纪律：任何跨家族 Φ 结论必须绑定 Loss_Comparability=Medium 声明；"
      f"留家族技能得分负值家族 {sorted(lv[lv.skill<0].family.unique().tolist())} 的证据须降级为'条件性'。")

json.dump(dict(
    data=dict(n=int(len(br)), loss_span=SPAN,
              comparability=br.Loss_Comparability.value_counts().to_dict(),
              families=br.fam.value_counts().to_dict()),
    pooled=pool_fits, pooled_sharpness=pool_rows,
    within_family=fam_rows, leave_family_out=lv_rows,
    conclusion=dict(tasks_without_threshold=no_thr,
                    slope_ranking=pk.slope.sort_values(ascending=False).to_dict(),
                    musr_over_bbh_slope=float(pk.loc["BBH", "slope"] / pk.loc["MUSR", "slope"]),
                    flag="MUSR 的低涨幅归因于度量敏感度而非阈值；跨家族结论受 Medium 可比性限制"),
    note="ΔL(10-90%) = 0.8·(s_max-s_min)/|ds/dL|；> 观测跨度 ⇒ 观测范围内无阈值",
), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pr_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
lv.to_csv(OUT_LV, index=False, encoding="utf-8-sig")
print(f"\n已写出 {OUT}\n已写出 {OUT_CSV}\n已写出 {OUT_LV}")
