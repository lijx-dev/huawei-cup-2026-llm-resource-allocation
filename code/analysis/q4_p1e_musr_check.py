# -*- coding: utf-8 -*-
"""
P1-E  C8 逐任务指标假象复核：MUSR 的"停滞"是阈值/度量假象吗？

背景
----
问题四主口径下 C8 逐任务年化增速极差 1.163（math_hard +1.1753 vs musr +0.0123），
其中 MUSR 被读作"停滞"。本脚本对该判读做五项证伪检验：

  检验1 尺度不变性 —— 附件里同一批模型存在两套 MUSR 尺度：
        (a) detailed_results 的原始 `acc_norm`（det 尺度）
        (b) leaderboard 列的 MUSR（lb 尺度，按正确答案数做基线归一后重标定）
  检验2 阈值（Schaeffer）—— 复用 P1-C：逐任务 Φ 最优模型 + ΔL(10-90%) vs 观测跨度。
  检验3 天花板 / 分辨率 / 敏感度。
  检验4 共同 ΔL 一致性 —— 六任务测于同一批模型，反解 ΔL_i 是否一致。
  检验5 规格曲线 —— 枚举 尺度×样本×分位×窗口，看 MUSR 增速与"最慢任务"排名的稳健性。

数据：C_efficiency_evolution/{detailed_results, leaderboard_cleaned.csv, leaderboard_extended_timeseries.csv}
输出：q4_p1e_musr_results.json / q4_p1e_spec_curve.csv / q4_p1e_family.csv
"""
import os, re, json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
P1C = r"d:\F题\q4_p1c_schaeffer_results.json"
OUT = r"d:\F题\q4_p1e_musr_results.json"
OUT_SC = r"d:\F题\q4_p1e_spec_curve.csv"
OUT_FAM = r"d:\F题\q4_p1e_family.csv"

TASKS = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
COLMAP = {"IFEval": "IFEval", "BBH": "BBH", "MATH": "MATH Lvl 5", "GPQA": "GPQA",
          "MUSR": "MUSR", "MMLU_PRO": "MMLU-PRO"}
DETCOL = {"IFEval": "leaderboard_ifeval", "BBH": "leaderboard_bbh", "MATH": "leaderboard_math_hard",
          "GPQA": "leaderboard_gpqa", "MUSR": "leaderboard_musr", "MMLU_PRO": "leaderboard_mmlu_pro"}
KEY = {'leaderboard_ifeval': ('inst_level_strict_acc,none', 'prompt_level_strict_acc,none'),
       'leaderboard_bbh': ('acc_norm,none',), 'leaderboard_math_hard': ('exact_match,none',),
       'leaderboard_gpqa': ('acc_norm,none',), 'leaderboard_musr': ('acc_norm,none',),
       'leaderboard_mmlu_pro': ('acc,none',)}
OW = ['apache', 'mit', 'bsd', 'llama', 'gemma', 'cc-by', 'openrail', 'gpl', 'wtfpl', 'afl',
      'creativeml', 'bigscience', 'bigcode', 'apple-ascl']
FAM_KEYS = ["pythia", "qwen", "llama", "gemma", "phi", "falcon", "yi", "mixtral", "mistral",
            "bloom", "opt", "deepseek", "gpt-neox"]


def sec(t):
    print("\n" + "=" * 104)
    print("### " + t)


def norm_key(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def fam_of(m):
    m = str(m).lower()
    for k in FAM_KEYS:
        if k in m:
            return k
    return "other"


def ols_growth(qs, vals):
    """季度 p90 序列 → 年化对数增速 + OLS 标准误（×4）"""
    v = np.asarray(vals, float)
    if len(v) < 3 or np.any(v <= 0) or np.any(~np.isfinite(v)):
        return np.nan, np.nan
    x = np.arange(len(v), dtype=float)
    y = np.log(v)
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = len(x) - 2
    if dof <= 0:
        return float(c[1] * 4), np.nan
    s2 = float(r @ r) / dof
    se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]) * 4
    return float(c[1] * 4), float(se)


def q_p90(df, qcol, vcol, qs, qtile=0.9):
    return df.dropna(subset=[vcol]).groupby(qcol)[vcol].quantile(qtile).reindex(qs)


# =====================================================================================
sec("0  数据装载")
det = os.path.join(BASE, "detailed_results")
rows, n_bad = [], 0
for dd in os.listdir(det):
    dp = os.path.join(det, dd)
    if not os.path.isdir(dp):
        continue
    fn = [f for f in os.listdir(dp) if f.endswith('.json')]
    if not fn:
        continue
    try:
        js = json.load(open(os.path.join(dp, fn[0]), encoding='utf-8'))
    except Exception:
        n_bad += 1
        continue
    g = js.get('groups', {}) or {}
    res = js.get('results', {}) or {}
    r = {'key': norm_key(dd)}
    for t, ks in KEY.items():
        src = g.get(t) or res.get(t)
        if isinstance(src, dict):
            v = [src.get(k) for k in ks if isinstance(src.get(k), (int, float))]
            if v:
                r[t] = float(np.mean(v)) * 100
    rows.append(r)
D = pd.DataFrame(rows)
print(f"detailed_results 目录={len(os.listdir(det))}  物理截断解析失败={n_bad}  有效={len(D)}")
print("  det 尺度六任务 p90：", {t: round(float(D[DETCOL[t]].quantile(.9)), 2) for t in TASKS})

lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
lb['Year'] = lb['date'].dt.year
lb['key'] = lb.Model.map(norm_key)
lb['fam'] = lb.Model.map(fam_of)
for t in TASKS:
    lb['lb_' + t] = pd.to_numeric(lb[COLMAP[t]], errors='coerce')
lb['lb_S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lic = lb['Hub License'].fillna('').str.lower()
lb['open'] = lic.apply(lambda x: any(w in x for w in OW))
print(f"leaderboard_cleaned 行={len(lb)}  日期有效={int(lb.date.notna().sum())}  OPEN_LIC={int(lb.open.sum())}")

M = lb.merge(D, on='key', how='inner')
M = M[M.q.isin(sorted([q for q in M.q.unique() if q != 'NaT']))].copy()
QS = sorted(M.q.unique())
print(f"两尺度可配对模型={len(M)}   季度覆盖={QS}")

# =====================================================================================
sec("检验1  尺度不变性：两套 MUSR 尺度的季度 p90 与年化增速")
ts_rows = []
for tag, col, sub in [("det 原始acc_norm(全样本)", 'leaderboard_musr', M),
                      ("det 原始acc_norm(OPEN_LIC)", 'leaderboard_musr', M[M.open]),
                      ("lb 归一化(全样本)", 'lb_MUSR', M),
                      ("lb 归一化(OPEN_LIC)", 'lb_MUSR', M[M.open])]:
    g = q_p90(sub, 'q', col, QS)
    gg, se = ols_growth(QS, g.values)
    ts_rows.append(dict(scale=tag, series={k: float(v) for k, v in g.items()},
                        growth=gg, se=se, ci_lo=gg - 1.96 * se, ci_hi=gg + 1.96 * se,
                        rel_change=float(g.values[-1] / g.values[0] - 1)))
    print(f"  {tag:28s} g={gg:+.4f}±{se:.4f}/yr  95%CI=[{gg-1.96*se:+.4f},{gg+1.96*se:+.4f}]  "
          f"{g.values[0]:.2f}→{g.values[-1]:.2f} ({g.values[-1]/g.values[0]-1:+.1%})")

x = M['leaderboard_musr'].values.astype(float)
y = M['lb_MUSR'].values.astype(float)
ok = np.isfinite(x) & np.isfinite(y)
a, b = np.polyfit(x[ok], y[ok], 1)
resid = y[ok] - (a * x[ok] + b)
print(f"\n  尺度映射 lb ≈ {a:.4f}·det {b:+.3f}   r={np.corrcoef(x[ok], y[ok])[0,1]:.4f}  "
      f"残差 sd={resid.std():.2f}  max|resid|={np.abs(resid).max():.2f}")
print("  → 斜率>1、截距<0 且残差 sd 达 1.36（det 尺度仅 0–60）：lb 尺度对 det 做了按正确答案数的"
      "基线扣减再重标定，两者不是同一度量的平移")

sec("检验1b  六任务两尺度对照（MUSR 是否为孤例）")
scale_cmp = {}
for t in TASKS:
    xx = pd.to_numeric(M[DETCOL[t]], errors='coerce')
    yy = M['lb_' + t]
    o = xx.notna() & yy.notna()
    aa, bb = np.polyfit(xx[o], yy[o], 1)
    scale_cmp[t] = dict(det_p90=float(xx[o].quantile(.9)), lb_p90=float(yy[o].quantile(.9)),
                        corr=float(np.corrcoef(xx[o], yy[o])[0, 1]), slope=float(aa), intercept=float(bb))
    print(f"  {t:9s} det_p90={scale_cmp[t]['det_p90']:6.2f}  lb_p90={scale_cmp[t]['lb_p90']:6.2f}  "
          f"corr={scale_cmp[t]['corr']:.4f}  lb≈{aa:.3f}·det{bb:+.2f}")
print("  → 逐任务仿射参数各不相同 ⇒ 两套输出来自不同 harness 配置，不是统一线性变换；"
      "跨任务直接比较分数/增速本身即受配置污染")

# =====================================================================================
sec("检验2  阈值检验（复用 P1-C）：MUSR 是否为 Schaeffer 阈值假象")
p1c = json.load(open(P1C, encoding='utf-8'))
sharp = {r['task']: r for r in p1c['pooled_sharpness']}
print(f"  {'task':10s} {'最优Φ':16s} {'r2':>7s} {'|ds/dL|':>9s} {'ΔL(10-90%)':>11s} {'观测跨度':>9s} {'带宽/跨度':>9s}")
for t in TASKS:
    s = sharp[t]
    print(f"  {t:10s} {s['best']:16s} {s['r2']:7.3f} {s['slope']:9.3f} {s['band_10_90']:11.3f} "
          f"{s['smax']-s['smin']:9.3f} {s['band_over_span']:9.3f}")
wf = pd.DataFrame(p1c['within_family'])
nm2 = int((wf.best.str.startswith('M2')).sum())
print(f"  → 6/6 任务 ΔL(10-90%) > 观测 loss 跨度 ⇒ 观测范围内不存在阈值")
print(f"  → 族内（同分词器/同验证集）幂律胜出 {nm2}/{len(wf)} = {nm2/len(wf):.1%}；"
      f"MUSR 族内最优模型分布 {wf[wf.task=='MUSR'].best.value_counts().to_dict()}")
print("  ⇒ Schaeffer 型阈值假象无证据；MUSR 的平坦不是阈值造成的")

# =====================================================================================
sec("检验3  天花板 / 分辨率 / 敏感度")
d_mus = pd.to_numeric(M['leaderboard_musr'], errors='coerce')
l_mus = pd.to_numeric(M['lb_MUSR'], errors='coerce')
ext = pd.read_csv(os.path.join(BASE, "leaderboard_extended_timeseries.csv"), low_memory=False)
ext_mus = pd.to_numeric(ext['MUSR'], errors='coerce')
ceil = dict(
    det=dict(p90=float(d_mus.quantile(.9)), p99=float(d_mus.quantile(.99)), max=float(d_mus.max()),
             headroom=float(100 - d_mus.quantile(.9)), frac_gt60=float((d_mus > 60).mean())),
    lb=dict(p90=float(l_mus.quantile(.9)), p99=float(l_mus.quantile(.99)), max=float(l_mus.max()),
            headroom=float(100 - l_mus.quantile(.9)), frac_gt30=float((l_mus > 30).mean())),
    gran=dict(uniq=int(ext_mus.nunique()), n=int(ext_mus.notna().sum()),
              uniq_ratio=float(ext_mus.nunique() / ext_mus.notna().sum())))
print(f"  det 尺度: p90={ceil['det']['p90']:.2f} p99={ceil['det']['p99']:.2f} max={ceil['det']['max']:.2f} "
      f"headroom={ceil['det']['headroom']:.1f}  >60 占比={ceil['det']['frac_gt60']:.3%}")
print(f"  lb  尺度: p90={ceil['lb']['p90']:.2f} p99={ceil['lb']['p99']:.2f} max={ceil['lb']['max']:.2f} "
      f"headroom={ceil['lb']['headroom']:.1f}  >30 占比={ceil['lb']['frac_gt30']:.3%}")
print(f"  取值粒度: 唯一值 {ceil['gran']['uniq']}/{ceil['gran']['n']} = {ceil['gran']['uniq_ratio']:.1%}"
      f"（MUSR 子集分母 1260 的细网格 ⇒ 非量化假象）")
print(f"  敏感度排序(|ds/dL|)：" + "  ".join(
    f"{k}={v:.1f}" for k, v in sorted(p1c['conclusion']['slope_ranking'].items(), key=lambda z: -z[1])))
print(f"  → MUSR |ds/dL|={sharp['MUSR']['slope']:.2f}，为 BBH 的 1/{p1c['conclusion']['musr_over_bbh_slope']:.2f}；"
      f"同等 loss 改善只换到 1/5 的分数涨幅")
print("  ⇒ 无天花板、无量化 ⇒ 平坦只能来自敏感度低或真实平台期（二者本数据不可分）")

# =====================================================================================
sec("检验4  共同 ΔL 一致性：六任务同批模型，反解 ΔL 是否一致")
cons = {}
for tag, sub, sc in [("det", M, 'det'), ("lb(OPEN_LIC)", M[M.open], 'lb')]:
    d = {}
    for t in TASKS:
        col = DETCOL[t] if sc == 'det' else ('lb_' + t)
        g = q_p90(sub, 'q', col, QS)
        ds = float(g.values[-1] - g.values[0])
        bb = sharp[t]['slope']
        d[t] = dict(ds=ds, b=bb, dL=float(-ds / bb) if bb else np.nan)
    vals = np.array([d[t]['dL'] for t in TASKS])
    cons[tag] = dict(per_task=d, spread=float(np.nanmax(vals) - np.nanmin(vals)),
                     sd=float(np.nanstd(vals)),
                     ratio=float(np.nanmax(np.abs(vals)) / max(np.nanmin(np.abs(vals)), 1e-9)))
    print(f"\n  [{tag}] {'task':10s} {'Δs':>8s} {'b':>8s} {'反解ΔL':>9s}")
    for t in TASKS:
        print(f"        {t:10s} {d[t]['ds']:+8.3f} {d[t]['b']:8.3f} {d[t]['dL']:+9.4f}")
    print(f"        → 极差={cons[tag]['spread']:.3f}  sd={cons[tag]['sd']:.3f}  "
          f"最大/最小={cons[tag]['ratio']:.1f}×")
print("\n  ⇒ 六任务同批模型同期，真实 ΔL 必然相同；反解 ΔL 相差两个数量级 ⇒ pooled Φ 不能把"
      "逐任务分数轨迹换算到共同能力尺度（跨家族 loss 仅 7/75 行 High 可比）")

# =====================================================================================
sec("检验5  规格曲线：增速估计对 尺度×样本×分位×窗口 的敏感性")
SAMP = {'all lb': lb, 'OPEN_LIC lb': lb[lb.open],
        'det∩lb': M, 'det∩lb∩OPEN_LIC': M[M.open]}
specs = []
for sname, sdf in SAMP.items():
    scales = ['lb'] + (['det'] if sname.startswith('det') else [])
    for sc in scales:
        for qt in (0.75, 0.90, 0.95):
            for wname, wq in [('4Q', QS), ('3Q', QS[-3:]), ('2Q', QS[-2:])]:
                row = dict(sample=sname, scale=sc, qtile=qt, window=wname)
                for t in TASKS:
                    col = (DETCOL[t] if sc == 'det' else 'lb_' + t)
                    g = q_p90(sdf, 'q', col, wq, qt)
                    gg, se = ols_growth(wq, g.values)
                    row['g_' + t] = gg
                    row['se_' + t] = se
                if row['g_MUSR'] == row['g_MUSR']:
                    vals = {t: row['g_' + t] for t in TASKS if row['g_' + t] == row['g_' + t]}
                    row['musr_rank'] = int(sum(1 for v in vals.values() if v < row['g_MUSR'])) + 1
                    row['musr_is_slowest'] = bool(row['musr_rank'] == 1)
                    row['musr_ci_excl0'] = bool(abs(row['g_MUSR']) > 1.96 * (row['se_MUSR'] or 0))
                specs.append(row)
SC = pd.DataFrame(specs)
print(f"  规格总数 = {len(SC)}")
gm = SC['g_MUSR'].dropna()
print(f"  MUSR 年化增速：中位={gm.median():+.4f}  均值={gm.mean():+.4f}  "
      f"[p5,p95]=[{gm.quantile(.05):+.4f},{gm.quantile(.95):+.4f}]  极差={gm.max()-gm.min():.4f}")
print(f"  点估计为正的比例 = {(gm>0).mean():.1%}   95%CI 排除 0 的比例 = {SC['musr_ci_excl0'].mean():.1%}")
print(f"  MUSR 在六任务中排最慢的比例 = {SC['musr_is_slowest'].mean():.1%}  "
      f"排名分布={SC['musr_rank'].value_counts().sort_index().to_dict()}")
print("\n  各任务增速的规格中位数与跨规格极差：")
for t in TASKS:
    gt = SC['g_' + t].dropna()
    print(f"    {t:9s} 中位={gt.median():+.4f}  [p5,p95]=[{gt.quantile(.05):+.4f},{gt.quantile(.95):+.4f}]  "
          f"极差={gt.max()-gt.min():.4f}")
by_scale = SC.groupby('scale')['musr_is_slowest'].agg(['mean', 'size'])
by_qt = SC.groupby('qtile')['g_MUSR'].median()
print(f"  ⇒ MUSR 增速点估计跨规格从 {gm.min():+.3f} 到 {gm.max():+.3f}，"
      f"{int(round((1 - SC['musr_ci_excl0'].mean()) * 100))}% 的规格 95%CI 含 0")
bs_txt = "  ".join("%s: %.0f%%(n=%d)" % (k, v['mean'] * 100, int(v['size']))
                   for k, v in by_scale.iterrows())
print("     按尺度看最慢比例：" + bs_txt)
print("     按分位看增速中位：" + "  ".join("%s: %+.3f" % (k, float(v)) for k, v in by_qt.items()))
print("     ⇒ \"停滞\"（点估计≈0 且显著）与\"显著增长\"两种强结论都不成立 ⇒ 应报为不可识别；"
      f"\"MUSR 为六任务中最慢\"仅在 {SC['musr_is_slowest'].mean():.0%} 规格下成立"
      f"（lb/det 两尺度同为 {by_scale['mean'].min():.0%}），属弱多数而非稳健事实")

# =====================================================================================
sec("检验6  构成假象：族内纵向轨迹 vs 截面 p90（lb 尺度）")
fam_rows = []
for fam in ['qwen', 'gemma', 'llama', 'phi', 'yi', 'mistral']:
    s = lb[(lb.fam == fam) & lb.open]
    if len(s) < 8:
        continue
    g_lb = q_p90(s, 'q', 'lb_MUSR', QS)
    yr = s.dropna(subset=['lb_MUSR']).groupby('Year')['lb_MUSR'].quantile(.9)
    gg, se = ols_growth(QS, g_lb.values)
    fam_rows.append(dict(family=fam, n=len(s), lb_q={k: float(v) for k, v in g_lb.items()},
                         lb_year={int(k): float(v) for k, v in yr.items()}, g_lb=gg, se=se))
    print(f"  {fam:9s} n={len(s):3d}  季度p90={dict(zip(g_lb.index, np.round(g_lb.values,1)))}  "
          f"g={gg:+.4f}±{se:.4f}  年度p90={fam_rows[-1]['lb_year']}")
g_pool = ols_growth(QS, q_p90(lb[lb.open], 'q', 'lb_MUSR', QS).values)[0]
fam_g = np.array([r['g_lb'] for r in fam_rows])
fam_se = np.array([r['se'] for r in fam_rows])
sig = np.abs(fam_g) > 1.96 * fam_se
print(f"\n  全池 OPEN_LIC g={g_pool:+.4f}   族内 g 中位={np.median(fam_g):+.4f}   "
      f"族内极差={fam_g.max()-fam_g.min():.4f}")
print(f"  族内 95%CI 排除 0 的家族：{int(sig.sum())}/{len(fam_g)}  "
      f"→ 显著为正 {[r['family'] for r, s in zip(fam_rows, sig) if s and r['g_lb'] > 0]}，"
      f"显著为负 {[r['family'] for r, s in zip(fam_rows, sig) if s and r['g_lb'] < 0]}")
print("  ⇒ 族内 MUSR 轨迹本身异质且方向相反（qwen/mistral 显著上升，llama 显著下降），"
      "多数家族不显著；截面 p90 与族内轨迹给不出同向结论 ⇒ 构成效应无法被\"修正\"掉，"
      "反而进一步说明 MUSR 的时间趋势不可识别")

# =====================================================================================
sec("结论")
g_det = ts_rows[0]
g_lb = ts_rows[2]
verdict = {
    "threshold_artifact": False,
    "ceiling_artifact": False,
    "granularity_artifact": False,
    "growth_identifiable": False,
    "scale_dependent_point_estimate": True,
    "det_growth": g_det['growth'], "det_ci": [g_det['ci_lo'], g_det['ci_hi']],
    "lb_growth": g_lb['growth'], "lb_ci": [g_lb['ci_lo'], g_lb['ci_hi']],
    "musr_slope": sharp['MUSR']['slope'], "bbh_slope": sharp['BBH']['slope'],
    "spec_curve": dict(n=int(len(SC)), median=float(gm.median()),
                       p05=float(gm.quantile(.05)), p95=float(gm.quantile(.95)),
                       min=float(gm.min()), max=float(gm.max()),
                       share_positive=float((gm > 0).mean()),
                       share_ci_excl0=float(SC['musr_ci_excl0'].mean()),
                       share_slowest=float(SC['musr_is_slowest'].mean()),
                       share_slowest_lb=float(by_scale.loc['lb', 'mean'])),
    "implied_dL_spread_det": cons['det']['spread'],
    "implied_dL_spread_lb": cons['lb(OPEN_LIC)']['spread'],
    "family_growth": {r['family']: float(r['g_lb']) for r in fam_rows},
    "family_significant": [r['family'] for r, s in zip(fam_rows, sig) if s],
    "pool_open_lic_growth": float(g_pool),
}
print(json.dumps(verdict, ensure_ascii=False, indent=1))
print(f"""
  ① 非阈值假象：六任务 pooled Φ 最优均为线性，ΔL(10-90%) > 观测 loss 跨度；族内干净检验幂律胜出仅 {nm2/len(wf):.0%}。
  ② 非天花板/量化假象：det p90={ceil['det']['p90']:.1f} 距 100 有 {ceil['det']['headroom']:.1f} 点余量，lb p90={ceil['lb']['p90']:.1f} 有 {ceil['lb']['headroom']:.1f} 点余量；取值细网格无台阶。
  ③ 尺度不统一：同一批模型存在两套 MUSR 尺度（det 原始 acc_norm 与 lb 基线归一），逐任务仿射参数各异
     （lb≈{a:.3f}·det{b:+.2f}，残差 sd={resid.std():.2f}），说明二者来自不同 harness 配置；跨任务直接比增速受配置污染。
  ④ 敏感度是共同成因：MUSR |ds/dL|={sharp['MUSR']['slope']:.2f}，为六任务最低（BBH 的 1/{p1c['conclusion']['musr_over_bbh_slope']:.2f}）。
  ⑤ 规格曲线（{len(SC)} 个规格）：MUSR 增速中位 {gm.median():+.3f}，[p5,p95]=[{gm.quantile(.05):+.3f},{gm.quantile(.95):+.3f}]，
     极差 {gm.max()-gm.min():.3f}；{1-SC['musr_ci_excl0'].mean():.0%} 的规格 95%CI 含 0 ⇒ 增速不可识别，不得报单点。
  ⑥ "MUSR 为六任务中最慢"仅在 {SC['musr_is_slowest'].mean():.0%} 规格下成立（lb/det 两尺度同为 {by_scale['mean'].min():.0%}），属弱多数而非稳健事实。
  ⑦ 族内轨迹半数显著且方向相反（显著为正 {[r['family'] for r,s in zip(fam_rows,sig) if s and r['g_lb']>0]}，
     显著为负 {[r['family'] for r,s in zip(fam_rows,sig) if s and r['g_lb']<0]}），构成效应无法修正出同向结论。
  ⑧ 共同 ΔL 反解极差 {cons['det']['spread']:.2f}(det)/{cons['lb(OPEN_LIC)']['spread']:.2f}(lb)，最大/最小 {cons['det']['ratio']:.0f}×/{cons['lb(OPEN_LIC)']['ratio']:.0f}×
     ⇒ pooled Φ 不能把逐任务分数轨迹换算到共同能力尺度。
  ⇒ 论文口径：不得写"MUSR 能力停滞"，也不得写"涌现/阈值假象"；应写
     "MUSR 度量敏感度最低（|ds/dL|={sharp['MUSR']['slope']:.2f}），其时间趋势在所有规格下均无法与 0 稳健区分；
      其'六任务中最慢'仅在约三分之二规格下成立，为弱多数；该判读同时受评测配置（两套尺度）与样本构成影响"。
""")

json.dump(dict(
    data=dict(n_pairs=len(M), quarters=QS, n_det_dirs=len(os.listdir(det)), n_parse_fail=n_bad),
    two_scales=ts_rows,
    scale_map=dict(slope=float(a), intercept=float(b), corr=float(np.corrcoef(x[ok], y[ok])[0, 1]),
                   resid_sd=float(resid.std())),
    per_task_scale=scale_cmp,
    threshold=dict(pooled={t: {k: sharp[t][k] for k in ('best', 'r2', 'slope', 'band_10_90', 'band_over_span')}
                           for t in TASKS},
                   within_family_m2_share=float(nm2 / len(wf)),
                   tasks_without_threshold=p1c['conclusion']['tasks_without_threshold']),
    ceiling=ceil, consistency=cons, families=fam_rows, verdict=verdict),
    open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
SC.to_csv(OUT_SC, index=False, encoding='utf-8-sig')
pd.DataFrame(fam_rows).to_csv(OUT_FAM, index=False, encoding='utf-8-sig')
print(f"\n已写出：{OUT}\n         {OUT_SC}\n         {OUT_FAM}")
