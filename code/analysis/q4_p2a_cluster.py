# -*- coding: utf-8 -*-
"""
P2-A  机构（发布者）聚类 Bootstrap：前沿与桥接系数的稳健标准误

动机
----
主口径下的 b_N / b_T、桥接斜率 k、近期前沿腿增速，此前都用**逐行 i.i.d. 重抽**给区间。
但榜单里同一发布者会提交大量高度相关的模型（最大簇 197 条，中位簇 2 条，715 个发布者），
i.i.d. 重抽把同一机构的相关记录当独立信息，**系统性低估标准误**。

本脚本
  1. 以 HuggingFace 发布者前缀为聚类单元，做**整簇重抽**（cluster bootstrap）；
  2. 与 i.i.d. 重抽对照，给出**设计效应 deff = (SE_cluster/SE_iid)^2** 与有效样本量 n_eff；
  3. 对三个关键量给出聚类修正后的区间：
       (a) 前沿 QR90 的 b_N / b_T（OPEN_LIC × 连续年）
       (b) C6 桥接斜率 k（lnA ~ lnL）
       (c) 近期前沿腿年化增速（C1 OPEN_LIC 季度 p90(S)）
  4. 用 Epoch_AI_Organization 作**外部交叉校验**（覆盖仅 443/4576，只能定性对照）。

输出：q4_p2a_cluster_results.json / q4_p2a_clusters.csv
"""
import os, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import linprog, minimize

warnings.filterwarnings("ignore")
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
OUT = r"d:\F题\q4_p2a_cluster_results.json"
OUT_CSV = r"d:\F题\q4_p2a_clusters.csv"
SEED = 20260926
RNG = np.random.default_rng(SEED)
B = 500

OW = ['apache', 'mit', 'bsd', 'llama', 'gemma', 'cc-by', 'openrail', 'gpl', 'wtfpl', 'afl',
      'creativeml', 'bigscience', 'bigcode', 'apple-ascl']


def sec(t):
    print("\n" + "=" * 104)
    print("### " + t)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def qr_lp(X, y, tau):
    n, k = X.shape
    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    res = linprog(c, A_eq=A_eq, b_eq=y,
                  bounds=[(None, None)] * k + [(0, None)] * (2 * n), method='highs')
    return res.x[:k] if res.success else None


def qr_pin(X, y, tau):
    def f(b):
        r = y - X @ b
        return np.sum(np.where(r >= 0, tau * r, (tau - 1) * r))
    return minimize(f, ols(X, y), method='L-BFGS-B').x


def boot_ci(mat, lo=2.5, hi=97.5):
    return np.percentile(mat, [lo, 50, hi], axis=0)


# =====================================================================================
sec("0  数据与聚类结构")
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)
enh = pd.read_csv(os.path.join(BASE, "leaderboard_enhanced.csv"), low_memory=False)
c6 = pd.read_csv(os.path.join(BASE, "loss_benchmark_bridge_expanded.csv"), low_memory=False)

lb['date'] = pd.to_datetime(lb['Submission Date'], errors='coerce')
lb['N'] = pd.to_numeric(lb['#Params (B)'], errors='coerce')
lb['S'] = pd.to_numeric(lb['Average ⬆️'], errors='coerce')
lb = lb.dropna(subset=['date', 'N', 'S']).copy()
lb = lb[(lb['N'] > 0) & (lb['S'] > 0)].copy()
lb['t_frac'] = (lb['date'] - pd.Timestamp('2022-01-01')).dt.days / 365.25
lb['lnN'] = np.log(lb['N']); lb['lnS'] = np.log(lb['S'])
lb['q'] = lb['date'].dt.to_period('Q').astype(str)
lb['pub'] = lb['Model'].astype(str).str.split('/').str[0]
lic = lb['Hub License'].fillna('').str.lower()
lb['open'] = lic.apply(lambda x: any(w in x for w in OW))

org = enh.drop_duplicates('Model').set_index('Model')['Epoch_AI_Organization']
lb['epoch_org'] = lb['Model'].map(org)

sub = lb[lb.open].reset_index(drop=True)
print(f"C1 全样本 n={len(lb)}   OPEN_LIC n={len(sub)}   窗口 {lb.date.min().date()} ~ {lb.date.max().date()}")
print(f"聚类单元（HF 发布者）总数 = {lb.pub.nunique()}")
vc = lb.pub.value_counts()
print(f"  簇大小：中位={vc.median():.0f}  均值={vc.mean():.1f}  最大={vc.max()}  "
      f"单条簇占比={(vc == 1).mean():.1%}")
print(f"  最大 5 簇：{vc.head(5).to_dict()}")
# 簇内相关强度：同一发布者内 S 的离散 vs 全样本
within = lb.groupby('pub')['lnS'].std().dropna()
big = vc[vc >= 5].index
print(f"  簇内 lnS 标准差（簇大小≥5，共 {len(big)} 簇）中位={within.reindex(big).median():.3f}  "
      f"全样本 lnS 标准差={lb.lnS.std():.3f}")
print("  → 簇内离散显著小于总体 ⇒ 同一发布者记录高度相关，i.i.d. 重抽会低估标准误")

cl = lb.groupby('pub').agg(n=('lnS', 'size'), med_S=('S', 'median'),
                           med_N=('N', 'median'), open_share=('open', 'mean')).sort_values('n', ascending=False)
cl.to_csv(OUT_CSV, encoding='utf-8-sig')

# =====================================================================================
sec("1  前沿 QR90（OPEN_LIC × 连续年）：i.i.d. vs 聚类重抽")
X = np.column_stack([np.ones(len(sub)), sub.lnN.values, sub.t_frac.values])
y = sub.lnS.values
b_hat = qr_lp(X, y, 0.9)
print(f"  点估计（精确 LP）：c={b_hat[0]:+.4f}  b_N={b_hat[1]:+.4f}  b_T={b_hat[2]:+.4f}")

pub_idx = sub.groupby('pub').indices
pubs = list(pub_idx.keys())
n_pub = len(pubs)

boot_iid = np.zeros((B, 3))
for i in range(B):
    j = RNG.integers(0, len(sub), len(sub))
    boot_iid[i] = qr_pin(X[j], y[j], 0.9)

boot_clu = np.zeros((B, 3))
n_bad = 0
for i in range(B):
    pick = RNG.integers(0, n_pub, n_pub)
    j = np.concatenate([pub_idx[pubs[t]] for t in pick])
    if len(j) < 20:
        n_bad += 1
        continue
    bb = qr_pin(X[j], y[j], 0.9)
    if not np.all(np.isfinite(bb)) or abs(bb[2]) > 20:
        n_bad += 1
        continue
    boot_clu[i] = bb
boot_clu = boot_clu[np.any(boot_clu != 0, axis=1)]

ci_i = boot_ci(boot_iid); ci_c = boot_ci(boot_clu)
print(f"  重抽有效次数：i.i.d.={B}  聚类={len(boot_clu)}（剔除退化 {n_bad}）")
print(f"  {'参数':6s} {'点估计':>9s} | {'iid SE':>8s} {'iid 95%CI':>22s} | {'cluster SE':>10s} {'cluster 95%CI':>22s} | {'deff':>6s} {'n_eff':>7s}")
qr_rows = {}
for j, nm in [(1, 'b_N'), (2, 'b_T')]:
    se_i = float(np.std(boot_iid[:, j], ddof=1)); se_c = float(np.std(boot_clu[:, j], ddof=1))
    deff = (se_c / se_i) ** 2
    qr_rows[nm] = dict(point=float(b_hat[j]), se_iid=se_i, se_cluster=se_c, deff=float(deff),
                       n_eff=float(len(sub) / deff),
                       ci_iid=[float(ci_i[0, j]), float(ci_i[2, j])],
                       ci_cluster=[float(ci_c[0, j]), float(ci_c[2, j])])
    print(f"  {nm:6s} {b_hat[j]:+9.4f} | {se_i:8.4f} [{ci_i[0,j]:+.4f},{ci_i[2,j]:+.4f}] | "
          f"{se_c:10.4f} [{ci_c[0,j]:+.4f},{ci_c[2,j]:+.4f}] | {deff:6.2f} {len(sub)/deff:7.0f}")
print("  → deff>1 表示 i.i.d. 区间过窄；n_eff 为聚类后的等效独立样本量")
print(f"  → b_T 聚类区间是否含 0：{ci_c[0,2] <= 0 <= ci_c[2,2]}"
      f"（i.i.d. 判读：{ci_i[0,2] <= 0 <= ci_i[2,2]}）")

# =====================================================================================
sec("2  C6 桥接 lnA ~ lnL 的斜率：i.i.d. vs 聚类重抽")
c6 = c6.dropna(subset=['Val_Loss', 'LB_Average']).copy()
c6['pub'] = c6['Model'].astype(str).str.split('/').str[0]
c6['lL'] = np.log(c6.Val_Loss); c6['lA'] = np.log(c6.LB_Average)
Xb = np.column_stack([np.ones(len(c6)), c6.lL.values]); yb = c6.lA.values
kb = ols(Xb, yb)
print(f"  C6 n={len(c6)}  簇数={c6.pub.nunique()}  k={kb[1]:+.4f}  b={kb[0]:+.4f}")
c6idx = c6.groupby('pub').indices; cp = list(c6idx.keys())
bi = np.zeros((B, 2)); bc = np.zeros((B, 2))
for i in range(B):
    j = RNG.integers(0, len(c6), len(c6))
    bi[i] = ols(Xb[j], yb[j])
    pick = RNG.integers(0, len(cp), len(cp))
    jj = np.concatenate([c6idx[cp[t]] for t in pick])
    if len(jj) >= 8:
        bc[i] = ols(Xb[jj], yb[jj])
bi = bi[np.any(bi != 0, axis=1)]; bc = bc[np.any(bc != 0, axis=1)]
cii = boot_ci(bi); cic = boot_ci(bc)
bridge = {}
for j, nm in [(1, 'k'), (0, 'intercept')]:
    se_i = float(np.std(bi[:, j], ddof=1)); se_c = float(np.std(bc[:, j], ddof=1))
    bridge[nm] = dict(point=float(kb[j]), se_iid=se_i, se_cluster=se_c,
                      deff=float((se_c / se_i) ** 2),
                      ci_iid=[float(cii[0, j]), float(cii[2, j])],
                      ci_cluster=[float(cic[0, j]), float(cic[2, j])])
    print(f"  {nm:10s} {kb[j]:+.4f} | iid SE={se_i:.4f} [{cii[0,j]:+.3f},{cii[2,j]:+.3f}] | "
          f"cluster SE={se_c:.4f} [{cic[0,j]:+.3f},{cic[2,j]:+.3f}] | deff={(se_c/se_i)**2:.2f}")

# =====================================================================================
sec("3  近期前沿腿增速（C1 OPEN_LIC 季度 p90）：i.i.d. vs 聚类重抽")
QS = sorted(sub.q.unique())
g0 = sub.groupby('q')['S'].quantile(0.9).reindex(QS)
print(f"  季度 p90(S) = {dict(zip(QS, np.round(g0.values, 2)))}")


def leg_growth(df, idxmap, keys):
    g = df.groupby('q')['S'].quantile(0.9).reindex(QS)
    v = g.values
    if len(v) < 3 or np.any(~np.isfinite(v)) or np.any(v <= 0):
        return None
    return float(np.polyfit(np.arange(len(v)), np.log(v), 1)[0] * 4)


g_point = leg_growth(sub, None, None)
print(f"  点估计 = {g_point:+.4f}/年（仅 4 个季度点）")
sidx = sub.groupby('pub').indices; sp = list(sidx.keys())
gi = []; gc = []
for i in range(B):
    j = RNG.integers(0, len(sub), len(sub))
    v = leg_growth(sub.iloc[j], None, None)
    if v is not None:
        gi.append(v)
    pick = RNG.integers(0, len(sp), len(sp))
    jj = np.concatenate([sidx[sp[t]] for t in pick])
    v2 = leg_growth(sub.iloc[jj], None, None)
    if v2 is not None:
        gc.append(v2)
gi, gc = np.array(gi), np.array(gc)
leg = dict(point=g_point, se_iid=float(gi.std(ddof=1)), se_cluster=float(gc.std(ddof=1)),
           deff=float((gc.std(ddof=1) / gi.std(ddof=1)) ** 2),
           ci_iid=[float(np.percentile(gi, 2.5)), float(np.percentile(gi, 97.5))],
           ci_cluster=[float(np.percentile(gc, 2.5)), float(np.percentile(gc, 97.5))])
print(f"  i.i.d.  SE={leg['se_iid']:.4f}  95%CI=[{leg['ci_iid'][0]:+.3f},{leg['ci_iid'][1]:+.3f}]")
print(f"  聚类    SE={leg['se_cluster']:.4f}  95%CI=[{leg['ci_cluster'][0]:+.3f},{leg['ci_cluster'][1]:+.3f}]"
      f"  deff={leg['deff']:.2f}")

# =====================================================================================
sec("4  外部交叉校验：Epoch_AI_Organization（覆盖 443/4576，仅定性）")
mm = lb.dropna(subset=['epoch_org']).copy()
print(f"  有 Epoch 机构标注的行 n={len(mm)}  机构数={mm.epoch_org.nunique()}")
grp = mm.groupby('pub')['epoch_org'].nunique()
print(f"  同一 HF 发布者映射到多个 Epoch 机构的比例 = {(grp > 1).mean():.1%}（共 {len(grp)} 个发布者）")
print("  → HF 前缀与 Epoch 机构大体一致；Epoch 覆盖过低，不能作主聚类键，仅作交叉校验")

# =====================================================================================
sec("结论")
res = dict(cluster_structure=dict(n_rows=len(lb), n_open=len(sub), n_publishers=int(lb.pub.nunique()),
                                  median_cluster=float(vc.median()), max_cluster=int(vc.max()),
                                  share_singleton=float((vc == 1).mean()),
                                  within_cluster_sd=float(within.reindex(big).median()),
                                  overall_sd=float(lb.lnS.std())),
           frontier_qr=qr_rows, bridge=bridge, leg2=leg,
           epoch_crosscheck=dict(n=len(mm), n_org=int(mm.epoch_org.nunique()),
                                 share_pub_multi_org=float((grp > 1).mean())))
print(json.dumps(res, ensure_ascii=False, indent=1))
print(f"""
  ① 聚类结构：{int(lb.pub.nunique())} 个发布者，最大簇 {int(vc.max())} 条，{(vc==1).mean():.0%} 为单条簇；
     簇内 lnS 标准差 {within.reindex(big).median():.3f} 远小于总体 {lb.lnS.std():.3f} ⇒ 组内相关显著。
  ② 前沿系数：b_N 设计效应 deff={qr_rows['b_N']['deff']:.2f}（n_eff≈{qr_rows['b_N']['n_eff']:.0f}），
     b_T deff={qr_rows['b_T']['deff']:.2f}（n_eff≈{qr_rows['b_T']['n_eff']:.0f}）。
  ③ 桥接斜率 k：deff={bridge['k']['deff']:.2f}；近期前沿腿 deff={leg['deff']:.2f}。
  ④ 主口径判读是否改变：b_T 聚类 95%CI=[{qr_rows['b_T']['ci_cluster'][0]:+.4f},{qr_rows['b_T']['ci_cluster'][1]:+.4f}]
     含 0={qr_rows['b_T']['ci_cluster'][0] <= 0 <= qr_rows['b_T']['ci_cluster'][1]}；
     i.i.d. 区间=[{qr_rows['b_T']['ci_iid'][0]:+.4f},{qr_rows['b_T']['ci_iid'][1]:+.4f}] 含 0={qr_rows['b_T']['ci_iid'][0] <= 0 <= qr_rows['b_T']['ci_iid'][1]}。
  ⇒ 论文口径：所有涉及 b_N / b_T / k / 腿增速的区间必须改报**聚类稳健区间**，
     并同时给出 deff 与 n_eff；不得再使用 i.i.d. 重抽区间作为主结果。
""")
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"已写出：{OUT}\n         {OUT_CSV}")
