# -*- coding: utf-8 -*-
"""
问题四 三条证据腿的权重重估（P0 收尾）

背景：
  q4_legs.py 用逆方差合成三条腿，得到 g_comb=0.5681、24M=82.04。
  但近期前沿腿的 sigma=0.0699 只是「4 个季度点」的 bootstrap 标准误，
  是纯抽样误差，不含模型不确定度；让它拿走 69.5% 权重不合理。

做法：
  定义模型不确定度下限 sigma_floor = 三条腿之间的离散度（ddof=1）。
  对「纯抽样型」的 sigma 施加下限 sigma_eff = max(sigma, sigma_floor)；
  对「已含模型不确定度」的腿（机制腿的 sigma 由桥接残差 sigma=0.5825 传播而来）不施加。

  主口径   ：仅对纯抽样腿（近期前沿腿、长历史腿）施加下限
  敏感性 A ：对全部三腿施加下限（等价于等权）
  对照     ：原始逆方差（不下限）

输出：q4_legs_reweight_results.json
"""
import json, warnings
import numpy as np
warnings.filterwarnings('ignore')

LEGS = [('腿1 机制', 0.49449377814887835, 0.1266895063459456, 'model'),
        ('腿2 近期前沿', 0.6326156401727455, 0.06987095205242991, 'sample'),
        ('腿3 长历史', 0.25398375861940115, 0.1908200801918722, 'sample')]
S0 = 40.424995320340855

g = np.array([l[1] for l in LEGS])
sd = np.array([l[2] for l in LEGS])
kind = [l[3] for l in LEGS]

# 模型不确定度下限 = 腿间离散度（ddof=1，样本标准差）
sigma_floor = float(np.std(g, ddof=1))
print(f"三条腿增速 = {np.round(g, 4).tolist()}")
print(f"腿间离散度 sigma_floor = std(ddof=1) = {sigma_floor:.4f}")
print(f"腿间极差 = {g.max() - g.min():.4f} (= sigma_floor 的 {(g.max()-g.min())/sigma_floor:.1f} 倍)")

def logistic24(g_target, S0=S0):
    A = (100 - S0) / S0
    r = g_target / (1 - S0 / 100)
    s12 = 100 / (1 + A * np.exp(-r * 1))
    s24 = 100 / (1 + A * np.exp(-r * 2))
    return float(s12), float(s24), float(r)

def combine(sd_eff, tag):
    w = 1 / sd_eff ** 2
    w = w / w.sum()
    g_comb = float((w * g).sum())
    sd_comb = float(np.sqrt(1 / (1 / sd_eff ** 2).sum()))
    s12, s24, r = logistic24(g_comb)
    lo, _, _ = logistic24(g_comb - 1.645 * sd_comb)
    hi, _, _ = logistic24(g_comb + 1.645 * sd_comb)
    print(f"\n【{tag}】")
    for i, (name, _, _, _) in enumerate(LEGS):
        print(f"  {name:10s} g={g[i]:+.4f}  sigma_raw={sd[i]:.4f}  sigma_eff={sd_eff[i]:.4f}  权重={w[i]*100:5.2f}%")
    print(f"  合成 g={g_comb:+.4f}/年   sigma={sd_comb:.4f}   12M={s12:.2f}   24M={s24:.2f}   90%CI(24M)=[{lo:.2f},{hi:.2f}]")
    return dict(tag=tag, sd_eff=sd_eff.tolist(), w=w.tolist(), g_comb=g_comb,
                sd_comb=sd_comb, s12=s12, s24=s24, lo=lo, hi=hi, r=r)

# ---- 对照：原始逆方差
r_old = combine(sd.copy(), "对照 原始逆方差（不下限）")

# ---- 主口径：仅对纯抽样腿施加下限
sd_main = np.array([sd[i] if kind[i] == 'model' else max(sd[i], sigma_floor) for i in range(3)])
r_main = combine(sd_main, "主口径 抽样型 sigma 施加模型不确定度下限")

# ---- 敏感性 A：对全部三腿施加下限（等权）
sd_all = np.maximum(sd, sigma_floor)
r_all = combine(sd_all, "敏感性A 全部三腿施加下限（等价等权）")

print("\n" + "=" * 88)
print("【结论】")
print(f"  24M 乐观（原始逆方差）      = {r_old['s24']:.2f}")
print(f"  24M 稳健（主口径）          = {r_main['s24']:.2f}")
print(f"  24M 稳健下界（敏感性A 等权）= {r_all['s24']:.2f}")
print(f"  → 稳健区间 = [{min(r_main['s24'], r_all['s24']):.2f}, {max(r_main['s24'], r_all['s24']):.2f}]")
print(f"  12M 乐观 {r_old['s12']:.2f} → 稳健 [{min(r_main['s12'], r_all['s12']):.2f}, {max(r_main['s12'], r_all['s12']):.2f}]")
print("  说明：无论用哪种下限口径，24M 都落在 76 分附近，而 82.04 只在『相信 4 个季度点的 SE』时成立。")

# ---------- 三源不确定性分解（随 g_comb 与 sigma_comb 变化，须重算）
sig_res = 0.4724                      # ln 尺度残差 sd（与 q4_bounded_forecast.py 一致）
def unc_split(res):
    S24 = res['s24']
    v_res = (S24 * sig_res) ** 2
    v_par = (S24 * res['sd_comb'] * 2 * (1 - S24 / 100) / (1 - S0 / 100)) ** 2
    v_scen = ((S24 - logistic24(res['g_comb'] * 0.5)[1]) / 1.645) ** 2
    tot = v_res + v_par + v_scen
    return dict(res=v_res / tot, par=v_par / tot, scen=v_scen / tot, total=tot)

print("\n" + "=" * 88)
print("【三源不确定性分解（24M）——随权重重估同步更新】")
u_old, u_main = unc_split(r_old), unc_split(r_main)
print(f"  {'口径':34s} {'残差':>8s} {'参数':>8s} {'情景':>8s}")
print(f"  {'乐观（原始逆方差，24M=%.2f）' % r_old['s24']:34s} {u_old['res']*100:7.1f}% {u_old['par']*100:7.1f}% {u_old['scen']*100:7.1f}%")
print(f"  {'稳健（主口径，24M=%.2f）' % r_main['s24']:34s} {u_main['res']*100:7.1f}% {u_main['par']*100:7.1f}% {u_main['scen']*100:7.1f}%")
print("  判读：权重重估把合成 sigma 从 0.0583 提到 0.0925，参数项占比由 0.5% 升到约 2%，"
      "但残差仍主导 → 结论（杠杆在模型形式与样本量，不在参数精度）不变。")

json.dump(dict(S0=S0, legs=[dict(name=l[0], g=l[1], sd_raw=l[2], kind=l[3]) for l in LEGS],
               sigma_floor=sigma_floor,
               old=r_old, main=r_main, sens_uniform=r_all,
               unc_24M=dict(optimistic=u_old, robust=u_main),
               headline=dict(s24_optimistic=r_old['s24'],
                             s24_robust=r_main['s24'],
                             s24_robust_lower=r_all['s24'],
                             s12_optimistic=r_old['s12'],
                             s12_robust=r_main['s12'])),
          open(r"d:\F题\q4_legs_reweight_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出 d:\\F题\\q4_legs_reweight_results.json")
