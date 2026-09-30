# -*- coding: utf-8 -*-
"""
P1-5 处置脚本：三套参数集在同一点 (N=1B, D=150B, Q_B=0.5, lambda_p=0) 的数值对照。

三套参数来源：
  SET_A 本稿主参数（B6 拟合集，Q 未中心化，质量参考点 Q*=1）
  SET_B 旧版 v3（对 B6∪B7-new 重拟合，Q 未中心化）
  SET_C V5 实验报告（对 B6∪B7-new 重拟合，以 Q0=0.5 中心化）

模型（未中心化口径）：
  L = E + A*N^(-alpha)*exp(-rhoN*Q) + B*D^(-beta)*exp(-rhoD*Q) - E1*Q
模型（Q0=0.5 中心化口径）：
  L = E + A*N^(-alpha)*exp(-rhoN*(Q-Q0)) + B*D^(-beta)*exp(-rhoD*(Q-Q0)) - E1*(Q-Q0)

弹性定义（与运行报告第 6 节一致）：eps_X = -(dL/dX)*X / R,  R = L - E
"""
import numpy as np

N, D, Q, LP = 1.0, 150.0, 0.5, 0.0

SETS = {
    "SET_A 本稿主参数(B6)": dict(
        E=1.7112, A=0.6519, alpha=0.2783, B=1.4207, beta=0.2834,
        rhoN=0.3555, rhoD=0.1428, E1=0.1027, q0=None,
        src="B6 拟合集，Q 未中心化"),
    "SET_B 旧版v3(B6∪B7-new)": dict(
        E=1.7473, A=0.6396, alpha=0.2827, B=1.4263, beta=0.2998,
        rhoN=0.3497, rhoD=0.1301, E1=0.1157, q0=None,
        src="B6∪B7-new 重拟合，Q 未中心化"),
    "SET_C V5(B6∪B7-new, Q0=0.5)": dict(
        E=1.68951, A=0.53697, alpha=0.282713, B=1.33649, beta=0.299794,
        rhoN=0.349709, rhoD=0.130121, E1=0.115652, q0=0.5,
        src="B6∪B7-new 重拟合，以 Q0=0.5 中心化"),
}


def evaluate(p):
    q0 = 0.0 if p["q0"] is None else p["q0"]
    dq = Q - q0
    n_term = p["A"] * N ** (-p["alpha"]) * np.exp(-p["rhoN"] * dq)
    d_term = p["B"] * D ** (-p["beta"]) * np.exp(-p["rhoD"] * dq)
    L = p["E"] + n_term + d_term - p["E1"] * dq
    R = L - p["E"]

    dL_dN = -p["alpha"] * p["A"] * N ** (-p["alpha"] - 1) * np.exp(-p["rhoN"] * dq)
    dL_dD = -p["beta"] * p["B"] * D ** (-p["beta"] - 1) * np.exp(-p["rhoD"] * dq)
    dL_dQ = (-p["rhoN"] * n_term - p["rhoD"] * d_term - p["E1"])

    epsN = -(dL_dN) * N / R
    epsD = -(dL_dD) * D / R
    epsQ = -(dL_dQ) * Q / R if Q != 0 else float("nan")
    return dict(L=L, R=R, Nterm=n_term, Dterm=d_term, E1term=-p["E1"] * dq,
                mN=-dL_dN, mD=-dL_dD, mQ=-dL_dQ,
                epsN=-epsN, epsD=-epsD, epsQ=-epsQ)


print(f"代表点: N={N}B, D={D}B tokens, Q_B={Q}, lambda_p={LP}")
print("=" * 96)
rows = {}
for k, v in SETS.items():
    r = evaluate(v)
    rows[k] = r
    print(f"\n{k}   [{v['src']}]")
    print(f"  E(常数项)      = {v['E']:.5f}")
    print(f"  配比无关项 N项 = {r['Nterm']:.5f}   D项 = {r['Dterm']:.5f}   质量线性项 = {r['E1term']:+.5f}")
    print(f"  预测 Loss L    = {r['L']:.5f}")
    print(f"  可变损失 R=L-E = {r['R']:.5f}")
    print(f"  边际 -dL/dN    = {r['mN']:.6f}   -dL/dD = {r['mD']:.3e}   -dL/dQ = {r['mQ']:.6f}")
    print(f"  弹性 epsN      = {r['epsN']:+.4f}  epsD = {r['epsD']:+.4f}  epsQ = {r['epsQ']:+.4f}")

print("\n" + "=" * 96)
print("对照汇总（以 SET_A 为基准）")
print("=" * 96)
keys = ["SET_A 本稿主参数(B6)", "SET_B 旧版v3(B6∪B7-new)", "SET_C V5(B6∪B7-new, Q0=0.5)"]
base = rows[keys[0]]
hdr = f"{'参数集':<28}{'L':>9}{'ΔL%':>8}{'R':>9}{'ΔR%':>8}{'epsN':>9}{'epsD':>9}{'epsQ':>9}"
print(hdr)
print("-" * 96)
for k in keys:
    r = rows[k]
    dL = (r["L"] / base["L"] - 1) * 100
    dR = (r["R"] / base["R"] - 1) * 100
    print(f"{k:<28}{r['L']:>9.5f}{dL:>8.2f}{r['R']:>9.5f}{dR:>8.2f}"
          f"{r['epsN']:>9.4f}{r['epsD']:>9.4f}{r['epsQ']:>9.4f}")

print("\n" + "=" * 96)
print("稳健性判读")
print("=" * 96)
Ls = np.array([rows[k]["L"] for k in keys])
Rs = np.array([rows[k]["R"] for k in keys])
eN = np.array([rows[k]["epsN"] for k in keys])
eD = np.array([rows[k]["epsD"] for k in keys])
eQ = np.array([rows[k]["epsQ"] for k in keys])
print(f"  L   极差/中位 = {100*(Ls.max()-Ls.min())/np.median(Ls):.3f}%   (max-min = {Ls.max()-Ls.min():.5f})")
print(f"  R   极差/中位 = {100*(Rs.max()-Rs.min())/np.median(Rs):.3f}%   (max-min = {Rs.max()-Rs.min():.5f})")
for nm, arr in [("epsN", eN), ("epsD", eD), ("epsQ", eQ)]:
    print(f"  {nm} 极差/中位 = {100*(arr.max()-arr.min())/abs(np.median(arr)):.3f}%   "
          f"区间 [{arr.min():+.4f}, {arr.max():+.4f}]  符号一致 = {np.all(arr > 0) or np.all(arr < 0)}")
print("\n  排序一致性: |epsQ| > |epsN| > |epsD| ?",
      [bool(abs(r["epsQ"]) > abs(r["epsN"]) > abs(r["epsD"])) for r in (rows[k] for k in keys)])
print("  L 排序:", [k.split()[0] for k in keys[np.argsort(Ls)]] if isinstance(keys, np.ndarray) else
      [keys[i].split()[0] for i in np.argsort(Ls)])
