# -*- coding: utf-8 -*-
"""
问题三核心计算：算力约束下的多维资源联合优化与结构性转移。

模型（目标函数 = 问题二 SET_A 广义标度律）
------------------------------------------
    L(N,D,Q;p) = E~ + A~ N^(-a) exp(-rN Q) + B~ D^(-b) exp(-rD Q) exp(lam_p h_p(p)) - E1 Q
    N 单位：B 参数；D 单位：B tokens；Q 为 B 端质量尺度

约束（题面三部分成本，全部线性作用于 N*D 或 D）
----------------------------------------------
    C_train = 6 N D,  C_Q = D [g(Q)-g(Q0)]_+,  C_attn = eta N D Lctx
    C_train + C_Q + C_attn <= C,     Q0 <= Q <= 1

降维（本脚本的解析核心）
------------------------
令 K = 6 + eta*Lctx，Psi(Q) = [g(Q)-g(Q0)]_+，总成本 = D (K N + Psi)。
L 对 D 单调递减 => 预算必被用尽 => D 可被约束精确消去：
    D(N,Q) = C / (K N + Psi(Q))
问题降为 (N,Q) 二维。固定 Q 时 N 的一阶条件为
    a A~ N^(-a-1) e^(-rN Q) = b K B~ C^(-b) e^(-rD Q) (K N + Psi)^(b-1)
记 h(N) = -dL/dN，则 h(N->0) -> +inf、h(N->inf) -> 0^-，且在 N 上单调 => 唯一根。
Q 维度：网格 + 黄金分割精修。

结构性转移（严格定义）
----------------------
状态 sigma(C) = (Q 的边界状态, 主导成本成分)，其中
  Q 状态: boundary  (Q*=Q0) | interior (Q0<Q*<1) | saturated (Q*=1)
  主导成本: argmax{s_train, s_attn, s_Q}
若 C 跨越某点使 sigma 改变，则称该点为结构性转移点（策略质变，而非量的渐变）。
识别：扫描定位 + 在切换区间上二分求精确阈值。
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import brentq

# ---------------- 问题二锁定参数（SET_A，B6 拟合集）----------------
E_T, A_T, ALPHA = 1.7112, 0.6519, 0.2783
B_T, BETA = 1.4207, 0.2834
RHO_N, RHO_D, E1 = 0.3555, 0.1428, 0.1027
ETA = 2.0e-4
L_CRIT = 6.0 / ETA                      # 注意力开销 = 训练开销 的临界上下文长度

# 稳健性对照端：E1/rhoD 分工的边界解（问题二 §4.6(d) profile likelihood）
PAR_B = (E_T, A_T, ALPHA, B_T, BETA, RHO_N, 0.2469, 0.0)

G = {
    'exp': (lambda Q: 1e7 * np.exp(6.0 * Q), lambda Q: 6e7 * np.exp(6.0 * Q)),
    'pow': (lambda Q: 5e9 * Q ** 4, lambda Q: 2e10 * Q ** 3),
    'log': (lambda Q: 2e9 * np.log1p(10.0 * Q), lambda Q: 2e10 / (1.0 + 10.0 * Q)),
}
G_LABEL = {'exp': '指数型', 'pow': '幂函数型', 'log': '对数渐进型'}

# C7（model_architecture_metadata.csv）中 max_position_embeddings 的实际取值
LCTX_C7 = [2048, 4096, 8192, 32768, 131072]

# 问题二校准支撑域（B6/B7）
BOX = dict(N=(0.07, 11.97), D=(10.0, 600.0), Q=(0.1, 1.0))


def par_of(par):
    return PAR_DEF if par is None else par


PAR_DEF = (E_T, A_T, ALPHA, B_T, BETA, RHO_N, RHO_D, E1)


def loss(ntil, dtil, Q, lam=0.0, hp=0.0, par=None):
    E, A, a, B, b, rN, rD, e1 = par_of(par)
    return (E + A * ntil ** (-a) * np.exp(-rN * Q)
            + B * dtil ** (-b) * np.exp(-rD * Q) * np.exp(lam * hp) - e1 * Q)


def k_of(Lctx):
    return 6.0 + ETA * Lctx


# ---------------- 固定 Q：N 维一阶条件求根 ----------------
def _foc(N, Q, Ctil, K, Psi_t, par=None):
    """h(N) = -dL/dN（D 已消去）。符号由 + 穿越到 -，唯一根即 L 的最小点。"""
    E, A, a, B, b, rN, rD, e1 = par_of(par)
    t1 = a * A * np.exp(-rN * Q) * N ** (-a - 1.0)
    t2 = (b * K * B * np.exp(-rD * Q) * Ctil ** (-b)
          * (K * N + Psi_t) ** (b - 1.0))
    return t1 - t2


def solve_N(Q, Ctil, K, Psi_t, par=None):
    """给定 Q，解出最优 N（B 参数）。"""
    lo, hi = 1e-9, 1e9
    flo = _foc(lo, Q, Ctil, K, Psi_t, par)
    fhi = _foc(hi, Q, Ctil, K, Psi_t, par)
    it = 0
    while flo < 0 and it < 60:          # 向小端扩展
        lo *= 0.1
        flo = _foc(lo, Q, Ctil, K, Psi_t, par)
        it += 1
    it = 0
    while fhi > 0 and it < 60:          # 向大端扩展
        hi *= 10
        fhi = _foc(hi, Q, Ctil, K, Psi_t, par)
        it += 1
    if flo < 0 or fhi > 0:
        return None
    return float(brentq(_foc, lo, hi, args=(Q, Ctil, K, Psi_t, par),
                        xtol=1e-14, rtol=1e-15, maxiter=300))


def solve_NDQ(C, gname, Q0, Lctx, lam=0.0, hp=0.0, par=None, nq=400):
    """联合最优 (N,D,Q) 与成本份额。"""
    g, _ = G[gname]
    K = k_of(Lctx)
    Ctil = C / 1e18
    lossv = lambda n, d, Q: loss(n, d, Q, lam, hp, par)

    def reduced(N, Q):
        Psi_t = max(g(Q) - g(Q0), 0.0) / 1e9
        D = Ctil / (K * N + Psi_t)
        return lossv(N, D, Q), D, Psi_t

    Qs = np.linspace(Q0, 1.0, nq)
    best = (np.inf, None, None)
    cache = []
    for Q in Qs:
        N = solve_N(Q, Ctil, K, max(g(Q) - g(Q0), 0.0) / 1e9, par)
        if N is None:
            cache.append(np.nan)
            continue
        v, D, _ = reduced(N, Q)
        cache.append(v)
        if v < best[0]:
            best = (v, N, Q)
    cache = np.array(cache)
    # 黄金分割精修（在最优网格点的相邻区间内）
    j = int(np.nanargmin(cache))
    a, b = Qs[max(j - 1, 0)], Qs[min(j + 1, nq - 1)]

    def fQ(Q):
        N = solve_N(Q, Ctil, K, max(g(Q) - g(Q0), 0.0) / 1e9, par)
        return reduced(N, Q)[0] if N is not None else np.inf

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = fQ(c), fQ(d)
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fQ(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fQ(d)
        if abs(b - a) < 1e-12:
            break
    Qstar = 0.5 * (a + b)
    if fQ(Q0) <= fQ(Qstar):                    # 与下边界比较
        Qstar = Q0
    if fQ(1.0) < fQ(Qstar):                    # 与上边界比较
        Qstar = 1.0
    Nstar = solve_N(Qstar, Ctil, K, max(g(Qstar) - g(Q0), 0.0) / 1e9, par)
    Lv, Dstar, Psi_t = reduced(Nstar, Qstar)
    s_tr = 6.0 * Nstar * Dstar / Ctil
    s_at = ETA * Lctx * Nstar * Dstar / Ctil
    s_Q = Dstar * Psi_t / Ctil
    return dict(C=C, g=gname, Lctx=Lctx, Q0=Q0, N_B=Nstar, D_B=Dstar, Q=Qstar,
                L=float(Lv), Psi=Psi_t * 1e9, K=K,
                s_train=float(s_tr), s_attn=float(s_at), s_Q=float(s_Q),
                ratio_attn_train=float(s_at / s_tr) if s_tr > 0 else np.inf,
                N_abs=Nstar * 1e9, D_abs=Dstar * 1e9)


# ---------------- 纯规模解析分支（Q == Q0）----------------
def pure_scale(C, Q0, Lctx, par=None):
    E, A, a, B, b, rN, rD, e1 = par_of(par)
    K = k_of(Lctx)
    Ctil = C / 1e18
    N = ((a * A * np.exp(-rN * Q0) * Ctil ** b)
         / (b * B * np.exp(-rD * Q0) * K ** b)) ** (1.0 / (a + b))
    D = Ctil / (K * N)
    Lv = E + A * N ** (-a) * np.exp(-rN * Q0) + B * D ** (-b) * np.exp(-rD * Q0) - e1 * Q0
    return dict(N_B=float(N), D_B=float(D), L=float(Lv), K=K,
                ratio_DN=float(D / N))


# ---------------- 质量通道 KKT 判据 ----------------
def mb_mc(Q, N, D, Q0, Lctx, gname, lam=0.0, hp=0.0, par=None):
    """Q 的边际收益 MB 与边际成本 MC（在给定 (N,D,Q) 上）。

    dL~/dQ = -MB + MC，其中 L~ 为消去 D 后的目标。
      MB = E1 + rN*A_N + rD*A_D
      MC = b*A_D*Psi'(Q)/(K*N + Psi(Q))
    激活判据：Q0 处 MB > MC；饱和判据：Q=1 处 MB <= MC。
    """
    E, A, a, B, b, rN, rD, e1 = par_of(par)
    g, gp = G[gname]
    K = k_of(Lctx)
    A_N = A * N ** (-a) * np.exp(-rN * Q)
    A_D = B * D ** (-b) * np.exp(-rD * Q) * np.exp(lam * hp)
    Psi = max(g(Q) - g(Q0), 0.0)
    MB = e1 + rN * A_N + rD * A_D
    MC = b * A_D * gp(Q) / (K * N * 1e9 + Psi) if (K * N * 1e9 + Psi) > 0 else np.inf
    return float(MB), float(MC)


def act_threshold(gname, Q0, Lctx, par=None, lo=1e14, hi=1e28):
    """质量激活阈值：在纯规模分支上 MB(Q0) - MC(Q0) 由负变正的 C。"""
    def f(lc):
        C = 10.0 ** lc
        ps = pure_scale(C, Q0, Lctx, par)
        MB, MC = mb_mc(Q0, ps['N_B'], ps['D_B'], Q0, Lctx, gname, par=par)
        return MB - MC
    flo, fhi = f(np.log10(lo)), f(np.log10(hi))
    if flo > 0:
        return None, '始终激活（下限处已 MB>MC）'
    if fhi < 0:
        return None, '始终不激活'
    r = brentq(f, np.log10(lo), np.log10(hi), xtol=1e-12, maxiter=400)
    return float(10.0 ** r), '唯一阈值'


# ---------------- 配比子问题（问题一 M2 冻结接口）----------------
def load_m2(path=r'd:\F题\q1_quality_results\mix_final_model_quad.csv'):
    d = pd.read_csv(path)
    d = d.rename(columns={d.columns[0]: 'term'}).set_index('term')
    lin = d.loc[['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
                  'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
                  'hackernews', 'pubmed_abstracts', 'other']]
    quad_terms = [t for t in d.index if t.endswith('^2')]
    quad = d.loc[quad_terms]
    return lin, quad, quad_terms


# ================================================================================
# 配比 p 的坐标约定与参考点 p0（两种口径，必须显式区分）
# ================================================================================
#
# 口径 A（RAW，与问题一冻结接口完全一致）
#   附件 A 的配方以 3 位小数存储，逐行和 ∈ [0.996, 1.003]，A4 均值行和 = 0.99988086。
#   问题一的 M2 回归与 Q1_M2响应接口.csv.gz 都在该原始值上计算，故**接口对账必须用 RAW**。
#
# 口径 B（NORMALIZED，本问配比子问题使用）
#   题面要求 p 满足 Σp_i = 1、p_i >= 0，即 p 落在 17 维单纯形 Δ^17 上。
#   RAW 行和偏离 1 是 3 位小数的舍入产物，直接使用会使 p0 不在 Δ^17 上，
#   进而使「单纯形 + L1 支持球」的可行域自相矛盾（顶点构造给出 Σx=0.99988）。
#   故配比子问题一律按行归一化；Q1_p0_17dim_manifest.json 已提供同一列
#   （p0_A4mean_normalized，本处用 P0_17_RAW/行和 复算，两者一致到 1e-16）。
#
# 两种口径下 h_p 的差异在 [10] 中量化（量级 O(1e-5)，远小于 h_p 支持半径 0.26）。
DOM17 = ['arxiv', 'freelaw', 'nih_exporter', 'pubmed_central', 'wikipedia_en',
         'dm_mathematics', 'github', 'philpapers', 'stackexchange', 'enron_emails',
         'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc', 'europarl', 'hackernews',
         'pubmed_abstracts', 'uspto_backgrounds']

# A4 训练配方 512 行的逐域算术均值（原始值，见 Q1_p0_17dim.csv 的 p0_A4mean 列）
P0_17_RAW = np.array([0.11300390625, 0.09680859375, 0.005888671875, 0.113728515625,
                      0.075078125, 0.0254609375, 0.110689453125, 0.005486328125,
                      0.099423828125, 0.0022109375, 0.046912109375, 0.119482421875,
                      0.019599609375, 0.012736328125, 0.0116484375, 0.066173828125,
                      0.075548828125])
P0_17 = P0_17_RAW / P0_17_RAW.sum()      # 归一到 Σ=1，用于配比子问题

# M2 回归的 13 维域顺序（与 load_m2 的 lin.index 一致；uspto_backgrounds 为参考域不进入）
M13 = ['arxiv', 'freelaw', 'pubmed_central', 'wikipedia_en', 'dm_mathematics',
       'github', 'stackexchange', 'gutenberg_pg_19', 'pile_cc', 'ubuntu_irc',
       'hackernews', 'pubmed_abstracts', 'other']

# 17 -> 13 聚合矩阵：4 个小域并入 'other'，uspto_backgrounds 作为参考域被排除
_AGG17 = {'nih_exporter': 'other', 'philpapers': 'other',
          'enron_emails': 'other', 'europarl': 'other'}
AMAT = np.zeros((13, 17))
for _j, _d in enumerate(DOM17):
    _m = _AGG17.get(_d, _d)
    if _m in M13:
        AMAT[M13.index(_m), _j] = 1.0

P0_13_RAW = AMAT @ P0_17_RAW              # 口径 A（= Q1_M2接口_manifest.json 的 p0_13coords）
P0_13 = AMAT @ P0_17                      # 口径 B（配比子问题使用）

_P_CACHE = None


def _fin(x):
    """数值有限性判据；求解器未找到可行解时返回 None，np.isfinite(None) 会抛 TypeError。"""
    return x is not None and bool(np.isfinite(x))


def p_coefs():
    """惰性加载并缓存 M2 冻结接口的等权 h_p 系数（全脚本共用同一份）。"""
    global _P_CACHE
    if _P_CACHE is None:
        lin, quad, qt = load_m2()
        bbar, gbar, idx6 = h_coef(lin, quad, qt)
        _P_CACHE = dict(lin=lin, quad=quad, qt=qt, bbar=bbar, gbar=gbar, idx6=idx6)
    return _P_CACHE


def h17(x, p0=None):
    """17 维配方输入的等权 h_p（p0 默认为归一化参考点 P0_13）。

    h_p(p) = bbar·(A p - p0) + Σ_{j∈I6} gbar_j ((A p)_j² - p0_j²)
    """
    c = p_coefs()
    x13 = AMAT @ np.asarray(x, float)
    p0 = P0_13 if p0 is None else np.asarray(p0, float)
    return float(c['bbar'] @ (x13 - p0)
                 + c['gbar'] @ (x13[c['idx6']] ** 2 - p0[c['idx6']] ** 2))


def h17_grad(x):
    """∇_p h_p（17 维）= Aᵀ ∇_{p13} h_p。"""
    c = p_coefs()
    x13 = AMAT @ np.asarray(x, float)
    g = c['bbar'].copy()
    g[c['idx6']] += 2.0 * c['gbar'] * x13[c['idx6']]
    return AMAT.T @ g


def vertex_enum_max(r, tol=1e-12):
    """max h_p 的**精确解**：凸二次型在凸多面体上的最大值必在顶点取得。

    F = {x >= 0, Σx = 1, ||x-p0||_1 <= r} 的顶点刻画：|Z|（压到 0 的坐标）+
    |W|（L1 的 kink 坐标）= 15，自由坐标恰 2 个 —— 一个「搬出余量」k、一个
    「搬入目的」i。故枚举 (Z,k,i) 覆盖 F 的全部顶点；对 (k,i) 的 17×17 网格闭式求值。

    注意：L1 与单纯形约束都必须用**归一化** p0（P0_17），否则 Σx=0.99988 的
    「顶点」并不落在 Δ^17 上（这正是旧版把极值报成 +0.6593 的根因之一）。
    """
    c = p_coefs()
    bbar, gbar, idx6 = c['bbar'], c['gbar'], c['idx6']
    AI = AMAT[idx6, :]                                   # (6,17)
    FI = (gbar[:, None] * AI ** 2).sum(axis=0)           # (17,)
    BBAR_A = AMAT.T @ bbar                               # (17,)
    BBAR_P0 = float(bbar @ P0_13)
    CONST_Q = float(np.sum(gbar * P0_13[idx6] ** 2))

    m = r / 2.0
    best, bx, nsub = -np.inf, None, 0
    for size in range(0, 18):
        for Z in combinations(range(17), size):
            if P0_17[list(Z)].sum() > m + tol:
                continue
            nsub += 1
            base = P0_13 - (AMAT[:, list(Z)] @ P0_17[list(Z)] if Z else 0.0)
            rest = m - (P0_17[list(Z)].sum() if Z else 0.0)
            ks = [k for k in range(17) if k not in Z and rest <= P0_17[k] + tol]
            if not ks:
                continue
            ks = np.array(ks)
            B = base[None, :] - rest * AMAT[:, ks].T     # (nk,13)
            BI = B[:, idx6]                              # (nk,6)
            linpart = B @ bbar
            C0 = (BI ** 2) @ gbar
            E = (BI * gbar[None, :]) @ AI                # (nk,6)@(6,17) -> (nk,17)
            VAL = (linpart + C0 - BBAR_P0 - CONST_Q)[:, None] \
                + m * BBAR_A[None, :] + 2.0 * m * E + m ** 2 * FI[None, :]
            kk, ii = np.unravel_index(int(np.argmax(VAL)), VAL.shape)
            if VAL[kk, ii] > best:
                x = P0_17.copy()
                if Z:
                    x[list(Z)] = 0.0
                x[ks[kk]] -= rest
                x[ii] += m
                best, bx = float(h17(x)), x      # 直接求值，避免常数项口径错误
    return best, bx, nsub


def slsqp_extreme_p(sign, r, nstart=80, seed=0, eps=1e-9):
    """min/max h_p 的多初值 SLSQP（L1 用光滑代理 Σ√(δ²+ε²) ≤ r + 17ε/2）。

    min 方向 h_p 是凸规划（局部即全局），本函数为主链路；max 方向由顶点枚举定值，
    本函数只作方向性复核。返回 (值, 解, 成功收敛数, 收敛初值离散度)。
    """
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    cons = [dict(type='eq', fun=lambda x: x.sum() - 1.0, jac=lambda x: np.ones(17))]
    if r is not None:
        l1s = lambda x: np.sqrt((x - P0_17) ** 2 + eps ** 2).sum()
        l1sj = lambda x: (x - P0_17) / np.sqrt((x - P0_17) ** 2 + eps ** 2)
        cons.append(dict(type='ineq', fun=lambda x: r + 17 * eps / 2 - l1s(x),
                         jac=lambda x: -l1sj(x)))
    bnds = [(0.0, 1.0)] * 17
    f = lambda x: sign * h17(x)
    jac = lambda x: sign * h17_grad(x)
    starts = [P0_17, np.full(17, 1 / 17)]
    starts += [np.eye(17)[i] for i in range(17)]
    starts += [rng.dirichlet(np.ones(17) * 0.6) for _ in range(nstart)]
    starts += [rng.dirichlet(np.ones(17) * 0.15) for _ in range(nstart)]
    best, bx, nok, vals = np.inf, None, 0, []
    for s in starts:
        try:
            rr = minimize(f, s, method='SLSQP', bounds=bnds, constraints=cons, jac=jac,
                          options=dict(maxiter=600, ftol=1e-16))
        except Exception:
            continue
        x = np.clip(rr.x, 0.0, None)
        if x.sum() <= 0:
            continue
        x = x / x.sum()
        if not feasible_p(x, r):
            continue
        if rr.success:
            nok += 1
        v = sign * h17(x)
        vals.append(float(v))
        if v < best:
            best, bx = v, x
    if bx is None:
        return None, None, 0, np.nan
    spread = float(np.max(vals) - np.min(vals)) if vals else np.nan
    return float(sign * best), bx, nok, spread


def feasible_p(x, r, tol=1e-7):
    """可行性判据（容差 1e-7：光滑 L1 代理会带来约 17ε/2 ≈ 8.5e-9 的松弛）。"""
    return (np.all(x >= -tol) and abs(x.sum() - 1.0) < tol
            and (r is None or np.abs(x - P0_17).sum() <= r + tol))


def penalty_extreme_p(sign, r, nstart=30, seed=1):
    """独立链路：二次罚函数 + L-BFGS-B。仅作方向性复核（对 L1 拐点分辨力弱）。"""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    bnds = [(0.0, 1.0)] * 17
    best, bx = np.inf, None
    starts = [P0_17, np.full(17, 1 / 17)]
    starts += [rng.dirichlet(np.ones(17) * 0.5) for _ in range(nstart)]
    for rho in (1e2, 1e4, 1e6):
        for s in starts:
            def obj(x, rho=rho):
                pen = (x.sum() - 1.0) ** 2 + np.clip(-x, 0, None).sum() ** 2
                if r is not None:
                    pen += np.clip(np.abs(x - P0_17).sum() - r, 0, None) ** 2
                return sign * h17(x) + rho * pen
            try:
                rr = minimize(obj, s, method='L-BFGS-B', bounds=bnds,
                              options=dict(maxiter=800))
            except Exception:
                continue
            x = np.clip(rr.x, 0.0, None)
            if x.sum() <= 0 or not feasible_p(x, r):
                continue
            v = sign * h17(x)
            if v < best:
                best, bx = v, x
    if bx is None:
        return None, None
    return float(sign * best), bx


def h_coef(lin, quad, quad_terms):
    """把 13 目标的 h_p 系数按等权聚合，得到 13 维标量响应及其梯度系数。

        h_eq(p) = bbar·(p-p0) + Σ_{i∈I6} gbar_i (p_i² - p0_i²)
    其中 bbar = 13 目标线性系数均值，gbar = 13 目标二次系数均值（仅 6 个域）。

    lin/quad 的形状为 (项 × 目标)，故“沿 13 个目标取均值”= 沿列取均值 = axis=1。
    """
    bbar = lin.values.mean(axis=1)          # (13,) 按域索引
    idx = [list(lin.index).index(t.replace('^2', '')) for t in quad_terms]
    gbar = quad.values.mean(axis=1)         # (6,) 按 quad_terms 顺序
    return bbar, gbar, idx


def h_eq(x13, lin, quad, quad_terms, p0=None):
    """等权聚合 h_p(x)，与问题一冻结接口定义一致：h_p(p0)=0。

    接口定义（q1_export_m2_interface.py）：
        h_{p,v}(p) = (p-p0)·β_v + Σ_{i∈I6} (p_i²-p0_i²)·γ_{v,i}
    本函数返回 13 目标等权平均。必须减去 p0 的常数项，否则 h_p 整体平移。
    """
    p0 = P0_13 if p0 is None else p0
    bbar, gbar, idx = h_coef(lin, quad, quad_terms)
    return float(bbar @ (x13 - p0) + gbar @ (x13[idx] ** 2 - p0[idx] ** 2))


def h_grad(x13, bbar, gbar, idx):
    """∇_p h_eq(p)（13 维）。"""
    g = bbar.copy()
    g[idx] += 2.0 * gbar * x13[idx]
    return g


def main():
    out = {}
    Q0 = 0.5          # 主口径：B6 质量网格设计中心（中位）
    Lctx = 4096       # 主口径：C7 中位附近
    budgets = [1e19, 1e22, 1e24]

    print('=' * 108)
    print('问题三核心计算（重写版）· 参数 SET_A：E=%.4f A=%.4f a=%.4f B=%.4f b=%.4f rN=%.4f rD=%.4f E1=%.4f'
          % (E_T, A_T, ALPHA, B_T, BETA, RHO_N, RHO_D, E1))
    print('eta=%.1e   L_crit = 6/eta = %.0f tokens   Q0=%.2f   Lctx=%d   K=%.4f'
          % (ETA, L_CRIT, Q0, Lctx, k_of(Lctx)))
    print('=' * 108)

    # ================= [1] 降维自洽性：固定 Q=Q0 的数值解 vs 解析解 =================
    print('\n[1] 降维与解析解自洽性（Q=Q0 纯规模分支）')
    chk = []
    for C in budgets:
        ps = pure_scale(C, Q0, Lctx)
        r = solve_NDQ(C, 'exp', Q0, Lctx, nq=2)     # nq=2 -> 只在 {Q0,1} 上取，此处用于对照
        num = solve_N(Q0, C / 1e18, k_of(Lctx), 0.0)
        Lnum = loss(num, C / 1e18 / (k_of(Lctx) * num), Q0)
        chk.append(dict(C='1e%d' % round(np.log10(C)), N_ana=ps['N_B'], N_num=num,
                        rel=abs(num - ps['N_B']) / ps['N_B'],
                        L_ana=ps['L'], L_num=float(Lnum),
                        dL=abs(Lnum - ps['L']), DN=ps['ratio_DN']))
        print('  C=%-5s N_ana=%.6fB N_num=%.6fB 相对偏差=%.2e | L_ana=%.8f L_num=%.8f |D/N=%.2f'
              % (chk[-1]['C'], ps['N_B'], num, chk[-1]['rel'], ps['L'], Lnum, ps['ratio_DN']))
    out['selfcheck_purescale'] = chk

    # ================= [2] 三档预算 × 三种成本函数 =================
    print('\n[2] 三档预算 × 三种质量成本函数：联合最优（Q0=%.2f, Lctx=%d）' % (Q0, Lctx))
    print('    %-6s %-6s %10s %12s %9s %10s   %-22s %s'
          % ('C', 'g(Q)', 'N*(B)', 'D*(B)', 'Q*', 'L*', '成本份额(训/质/注)', '纯规模L*'))
    rows = []
    for C in budgets:
        ps = pure_scale(C, Q0, Lctx)
        for gn in ['exp', 'pow', 'log']:
            r = solve_NDQ(C, gn, Q0, Lctx)
            r['g_label'] = G_LABEL[gn]
            r['L_pure'] = ps['L']
            r['gain_vs_pure'] = (ps['L'] - r['L']) / ps['L']
            rows.append(r)
            print('    %-6s %-6s %10.4f %12.3f %9.4f %10.4f   (%.3f,%.3f,%.3f)      %s'
                  % ('1e%d' % round(np.log10(C)), G_LABEL[gn], r['N_B'], r['D_B'], r['Q'], r['L'],
                     r['s_train'], r['s_Q'], r['s_attn'],
                     '%.4f (%+.2f%%)' % (ps['L'], 100 * r['gain_vs_pure'])))
    out['budget_table'] = rows

    # ================= [3] 预算弹性：数值 vs 解析 =================
    #
    # 口径要点：弹性必须对「模型自身的渐近地板」E_eff(Q) = E - E1*Q 取。
    # 若误用固定地板 E - E1*Q0，在 Q* 饱和到 1 的区间会多算 E1*(1-Q0)=0.0514，
    # 把弹性系统性放大（旧口径给出 -0.166，真值 -0.140）。
    # 因此拟合区间必须限制在 Q* 已饱和的高预算段，并逐点用 Q* 计算 E_eff。
    print('\n[3] 预算—最优损失弹性：d ln(L*-E_eff)/d ln C，E_eff = E - E1*Q*（自洽地板）')
    print('    解析基准（Q 饱和后 Q 通道冻结）：-ab/(a+b) = %.6f'
          % (-ALPHA * BETA / (ALPHA + BETA)))
    for gn in ['exp', 'pow', 'log']:
        Cs = np.logspace(21.0, 26.0, 13)          # 三形式在该区间均已饱和
        rr = [solve_NDQ(c, gn, Q0, Lctx, nq=120) for c in Cs]
        Lv = np.array([x['L'] for x in rr])
        Eeff = np.array([E_T - E1 * x['Q'] for x in rr])
        sl = np.polyfit(np.log(Cs), np.log(Lv - Eeff), 1)[0]
        print('    g=%-4s 饱和区斜率=%.5f   解析=%.5f   差值=%+.5f   (区间内 Q*∈[%.3f,%.3f])'
              % (G_LABEL[gn], sl, -ALPHA * BETA / (ALPHA + BETA),
                 sl + ALPHA * BETA / (ALPHA + BETA),
                 min(x['Q'] for x in rr), max(x['Q'] for x in rr)))
        out.setdefault('budget_elasticity', []).append(
            dict(g=gn, numeric=float(sl), analytic=float(-ALPHA * BETA / (ALPHA + BETA)),
                 C_lo=float(Cs[0]), C_hi=float(Cs[-1])))
    # 低预算段（Q 内点）：弹性被质量通道放大，且依赖工作点，必须逐点报
    print('    低预算段（Q 内点，弹性被质量通道放大，强依赖工作点）：')
    for gn in ['exp', 'pow']:
        for C in [1e18, 1e19]:
            r = solve_NDQ(C, gn, Q0, Lctx, nq=400)
            h = 1e-4 * np.log(10.0)
            def _L(lc):
                return solve_NDQ(10.0 ** lc, gn, Q0, Lctx, nq=400)['L']
            sl = (_L(np.log10(C) + h) - _L(np.log10(C) - h)) / (2 * h)
            print('        g=%-4s C=1e%-2d  Q*=%.4f  局部弹性=%.5f'
                  % (G_LABEL[gn], round(np.log10(C)), r['Q'], sl))
    # 纯规模区间（Q 被冻结在 Q0 的预算段）
    Cs = np.logspace(15, 17.5, 12)
    Ls = np.array([pure_scale(c, Q0, Lctx)['L'] for c in Cs])
    sl = np.polyfit(np.log(Cs), np.log(Ls - (E_T - E1 * Q0)), 1)[0]
    print('    纯规模段（Q≡Q0）数值斜率=%.5f   解析=%.5f'
          % (sl, -ALPHA * BETA / (ALPHA + BETA)))

    # ================= [4] 质量激活阈值 =================
    print('\n[4] 质量激活阈值 C_act（纯规模分支上 MB(Q0)=MC(Q0) 的临界预算）')
    act = {}
    for gn in ['exp', 'pow', 'log']:
        Ca, kind = act_threshold(gn, Q0, Lctx)
        ps = pure_scale(1e19, Q0, Lctx)
        MB, MC = mb_mc(Q0, ps['N_B'], ps['D_B'], Q0, Lctx, gn)
        act[gn] = Ca
        print('    g=%-6s  C_act=%s   [%s]   (C=1e19 处 MB/MC=%.3f)'
              % (G_LABEL[gn], '%.3e' % Ca if Ca else '—', kind, MB / MC))
    out['activation'] = act

    # ================= [5] 结构性转移扫描与精确阈值 =================
    print('\n[5] 结构性转移：状态 sigma(C) = (Q 边界状态, 主导成本成分)')
    for gn in ['exp', 'pow', 'log']:
        prev, marks = None, []
        for lc in np.arange(15.0, 27.01, 0.125):
            r = solve_NDQ(10.0 ** lc, gn, Q0, Lctx, nq=120)
            Qs = ('boundary' if r['Q'] <= Q0 + 1e-9 else
                  ('saturated' if r['Q'] >= 1 - 1e-9 else 'interior'))
            sh = dict(train=r['s_train'], attn=r['s_attn'], Q=r['s_Q'])
            dom = max(sh, key=sh.get)
            cur = (Qs, dom)
            if prev is not None and cur != prev:
                marks.append((lc, prev, cur))
            prev = cur
        print('    g=%-6s' % G_LABEL[gn])
        if not marks:
            print('        全程单一状态 %s' % (prev,))
        for lc, p0, c0 in marks:
            print('        log10C≈%6.3f : %-22s -> %s' % (lc, str(p0), str(c0)))
        out.setdefault('transitions', []).append(
            dict(g=gn, marks=[dict(log10C=m[0], frm=str(m[1]), to=str(m[2])) for m in marks]))

    # 精确阈值：激活 / 饱和（对 Q* 直接二分）
    print('\n[5b] 精确阈值（对 Q* 直接二分，容差 1e-6 个数量级）')
    sat = {}
    thr = {}
    for gn in ['exp', 'pow', 'log']:
        def qstar(lc):
            return solve_NDQ(10.0 ** lc, gn, Q0, Lctx, nq=200)['Q']
        # 激活
        lo, hi = 15.0, 27.0
        if qstar(lo) > Q0 + 1e-9:
            Cact = None
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if qstar(mid) > Q0 + 1e-9:
                    hi = mid
                else:
                    lo = mid
            Cact = 10.0 ** hi
        # 饱和
        lo2, hi2 = 15.0, 27.0
        if qstar(hi2) < 1 - 1e-9:
            Csat = None
        else:
            while qstar(lo2) >= 1 - 1e-9 and lo2 > 5.0:
                lo2 -= 1.0
            for _ in range(60):
                mid = 0.5 * (lo2 + hi2)
                if qstar(mid) >= 1 - 1e-9:
                    hi2 = mid
                else:
                    lo2 = mid
            Csat = 10.0 ** hi2
        sat[gn] = Csat
        thr[gn] = dict(C_act=Cact, C_sat=Csat)
        print('    g=%-6s C_act=%s   C_sat=%s'
              % (G_LABEL[gn],
                 ('%.3e' % Cact) if Cact else '—',
                 ('%.3e' % Csat) if Csat else '—'))
    out['saturation'] = sat
    out['thresholds'] = thr

    # ================= [5c] 全局性核验：Q 维密集扫描 vs 网格+黄金分割 =================
    print('\n[5c] 全局性核验（Q 维密集扫描，检查归约后损失在 Q 上是否非凸）')
    for gn in ['exp', 'pow', 'log']:
        g, _ = G[gn]
        for C in budgets:
            Ctil, K = C / 1e18, k_of(Lctx)
            Qg = np.linspace(Q0, 1.0, 2001)
            Lg = np.empty_like(Qg)
            for i, Q in enumerate(Qg):
                Psi_t = max(g(Q) - g(Q0), 0.0) / 1e9
                N = solve_N(Q, Ctil, K, Psi_t)
                Lg[i] = loss(N, Ctil / (K * N + Psi_t), Q) if N is not None else np.inf
            jd = int(np.nanargmin(Lg))
            sg = np.sign(np.diff(Lg))
            nloc = int(np.sum((sg[:-1] < 0) & (sg[1:] > 0)))      # 内部局部极小个数
            r = solve_NDQ(C, gn, Q0, Lctx, nq=200)
            print('    g=%-6s C=1e%-2d 密集扫描 Q*=%.4f L*=%.5f | 求解器 Q*=%.4f L*=%.5f | '
                  'ΔL=%.1e | 内部局部极小=%d'
                  % (G_LABEL[gn], round(np.log10(C)), Qg[jd], Lg[jd], r['Q'], r['L'],
                     Lg[jd] - r['L'], nloc))
            out.setdefault('global_check', []).append(
                dict(g=gn, C=C, Q_dense=float(Qg[jd]), L_dense=float(Lg[jd]),
                     Q_solver=r['Q'], L_solver=float(r['L']), dL=float(Lg[jd] - r['L']),
                     n_local_min=nloc))

    # ================= [6] L_ctx 敏感性 + 临界点核验 =================
    print('\n[6] L_ctx 敏感性（C7 实际取值；g=pow, C=1e19, Q0=%.2f）' % Q0)
    print('    %8s %10s %12s %9s %10s %10s %14s' %
          ('Lctx', 'N*(B)', 'D*(B)', 'Q*', 's_train', 's_attn', 's_attn/s_train'))
    lrows = []
    for Lc in LCTX_C7:
        r = solve_NDQ(1e19, 'pow', Q0, Lc)
        lrows.append(dict(Lctx=Lc, **{k: r[k] for k in
                                      ['N_B', 'D_B', 'Q', 'L', 's_train', 's_attn', 's_Q',
                                       'ratio_attn_train']}))
        print('    %8d %10.4f %12.3f %9.4f %10.4f %10.4f %14.4f   (=eta*Lctx/6=%.4f)'
              % (Lc, r['N_B'], r['D_B'], r['Q'], r['s_train'], r['s_attn'],
                 r['ratio_attn_train'], ETA * Lc / 6.0))
    print('    解析临界点 L_crit = 6/eta = %.0f（落在 C7 可行区间 [2048, 131072] 内部，'
          '位于 8192 与 32768 之间）' % L_CRIT)
    out['lctx_sensitivity'] = lrows

    # ================= [7] Q0 敏感性 =================
    print('\n[7] Q0 敏感性（g=exp, Lctx=%d）' % Lctx)
    q0rows = []
    for Q0v in [0.3, 0.4, 0.5, 0.6, 0.7]:
        line = []
        for C in budgets:
            r = solve_NDQ(C, 'exp', Q0v, Lctx, nq=200)
            line.append(r)
        q0rows.append(dict(Q0=Q0v, rows=line))
        print('    Q0=%.4f  ' % Q0v + ' | '.join(
            'C=1e%d: Q*=%.4f L*=%.4f' % (round(np.log10(x['C'])), x['Q'], x['L']) for x in line))
    out['q0_sensitivity'] = [dict(Q0=r['Q0'],
                                  rows=[dict(C=x['C'], Q=x['Q'], L=x['L'], N_B=x['N_B'],
                                             D_B=x['D_B']) for x in r['rows']])
                             for r in q0rows]

    # ================= [8] 参数集敏感性（E1/rhoD 分工）=================
    print('\n[8] 参数集敏感性：SET_A(E1=%.4f,rD=%.4f) vs 边界端(E1=0,rD=%.4f)'
          % (E1, RHO_D, 0.2469))
    for gn in ['exp', 'pow', 'log']:
        Ca, _ = act_threshold(gn, Q0, Lctx)
        Cb, _ = act_threshold(gn, Q0, Lctx, par=PAR_B)
        print('    g=%-6s C_act: SET_A=%s | 边界端=%s'
              % (G_LABEL[gn], ('%.3e' % Ca) if Ca else '—', ('%.3e' % Cb) if Cb else '—'))
        for C in budgets:
            ra = solve_NDQ(C, gn, Q0, Lctx, nq=200)
            rb = solve_NDQ(C, gn, Q0, Lctx, par=PAR_B, nq=200)
            print('        C=1e%-2d  SET_A: N*=%.3fB D*=%.2fB Q*=%.4f L*=%.4f | '
                  '边界端: N*=%.3fB D*=%.2fB Q*=%.4f L*=%.4f'
                  % (round(np.log10(C)), ra['N_B'], ra['D_B'], ra['Q'], ra['L'],
                     rb['N_B'], rb['D_B'], rb['Q'], rb['L']))

    # ================= [9] 支撑域核对 =================
    print('\n[9] 支撑域核对（问题二校准域 N∈[%.2f,%.2f]B, D∈[%.0f,%.0f]B, Q∈[%.1f,%.1f]）'
          % (BOX['N'][0], BOX['N'][1], BOX['D'][0], BOX['D'][1], BOX['Q'][0], BOX['Q'][1]))
    sup = []
    for C in budgets:
        r = solve_NDQ(C, 'exp', Q0, Lctx)
        fl = []
        if not (BOX['N'][0] <= r['N_B'] <= BOX['N'][1]):
            fl.append('N 越界')
        if not (BOX['D'][0] <= r['D_B'] <= BOX['D'][1]):
            fl.append('D 越界')
        if not (BOX['Q'][0] <= r['Q'] <= BOX['Q'][1]):
            fl.append('Q 越界')
        sup.append(dict(C=r['C'], N_B=r['N_B'], D_B=r['D_B'], Q=r['Q'],
                        in_box=not fl, flags=fl))
        print('    C=1e%-2d  N*=%9.4fB  D*=%10.3fB  Q*=%.4f  -> %s'
              % (round(np.log10(C)), r['N_B'], r['D_B'], r['Q'],
                 '在支撑域内' if not fl else '、'.join(fl)))
    # 使最优配置落在支撑域内的预算窗口
    lo_c, hi_c = None, None
    for lc in np.arange(16.0, 26.01, 0.01):
        r = solve_NDQ(10.0 ** lc, 'exp', Q0, Lctx, nq=60)
        ok = (BOX['N'][0] <= r['N_B'] <= BOX['N'][1] and BOX['D'][0] <= r['D_B'] <= BOX['D'][1])
        if ok and lo_c is None:
            lo_c = lc
        if ok:
            hi_c = lc
    print('    最优配置落在支撑域内的预算窗口：C ∈ [%.2e, %.2e]' % (10 ** lo_c, 10 ** hi_c))
    out['support'] = dict(rows=sup, window=[float(10 ** lo_c), float(10 ** hi_c)])

    # ================= [10] 配比子问题（17 维单纯形 + 13 维 h_p 响应）=================
    print('\n[10] 配比子问题（问题一 M2 冻结接口，等权 h_p；支持半径取训练留出 L1 分位）')
    c = p_coefs()
    lin, quad, qt = c['lin'], c['quad'], c['qt']
    bbar, gbar, idx6 = c['bbar'], c['gbar'], c['idx6']
    assert list(lin.index) == M13, '模块级 M13 与 load_m2 的域顺序不一致'

    # --- 10.0 口径自检：p0 必须落在单纯形上 ---
    print('    口径自检：Σ P0_17_RAW = %.10f（原始 3 位小数数据）-> Σ P0_17 = %.12f（归一化）'
          % (P0_17_RAW.sum(), P0_17.sum()))
    print('    h17(p0) = %.3e （自检，应为 0）' % h17(P0_17))
    print('    AMAT@p0_17 vs p0_13 最大偏差 = %.3e （应为 0，核验 17→13 聚合口径）'
          % np.max(np.abs(AMAT @ P0_17 - P0_13)))

    # --- 10.1 与冻结接口逐行对账（必须用 RAW 口径，才能与问题一产物同源）---
    iface = r'd:\F题\q1_quality_results\Q1_M2响应接口.csv.gz'
    try:
        dd = pd.read_csv(iface)
        rec = dd[dd.split == 'p0 (A4 mean)'].iloc[0]
        dev0 = abs(h_eq(rec[[f'p_{k}' for k in M13]].values.astype(float), lin, quad, qt,
                        p0=P0_13_RAW) - float(rec['h_p_eq']))
        sub = dd[dd.split != 'p0 (A4 mean)'].sample(12, random_state=0)
        devt = max(abs(h_eq(r[[f'p_{k}' for k in M13]].values.astype(float), lin, quad, qt,
                            p0=P0_13_RAW) - float(r['h_p_eq'])) for _, r in sub.iterrows())
        print('    接口对账（RAW 口径）：p0 行 |Δh_p_eq| = %.3e ；随机 12 个检验配方 max|Δh_p_eq| = %.3e'
              % (dev0, devt))
        # 硬校验：实现必须与问题一冻结接口逐行一致（防止静默的口径漂移）
        if devt > 1e-8:
            raise AssertionError('h_eq 实现与 Q1 冻结接口不一致：max|Δ|=%.3e' % devt)
        hv = dd.loc[dd.split != 'p0 (A4 mean)', 'h_p_eq'].values.astype(float)
        iq = np.percentile(hv, [0, 5, 50, 95, 100])
        print('    接口内 %d 个检验配方 h_p_eq 分布（RAW）：min=%.4f p05=%.4f 中位=%.4f p95=%.4f max=%.4f'
              % (len(hv), iq[0], iq[1], iq[2], iq[3], iq[4]))
        out['p_iface_reconcile'] = dict(p0=float(dev0), test_max=float(devt),
                                        n=int(len(hv)), q=[float(v) for v in iq])
        # --- 10.2 两种口径的差异量化：归一化对 h_p 的影响（用 17 维原始配方，能拿到行和）---
        try:
            rt = pd.read_csv(r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
                             r'\test_mixture_1m.csv')
            cols17 = [f'train_the_pile_{d}' for d in DOM17]
            Xr = rt[cols17].values.astype(float)
            dv = np.array([abs(h17(x / x.sum()) - h17(x, p0=P0_13_RAW)) for x in Xr])
            print('    口径 A→B（行归一化）对 h_p 的影响：max|Δh_p| = %.3e，中位 = %.3e（%d 配方）'
                  % (dv.max(), float(np.median(dv)), len(Xr)))
            out['p_convention_gap'] = dict(max=float(dv.max()),
                                           median=float(np.median(dv)))
        except FileNotFoundError:
            print('    [警告] 未找到 17 维配方表，跳进口径差异量化')
    except FileNotFoundError:
        print('    [警告] 未找到冻结接口文件，跳过对账')

    # --- 10.3 支持半径下的极值：max 用顶点枚举（精确），min 用多初值 SLSQP（凸规划）---
    prows = []
    for lab, rad in [('无支持约束（仅单纯形）', None),
                     ('L1<=0.2601（训练留出 p95）', 0.2601),
                     ('L1<=0.1781（训练留出中位）', 0.1781)]:
        if rad is None:
            # 无支持约束时 h_p 凸 => max 在单纯形顶点；min 仍在单纯形内部
            vmax = float(np.max([h17(np.eye(17)[i]) for i in range(17)]))
            vx = np.eye(17)[int(np.argmax([h17(np.eye(17)[i]) for i in range(17)]))]
            nsub = 17
        else:
            vmax, vx, nsub = vertex_enum_max(rad)
        smin, sx, nok_min, spread_min = slsqp_extreme_p(+1, rad, nstart=80)
        smax, _, nok_max, _ = slsqp_extreme_p(-1, rad, nstart=30)
        pmax, _ = penalty_extreme_p(-1, rad, nstart=15)
        pmin, _ = penalty_extreme_p(+1, rad, nstart=30)
        ok_max = bool(_fin(smax) and _fin(pmax)
                      and vmax >= smax - 1e-5 and vmax >= pmax - 1e-5)
        ok_min = bool(_fin(smin) and spread_min < 1e-9
                      and (not _fin(pmin) or smin <= pmin + 1e-6))
        prows.append(dict(label=lab, r=rad, h_min=float(smin) if _fin(smin) else None,
                          h_max=float(vmax),
                          h_max_slsqp=float(smax) if _fin(smax) else None,
                          h_max_penalty=float(pmax) if _fin(pmax) else None,
                          h_min_penalty=float(pmin) if _fin(pmin) else None,
                          n_vertex_subsets=int(nsub), n_ok_min=int(nok_min),
                          min_spread=float(spread_min) if _fin(spread_min) else None,
                          consistent=bool(ok_max and ok_min),
                          l1_min=float(np.abs(sx - P0_17).sum()) if sx is not None else None,
                          l1_max=float(np.abs(vx - P0_17).sum()),
                          x_min=sx.tolist() if sx is not None else None,
                          x_max=vx.tolist()))
        print('    %-26s min h_p=%s (L1=%s, 收敛%d, 离散度=%.1e)'
              % (lab, ('%+.4f' % smin) if _fin(smin) else 'nan',
                 ('%.4f' % np.abs(sx - P0_17).sum()) if sx is not None else 'nan',
                 nok_min, spread_min if _fin(spread_min) else np.nan))
        print('    %-26s max h_p=%+.4f (L1=%.4f, 顶点枚举 %d 子集) | SLSQP 复核=%+.4f | 罚函数=%+.4f'
              % ('', vmax, np.abs(vx - P0_17).sum(), nsub,
                 smax if _fin(smax) else np.nan,
                 pmax if _fin(pmax) else np.nan))
    print('    说明：max 由顶点枚举精确给出（凸二次型在凸多面体上的最大值必在顶点取得）；')
    print('          min 为凸规划，多初值 SLSQP 的收敛初值离散度 < 1e-9 即全局性自证。')
    print('          支持半径取自问题一 manifest：训练留出最近邻 L1 p95=0.2601、中位=0.1781（17 维）')
    out['p_subproblem'] = prows

    # 10b 配比通道在 λ_p 情景下的损失包络（工作点 C=1e22, g=exp）
    print('\n[10b] 配比通道的损失包络（工作点 C=1e22, g=exp 的最优配置）')
    base = solve_NDQ(1e22, 'exp', Q0, Lctx)
    hmin, hmax = prows[1]['h_min'], prows[1]['h_max']
    for lam in [0.0, 0.5, 1.0, 1.5]:
        Llo = loss(base['N_B'], base['D_B'], base['Q'], lam, hmin)
        Lhi = loss(base['N_B'], base['D_B'], base['Q'], lam, hmax)
        print('    lam_p=%.1f  h_p∈[%+.4f,%+.4f] -> L∈[%.5f, %.5f]  极差=%.5f (%.2f%%)'
              % (lam, hmin, hmax, Llo, Lhi, abs(Lhi - Llo), 100 * abs(Lhi - Llo) / base['L']))
        out.setdefault('p_envelope', []).append(
            dict(lam=lam, h_min=hmin, h_max=hmax, L_lo=float(Llo), L_hi=float(Lhi)))

    # 10c p 通道解耦：h_p 只以 B~ → B~·exp(λ_p h_p) 进入，且 p 不消耗预算
    print('\n[10c] p 通道解耦（λ_p>0 取 h_min，λ_p<0 取 h_max；p 不进入预算约束）')
    print('    %-7s %10s %12s %12s %9s %10s %12s' %
          ('lam_p', 'B~_eff', 'N*(B)', 'D*(B)', 'Q*', 'L*', 'vs lam=0'))
    base0 = solve_NDQ(1e22, 'exp', Q0, Lctx, par=PAR_DEF)
    for lam in [0.0, 0.5, 1.0, 1.5]:
        hh = hmin if lam > 0 else (hmax if lam < 0 else 0.0)
        Bf = B_T * np.exp(lam * hh)
        r = solve_NDQ(1e22, 'exp', Q0, Lctx, nq=200,
                      par=(E_T, A_T, ALPHA, Bf, BETA, RHO_N, RHO_D, E1))
        print('    %-7.1f %10.5f %12.4f %12.4f %9.4f %10.5f %+12.5f'
              % (lam, Bf, r['N_B'], r['D_B'], r['Q'], r['L'], r['L'] - base0['L']))
        out.setdefault('p_decoupled', []).append(
            dict(lam=lam, h=hh, B_eff=Bf, N_B=r['N_B'], D_B=r['D_B'], Q=r['Q'],
                 L=float(r['L']), dL=float(r['L'] - base0['L'])))

    # ================= [11] 联合优化 vs 顺序决策 =================
    print('\n[11] 联合优化 vs 顺序决策（先定规模、质量无预算余量 -> 退化为纯规模）')
    for gn in ['exp', 'pow', 'log']:
        for C in budgets:
            r = solve_NDQ(C, gn, Q0, Lctx)
            ps = pure_scale(C, Q0, Lctx)
            print('    g=%-6s C=1e%-2d  联合 L*=%.4f | 顺序 L=%.4f | 联合收益=%+.3f (%+.2f%%)'
                  % (G_LABEL[gn], round(np.log10(C)), r['L'], ps['L'],
                     ps['L'] - r['L'], 100 * (ps['L'] - r['L']) / ps['L']))

    # ================= [12] 成本函数选择的影响 =================
    print('\n[12] 成本函数选择对最优解的影响（同一预算下三形式对照）')
    for C in budgets:
        qs = {gn: solve_NDQ(C, gn, Q0, Lctx) for gn in ['exp', 'pow', 'log']}
        span = max(x['L'] for x in qs.values()) - min(x['L'] for x in qs.values())
        print('    C=1e%-2d  Q*: %s | L* 极差=%.5f (%.3f%% of %.4f)'
              % (round(np.log10(C)),
                 ', '.join('%s=%.4f' % (G_LABEL[g], qs[g]['Q']) for g in qs),
                 span, 100 * span / np.mean([x['L'] for x in qs.values()]),
                 np.mean([x['L'] for x in qs.values()])))

    # ================= [13] 质量通道转移带宽 =================
    #
    # 旧口径（Q*∈[Q0+0.2, Q0+0.8]=[0.7,1.3]）有缺陷：Q* 上限为 1，饱和后的预算点
    # 会一直落在该区间内，导致把「饱和之后的全部预算」都算成过渡带（exp 虚高到 7.2 decade）。
    # 正确定义：过渡带 = Q* 严格处于内点 (Q0, 1) 的预算跨度，即 [C_act, C_sat]。
    print('\n[13] 质量通道转移带宽（严格内点区间 Q0<Q*<1 对应的预算跨度，即 C_act→C_sat）')
    bw = {}
    for gn in ['exp', 'pow', 'log']:
        Cact, Csat = thr[gn]['C_act'], thr[gn]['C_sat']
        if Cact is None or Csat is None:
            bw[gn] = None
            print('    g=%-6s 阈值缺失，跳过' % G_LABEL[gn])
            continue
        width = float(np.log10(Csat) - np.log10(Cact))
        bw[gn] = width
        print('    g=%-6s C_act=%.4e  C_sat=%.4e  带宽=%.3f decade%s'
              % (G_LABEL[gn], Cact, Csat, width,
                 '  （几乎无过渡带：激活即饱和）' if width < 1e-3 else ''))
    _bw = {k: v for k, v in bw.items() if v is not None}
    print('    带宽序：%s（exp/pow 有可观测的渐进过渡带，log 因高 Q 区成本饱和快而无过渡带）'
          % ' > '.join(sorted(_bw, key=_bw.get, reverse=True)))
    out['bandwidth'] = [dict(g=k, width=(None if v is None else float(v))) for k, v in bw.items()]

    with open(r'd:\F题\q3_core_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print('\n已写出 q3_core_results.json')


if __name__ == '__main__':
    main()
