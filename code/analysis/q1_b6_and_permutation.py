# -*- coding: utf-8 -*-
"""审阅用（最终）：
  (1) 拟合 B6 质量模型，核对论文 5.2.2 / 表5.6-5.7 的隐含参数（含 beta_6 是否塌缩）
  (2) 复现 4.5 的 Qmapped 置换诊断（用论文自报的域级 QH）
  (3) h_agg 对参考点的敏感性 -> 对问题二表5.6 情景 Loss 的影响
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

AB = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
BB = r'd:\F题\F题\real_attachments\B_scaling_laws'
load = lambda p: pd.read_csv(p)

# ---------------- (1) B6 质量模型 ----------------
b6 = load(f'{BB}/supplementary_NQ_experiment.csv')
print('=' * 92)
print('[1] B6 半合成数据与质量模型 M_Q: L = E + A N^-a + B D^-b exp{g_Q (Q - 0.5)}')
print('=' * 92)
print(f"  B6: {b6.shape}, N 取值 {sorted(b6.N_params_B.unique())}")
print(f"      D 取值 {sorted(b6.D_tokens_B.unique())}")
print(f"      Q 取值 {sorted(b6.Q_score.unique())}")
print(f"  (N,D) 组合数 = {b6.groupby(['N_params_B','D_tokens_B']).ngroups}, "
      f"中位 D = {b6.D_tokens_B.median()}, 中位 N = {b6.N_params_B.median()}")

N, D, Q, L = (b6[c].values for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
Q0 = 0.5


def resid(th):
    E, A, a, B, b, g = np.exp(th)
    return E + A * N ** -a + B * D ** -b * np.exp(g * (Q - Q0)) - L


best = None
for s in range(40):
    rng = np.random.default_rng(s)
    th0 = np.log([1.6, 0.5, 0.3, 1.3, 0.28, 0.4]) + rng.normal(0, [0.3, 0.3, 0.1, 0.2, 0.05, 0.6])
    r = least_squares(resid, th0, method='lm', max_nfev=200000)
    if best is None or r.cost < best.cost:
        best = r
E6, A6, a6, B6v, b6v, g6 = np.exp(best.x)
pred = E6 + A6 * N ** -a6 + B6v * D ** -b6v * np.exp(g6 * (Q - Q0))
rmse = np.sqrt(((pred - L) ** 2).mean())
print(f'\n  自由拟合：E={E6:.4f}  A={A6:.4f}  alpha={a6:.5f}  B={B6v:.4f}  beta={b6v:.5f}  gamma_Q={g6:.6f}')
print(f'    RMSE={rmse:.6f}  R2={1-((pred-L)**2).sum()/((L-L.mean())**2).sum():.6f}')
print(f'  论文 5.4.2 报告 gamma_Q = 0.317893  ->  本次 {g6:.6f}')
print(f'\n  对照 B1 经典标度律：E=1.689798  A=0.353980  alpha=0.339977  B=1.240306  beta=0.279878')
print(f'  => beta 由 {0.279878:.4f} 变为 {b6v:.4f}（{(b6v/0.279878-1)*100:+.1f}%）；'
      f'alpha 由 {0.339977:.4f} 变为 {a6:.4f}（{(a6/0.339977-1)*100:+.1f}%）')

# 表5.7 反演的 beta_6
rows = [(0.7, 10.0, 0.4536), (0.41, 300.0, 0.3232), (6.9, 300.0, 0.3232)]
b_imp = np.log(rows[0][2] / rows[1][2]) / np.log(300.0 / 10.0)
print(f'\n  由论文表5.7 的 M_Q 反演 beta_6 = ln({rows[0][2]}/{rows[1][2]})/ln(30) = {b_imp:.5f}')
print(f'    （表5.7 中 (0.41,300) 与 (6.9,300) 的 M_Q 完全相同 = 模型强制的结构恒等）')

# 工作点核对
print(f'\n  表5.6/5.7 工作点反解（用 Form A、lambda_p=0、h=0）：')
L0 = 2.61862
T = (2.66676 - L0) / (np.exp(0.5 * 0.09252) - 1)     # BD^-beta
AN = (2.69956 - L0) / (np.exp(0.5 * 0.09252) - 1) - T  # AN^-alpha
print(f'    B D^-beta = {T:.4f},  A N^-alpha = {AN:.4f},  E = {L0-T-AN:.4f}')
print(f'    以本次 beta_6={b6v:.4f}, B={B6v:.4f} 反解 D = {np.exp(-np.log(T/B6v)/b6v):.1f}')
print(f'    以本次 alpha_6={a6:.4f}, A={A6:.4f} 反解 N = {np.exp(-np.log(AN/A6)/a6):.4f}')

# ---------------- (2) Qmapped 置换诊断 ----------------
print('\n' + '=' * 92)
print('[2] 复现 4.5 Qmapped 置换诊断（Q 用论文表4.4 的域级 QH，映射用 A16）')
print('=' * 92)
mix = load(f'{AB}/train_mixture_1m.csv')
loss = load(f'{AB}/train_pile_loss_1m.csv')
mxc = [c for c in mix.columns if c != 'index']
lsc = [c for c in loss.columns if c != 'index']
P = mix[mxc].values
Y = np.log(loss[lsc].values)
QH = dict(arxiv=64.2046, github=54.0570, stackexchange=64.8150,
          wikipedia=60.2105, book=45.8456, commoncrawl=63.6085)
MAP = {'train_the_pile_arxiv': 'arxiv', 'train_the_pile_github': 'github',
       'train_the_pile_stackexchange': 'stackexchange',
       'train_the_pile_wikipedia_en': 'wikipedia',
       'train_the_pile_gutenberg_pg_19': 'book',
       'train_the_pile_pile_cc': 'commoncrawl'}
Mcol = [mxc.index(k) for k in MAP]
qvals = np.array([QH[v] for v in MAP.values()])


def qmapped(Pm, qs):
    w = Pm[:, Mcol]
    cov = w.sum(1)
    num = (w * qs).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(cov > 0, num / np.maximum(cov, 1e-12), np.nan)
    return q, cov


REF = mxc.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]


def ridge_fit(X, Yv, alphas=np.logspace(-4, 4, 41)):
    mu, sd = X.mean(0), X.std(0, ddof=1); sd[sd == 0] = 1
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    G, ZtY, n = Z.T @ Z, Z.T @ Yv, len(Z)
    best, bg = None, np.inf
    for a in alphas:
        A = G + a * np.eye(Z.shape[1]); A[0, 0] -= a
        B = np.linalg.solve(A, ZtY)
        tr = np.trace(Z @ np.linalg.solve(A, Z.T))
        g = ((Yv - Z @ B) ** 2).sum(0) / n / (1 - tr / n) ** 2
        if g.mean() < bg:
            bg, best = g.mean(), (mu, sd, B)
    return best


def ridge_pred(m, X):
    mu, sd, B = m
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ B


rng = np.random.default_rng(7)
fold = np.array_split(rng.permutation(512), 5)


def cv_r2(X, Yv):
    pr = np.zeros_like(Yv)
    for f in range(5):
        va = fold[f]; tr = np.concatenate([fold[j] for j in range(5) if j != f])
        pr[va] = ridge_pred(ridge_fit(X[tr], Yv[tr]), X[va])
    return 1 - ((pr - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()


q, cov = qmapped(P, qvals)
ok = ~np.isnan(q)
print(f'  可计算覆盖率 c>0 的配方数 = {ok.sum()}/512，平均覆盖率 = {cov[ok].mean():.4f}'
      f'（论文 507/512，0.5646）')
Xs, Ys, qs = P[ok][:, KEEP], Y[ok], q[ok]
fold = np.array_split(rng.permutation(len(Xs)), 5)
base_r2 = cv_r2(Xs, Ys)
Xq = np.column_stack([Xs, qs])
full_r2 = cv_r2(Xq, Ys)
print(f'  Ridge(p) 训练内CV pooled R2 = {base_r2:.5f}')
print(f'  Ridge(p,Qmapped)      pooled R2 = {full_r2:.5f}   增量 = {full_r2-base_r2:+.5f}'
      f'（论文 +0.000921）')

deltas = []
for r in range(1000):
    perm = rng.permutation(6)
    qp, _ = qmapped(P[ok], qvals[perm])
    deltas.append(cv_r2(np.column_stack([Xs, qp]), Ys) - base_r2)
deltas = np.array(deltas)
real = full_r2 - base_r2
print(f'  置换(1000次) 增量均值 = {deltas.mean():+.5f}，2.5/97.5 分位 = '
      f'{np.quantile(deltas,0.025):+.5f}/{np.quantile(deltas,0.975):+.5f}')
print(f'  真实增量百分位 = {(deltas < real).mean():.3f}，单侧经验 p = {(1+deltas[deltas>=real].size)/(1000+1):.4f}'
      f'（论文 0.146 / 0.854）')

# ---------------- (3) h_agg 的参考点敏感性 ----------------
print('\n' + '=' * 92)
print('[3] h_agg 参考点敏感性 -> 问题二表5.6 情景 Loss 的幅度')
print('=' * 92)


def _bs(X, g, h, ml):
    n, d = X.shape; G, H = g.sum(), h.sum(); best = (None, None, -np.inf)
    for j in range(d):
        o = np.argsort(X[:, j]); xs, gs, hs = X[o, j], g[o], h[o]
        Gl, Hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
        Gr, Hr = G - Gl, H - Hl
        k_ = np.arange(1, n)
        v = (k_ >= ml) & (k_ <= n - ml) & (xs[1:] > xs[:-1])
        if not v.any():
            continue
        gn = np.where(v, Gl**2/(Hl+1e-6)+Gr**2/(Hr+1e-6)-G**2/(H+1e-6), -np.inf)
        k = int(np.argmax(gn))
        if gn[k] > best[2]:
            best = (j, (xs[k]+xs[k+1])/2, gn[k])
    return best


def _bd(X, g, h, dep, md, ml, lam):
    nd = {'leaf': -g.sum()/(h.sum()+lam)}
    if dep >= md or len(X) < 2*ml:
        return nd
    j, thr, gn = _bs(X, g, h, ml)
    if j is None or gn <= 0:
        return nd
    m = X[:, j] <= thr
    if m.sum() < ml or (~m).sum() < ml:
        return nd
    nd.update(j=j, thr=thr, L=_bd(X[m], g[m], h[m], dep+1, md, ml, lam),
              R=_bd(X[~m], g[~m], h[~m], dep+1, md, ml, lam))
    return nd


def _ap(nd, X, out, idx):
    if 'j' not in nd:
        out[idx] = nd['leaf']; return
    m = X[:, nd['j']] <= nd['thr']
    _ap(nd['L'], X[m], out, idx[m]); _ap(nd['R'], X[~m], out, idx[~m])


def gfit(X, y, nt=220, lr=0.03, md=4, ml=20, lam=1.0):
    base, pred, tr = y.mean(), np.full(len(y), y.mean()), []
    for _ in range(nt):
        t = _bd(X, pred-y, np.ones(len(y)), 0, md, ml, lam)
        o = np.empty(len(y)); _ap(t, X, o, np.arange(len(y)))
        pred = pred + lr*o; tr.append(t)
    return (base, tr, lr)


def gpred(M, X):
    base, tr, lr = M
    p = np.full(len(X), base)
    for t in tr:
        o = np.empty(len(X)); _ap(t, X, o, np.arange(len(X)))
        p = p + lr*o
    return p


Xtr = P[:, KEEP]
ms = [gfit(Xtr, Y[:, j]) for j in range(13)]
gm = lambda X: np.column_stack([gpred(m, X) for m in ms])
for tag, ref in [('A4 算术均值 p0', Xtr.mean(0, keepdims=True)),
                 ('A4 最优（预测 Loss 最低）', Xtr[np.argmin(gm(Xtr).mean(1))][None, :]),
                 ('A4 中位数（按预测 Loss）', Xtr[np.argsort(gm(Xtr).mean(1))[256]][None, :])]:
    h = (gm(Xtr) - gm(ref)[0]).mean(1)
    print(f'  参考点 = {tag:<26} h_agg: min={h.min():+.5f} med={np.median(h):+.5f} max={h.max():+.5f}')
print('\n  论文表5.6 取 h_agg = 0.09252 并乘 lambda_p 得 Form A 的 Loss 增量：')
for lam in (0.0, 0.5, 1.0, 1.5):
    for h in (0.09252, 0.01335, 0.0488, 0.1270):
        T = 1.0167
        dL = T * (np.exp(lam * h) - 1)
        print(f'    lambda_p={lam:<4} h_agg={h:<8.5f} -> Loss = {2.61862+dL:.5f} (增量 {dL:+.5f})', end='')
        if h == 0.01335:
            print('   <- 表5.5 的 A4 下界')
        else:
            print()
