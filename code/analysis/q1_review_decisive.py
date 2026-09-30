# -*- coding: utf-8 -*-
"""审阅用（决定性检验）：
  [E] B6 上"质量项挂在哪里"是否可识别 —— 论文采用"只挂 D 项"(FormA/单通道)，
      但 B6 有 9x5x8 全网格，可直接检验四种挂载形式。
  [F] 问题一冻结配比模型的"配比效应量级"能否跨规模迁移 —— h_p 是 1M 尺度的
      对数损失比，论文把它直接当 150B/300B 尺度的可加项使用。
  [G] 论文表4.11 只列 6 个领域的 AME；此处补全 17 域，检查符号与量级的对称性。
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

B = r'd:\F题\F题\real_attachments\B_scaling_laws'
b6 = pd.read_csv(f'{B}/supplementary_NQ_experiment.csv')
N, D, Q, L = (b6[c].values.astype(float) for c in ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss'])
Q0 = 0.5
uQ = Q - Q0

print('=' * 92)
print('[E] B6 上质量项挂载位置的可识别性（9x5x8 全网格，360 点）')
print('=' * 92)


def fit(forms, npar, seed=0, tries=40):
    def r(th):
        return forms(th) - L
    best = None
    for s in range(tries):
        rng = np.random.default_rng(seed + s)
        x0 = np.log(np.abs(np.array(npar))) + rng.normal(0, 0.6, len(npar))
        try:
            sol = least_squares(r, x0, method='lm', max_nfev=600000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    k = len(npar)
    rss = (r(best.x) ** 2).sum()
    n = len(L)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return np.exp(best.x), np.sqrt(rss / n), aic, bic


def f_M0(th):
    E, A, a, B_, b = np.exp(th)
    return E + A * N ** -a + B_ * D ** -b


def f_MA(th):      # 论文采用：质量只挂 D 项
    E, A, a, B_, b, g = np.exp(th)
    return E + A * N ** -a + B_ * D ** -b * np.exp(g * (Q0 - Q))


def f_MB(th):      # 双通道：质量同时挂 N 项与 D 项
    E, A, a, B_, b, gN, gD = np.exp(th)
    return E + A * N ** -a * np.exp(gN * (Q0 - Q)) + B_ * D ** -b * np.exp(gD * (Q0 - Q))


def f_MC(th):      # 全局乘子：质量作用于全部可约损失
    E, A, a, B_, b, g = np.exp(th)
    return E + (A * N ** -a + B_ * D ** -b) * np.exp(g * (Q0 - Q))


def f_MD(th):      # 加性线性质量项
    E, A, a, B_, b, g = np.exp(th)
    return E + A * N ** -a + B_ * D ** -b - g * uQ


res = {}
res['M0 无质量项'] = fit(f_M0, [1.66, 0.54, 0.28, 1.32, 0.28])
res['MA 只挂D项(论文)'] = fit(f_MA, [1.66, 0.54, 0.28, 1.32, 0.28, 0.3])
res['MB 双通道N+D'] = fit(f_MB, [1.66, 0.54, 0.28, 1.32, 0.28, 0.2, 0.2], tries=60)
res['MC 全局乘子'] = fit(f_MC, [1.66, 0.54, 0.28, 1.32, 0.28, 0.3])
res['MD 加性线性'] = fit(f_MD, [1.66, 0.54, 0.28, 1.32, 0.28, 0.5])

print(f"  {'模型':<22}{'k':>3}{'RMSE':>10}{'AIC':>12}{'BIC':>12}   参数")
for k, (p, rm, ai, bi) in res.items():
    ps = ' '.join(f'{v:.4f}' for v in p)
    print(f'  {k:<22}{len(p):>3}{rm:>10.5f}{ai:>12.1f}{bi:>12.1f}   {ps}')

print(f"\n  B1 对照: E=1.68980 A=0.35398 alpha=0.33998 B=1.24031 beta=0.27988")
print('  判读：若 MA（只挂 D 项）不是最优，则论文的 FormA/单通道选择未做检验，')
print('        而该选择会强制 beta 由 ~0.28 畸变到 ~0.10，并压低 E。')

print('\n' + '=' * 92)
print('[F] 配比效应的跨规模可迁移性（问题一冻结模型 -> 60M/1B）')
print('=' * 92)
BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
ld = lambda n: pd.read_csv(f'{BASE}/{n}.csv')
tr_mx, tr_ls = ld('train_mixture_1m'), ld('train_pile_loss_1m')
te_mx, te_ls = ld('test_mixture_1m'), ld('test_pile_loss_1m')
t6_mx, t6_ls = ld('test_mixture_60m'), ld('test_pile_loss_60m')
t1_mx, t1_ls = ld('test_mixture_1B'), ld('test_pile_loss_1B')
mx = [c for c in tr_mx.columns if c != 'index']
ls = [c for c in tr_ls.columns if c != 'index']
TGT = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in ls]
P, Y = tr_mx[mx].values, np.log(tr_ls[ls].values)
P6, Y6 = t6_mx[mx].values, np.log(t6_ls[ls].values)
P1, Y1 = t1_mx[mx].values, np.log(t1_ls[ls].values)

REF = mx.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]


class Ridge:
    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0, ddof=1)
        self.sd[self.sd == 0] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        G = Z.T @ Z
        best, bg = None, np.inf
        for al in np.logspace(-4, 4, 41):
            A = G + al * np.eye(Z.shape[1]); A[0, 0] -= al
            Bm = np.linalg.solve(A, Z.T @ Y)
            tr = np.trace(Z @ np.linalg.solve(A, Z.T))
            g = (((Y - Z @ Bm) ** 2).sum(0) / len(Z)) / (1 - tr / len(Z)) ** 2
            if g.mean() < bg:
                bg, best = g.mean(), Bm
        self.Bm = best
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd]) @ self.Bm


rd = Ridge().fit(P[:, KEEP], Y)
print(f"  {'目标':<20}{'1M std':>9}{'60M std':>9}{'1B std':>9}{'比值60M/1M':>12}{'斜率60M':>9}{'斜率1B':>9}")
sl6, sl1 = [], []
for j, t in enumerate(TGT):
    p6 = rd.predict(P6[:, KEEP])[:, j]
    p1 = rd.predict(P1[:, KEEP])[:, j]
    s6 = np.polyfit(p6, Y6[:, j], 1)[0]
    s1 = np.polyfit(p1, Y1[:, j], 1)[0]
    sl6.append(s6); sl1.append(s1)
    print(f'  {t:<20}{Y[:, j].std(ddof=1):>9.4f}{Y6[:, j].std(ddof=1):>9.4f}{Y1[:, j].std(ddof=1):>9.4f}'
          f'{Y6[:, j].std(ddof=1) / Y[:, j].std(ddof=1):>12.3f}{s6:>9.3f}{s1:>9.3f}')
print(f'\n  跨目标平均：std 1M={Y.std(0, ddof=1).mean():.4f}  60M={Y6.std(0, ddof=1).mean():.4f}  '
      f'1B={Y1.std(0, ddof=1).mean():.4f}')
print(f'              实测 log L 对模型预测 log L 的回归斜率：60M={np.mean(sl6):.3f}  1B={np.mean(sl1):.3f}')
print('  斜率<1 => 冻结模型在更大规模上把配比效应"放大"了；')
print('  而论文把 1M 尺度算出的 h_p 直接作为 150B/300B 尺度的可加项（表5.6）。')

print('\n' + '=' * 92)
print('[G] 表4.11 完整 17 域 AME（δ=0.03，共同支持子集，GBM 代理）')
print('=' * 92)


def _bs(X, g, h, ml):
    n, d = X.shape
    G, H = g.sum(), h.sum()
    best = (None, None, -np.inf)
    for j in range(d):
        o = np.argsort(X[:, j]); xs, gs, hs = X[o, j], g[o], h[o]
        Gl, Hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
        Gr, Hr = G - Gl, H - Hl
        k_ = np.arange(1, n)
        v = (k_ >= ml) & (k_ <= n - ml) & (xs[1:] > xs[:-1])
        if not v.any():
            continue
        gain = np.where(v, Gl ** 2 / (Hl + 1e-6) + Gr ** 2 / (Hr + 1e-6) - G ** 2 / (H + 1e-6), -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best[2]:
            best = (j, (xs[k] + xs[k + 1]) / 2, gain[k])
    return best


def _bd(X, g, h, dep, md, ml, lam):
    nd = {'leaf': -g.sum() / (h.sum() + lam)}
    if dep >= md or len(X) < 2 * ml:
        return nd
    j, thr, gn = _bs(X, g, h, ml)
    if j is None or gn <= 0:
        return nd
    m = X[:, j] <= thr
    if m.sum() < ml or (~m).sum() < ml:
        return nd
    nd.update(j=j, thr=thr, L=_bd(X[m], g[m], h[m], dep + 1, md, ml, lam),
              R=_bd(X[~m], g[~m], h[~m], dep + 1, md, ml, lam))
    return nd


def _ap(nd, X, out, idx):
    if 'j' not in nd:
        out[idx] = nd['leaf']; return
    m = X[:, nd['j']] <= nd['thr']
    _ap(nd['L'], X[m], out, idx[m]); _ap(nd['R'], X[~m], out, idx[~m])


def gfit(X, y, nt=220, lr=0.03, md=4, ml=20, lam=1.0):
    base, pred, tr = y.mean(), np.full(len(y), y.mean()), []
    for _ in range(nt):
        t = _bd(X, pred - y, np.ones(len(y)), 0, md, ml, lam)
        o = np.empty(len(y)); _ap(t, X, o, np.arange(len(y)))
        pred = pred + lr * o; tr.append(t)
    return base, tr, lr


def gpred(M, X):
    base, tr, lr = M
    pr = np.full(len(X), base)
    for t in tr:
        o = np.empty(len(X)); _ap(t, X, o, np.arange(len(X))); pr = pr + lr * o
    return pr


G = [gfit(P[:, KEEP], Y[:, j]) for j in range(13)]
GP = lambda X: np.column_stack([gpred(m, X) for m in G])
sd_tr = Y.std(0, ddof=1)
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(Dm, np.inf)
nn = Dm.min(1); thr = np.quantile(nn, 0.95)

delta = 0.03
keep = []
for i, p in enumerate(P):
    ok = True
    for j in range(17):
        rest = 1 - p[j]
        if rest <= 1e-12:
            ok = False; break
        q = p * (1 - delta / rest); q[j] = p[j] + delta
        if np.sqrt(((P - q) ** 2).sum(1)).min() > thr:
            ok = False; break
    if ok:
        keep.append(i)
keep = np.array(keep)
print(f'  共同支持子集大小 = {len(keep)}（论文 δ=0.03 为 475）')

base = GP(P[keep][:, KEEP])
rows = []
for j in range(17):
    q = P[keep] * (1 - delta / (1 - P[keep][:, j]))[:, None]
    q[:, j] = P[keep][:, j] + delta
    dL = GP(q[:, KEEP]) - base
    rows.append((mx[j].replace('train_the_pile_', ''), (dL / sd_tr).mean()))
rows.sort(key=lambda r: r[1])
print(f"  {'领域':<22}{'ΔJ (标准化)':>14}   论文表4.11 值")
paper = {'ubuntu_irc': -0.0895, 'stackexchange': -0.0873, 'hackernews': -0.0833,
         'enron_emails': 0.0026, 'europarl': 0.0064, 'philpapers': 0.0077}
for nm, v in rows:
    print(f'  {nm:<22}{v:>14.5f}   ' + (f'{paper[nm]:+.4f}' if nm in paper else ''))
neg = sum(1 for _, v in rows if v < 0)
print(f'\n  ΔJ<0（增加该域份额降低预测Loss）的领域数 = {neg}/17；'
      f'最小 {rows[0][1]:+.5f}，最大 {rows[-1][1]:+.5f}')
print('  论文表4.11 只列两端各 3 个；若 17 域中绝大多数为负，则该表会给出')
print('  "增加几乎任何领域都能降损"的印象，需要显式说明这是份额再分配下的联合结果。')
