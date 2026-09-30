# -*- coding: utf-8 -*-
"""审阅用补充：
  [H] ΔJ 的符号不对称是否来自树模型 —— 用线性模型与 GBM 代理并列比较
  [I] 复现论文 4.4.2 的 481/475/472（尝试多种支持门槛规则变体）
"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
ld = lambda n: pd.read_csv(f'{BASE}/{n}.csv')
tr_mx, tr_ls = ld('train_mixture_1m'), ld('train_pile_loss_1m')
mx = [c for c in tr_mx.columns if c != 'index']
ls = [c for c in tr_ls.columns if c != 'index']
P, Y = tr_mx[mx].values, np.log(tr_ls[ls].values)
NAME = [c.replace('train_the_pile_', '') for c in mx]
REF = mx.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]
sd = Y.std(0, ddof=1)


class Ridge:
    def fit(self, X, Y):
        self.mu, self.s = X.mean(0), X.std(0, ddof=1)
        self.s[self.s == 0] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.s])
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
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.s]) @ self.Bm


def _bs(X, g, h, ml):
    n, d = X.shape; G, H = g.sum(), h.sum(); best = (None, None, -np.inf)
    for j in range(d):
        o = np.argsort(X[:, j]); xs, gs, hs = X[o, j], g[o], h[o]
        Gl, Hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]; Gr, Hr = G - Gl, H - Hl
        k_ = np.arange(1, n)
        v = (k_ >= ml) & (k_ <= n - ml) & (xs[1:] > xs[:-1])
        if not v.any():
            continue
        gn = np.where(v, Gl ** 2 / (Hl + 1e-6) + Gr ** 2 / (Hr + 1e-6) - G ** 2 / (H + 1e-6), -np.inf)
        k = int(np.argmax(gn))
        if gn[k] > best[2]:
            best = (j, (xs[k] + xs[k + 1]) / 2, gn[k])
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


rd = Ridge().fit(P[:, KEEP], Y)
G = [gfit(P[:, KEEP], Y[:, j]) for j in range(13)]
GP = lambda X: np.column_stack([gpred(m, X) for m in G])
RP = lambda X: rd.predict(X)

Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(Dm, np.inf)
nn = Dm.min(1); thr = np.quantile(nn, 0.95)
delta = 0.03

print('=' * 92)
print('[H] ΔJ 符号分布：线性(Ridge) vs GBM 代理（δ=0.03，全集 512）')
print('=' * 92)


def ame(fn, X, idx):
    b = fn(X[idx][:, KEEP])
    out = []
    for j in range(17):
        q = X[idx] * (1 - delta / (1 - X[idx][:, j]))[:, None]
        q[:, j] = X[idx][:, j] + delta
        out.append(((fn(q[:, KEEP]) - b) / sd).mean())
    return np.array(out)


allidx = np.arange(len(P))
for nm, fn in [('Ridge(线性)', RP), ('GBM代理', GP)]:
    v = ame(fn, P, allidx)
    o = np.argsort(v)
    print(f'  {nm}: 负 {int((v < 0).sum())}/17  正 {int((v > 0).sum())}/17   '
          f'min={v.min():+.5f} max={v.max():+.5f}')
    print('     ' + '  '.join(f'{NAME[j]}={v[j]:+.4f}' for j in o[:5]))
    print('     ' + '  '.join(f'{NAME[j]}={v[j]:+.4f}' for j in o[-5:]))

# 反向扰动：减少 j、等比例增加其余（考察是否近似反对称）
print('\n  反向扰动（减少域 j，等比例增加其余）对照：')
for nm, fn in [('Ridge(线性)', RP), ('GBM代理', GP)]:
    b = fn(P[:, KEEP])
    out = []
    for j in range(17):
        q = P.copy()
        rest = 1 - P[:, j]
        q[:, j] = P[:, j] - delta
        add = delta / rest
        for i in range(17):
            if i != j:
                q[:, i] = P[:, i] * (1 + add)
        out.append(((fn(q[:, KEEP]) - b) / sd).mean())
    v = np.array(out)
    print(f'  {nm}: 负 {int((v < 0).sum())}/17  正 {int((v > 0).sum())}/17  '
          f'min={v.min():+.5f} max={v.max():+.5f}')

print('\n' + '=' * 92)
print('[I] 支持门槛规则变体，尝试复现论文 481/475/472')
print('=' * 92)


def perturb(p, j, d):
    rest = 1 - p[j]
    if rest <= 1e-12:
        return None
    q = p * (1 - d / rest); q[j] = p[j] + d
    return q


def dist_to(q, idx_excl=None):
    d = np.sqrt(((P - q) ** 2).sum(1))
    if idx_excl is not None:
        d = np.delete(d, idx_excl)
    return d.min()


for d in (0.01, 0.03, 0.05):
    line = f'  δ={d:.2f}: '
    # V1 扰动点对全训练集
    c1 = sum(all(dist_to(perturb(P[i], j, d)) <= thr for j in range(17)) for i in range(len(P)))
    # V2 扰动点对训练集但排除自身
    c2 = sum(all(dist_to(perturb(P[i], j, d), i) <= thr for j in range(17)) for i in range(len(P)))
    # V3 V1 且自身也在支持范围内
    c3 = sum((nn[i] <= thr) and all(dist_to(perturb(P[i], j, d)) <= thr for j in range(17))
             for i in range(len(P)))
    # V4 V2 且自身也在支持范围内
    c4 = sum((nn[i] <= thr) and all(dist_to(perturb(P[i], j, d), i) <= thr for j in range(17))
             for i in range(len(P)))
    print(line + f'V1={c1}  V2={c2}  V3={c3}  V4={c4}   （论文 481/475/472 随 δ 递减）')
print('  注：论文阈值 0.2600（留一最近邻距离 95 分位），此处 %.6f' % thr)
