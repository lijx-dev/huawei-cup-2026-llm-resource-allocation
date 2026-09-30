# -*- coding: utf-8 -*-
"""审阅用：判定问题一配比模型输出的 h_p 是"真实配比效应"还是"模型类别的产物"。

核心检验：
  (1) 固定规模下，配方间 log L 的实测离散度有多大？（配比效应能否支撑 h_p 的量级）
  (2) 模型对留出配方的排序/水平解释力
  (3) 不同模型类（线性 / 二次 / 树集成）给出的 h_p 差异
  (4) 支持门槛规则变体，核对 481/475/472
"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
load = lambda n: pd.read_csv(f'{BASE}/{n}.csv')

tr_mx, tr_ls = load('train_mixture_1m'), load('train_pile_loss_1m')
te_mx, te_ls = load('test_mixture_1m'), load('test_pile_loss_1m')
t60_mx, t60_ls = load('test_mixture_60m'), load('test_pile_loss_60m')
t1b_mx, t1b_ls = load('test_mixture_1B'), load('test_pile_loss_1B')

mx_cols = [c for c in tr_mx.columns if c != 'index']
ls_cols = [c for c in tr_ls.columns if c != 'index']
TGT = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in ls_cols]

P, Y = tr_mx[mx_cols].values, np.log(tr_ls[ls_cols].values)
Pt, Yt = te_mx[mx_cols].values, np.log(te_ls[ls_cols].values)
P6, Y6 = t60_mx[mx_cols].values, np.log(t60_ls[ls_cols].values)
P1, Y1 = t1b_mx[mx_cols].values, np.log(t1b_ls[ls_cols].values)

print('=' * 92)
print('[1] 固定规模下配方间 log L 的实测离散度（这是配比效应能被"看到"的上限）')
print('=' * 92)
print(f"  {'目标':<20}{'训练512 std':>13}{'1M-256 std':>13}{'60M-256 std':>13}{'1B-64 std':>12}{'1M mean':>10}")
for j, t in enumerate(TGT):
    print(f'  {t:<20}{Y[:, j].std(ddof=1):>13.4f}{Yt[:, j].std(ddof=1):>13.4f}'
          f'{Y6[:, j].std(ddof=1):>13.4f}{Y1[:, j].std(ddof=1):>12.4f}{Yt[:, j].mean():>10.4f}')
print(f'\n  跨目标平均：训练 {Y.std(0, ddof=1).mean():.4f}  1M {Yt.std(0, ddof=1).mean():.4f}  '
      f'60M {Y6.std(0, ddof=1).mean():.4f}  1B {Y1.std(0, ddof=1).mean():.4f}')
print('  => 同一规模、仅配比不同的配方，其 log L 标准差即为"配比可解释的变动上限"。')

# ---------- 模型 ----------
REF = mx_cols.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]


class Ridge:
    def __init__(self, alphas=np.logspace(-4, 4, 41)):
        self.alphas = alphas

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0, ddof=1)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        Z = np.column_stack([np.ones(len(Z)), Z])
        G, ZtY, n = Z.T @ Z, Z.T @ Y, len(Z)
        best, bg = None, np.inf
        for a in self.alphas:
            A = G + a * np.eye(Z.shape[1]); A[0, 0] -= a
            B = np.linalg.solve(A, ZtY)
            tr = np.trace(Z @ np.linalg.solve(A, Z.T))
            gcv = ((Y - Z @ B) ** 2).sum(0) / n / (1 - tr / n) ** 2
            if gcv.mean() < bg:
                bg, best = gcv.mean(), B
        self.B = best
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        return np.column_stack([np.ones(len(Z)), Z]) @ self.B


class Poly2:      # 二次（含平方项，不含交叉项），岭正则，闭式
    def __init__(self, alpha=1e-3):
        self.alpha = alpha

    def _feat(self, X):
        return np.column_stack([np.ones(len(X)), X, X ** 2])

    def fit(self, X, Y):
        Z = self._feat(X)
        self.mu, self.sd = Z[:, 1:].mean(0), Z[:, 1:].std(0, ddof=1)
        self.sd[self.sd == 0] = 1.0
        Zs = np.column_stack([np.ones(len(Z)), (Z[:, 1:] - self.mu) / self.sd])
        G = Zs.T @ Zs
        A = G + self.alpha * np.eye(Zs.shape[1]); A[0, 0] -= self.alpha
        self.B = np.linalg.solve(A, Zs.T @ Y)
        return self

    def predict(self, X):
        Z = self._feat(X)
        Zs = np.column_stack([np.ones(len(Z)), (Z[:, 1:] - self.mu) / self.sd])
        return Zs @ self.B


def _best_split(X, g, h, min_leaf):
    n, d = X.shape
    G, H = g.sum(), h.sum()
    best = (None, None, -np.inf)
    for j in range(d):
        o = np.argsort(X[:, j]); xs, gs, hs = X[o, j], g[o], h[o]
        Gl, Hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
        Gr, Hr = G - Gl, H - Hl
        k_ = np.arange(1, n)
        valid = (k_ >= min_leaf) & (k_ <= n - min_leaf) & (xs[1:] > xs[:-1])
        if not valid.any():
            continue
        gain = np.where(valid, Gl ** 2 / (Hl + 1e-6) + Gr ** 2 / (Hr + 1e-6) - G ** 2 / (H + 1e-6), -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best[2]:
            best = (j, (xs[k] + xs[k + 1]) / 2, gain[k])
    return best


def _build(X, g, h, dep, md, ml, lam):
    node = {'leaf': -g.sum() / (h.sum() + lam)}
    if dep >= md or len(X) < 2 * ml:
        return node
    j, thr, gain = _best_split(X, g, h, ml)
    if j is None or gain <= 0:
        return node
    m = X[:, j] <= thr
    if m.sum() < ml or (~m).sum() < ml:
        return node
    node.update(j=j, thr=thr, L=_build(X[m], g[m], h[m], dep + 1, md, ml, lam),
                R=_build(X[~m], g[~m], h[~m], dep + 1, md, ml, lam))
    return node


def _apply(node, X, out, idx):
    if 'j' not in node:
        out[idx] = node['leaf']; return
    m = X[:, node['j']] <= node['thr']
    _apply(node['L'], X[m], out, idx[m]); _apply(node['R'], X[~m], out, idx[~m])


def gbm_fit(X, y, n_trees=220, lr=0.03, md=4, ml=20, lam=1.0):
    base, pred, trees = y.mean(), np.full(len(y), y.mean()), []
    for _ in range(n_trees):
        t = _build(X, pred - y, np.ones(len(y)), 0, md, ml, lam)
        o = np.empty(len(y)); _apply(t, X, o, np.arange(len(y)))
        pred = pred + lr * o; trees.append(t)
    return base, trees, lr


def gbm_pred(M, X):
    base, trees, lr = M
    pred = np.full(len(X), base)
    for t in trees:
        o = np.empty(len(X)); _apply(t, X, o, np.arange(len(X)))
        pred = pred + lr * o
    return pred


Xtr, Xte, X60, X1 = P[:, KEEP], Pt[:, KEEP], P6[:, KEEP], P1[:, KEEP]
ridge = Ridge().fit(Xtr, Y)
poly = Poly2(alpha=1e-3).fit(Xtr, Y)
gbm = [gbm_fit(Xtr, Y[:, j]) for j in range(13)]
gbm_p = lambda X: np.column_stack([gbm_pred(m, X) for m in gbm])

print('\n' + '=' * 92)
print('[2] 模型对留出配方的解释力（per-target Pearson/Spearman of log L）')
print('=' * 92)


def sp(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return np.corrcoef(ra, rb)[0, 1]


for nm, fn in [('Ridge', ridge.predict), ('Poly2', poly.predict), ('GBM', gbm_p)]:
    for sn, X, Yo in [('1M-256', Xte, Yt), ('60M-256', X60, Y6), ('1B-64', X1, Y1)]:
        pr = fn(X)
        rs = [sp(pr[:, j], Yo[:, j]) for j in range(13)]
        r2 = 1 - ((pr - Yo) ** 2).sum() / ((Yo - Yo.mean(0)) ** 2).sum()
        print(f'  {nm:<7}{sn:<9} 平均Spearman={np.mean(rs):.4f}  pooledR2={r2:.4f}  '
              f'RMSE={np.sqrt(((pr - Yo) ** 2).mean()):.4f}')

print('\n' + '=' * 92)
print('[3] 三种模型类给出的接口量 h_p（A4 512 条，参考点 p0 = A4 均值）')
print('=' * 92)
p0 = Xtr.mean(0, keepdims=True)
for nm, fn in [('Ridge', ridge.predict), ('Poly2', poly.predict), ('GBM', gbm_p)]:
    H = fn(Xtr) - fn(p0)[0]
    h = H.mean(1)
    q = np.quantile(h, [0.05, 0.5, 0.95])
    print(f'  {nm:<7} h_agg: min={h.min():+.5f} med={np.median(h):+.5f} max={h.max():+.5f} | '
          f'5/50/95%={q[0]:+.4f}/{q[1]:+.4f}/{q[2]:+.4f}')

# 论文表5.6 用的 h_agg=0.09252
print('\n  A4 中有多少条配方的 h_agg 落在论文表5.6 所用值 0.09252 附近(±0.005)：')
for nm, fn in [('Ridge', ridge.predict), ('Poly2', poly.predict), ('GBM', gbm_p)]:
    h = (fn(Xtr) - fn(p0)[0]).mean(1)
    print(f'    {nm:<7}{np.sum(np.abs(h - 0.09252) < 0.005):>4} / 512 条')

print('\n' + '=' * 92)
print('[4] h_p 与"实测损失差"的一致性（1M 留出，256 条，参考=留出集均值配比）')
print('=' * 92)
pt_bar = Xte.mean(0, keepdims=True)
for nm, fn in [('Ridge', ridge.predict), ('Poly2', poly.predict), ('GBM', gbm_p)]:
    Hm = fn(Xte) - fn(pt_bar)[0]
    Ho = Yt - Yt.mean(0, keepdims=True)
    # 逐目标相关 + 汇聚相关
    rs = [sp(Hm[:, j], Ho[:, j]) for j in range(13)]
    print(f'  {nm:<7} 逐目标 Spearman 中位={np.median(rs):+.4f}  min={min(rs):+.4f}  '
          f'| h_agg 与实测 h_agg 的 Spearman={sp(Hm.mean(1), Ho.mean(1)):+.4f}')

print('\n' + '=' * 92)
print('[5] 支持门槛规则变体（论文 481/475/472）')
print('=' * 92)
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(Dm, np.inf)
nn = Dm.min(1)
thr = np.quantile(nn, 0.95)


def perturb(p, j, delta):
    rest = 1 - p[j]
    if rest <= 1e-12:
        return None
    q = p * (1 - delta / rest)
    q[j] = p[j] + delta
    return q


for need_self in (False, True):
    out = []
    for delta in (0.01, 0.03, 0.05):
        c = 0
        for i, p in enumerate(P):
            if need_self and nn[i] > thr:
                continue
            ok = True
            for j in range(17):
                q = perturb(p, j, delta)
                if q is None or np.sqrt(((P - q) ** 2).sum(1)).min() > thr:
                    ok = False; break
            c += ok
        out.append(c)
    tag = '要求原配方自身也在支持范围内' if need_self else '仅要求 17 个扰动在支持范围内'
    print(f'  {tag}: δ=0.01/0.03/0.05 -> {out[0]}/{out[1]}/{out[2]}')
print('  论文报告:                                  δ=0.01/0.03/0.05 -> 481/475/472')
