# -*- coding: utf-8 -*-
"""审阅用：复核论文 4.5 的 Qmapped 覆盖率与"加入质量代理的增量"及其置换检验。"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
ld = lambda n: pd.read_csv(f'{BASE}/{n}.csv')
tr_mx, tr_ls = ld('train_mixture_1m'), ld('train_pile_loss_1m')
te_mx, te_ls = ld('test_mixture_1m'), ld('test_pile_loss_1m')
mx = [c for c in tr_mx.columns if c != 'index']
ls = [c for c in tr_ls.columns if c != 'index']
NAME = [c.replace('train_the_pile_', '') for c in mx]
P, Y = tr_mx[mx].values, np.log(tr_ls[ls].values)
Pt, Yt = te_mx[mx].values, np.log(te_ls[ls].values)
sd = Y.std(0, ddof=1)

QD = pd.read_csv(r'd:\F题\q1_quality_results\domain_Q.csv')
qmap = dict(zip(QD['domain'], QD['Q_group_mean']))
# 论文表4.4 / q1 报告的主评分 Q_hierarchical_balanced（0-100 尺度）
QH = {'arxiv': 64.2046, 'book': 45.8456, 'c4': 62.2814, 'commoncrawl': 63.6085,
      'github': 54.0570, 'stackexchange': 64.8150, 'wikipedia': 60.2105}
print('域级主评分 Q_H（论文表4.4）:', {k: round(v, 4) for k, v in QH.items()})

# A16 映射：mixture_domain -> quality_domain（direct / near_direct）
MAP = {'arxiv': 'arxiv', 'github': 'github', 'stackexchange': 'stackexchange',
       'wikipedia_en': 'wikipedia', 'gutenberg_pg_19': 'book', 'pile_cc': 'commoncrawl'}
idx = {nm: i for i, nm in enumerate(NAME)}
M = [idx[k] for k in MAP if k in idx]
print('可靠映射域:', [(NAME[i], MAP[NAME[i]], round(qmap[MAP[NAME[i]]], 4)) for i in M])
print('未映射域:', [NAME[i] for i in range(17) if i not in M])

cov = P[:, M].sum(1)
print(f'\n训练集覆盖率: 可计算(c>0) {int((cov > 0).sum())}/512  均值 {cov.mean():.4f}  中位 {np.median(cov):.4f}'
      f'   （论文 507/512, 0.5646, 0.6004）')
covt = Pt[:, M].sum(1)
print(f'1M留出覆盖率: 可计算 {int((covt > 0).sum())}/256  均值 {covt.mean():.4f} 中位 {np.median(covt):.4f}'
      f'   （论文 256/256, 0.5690, 0.5830）')


def qmapped(Qv):
    w = P[:, M]
    s = w.sum(1)
    out = np.full(len(P), np.nan)
    ok = s > 0
    out[ok] = (w[ok] * Qv).sum(1) / s[ok]
    return out, ok


Qv0 = np.array([QH[MAP[NAME[i]]] for i in M])
print('进入 Qmapped 的六域 Q_H:', Qv0)


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


def cv_r2(X, Y, folds=5, seed=7):
    rng = np.random.default_rng(seed)
    fs = np.array_split(rng.permutation(len(X)), folds)
    pr = np.empty_like(Y)
    for f in range(folds):
        va = fs[f]; tr = np.concatenate([fs[j] for j in range(folds) if j != f])
        pr[va] = Ridge().fit(X[tr], Y[tr]).predict(X[va])
    return 1 - ((pr - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


REF = mx.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]
q0, ok = qmapped(Qv0)
sel = ok
print(f'\n[增量] 在 {int(sel.sum())} 个可映射训练配方上（论文 507）')
Xp = P[sel][:, KEEP]
Xpq = np.column_stack([P[sel][:, KEEP], q0[sel]])
r2_p = cv_r2(Xp, Y[sel])
r2_pq = cv_r2(Xpq, Y[sel])
print(f'  Ridge(p)        CV 汇总R2 = {r2_p:.5f}')
print(f'  Ridge(p,Qmapped) CV 汇总R2 = {r2_pq:.5f}')
print(f'  增量 = {r2_pq - r2_p:+.5f}   （论文 +0.00092）')

# 置换检验
nperm = 200
deltas = []
for s in range(nperm):
    rng = np.random.default_rng(1000 + s)
    perm = rng.permutation(len(M))
    Qv = Qv0[perm]
    qp, _ = qmapped(Qv)
    Xp2 = np.column_stack([P[sel][:, KEEP], qp[sel]])
    deltas.append(cv_r2(Xp2, Y[sel]) - r2_p)
deltas = np.array(deltas)
real = r2_pq - r2_p
print(f'\n[置换] {nperm} 次随机域-质量错位：均值 {deltas.mean():+.5f}  '
      f'真实增量 {real:+.5f}  百分位 {(deltas < real).mean():.3f}  '
      f'单侧p = {(1 + (deltas >= real).sum()) / (nperm + 1):.3f}')
print('  论文：真实 +0.000921，置换均值 +0.00510，百分位 0.146，p=0.854')
