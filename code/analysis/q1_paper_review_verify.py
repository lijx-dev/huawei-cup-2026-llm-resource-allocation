# -*- coding: utf-8 -*-
"""审阅用（numpy 自实现，无 sklearn 依赖）：
   核验论文问题一 4.4 / 4.5 的关键数值，并检验 5.2.3 接口量 h_p 的模型依赖性。
"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
np.set_printoptions(suppress=True)


def load_mix(n):
    return pd.read_csv(f'{BASE}/{n}.csv')


tr_mx, tr_ls = load_mix('train_mixture_1m'), load_mix('train_pile_loss_1m')
te_mx, te_ls = load_mix('test_mixture_1m'), load_mix('test_pile_loss_1m')
t1b_mx, t1b_ls = load_mix('test_mixture_1B'), load_mix('test_pile_loss_1B')
e10_mx, e70_mx = load_mix('est_mixture_10b'), load_mix('est_mixture_70b')

mx_cols = [c for c in tr_mx.columns if c != 'index']
ls_cols = [c for c in tr_ls.columns if c != 'index']
P, Y = tr_mx[mx_cols].values, np.log(tr_ls[ls_cols].values)
Pt, Yt = te_mx[mx_cols].values, np.log(te_ls[ls_cols].values)
P1b, Y1b = t1b_mx[mx_cols].values, np.log(t1b_ls[ls_cols].values)

# ---------------- Ridge（闭式，标准化，参考域省略） ----------------
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
        G = Z.T @ Z
        ZtY = Z.T @ Y
        best, bestgcv = None, np.inf
        n = len(Z)
        for a in self.alphas:
            A = G + a * np.eye(Z.shape[1])
            A[0, 0] -= a                      # 不惩罚截距
            B = np.linalg.solve(A, ZtY)
            H = Z @ np.linalg.solve(A, Z.T)   # 帽子矩阵
            tr = np.trace(H)
            resid = Y - Z @ B
            rss = (resid ** 2).sum(0)
            gcv = (rss / n) / (1 - tr / n) ** 2
            s = gcv.mean()
            if s < bestgcv:
                bestgcv, best = s, B
        self.B = best
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        return np.column_stack([np.ones(len(Z)), Z]) @ self.B


# ---------------- 深度受限回归树 + GBM（LightGBM 代理） ----------------
def _best_split(X, g, h, min_leaf):
    n, d = X.shape
    G, H = g.sum(), h.sum()
    best = (None, None, -np.inf)
    for j in range(d):
        o = np.argsort(X[:, j])
        xs, gs, hs = X[o, j], g[o], h[o]
        Gl = np.cumsum(gs)[:-1]
        Hl = np.cumsum(hs)[:-1]
        Gr, Hr = G - Gl, H - Hl
        valid = (np.arange(1, n) >= min_leaf) & (np.arange(1, n) <= n - min_leaf)
        # 相邻取值相同则不能分裂
        valid &= xs[1:] > xs[:-1]
        if not valid.any():
            continue
        gain = Gl ** 2 / (Hl + 1e-6) + Gr ** 2 / (Hr + 1e-6) - G ** 2 / (H + 1e-6)
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best[2]:
            best = (j, (xs[k] + xs[k + 1]) / 2, gain[k])
    return best


def _build(X, g, h, depth, max_depth, min_leaf, lam):
    node = {'leaf': -g.sum() / (h.sum() + lam)}
    if depth >= max_depth or len(X) < 2 * min_leaf:
        return node
    j, thr, gain = _best_split(X, g, h, min_leaf)
    if j is None or gain <= 0:
        return node
    m = X[:, j] <= thr
    if m.sum() < min_leaf or (~m).sum() < min_leaf:
        return node
    node.update(j=j, thr=thr,
                L=_build(X[m], g[m], h[m], depth + 1, max_depth, min_leaf, lam),
                R=_build(X[~m], g[~m], h[~m], depth + 1, max_depth, min_leaf, lam))
    return node


def _apply(node, X, out, idx):
    if 'leaf' in node and 'j' not in node:
        out[idx] = node['leaf']
        return
    m = X[:, node['j']] <= node['thr']
    _apply(node['L'], X[m], out, idx[m])
    _apply(node['R'], X[~m], out, idx[~m])


class GBM:
    """论文报告的 LightGBM 配置代理：220 树 / lr 0.03 / 深度 4 / 叶≥20 样本 / L2 正则"""

    def __init__(self, n_trees=220, lr=0.03, max_depth=4, min_leaf=20, lam=1.0):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth
        self.min_leaf, self.lam = min_leaf, lam

    def fit(self, X, y):
        self.base = y.mean()
        self.trees = []
        pred = np.full(len(y), self.base)
        for _ in range(self.n_trees):
            g = pred - y            # L2 梯度
            h = np.ones(len(y))
            t = _build(X, g, h, 0, self.max_depth, self.min_leaf, self.lam)
            o = np.empty(len(y))
            _apply(t, X, o, np.arange(len(y)))
            pred = pred + self.lr * o
            self.trees.append(t)
        return self

    def predict(self, X):
        pred = np.full(len(X), self.base)
        for t in self.trees:
            o = np.empty(len(X))
            _apply(t, X, o, np.arange(len(X)))
            pred = pred + self.lr * o
        return pred


def fit_gbm_multi(X, Y):
    return [GBM().fit(X, Y[:, j]) for j in range(Y.shape[1])]


def pred_gbm_multi(models, X):
    return np.column_stack([m.predict(X) for m in models])


def nrmse(pred, true, sd):
    return np.mean(np.sqrt(np.mean((pred - true) ** 2, axis=0)) / sd)


Xtr = P[:, KEEP]
sd_tr = Y.std(0, ddof=1)
sd_fold = None

print('=' * 88)
print('[A] 复现 4.4.1 的模型比较（NRMSE，分母=该折训练集标准差）')
print('=' * 88)

# 训练内 5 折 × 3 次重复（自实现，为控制耗时只做 3 次重复）
rng = np.random.default_rng(7)
errs_r, errs_g = [], []
for rep in range(3):
    folds = np.array_split(rng.permutation(len(Xtr)), 5)
    for f in range(5):
        va = folds[f]
        tr = np.concatenate([folds[j] for j in range(5) if j != f])
        sd = Y[tr].std(0, ddof=1)
        errs_r.append(nrmse(Ridge().fit(Xtr[tr], Y[tr]).predict(Xtr[va]), Y[va], sd))
        errs_g.append(nrmse(pred_gbm_multi(fit_gbm_multi(Xtr[tr], Y[tr]), Xtr[va]), Y[va], sd))
print(f'  Ridge    训练内CV 宏NRMSE = {np.mean(errs_r):.4f}   （论文 0.6280）')
print(f'  GBM代理  训练内CV 宏NRMSE = {np.mean(errs_g):.4f}   （论文 LightGBM 0.1662）')

ridge_full = Ridge().fit(Xtr, Y)
gbm_full = fit_gbm_multi(Xtr, Y)
pr, pg = ridge_full.predict(Pt[:, KEEP]), pred_gbm_multi(gbm_full, Pt[:, KEEP])
print(f'\n  1M 留出 Ridge 宏NRMSE = {nrmse(pr, Yt, sd_tr):.4f}  pooledR2 = {1 - ((pr - Yt) ** 2).sum() / ((Yt - Yt.mean()) ** 2).sum():.4f}   （论文 0.5770 / 0.5431）')
print(f'  1M 留出 GBM   宏NRMSE = {nrmse(pg, Yt, sd_tr):.4f}  pooledR2 = {1 - ((pg - Yt) ** 2).sum() / ((Yt - Yt.mean()) ** 2).sum():.4f}   （论文 0.1309 / 0.9817）')

# ---------------- [B] h_p 接口量的模型依赖性 ----------------
print('\n' + '=' * 88)
print('[B] 接口量 h_p(p) = log Lhat(p) - log Lhat(p0) 的模型依赖性（论文 5.2.3 / 表5.5-5.6）')
print('=' * 88)
p0 = P[:, KEEP].mean(0, keepdims=True)


def hstats(models_pred_fn, X, label):
    base = models_pred_fn(X[:1] * 0 + p0)[0]
    H = models_pred_fn(X) - base
    h = H.mean(1)
    q = np.quantile(h, [0.05, 0.5, 0.95])
    print(f'  {label:<22} A4: min={h.min():+.5f} med={np.median(h):+.5f} max={h.max():+.5f} | '
          f'5/50/95%={q[0]:+.4f}/{q[1]:+.4f}/{q[2]:+.4f}')
    return H, h


H_r, h_r = hstats(lambda X: ridge_full.predict(X), Xtr, 'Ridge（线性）')
H_g, h_g = hstats(lambda X: pred_gbm_multi(gbm_full, X), Xtr, 'GBM 代理（LightGBM）')

# 论文口径：A5 留出 256 条
Hg_te = pred_gbm_multi(gbm_full, Pt[:, KEEP]) - pred_gbm_multi(gbm_full, p0)[0]
h_te = Hg_te.mean(1)
q = np.quantile(h_te, [0.05, 0.5, 0.95])
print(f'\n  GBM 代理 A5-256 h_agg 的 5/50/95% = {q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f}')
print(f'  （实跑报告：-0.0256 / +0.0488 / +0.1160；论文表5.5 称 A4 范围 [0.01335, 0.33301]）')

# 逐目标 h_p 极值，对照论文表5.5
TGT = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in ls_cols]
print('\n  逐目标 h_p 在 A4 上的 [min, median, max]：')
print(f"    {'目标':<20}{'GBM代理':>30}{'Ridge':>30}")
for j, t in enumerate(TGT):
    print(f'    {t:<20}[{H_g[:, j].min():+.4f},{np.median(H_g[:, j]):+.4f},{H_g[:, j].max():+.4f}]'.ljust(50)
          + f'[{H_r[:, j].min():+.4f},{np.median(H_r[:, j]):+.4f},{H_r[:, j].max():+.4f}]')

# ---------------- [C] 10B/70B 是否等于训练配方 ----------------
print('\n' + '=' * 88)
print('[C] 4.4.1 称“10B/70B 的估算配方与训练配方重合”')
print('=' * 88)
tr_set = {tuple(np.round(r, 6)) for r in P}
for nm, M in [('est_10b', e10_mx), ('est_70b', e70_mx), ('test_1B', t1b_mx)]:
    A = M[[c for c in M.columns if c != 'index']].values
    hit = sum(tuple(np.round(r, 6)) in tr_set for r in A)
    print(f'  {nm}: {hit}/{len(A)} 条与 A4 训练配方完全相同')

# ---------------- [D] 共同支持样本数（表4.11 的 475 组） ----------------
print('\n' + '=' * 88)
print('[D] 4.4.2 支持门槛与共同样本（论文称 δ=0.03 时保留 475 组）')
print('=' * 88)
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(Dm, np.inf)
nn_train = Dm.min(1)
thr = np.quantile(nn_train, 0.95)
print(f'  训练配方留一最近邻距离的 95 分位 = {thr:.6f}   （论文 0.2600）')
print(f'  训练集自身满足该门槛的比例 = {(nn_train <= thr).mean():.3f}（按定义恰为 95%）')

for delta in (0.01, 0.03, 0.05):
    cnt = 0
    for p in P:
        ok = True
        for j in range(17):
            if p[j] + delta > 1 + 1e-12:
                ok = False
                break
            q = p.copy()
            rest = 1 - p[j]
            if rest <= 1e-12:
                ok = False
                break
            q = p * (1 - delta / rest)
            q[j] = p[j] + delta
            dmin = np.sqrt(((P - q) ** 2).sum(1))
            if dmin.min() > thr:
                ok = False
                break
        cnt += ok
    print(f'  δ={delta}: 全部 17 个单域扰动均可行且均在支持范围内的配方数 = {cnt}   （论文 481/475/472）')
