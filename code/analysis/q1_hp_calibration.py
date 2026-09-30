# -*- coding: utf-8 -*-
"""审阅用（决定性检验）：
  (A) h_p 的量级是否经得起样本外校准？（384/128 切分，模型 h_p vs 实测 h_p 的斜率）
  (B) h_p 是否只是"离均值配比的距离"（Jensen 效应）而非"配比好坏"？
  (C) 支持门槛计数 481/475/472 的规则复原
"""
import numpy as np
import pandas as pd

BASE = r'd:\F题\F题\real_attachments\A_data_value\regmix_tables'
load = lambda n: pd.read_csv(f'{BASE}/{n}.csv')
tr_mx, tr_ls = load('train_mixture_1m'), load('train_pile_loss_1m')
te_mx, te_ls = load('test_mixture_1m'), load('test_pile_loss_1m')
t60_mx, t60_ls = load('test_mixture_60m'), load('test_pile_loss_60m')

mx_cols = [c for c in tr_mx.columns if c != 'index']
ls_cols = [c for c in tr_ls.columns if c != 'index']
TGT = [c.replace('metric/the_pile_', '').replace('_val_loss', '') for c in ls_cols]
P, Y = tr_mx[mx_cols].values, np.log(tr_ls[ls_cols].values)
Pt, Yt = te_mx[mx_cols].values, np.log(te_ls[ls_cols].values)
P6, Y6 = t60_mx[mx_cols].values, np.log(t60_ls[ls_cols].values)
REF = mx_cols.index('train_the_pile_uspto_backgrounds')
KEEP = [i for i in range(17) if i != REF]


def _best_split(X, g, h, ml):
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


def gbm_fit(X, y, nt=220, lr=0.03, md=4, ml=20, lam=1.0):
    base, pred, trees = y.mean(), np.full(len(y), y.mean()), []
    for _ in range(nt):
        t = _build(X, pred - y, np.ones(len(y)), 0, md, ml, lam)
        o = np.empty(len(y)); _apply(t, X, o, np.arange(len(y)))
        pred = pred + lr * o; trees.append(t)
    return (base, trees, lr)


def gbm_pred(M, X):
    base, trees, lr = M
    pred = np.full(len(X), base)
    for t in trees:
        o = np.empty(len(X)); _apply(t, X, o, np.arange(len(X)))
        pred = pred + lr * o
    return pred


def fit_multi(X, Y):
    return [gbm_fit(X, Y[:, j]) for j in range(Y.shape[1])]


def pred_multi(ms, X):
    return np.column_stack([gbm_pred(m, X) for m in ms])


print('=' * 92)
print('[A] h_p 量级的样本外校准：384 训练 / 128 留出，参考点 = 训练侧均值配比')
print('=' * 92)
rng = np.random.default_rng(7)
idx = rng.permutation(512)
tr, va = idx[:384], idx[384:]
Xtr, Xva = P[tr][:, KEEP], P[va][:, KEEP]
Ytr, Yva = Y[tr], Y[va]
ms = fit_multi(Xtr, Ytr)
p0 = Xtr.mean(0, keepdims=True)
Hm = pred_multi(ms, Xva) - pred_multi(ms, p0)[0]      # 模型 h_p（留出点）
Ho = Yva - Yva.mean(0, keepdims=True)                 # 实测 h_p（同一留出集均值作参考）
for tag, hm, ho in [('逐目标(13x128)',
                     Hm.reshape(-1), Ho.reshape(-1)),
                    ('聚合 h_agg(128)', Hm.mean(1), Ho.mean(1))]:
    s = np.polyfit(hm, ho, 1)
    r = np.corrcoef(hm, ho)[0, 1]
    print(f'  {tag:<18} 实测h ~ 模型h 斜率={s[0]:.4f}  截距={s[1]:+.4f}  Pearson={r:.4f}  '
          f'|模型h|均值={np.abs(hm).mean():.4f} |实测h|均值={np.abs(ho).mean():.4f}')
print('  斜率≈1 表示模型 h_p 的量级与实测一致；斜率<1 表示模型放大；>1 表示模型压缩。')

print('\n  同一检验在不同训练量下（检验"模型越大 h_p 越大"是否持续）：')
for ntr in (128, 256, 384, 448):
    tr2, va2 = idx[:ntr], idx[ntr:]
    m2 = fit_multi(P[tr2][:, KEEP], Y[tr2])
    p02 = P[tr2][:, KEEP].mean(0, keepdims=True)
    hm = (pred_multi(m2, P[va2][:, KEEP]) - pred_multi(m2, p02)[0]).mean(1)
    ho = (Y[va2] - Y[va2].mean(0, keepdims=True)).mean(1)
    s = np.polyfit(hm, ho, 1)
    print(f'    n_train={ntr:<4} 留出{len(va2):<4} 斜率={s[0]:.4f} |模型h|={np.abs(hm).mean():.4f} '
          f'|实测h|={np.abs(ho).mean():.4f}')

print('\n' + '=' * 92)
print('[B] h_p 是否只是"离均值配比的距离"（Jensen 效应）')
print('=' * 92)
ms_full = fit_multi(P[:, KEEP], Y)
p0f = P[:, KEEP].mean(0, keepdims=True)
Hf = pred_multi(ms_full, P[:, KEEP]) - pred_multi(ms_full, p0f)[0]
h_agg = Hf.mean(1)
print(f'  A4 512 条 h_agg: 负值条数 = {(h_agg < 0).sum()} / 512,  均值 = {h_agg.mean():+.5f}')
print(f'  与"离均值配比的欧氏距离"的相关: Pearson = {np.corrcoef(h_agg, np.sqrt(((P[:, KEEP] - p0f) ** 2).sum(1)))[0,1]:+.4f}')
print('  说明：以训练均值配比为参考点，若 log L 关于 p 近似凸，则 h_agg 必然系统性为正，')
print('        此时 h_p 度量的是"配方离均值的远近"，而不是"配方的好坏"。')
# 实测侧的同构证据：留出集内 log L 的均值 vs 参考点预测
print(f'\n  实测侧同构检查（1M 留出 256 条）：')
print(f'    留出集 log L 的逐目标均值 vs 留出集均值配比处模型预测（GBM）之差：')
d = Yt.mean(0) - pred_multi(ms_full, Pt[:, KEEP].mean(0, keepdims=True))[0]
print(f'      mean_v [ mean_i logL_i - logLhat(pbar) ] = {d.mean():+.5f}  '
      f'(>0 说明"平均而言，偏离均值配比会变差"这一读数来自参考点选择)')

print('\n' + '=' * 92)
print('[C] 支持门槛计数复原（论文 481/475/472）')
print('=' * 92)
Dm = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(Dm, np.inf)
nn = Dm.min(1)
for q in (0.90, 0.95):
    thr = np.quantile(nn, q)
    res = []
    for delta in (0.01, 0.03, 0.05):
        c = 0
        for i, p in enumerate(P):
            ok = True
            for j in range(17):
                rest = 1 - p[j]
                if rest <= 1e-12:
                    ok = False; break
                qq = p * (1 - delta / rest); qq[j] = p[j] + delta
                if np.sqrt(((P - qq) ** 2).sum(1)).min() > thr:
                    ok = False; break
            c += ok
        res.append(c)
    print(f'  thr = 训练留一NN的{q:.0%}分位 = {thr:.6f}: δ=0.01/0.03/0.05 -> {res[0]}/{res[1]}/{res[2]}')
print('  论文报告: δ=0.01/0.03/0.05 -> 481/475/472')
