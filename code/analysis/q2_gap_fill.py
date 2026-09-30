# -*- coding: utf-8 -*-
"""补齐问题二的两处论证缺口：

(A) §2.7 领域间替代/互补：由问题一冻结的 M2 平均模型（仅对角二次项）计算
    "份额由 k 转移到 d" 的对数损失边际 M_{d<-k}，并给出替代/互补结构。
(B) §2.4(c) 质量尺度映射情景：用问题一域级 Q_A 给出 φ_lin / φ_pow 的显式映射，
    并量化 Q_A 单位变化在 Q_B 尺度上的放大倍数。
"""
import os
import numpy as np
import pandas as pd

A = r'F题\real_attachments\A_data_value\regmix_tables'
m = pd.read_csv(os.path.join(A, 'train_mixture_1m.csv'))
X = m.iloc[:, 1:]
dom = list(X.columns)
p0 = X.values.astype(float).mean(axis=0)
print('A4 训练配方：', m.shape, '域数', len(dom))
print('p0 行和 =', round(float(p0.sum()), 7))

# ---- 问题一冻结 M2 平均模型 ----
mm = pd.read_csv(r'd:\F题\q1_quality_results\mix_avg_model_quad.csv', index_col=0).iloc[:, 0]
inter = float(mm['intercept'])
beta = {k: float(mm[k]) for k in mm.index if not k.endswith('^2') and k != 'intercept'}
gam = {k[:-2]: float(mm[k]) for k in mm.index if k.endswith('^2')}
print('\nM2 平均模型: intercept=%.4f' % inter)
print('  β 覆盖组数 =', len(beta), ' γ 覆盖域数 =', len(gam), '（无交叉项 Γ_dk, d≠k）')
print('  γ:', {k: round(v, 3) for k, v in gam.items()})

# p0 对齐：A4 列名为 train_the_pile_<域>；4 个无 Loss 域聚合为 other；uspto_backgrounds 为参考域
PRE = 'train_the_pile_'
strip = [c[len(PRE):] if c.startswith(PRE) else c for c in dom]
p0raw = {s: float(v) for s, v in zip(strip, p0)}
NO_LOSS = ['nih_exporter', 'philpapers', 'enron_emails', 'europarl']
p0d = {k: 0.0 for k in beta}
for k in beta:
    if k == 'other':
        p0d[k] = sum(p0raw.get(d, 0.0) for d in NO_LOSS)
    else:
        p0d[k] = p0raw.get(k, 0.0)
print('\nA4 域(去前缀):', strip)
print('模型组:', list(beta.keys()))
print('other = 4 个无 Loss 域之和 =', round(p0d['other'], 5),
      f"（明细 { {d: round(p0raw.get(d,0),4) for d in NO_LOSS} }）")
print('参考域 uspto_backgrounds 的 p0 =', round(p0raw.get('uspto_backgrounds', 0.0), 5))
print('p0 映射后行和（13 组）=', round(sum(p0d.values()), 7),
      '；加参考域后 =', round(sum(p0d.values()) + p0raw.get('uspto_backgrounds', 0.0), 7))


def dlogL_dp(k):
    return beta[k] + 2.0 * gam.get(k, 0.0) * p0d[k]


# ---- (A) 份额转移的边际效应 M_{d<-k} = ∂logL/∂p_d - ∂logL/∂p_k ----
keys = list(beta.keys())
D = np.array([dlogL_dp(k) for k in keys])
M = D[None, :] - D[:, None]          # M[i,j] = M_{keys[i] <- keys[j]}
Md = pd.DataFrame(M, index=keys, columns=keys)
print('\n' + '=' * 96)
print('(A) 份额由 k 转移到 d 的对数损失边际 M_{d<-k}（负值 = 该转移降低损失，即 d 相对 k 有利）')
print('=' * 96)
print('  各域在 p0 处的边际 ∂logL/∂p_k：')
for k in sorted(keys, key=lambda z: D[keys.index(z)]):
    print(f'    {k:<20}{D[keys.index(k)]:+.4f}')
print('\n  最有利的 8 组转移（M 最小）：')
flat = [(Md.index[i], Md.columns[j], Md.values[i, j])
        for i in range(len(keys)) for j in range(len(keys)) if i != j]
flat.sort(key=lambda t: t[2])
for d, k, v in flat[:8]:
    print(f'    由 {k:<20}→ {d:<20} M={v:+.4f}')
print('\n  最不利的 4 组转移（M 最大）：')
for d, k, v in flat[-4:]:
    print(f'    由 {k:<20}→ {d:<20} M={v:+.4f}')
off = M[~np.eye(len(keys), dtype=bool)]
print(f'\n  非对角元素：均值={off.mean():+.4f} 标准差={off.std():.4f} '
      f'正/负={int((off>0).sum())}/{int((off<0).sum())}')
print('  注：M 完全由 β 与对角 γ 决定；M2 模型无交叉项 ⇒ 不存在"配对互补/替代"的独立项，')
print('      替代关系只来自 β 的相对大小与各域自身的饱和项 γ_k p_k。')

# ---- (B) 质量尺度映射情景 ----
print('\n' + '=' * 96)
print('(B) 质量尺度映射情景 φ：Q_A（问题一域级质量分）→ Q_B（附件 B 的 Q_score 尺度）')
print('=' * 96)
q = pd.read_csv(r'd:\F题\q1_quality_results\domain_Q.csv')
qa = q[q.source == 'A1抽样'][['domain', 'Q_group_mean']].set_index('domain')['Q_group_mean']
lo, hi = float(qa.min()), float(qa.max())
print(f'  7 个质量域的 Q_A：{ {k: round(v,3) for k,v in qa.items()} }')
print(f'  观测跨度 = [{lo:.3f}, {hi:.3f}]，宽度 = {hi-lo:.4f}（远窄于 Q_B 的 [0.1,1.0]）')
print('\n  情景一 φ_lin：u=(Q_A-{:.3f})/{:.4f}，Q_B=u'.format(lo, hi - lo))
print('  情景二 φ_pow：Q_B=u^γ（γ<1 抬升中低质量端，γ>1 压低下端）')
print(f"\n  {'域':<18}{'Q_A':>8}{'u':>8}{'Q_B(γ=1)':>11}{'Q_B(γ=0.5)':>13}{'Q_B(γ=2)':>11}")
for k, v in qa.items():
    u = (v - lo) / (hi - lo)
    print(f'  {k:<18}{v:>8.3f}{u:>8.3f}{u:>11.3f}{u**0.5:>13.3f}{u**2:>11.3f}')
print('\n  放大倍数（φ_lin）：ΔQ_B/ΔQ_A = 1/{:.4f} = {:.2f}'.format(hi - lo, 1 / (hi - lo)))
print('  ⇒ 问题一尺度上 0.1 的质量改善，在附件 B 尺度上相当于 {:.3f}'.format(0.1 / (hi - lo)))
print('  ⇒ 若把 Q_A 直接当 Q_B 用，会低估质量效应约 {:.1f} 倍'.format(1 / (hi - lo)))
print('\n  该映射不可由数据识别：附件 B（B6–B8）无域标签，不存在 (Q_A, L) 配对，')
print('  故 φ 只能作为情景声明，不能拟合其参数。')
