# -*- coding: utf-8 -*-
"""
P2-B  R / R+ 首批资格审计：424 → 56 → 合格数

背景（队友交接包 + 我方融合方案 §2.3）
--------------------------------------
三轨资格定义：
  W  ：C2 `Epoch_AI_Open_Weights == Yes`                      → 424 条 / 420 个模型名
  R  ：W ∧ 分数可对应固定评测运行与权重 revision ∧ 权重可取得且哈希完整
        ∧ 固定版本许可允许下载/运行/评测/研究复现 ∧ tokenizer/模板/任务/评分器/依赖可追溯
  R+ ：R ∧ 固定版本许可还允许商业使用

关键纪律：**56 条是"六项分数全匹配"的优先审计名单，不是 R/R+ 合格数。**

本脚本做三件事：
  1. 复现漏斗 424 → 207 → 198 → 194 → 56，并做**损失瀑布分解**（每一档丢掉多少、为什么）；
  2. 对 424 与 56 做**许可证分级筛查**，给出 R / R+ 的**许可证上界**（非合格数）；
  3. 逐闸门判定 R / R+，输出**已确认合格数**（当前应为 0），并明确指出卡在哪一闸门。

输入：_team4_handoff/.../q4_w_version_audit_queue.csv、q4_score_normalization_recheck.json、
      q4_exact_version_score_link_audit.csv、C2 leaderboard_cleaned.csv
输出：q4_p2b_eligibility_results.json / q4_p2b_funnel.csv / q4_p2b_gates.csv
"""
import os, json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
BASE = r"d:\F题\F题\real_attachments\C_efficiency_evolution"
HAND = r"d:\F题\_team4_handoff\outputs\第四问交接包\结果表"
OUT = r"d:\F题\q4_p2b_eligibility_results.json"
OUT_F = r"d:\F题\q4_p2b_funnel.csv"
OUT_G = r"d:\F题\q4_p2b_gates.csv"

RESEARCH_OK = {'apache-2.0', 'mit', 'bsd-3-clause-clear', 'cc-by-4.0', 'cc-by-sa-4.0', 'wtfpl',
               'gpl-3.0', 'afl-3.0', 'osl-3.0', 'gemma', 'llama2', 'llama3', 'llama3.1',
               'llama3.2', 'llama3.3', 'openrail', 'creativeml-openrail-m', 'bigcode-openrail-m',
               'bigscience-bloom-rail-1.0', 'apple-ascl'}
COMMERCIAL_OK = {'apache-2.0', 'mit', 'bsd-3-clause-clear', 'cc-by-4.0', 'cc-by-sa-4.0',
                 'wtfpl', 'gpl-3.0', 'afl-3.0', 'osl-3.0'}
COMMERCIAL_COND = {'gemma', 'llama2', 'llama3', 'llama3.1', 'llama3.2', 'llama3.3', 'openrail',
                   'creativeml-openrail-m', 'bigcode-openrail-m', 'bigscience-bloom-rail-1.0',
                   'bigscience-openrail-m', 'apple-ascl'}
NONCOMMERCIAL = {'cc-by-nc-4.0', 'cc-by-nc-sa-4.0', 'cc-by-nc-nd-4.0'}


def sec(t):
    print("\n" + "=" * 104)
    print("### " + t)


def tier(l):
    l = str(l).strip().lower()
    if l in COMMERCIAL_OK:
        return 'commercial_ok'
    if l in COMMERCIAL_COND:
        return 'commercial_conditional'
    if l in NONCOMMERCIAL:
        return 'noncommercial'
    if l in RESEARCH_OK:
        return 'research_ok'
    return 'unknown'


# =====================================================================================
sec("0  输入装载")
q = pd.read_csv(os.path.join(HAND, "q4_w_version_audit_queue.csv"), low_memory=False)
rc = json.load(open(os.path.join(HAND, "q4_score_normalization_recheck.json"), encoding='utf-8'))
ex = pd.read_csv(os.path.join(HAND, "q4_exact_version_score_link_audit.csv"), low_memory=False)
lb = pd.read_csv(os.path.join(BASE, "leaderboard_cleaned.csv"), low_memory=False)

print(f"W 候选队列 = {len(q)} 条   （Epoch_AI_Open_Weights == Yes）")
print(f"  候选轨类型：{q.model_type.value_counts().to_dict()}")
print(f"  队列内 license_label_unverified 缺失 = {q.license_label_unverified.isna().sum()}")
recs = pd.DataFrame(rc['records'])
print(f"评分归一化复核记录 = {len(recs)} 条（有同名修订号线索）")
print(f"  summary：{rc['summary']}")

# =====================================================================================
sec("1  漏斗复现与损失瀑布：424 → 56")
M = recs.copy()
mk = M['matches_1e_minus_6'].apply(pd.Series)
M = pd.concat([M.drop(columns=['matches_1e_minus_6']), mk], axis=1)
TASKS6 = ['IFEval', 'BBH', 'GPQA', 'MUSR', 'MMLU-PRO', 'MATH Lvl 5']
NONMATH = ['IFEval', 'BBH', 'GPQA', 'MUSR', 'MMLU-PRO']
M['six'] = M[TASKS6].all(axis=1)
M['five'] = M[NONMATH].all(axis=1)
n_w, n_rev = len(q), len(M)
n_bbh = int(M['BBH'].sum())
n_five = int(M['five'].sum())
n_six = int(M['six'].sum())
print(f"  W 候选                     {n_w:4d}")
print(f"  └ 有同名修订号线索          {n_rev:4d}   （−{n_w-n_rev}，无详细 JSON 或同名多份）")
print(f"    └ BBH 归一化后匹配        {n_bbh:4d}   （−{n_rev-n_bbh}，跨代际协议差异）")
print(f"      └ 五项非 MATH 全匹配     {n_five:4d}   （−{n_bbh-n_five}，其他任务仍有差）")
print(f"        └ 六项全匹配（优先名单）{n_six:4d}   （−{n_five-n_six}，仅 MATH 不匹配）")
print(f"  交叉核对：MATH 仅一项不匹配 {int(rc['math_only_unmatched_records'])}"
      f"  其他任务不匹配 {int(rc['other_unmatched_records'])}"
      f"  五项匹配 {int(rc['five_non_math_all_match_records'])}  六项匹配 {int(rc['all_six_match_records'])}")
print(f"  一致性检查：194−56={n_five-n_six} vs math_only={int(rc['math_only_unmatched_records'])}  "
      f"{'✓' if n_five-n_six == int(rc['math_only_unmatched_records']) else '✗'}")
print(f"  逐任务匹配数：{ {k: v['matched'] for k, v in rc['summary'].items()} }")

funnel = [dict(stage='W 候选（Epoch 开放权重=Yes）', n=n_w, lost=0, reason='起点'),
          dict(stage='有同名修订号线索', n=n_rev, lost=n_w - n_rev, reason='无详细评测 JSON / 同名多份'),
          dict(stage='BBH 归一化后匹配', n=n_bbh, lost=n_rev - n_bbh, reason='跨代际评测协议差异'),
          dict(stage='五项非 MATH 全匹配', n=n_five, lost=n_bbh - n_five, reason='其他任务仍不匹配'),
          dict(stage='六项全匹配（优先名单）', n=n_six, lost=n_five - n_six, reason='仅 MATH 不匹配（评分器/快照）')]
pd.DataFrame(funnel).to_csv(OUT_F, index=False, encoding='utf-8-sig')

# =====================================================================================
sec("2  许可证分级筛查：R / R+ 的许可证上界")
six_models = set(M.loc[M['six'], 'model'])
q['six_match'] = q.model_id.isin(six_models)
q['tier'] = q.license_label_unverified.map(tier)
q_u = q.drop_duplicates('model_id').copy()
print(f"  候选队列 {len(q)} 条 → 去重模型名 {len(q_u)} 个"
      f"（同名多次提交 {len(q)-len(q_u)} 条，按模型名去重后做许可判定）")
print(f"  全 424 条候选许可证分级（按记录）：{q.tier.value_counts().to_dict()}")
print(f"  全 {len(q_u)} 个候选许可证分级（按模型）：{q_u.tier.value_counts().to_dict()}")
print(f"  优先名单 {n_six} 条许可证分级：{q_u.loc[q_u.six_match,'tier'].value_counts().to_dict()}")
print()
for tag, s in [('全 424 条 W（按记录）', q), (f'优先名单 {n_six} 条（按模型）', q_u[q_u.six_match])]:
    n_res = int(s.tier.isin(['commercial_ok', 'commercial_conditional', 'research_ok']).sum())
    n_com = int((s.tier == 'commercial_ok').sum())
    n_con = int((s.tier == 'commercial_conditional').sum())
    n_nc = int((s.tier == 'noncommercial').sum())
    n_uk = int((s.tier == 'unknown').sum())
    print(f"  [{tag}] 允许研究/复现（R 许可上界）={n_res}/{len(s)} = {n_res/len(s):.1%}")
    print(f"           明确允许商用（R+ 严格许可上界）={n_com}   条件商用={n_con}   "
          f"非商用={n_nc}   未知/其他={n_uk}")
print("  → 这些只是**许可证维度的上界**，未含权重哈希与 harness 可追溯性，不能当合格数")

# =====================================================================================
sec("3  逐闸门判定：已确认的 R / R+")
print("  R 需四闸门全通过：g1 分数联结  g2 权重哈希  g3 固定版本许可文本  g4 harness 可追溯")
print(f"  深度取证记录（q4_exact_version_score_link_audit.csv）n={len(ex)}：")
for _, r in ex.iterrows():
    print(f"    {r.model_id}")
    print(f"      C2 来源代际 = {r.C2_source_generation}")
    print(f"      原始评测代际 = {r.raw_eval_source_generation}")
    print(f"      BBH 差 = {r.BBH_gap_raw_minus_C2:+.2f}   MATH 差 = {r.MATH_gap_raw_minus_C2:+.2f}")
    print(f"      分数→revision 联结 = {r.C2_score_to_revision_link}")
    print(f"      严格 R 资格 = {r.strict_R_C2_eligibility}")
n_link_ok = int((ex.C2_score_to_revision_link == 'established').sum()) if len(ex) else 0
n_r_conf = int((ex.strict_R_C2_eligibility == 'established').sum()) if len(ex) else 0
print(f"\n  深度取证中分数联结成立 = {n_link_ok}/{len(ex)}   严格 R 成立 = {n_r_conf}/{len(ex)}")
print(f"  其余 {n_six - len(ex)} 条优先名单记录：g2 权重哈希与 g3 版本许可文本**均未取证**")

gates = pd.DataFrame(dict(
    model_id=sorted(six_models),
    g1_score_link=True,
    g2_weight_hash=False,
    g3_version_license=False,
    g4_harness_trace=False))
gates['R'] = gates[['g1_score_link', 'g2_weight_hash', 'g3_version_license', 'g4_harness_trace']].all(axis=1)
tier_map = q.drop_duplicates('model_id').set_index('model_id')['tier']
gates['license_tier'] = gates.model_id.map(tier_map)
gates['R_plus'] = gates.R & gates.license_tier.isin(['commercial_ok'])
gates.to_csv(OUT_G, index=False, encoding='utf-8-sig')
print(f"  → 确认 R = {int(gates.R.sum())}   确认 R+ = {int(gates.R_plus.sum())}（卡在 g2/g3/g4 未取证）")
print(f"  优先名单的许可维度上界：R ≤ {int(gates.license_tier.isin(['commercial_ok','commercial_conditional','research_ok']).sum())}，"
      f"R+ ≤ {int((gates.license_tier == 'commercial_ok').sum())}")

# =====================================================================================
sec("结论")
res = dict(
    funnel=funnel, n_six=n_six,
    license_tier_all424={k: int(v) for k, v in q.tier.value_counts().items()},
    license_tier_priority={k: int(v) for k, v in q.loc[q.six_match, 'tier'].value_counts().items()},
    license_upper_bound=dict(
        all424_research=int(q.tier.isin(['commercial_ok', 'commercial_conditional', 'research_ok']).sum()),
        all424_commercial_strict=int((q.tier == 'commercial_ok').sum()),
        priority_research=int(q.loc[q.six_match, 'tier'].isin(['commercial_ok', 'commercial_conditional', 'research_ok']).sum()),
        priority_commercial_strict=int((q.loc[q.six_match, 'tier'] == 'commercial_ok').sum())),
    confirmed_R=int(gates.R.sum()), confirmed_R_plus=int(gates.R_plus.sum()),
    deep_audit=dict(n=len(ex), link_ok=n_link_ok, R_ok=n_r_conf,
                    blocked_gate='C2 与原始评测分属不同 leaderboard 代际 → 分数无法归属固定 revision'),
    binding_constraint='g2 权重哈希 + g3 固定版本许可文本 + g4 harness 可追溯性，均未取证')
print(json.dumps(res, ensure_ascii=False, indent=1))
print(f"""
  ① 漏斗：{n_w} → {n_rev} → {n_bbh} → {n_five} → {n_six}。最大损失在两处：
     无修订号线索丢 {n_w-n_rev} 条；仅 MATH 不匹配丢 {n_five-n_six} 条（评分器/快照口径）。
  ② 许可维度上界（优先名单 {n_six} 条）：R ≤ {res['license_upper_bound']['priority_research']}，
     R+ ≤ {res['license_upper_bound']['priority_commercial_strict']}（严格商用）或
     +{int((q.loc[q.six_match,'tier']=='commercial_conditional').sum())} 条条件商用。
  ③ 已确认合格数：**R = 0，R+ = 0**。卡点是 g2/g3/g4 三项取证，而非许可证。
  ④ 已深度取证的两款 Qwen 均因 C2 与原始评测**分属不同 leaderboard 代际**而无法把分数
     归属到固定 revision，strict_R_C2_eligibility = not_established。
  ⇒ 论文口径：只能写"W 候选 424、优先审计名单 56、许可上界 R≤X / R+≤Y、**已确认 R/R+ 合格数为 0**"；
     不得把 56 当作 R/R+ 合格数，不得据此宣称"严格 R/R+ 前沿"。这与队友交接包结论一致。
""")
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"已写出：{OUT}\n         {OUT_F}\n         {OUT_G}")
