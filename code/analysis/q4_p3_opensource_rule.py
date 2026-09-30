# -*- coding: utf-8 -*-
"""问题四 · 开源口径「另立规则 R*」冻结与分档执行

背景
----
问题四原文要求"开源口径须考虑权重、许可证"；《数据说明》补充"开源口径依据权重或许可证，
可以只选择其中一个衡量，也可以二者都要选择，或者另立规则"。本脚本执行**另立规则 R***：

    R*（三维证据分级，事前冻结）
      D1 权重可得性 : open_weights_label == "Yes"
      D2 许可明确性 : Hub License ∈ 开源/开放内容许可白名单（敏感性口径：非空即可）
      D3 版本证据   : model_sha 可核（**加分项，不作门槛**）

    分档：A = D1∧D2（双证齐） > B = 仅 D2 > C = 仅 D1 > 排除 = D1∨D2 皆无
    主口径 = A∪B∪C；严格子集 = A；敏感性 = D2 放宽为"非空"

纪律（写死在本脚本里，不允许在结果中违背）
  1. 规则阈值与白名单**事前冻结**，冻结哈希随结果一并落盘；改规则必须换哈希。
  2. D3（版本 SHA）**不是**权重证据、不是许可证据 —— 依据官方快照 JSON 自带的
     interpretation 原文："Restored model SHA is version evidence ... not proof of
     license, weight availability, or original score visibility date." 故只能作加分项。
  3. 分档 + 敏感性**并列报告**，禁止只报最有利档位。
  4. 各档冠军**不得**外推为"全行业最强"；样本是 HF Open LLM Leaderboard 平台样本，
     不是行业代表性抽样。

输出
----
  q4_p3_opensource_rule_results.json   机读结果（含冻结哈希、分档表、冠军、对照）
  q4_p3_opensource_rule_tiers.csv      逐行分档归属（可复核）
  _q4_p3_rule.log                      运行日志
"""
import hashlib
import json
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ==========================================================================================
# 0. 输入路径
# ==========================================================================================
C2_CSV = r"d:\F题\F题\real_attachments\C_efficiency_evolution\leaderboard_cleaned.csv"
ROWLINK = (r"d:\F题\第四问交接包_联合校准版_2026-09-26\第四问交接包_联合校准版_2026-09-26"
           r"\outputs\第四问继续建模_2026-09-26\q4_c2_official_sha_rowlink.json")
OUT_JSON = r"d:\F题\q4_p3_opensource_rule_results.json"
OUT_CSV = r"d:\F题\q4_p3_opensource_rule_tiers.csv"

# 团队版（联合校准版）已公布的权重轨口径与冠军，用于交叉验证
TEAM_REF = dict(source="团队版《第四问继续建模结果》§21/§23",
                W_candidate_rows=424,
                track_counts=dict(pretrained=67, chat_finetuned=231, merges=126),
                champion=dict(pretrained=38.4411, chat_finetuned=48.1065))

# ==========================================================================================
# 1. 规则规格（事前冻结）
# ==========================================================================================
RULE_SPEC = dict(
    rule_id="R*_open_source_3dim_tiering",
    version="1.0-frozen",
    dims=dict(
        D1_weight_availability=dict(
            field="open_weights_label", op="equals", value="Yes",
            note="C2 官方快照回补字段；缺失(nan)按不可得处理，不补全"),
        D2_license_explicitness=dict(
            field="Hub License", op="in_whitelist",
            whitelist=sorted(["apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-4.0",
                              "gpl-3.0", "bsd-3-clause", "wtfpl", "cc0-1.0",
                              "mpl-2.0"]),
            relaxed_op="not_null",
            note="白名单收开源软件许可（Apache-2.0/MIT/BSD-3/GPL-3.0/MPL-2.0/WTFPL，"
                 "含宽松与著佐权两类）与开放内容许可（CC0-1.0/CC-BY-4.0/CC-BY-SA-4.0）；"
                 "自定义社区许可（llama3.x/gemma 等）与非商业许可（CC-BY-NC）归入"
                 "'非宽松'，仅在敏感性口径纳入"),
        D3_version_evidence=dict(
            field="model_sha", op="len_eq_40",
            note="**加分项，不作门槛**；官方 interpretation 明示其非许可/权重可得性证据"),
    ),
    tiers=dict(
        A="D1 and D2", B="(not D1) and D2", C="D1 and (not D2)",
        excluded="(not D1) and (not D2)",
    ),
    main_scope="A | B | C",
    strict_subset="A",
    sensitivity="D2 放宽为 Hub License 非空",
    disciplines=[
        "规则阈值事前冻结，改规则必须换冻结哈希",
        "D3 版本 SHA 不得当作权重或许可证据",
        "分档与敏感性并列报告，禁止只报最有利档位",
        "各档冠军不得外推为全行业最强；样本为平台样本，非行业代表性抽样",
    ],
)

_RULE_CANON = json.dumps(RULE_SPEC, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
RULE_HASH = hashlib.sha256(_RULE_CANON).hexdigest().upper()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest().upper()


# ==========================================================================================
# 2. 读入并构造证据维度
# ==========================================================================================
print("=" * 104)
print("问题四 · 开源口径「另立规则 R*」冻结与分档")
print("=" * 104)
print(f"冻结哈希 RULE_HASH = {RULE_HASH}")
print(f"  rule_id = {RULE_SPEC['rule_id']}  version = {RULE_SPEC['version']}")
print(f"  D1 权重可得性 : {RULE_SPEC['dims']['D1_weight_availability']['field']} == 'Yes'")
print(f"  D2 许可明确性 : Hub License ∈ {len(RULE_SPEC['dims']['D2_license_explicitness']['whitelist'])} 项开源/开放内容许可白名单")
print(f"  D3 版本证据   : model_sha 长度 40（加分项，不作门槛）")
print()

c2 = pd.read_csv(C2_CSV)
c2 = c2.reset_index().rename(columns={"index": "C2_row_index"})
raw = json.load(open(ROWLINK, encoding="utf-8"))
rl = pd.DataFrame(raw["rows"])

print("--- 输入校验 ---")
print(f"  C2 行数            = {len(c2)}   （官方快照 local_rows={raw['local_rows']}）")
print(f"  rowlink 行数       = {len(rl)}   （official_rows={raw['official_rows']}）")
assert len(c2) == raw["local_rows"], "C2 行数与快照不一致"
assert len(rl) == raw["official_rows"], "rowlink 行数与快照不一致"

c2["D1"] = c2["Model"].map(
    dict(zip(rl["model"], rl["open_weights_label"].astype(str) == "Yes"))
).fillna(False).astype(bool)

WHITE = set(RULE_SPEC["dims"]["D2_license_explicitness"]["whitelist"])
c2["D2"] = c2["Hub License"].isin(WHITE)
c2["D2_relaxed"] = c2["Hub License"].notna()

sha_map = dict(zip(rl["model"], rl["model_sha"].astype(str)))
c2["D3"] = c2["Model"].map(lambda m: len(str(sha_map.get(m, ""))) == 40).astype(bool)

n = len(c2)
print(f"  D1 权重可得   = {int(c2.D1.sum()):5d}  ({c2.D1.mean()*100:5.1f}%)")
print(f"  D2 许可明确   = {int(c2.D2.sum()):5d}  ({c2.D2.mean()*100:5.1f}%)")
print(f"  D2 许可非空   = {int(c2.D2_relaxed.sum()):5d}  ({c2.D2_relaxed.mean()*100:5.1f}%)")
print(f"  D3 版本证据   = {int(c2.D3.sum()):5d}  ({c2.D3.mean()*100:5.1f}%)  ← 仅加分项")

# ==========================================================================================
# 3. 分档
# ==========================================================================================
c2["tier"] = "excluded"
c2.loc[c2.D1 & ~c2.D2, "tier"] = "C"
c2.loc[~c2.D1 & c2.D2, "tier"] = "B"
c2.loc[c2.D1 & c2.D2, "tier"] = "A"

TIER_DEF = {"A": "D1∧D2 双证齐", "B": "仅 D2 许可明确", "C": "仅 D1 权重可得",
            "excluded": "D1∨D2 皆无"}

print()
print("--- 分档表（主口径：宽松许可白名单）---")
tiers = []
for t in ["A", "B", "C", "excluded"]:
    m = c2.tier == t
    tiers.append(dict(tier=t, definition=TIER_DEF[t], n=int(m.sum()),
                      pct=float(m.mean() * 100)))
    print(f"  {t:9s} {TIER_DEF[t]:16s} N = {int(m.sum()):5d}  ({m.mean()*100:5.1f}%)")
main = int((c2.tier != "excluded").sum())
print(f"  主口径 A∪B∪C = {main}  ({main/n*100:.1f}%)")
print(f"  严格子集 A   = {int((c2.tier=='A').sum())}")

print()
print("--- 敏感性：D2 放宽为『许可非空』---")
c2["tier_rx"] = "excluded"
c2.loc[c2.D1 & ~c2.D2_relaxed, "tier_rx"] = "C"
c2.loc[~c2.D1 & c2.D2_relaxed, "tier_rx"] = "B"
c2.loc[c2.D1 & c2.D2_relaxed, "tier_rx"] = "A"
main_rx = int((c2.tier_rx != "excluded").sum())
for t in ["A", "B", "C", "excluded"]:
    m = c2.tier_rx == t
    print(f"  {t:9s} N = {int(m.sum()):5d}  ({m.mean()*100:5.1f}%)")
print(f"  主口径 A∪B∪C = {main_rx}  ({main_rx/n*100:.1f}%)")

# ==========================================================================================
# 4. 与团队版权重轨交叉验证（W = D1）
# ==========================================================================================
print()
print("--- 交叉验证：本规则 D1 轨 与 团队版 W 候选 ---")
d1_n = int(c2.D1.sum())
print(f"  本规则 D1（权重可得）        = {d1_n}")
print(f"  团队版 W_candidate_rows      = {TEAM_REF['W_candidate_rows']}")
print(f"  ⇒ 一致：{d1_n == TEAM_REF['W_candidate_rows']}"
      f"  （A∪C = {int((c2.tier=='A').sum())} + {int((c2.tier=='C').sum())} = {d1_n}）")


def track_of(t):
    t = str(t)
    if "pretrained" in t:
        return "pretrained"
    if "chat" in t or "fine-tuned" in t:
        return "chat_finetuned"
    if "merge" in t:
        return "merges"
    return "other"


c2["track"] = c2["Type"].map(track_of)
SCOL = "Average ⬆️"


def champ(df, tag):
    if not len(df):
        return dict(tag=tag, n=0, champ=None, champ_score=None)
    r = df.loc[df[SCOL].idxmax()]
    return dict(tag=tag, n=int(len(df)), champ=str(r["Model"]),
                champ_score=float(r[SCOL]))


print()
print("--- 各档冠军（按 Average，仅用于口径对照，不构成全行业结论）---")
champs = []
for t in ["A", "B", "C"]:
    champs.append(champ(c2[c2.tier == t], f"档{t} {TIER_DEF[t]}"))
    c = champs[-1]
    print(f"  {c['tag']:28s} n={c['n']:5d}  冠军 {str(c['champ']):46s} {c['champ_score']}")
champs.append(champ(c2[c2.tier != "excluded"], "主口径 A∪B∪C"))
print(f"  {champs[-1]['tag']:28s} n={champs[-1]['n']:5d}  冠军 {str(champs[-1]['champ']):46s} {champs[-1]['champ_score']}")

print()
print("--- 分类型轨冠军（团队版口径：pretrained / chat-finetuned / merges）---")
tracks = []
for tk, ref in [("pretrained", TEAM_REF["champion"]["pretrained"]),
                ("chat_finetuned", TEAM_REF["champion"]["chat_finetuned"]),
                ("merges", None)]:
    sub = c2[(c2.tier == "A") | (c2.tier == "C")]        # = D1 权重轨
    sub = sub[sub.track == tk]
    c = champ(sub, f"权重轨/{tk}")
    c["team_ref_score"] = ref
    c["delta_vs_team"] = (None if ref is None or c["champ_score"] is None
                          else round(c["champ_score"] - ref, 4))
    tracks.append(c)
    print(f"  {tk:16s} n={c['n']:5d}  冠军 {str(c['champ']):46s} {c['champ_score']}"
          f"  团队版 {ref}  Δ={c['delta_vs_team']}")

print()
print("--- 许可轨（D2 明确）冠军 vs 权重轨冠军 是否一致 ---")
lic_ch = champ(c2[c2.tier.isin(["A", "B"])], "许可轨")
w_scope = c2[c2.tier.isin(["A", "C"])]
w_ch = champ(w_scope, "权重轨")
print(f"  许可轨 n={lic_ch['n']:5d}  冠军 {str(lic_ch['champ']):46s} {lic_ch['champ_score']}")
print(f"  权重轨 n={w_ch['n']:5d}  冠军 {str(w_ch['champ']):46s} {w_ch['champ_score']}")
same = lic_ch["champ"] == w_ch["champ"]
print(f"  ⇒ 两轨冠军一致：{same}")

# ==========================================================================================
# 4.5 口径敏感性诊断：为什么两轨冠军不一致
# ==========================================================================================
print()
print("--- 诊断 1：D1 字段的缺失性质（权重轨是否被系统性收窄）---")
miss = int((~c2.D1).sum())
print(f"  open_weights_label 非 'Yes' 的行 = {miss}（其中 'No' 仅 "
      f"{int((c2.Model.map(dict(zip(rl['model'], rl['open_weights_label'].astype(str))))=='No').sum())} 条，"
      f"其余为字段缺失 nan）")
print(f"  ⇒ D1 的 424 是『明确标注可得』，**不等于**『权重确实不可得』；"
      f"字段缺失 {miss} 条（{miss/n*100:.1f}%）是主要瓶颈。")

print()
print("--- 诊断 2：许可轨冠军为何进不了权重轨 ---")
top = c2.nlargest(5, SCOL)[["Model", "Type", "Hub License", "D1", "D2", "tier", SCOL]]
for _, r in top.iterrows():
    print(f"  {r['Model'][:44]:44s} lic={str(r['Hub License'])[:14]:14s} "
          f"D1={int(r['D1'])} D2={int(r['D2'])} tier={r['tier']:8s} {r[SCOL]:.4f}")

print()
print("--- 诊断 3：B 档（仅许可明确、权重标注缺失）对前沿的影响 ---")
b = c2[c2.tier == "B"]
print(f"  B 档 n={len(b)}  最高 {b[SCOL].max():.4f}  均值 {b[SCOL].mean():.4f}")
print(f"  权重轨(D1) n={d1_n}  最高 {c2[c2.D1][SCOL].max():.4f}  均值 {c2[c2.D1][SCOL].mean():.4f}")
print(f"  许可轨(D2) n={int(c2.D2.sum())}  最高 {c2[c2.D2][SCOL].max():.4f}  均值 {c2[c2.D2][SCOL].mean():.4f}")
gap = float(c2[c2.D2][SCOL].max() - c2[c2.D1][SCOL].max())
print(f"  ⇒ 两轨最高分差 = {gap:+.4f} 分 ⇒ **开源口径的选择会改变冠军**，"
      f"故『只选权重』与『只选许可』必须并列报告，不得择一。")

print()
print("--- 诊断 4：严格白名单是否漏掉全局最强模型 ---")
gmax = c2.loc[c2[SCOL].idxmax()]
print(f"  C2 全局最高分 = {gmax[SCOL]:.4f}  {gmax['Model']}  lic={gmax['Hub License']}")
print(f"  其分档 = {gmax['tier']}（D1={int(gmax['D1'])}, D2={int(gmax['D2'])}）")
rx_scope = c2[c2.tier_rx != "excluded"]
rx_ch = champ(rx_scope, "敏感性口径 许可非空")
print(f"  敏感性口径 主口径 n={rx_ch['n']}  冠军 {rx_ch['champ']}  {rx_ch['champ_score']:.4f}")
print(f"  ⇒ 严格白名单把 lic='other' 的 {int((c2['Hub License']=='other').sum())} 条排除；"
      f"其中含全局最高分模型 ⇒ **'other' 类必须单列处置**，"
      f"既不能简单排除（漏掉真开源），也不能简单纳入（混入不明许可）。")

print()
print("--- 三口径冠军并列（口径敏感性是一阶效应，不得择一报告）---")
three = [dict(track="权重轨 D1（=团队版 W）", n=d1_n,
              champ=str(c2[c2.D1].loc[c2[c2.D1][SCOL].idxmax(), "Model"]),
              score=float(c2[c2.D1][SCOL].max())),
         dict(track="许可轨 D2 严格白名单", n=int(c2.D2.sum()),
              champ=str(c2[c2.D2].loc[c2[c2.D2][SCOL].idxmax(), "Model"]),
              score=float(c2[c2.D2][SCOL].max())),
         dict(track="许可轨 放宽（非空，含 other）", n=int(c2.D2_relaxed.sum()),
              champ=str(c2[c2.D2_relaxed].loc[c2[c2.D2_relaxed][SCOL].idxmax(), "Model"]),
              score=float(c2[c2.D2_relaxed][SCOL].max()))]
for t in three:
    print(f"  {t['track']:28s} n={t['n']:5d}  冠军 {t['champ'][:44]:44s} {t['score']:.4f}")
print(f"  三口径最高分极差 = {three[2]['score'] - three[0]['score']:+.4f} 分")


diag = dict(
    d1_field_missing=dict(not_yes_rows=miss, explicit_no_rows=int(
        (c2.Model.map(dict(zip(rl["model"], rl["open_weights_label"].astype(str)))) == "No").sum()),
        note="D1=424 是『明确标注可得』，不等于『权重不可得』"),
    top5_by_average=[dict(model=str(r["Model"]), license=str(r["Hub License"]),
                          D1=bool(r["D1"]), D2=bool(r["D2"]), tier=str(r["tier"]),
                          average=float(r[SCOL])) for _, r in top.iterrows()],
    tierB_impact=dict(n=int(len(b)), max=float(b[SCOL].max()), mean=float(b[SCOL].mean())),
    weight_track=dict(n=d1_n, max=float(c2[c2.D1][SCOL].max()),
                      mean=float(c2[c2.D1][SCOL].mean())),
    license_track=dict(n=int(c2.D2.sum()), max=float(c2[c2.D2][SCOL].max()),
                       mean=float(c2[c2.D2][SCOL].mean())),
    max_score_gap=gap,
    note="两轨冠军与最高分不同 ⇒ 开源口径选择本身是结论敏感项，必须并列报告",
    global_max=dict(model=str(gmax["Model"]), license=str(gmax["Hub License"]),
                    average=float(gmax[SCOL]), tier=str(gmax["tier"]),
                    D1=bool(gmax["D1"]), D2=bool(gmax["D2"]),
                    note="C2 全局最高分模型许可标为 'other'，被严格白名单排除；"
                         "'other' 类须单列处置，不得简单排除或纳入"),
    sensitivity_main_champion=rx_ch,
    three_track_champions=three,
    three_track_score_range=float(three[2]["score"] - three[0]["score"]),
)


# ==========================================================================================
# 5. 落盘
# ==========================================================================================
RES = dict(
    rule=RULE_SPEC,
    rule_hash=RULE_HASH,
    inputs=dict(
        c2_csv=dict(path=C2_CSV, rows=int(len(c2)), sha256=sha256_file(C2_CSV)),
        rowlink=dict(path=ROWLINK, rows=int(len(rl)), sha256=sha256_file(ROWLINK)),
    ),
    dims=dict(
        D1_weight_available=int(c2.D1.sum()),
        D2_license_explicit=int(c2.D2.sum()),
        D2_license_notnull=int(c2.D2_relaxed.sum()),
        D3_version_evidence=int(c2.D3.sum()),
        D3_note="加分项，非门槛；不得作为权重/许可证据",
    ),
    tiers_main=tiers,
    main_scope_n=main,
    strict_subset_n=int((c2.tier == "A").sum()),
    sensitivity_license_notnull=dict(
        main_scope_n=main_rx,
        tiers=[dict(tier=t, n=int((c2.tier_rx == t).sum()),
                    pct=float((c2.tier_rx == t).mean() * 100))
               for t in ["A", "B", "C", "excluded"]],
    ),
    crosscheck_team=dict(
        our_D1=d1_n,
        team_W_candidate_rows=TEAM_REF["W_candidate_rows"],
        consistent=bool(d1_n == TEAM_REF["W_candidate_rows"]),
        team_source=TEAM_REF["source"],
    ),
    champions=champs,
    track_champions=tracks,
    license_vs_weight=dict(license_track=lic_ch, weight_track=w_ch,
                           champion_identical=bool(same)),
    diagnostics=diag,
    verdict=("严格 R/R+ 仍为 0（要求权重内容哈希+许可正文+harness 三者齐备，附件不支持）；"
             f"但按 R* 可给出非 0 的分档合格样本：严格子集 A={int((c2.tier=='A').sum())}、"
             f"主口径 A∪B∪C={main}、许可放宽口径={main_rx}。"
             "分档与敏感性必须并列报告，禁止只报最有利档位；"
             "各档冠军不得外推为全行业最强。"),
    disciplines=RULE_SPEC["disciplines"],
)
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
c2[["C2_row_index", "Model", "Type", "track", "Hub License", "D1", "D2", "D2_relaxed",
    "D3", "tier", "tier_rx", SCOL]].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print()
print("=" * 104)
print(f"结果已写入 {OUT_JSON}")
print(f"逐行分档已写入 {OUT_CSV}")
print(f"冻结哈希 RULE_HASH = {RULE_HASH}")
print("=" * 104)
