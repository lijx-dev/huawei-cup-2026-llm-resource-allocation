"""根据真实计算输出 P1 实验说明。"""


def write_report(path,metadata,domains,comparisons,robustness,config,domain_exclusions):
    counts=metadata["scored_counts"]
    lines=["# 第一问 P1 综合质量评价", "",
           "本报告由 `PYTHONPATH=src python -m q1.quality.run` 生成。质量分是本方案的统计代理，不是原始附件直接给出的真实质量标签，也不是训练收益的因果估计。", "",
           "## 输入与样本", "",
           f"- A1 拟合：{metadata['fit_count_A1']} 条；A2/A3 使用相同归一化参数和权重。",
           f"- 成功评分：{counts}；P1 额外排除：{metadata['p1_exclusions']} 条。",
           f"- P0 状态：{metadata['p0_status_counts']}；原始附件 SHA-256 见 `quality_model_metadata.json`。",
           "- 扩展集独立统计含 A1 的精确重叠；联合视图只统计 P0 的 `valid`，不把 `duplicate_exact` 当独立样本。", "",
           "### A1 各域排除情况", "",
           "|领域|原始行数|P0 合格|损坏|可疑|P1 评分|总排除|排除率|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in domain_exclusions:
        lines.append(f"|{r['domain']}|{r['raw_n']}|{r['p0_valid_n']}|{r['rejected_corrupt_n']}|{r['suspected_unreliable_n']}|{r['p1_scored_n']}|{r['excluded_n']}|{r['excluded_rate']:.4%}|")
    lines.extend(["", "解析失败且无法取得领域的记录归于 `unknown`，不按领域猜测分配。", "",
           "## 处理口径", "",
           "- 22 项具体字段、方向和变换见 `indicator_metadata.csv` 与 `configs/q1/indicator_schema.json`。方向中涉及自然语言偏好的条目是预先声明的建模假设，尤其可能影响代码和数学文本。",
           "- `fineweb_edu` 是单值列表，直接提取；`qurater` 是四个无序评价维度，主方案采用教育价值第 4 项，四维均值作敏感性对照；两者不能用有序等级公式。",
           "- 四个 ModernBERT 0–5 logits 使用稳定 Softmax 的期望等级。数据卡示例使用 argmax；本实验遵循用户文档的平滑期望方法，属于明确的模型选择。两个二分类 logits 转为正类概率。",
           "- 单词数与平均词长使用 Gopher 参考区间内得分 1、区间外连续衰减的预设函数。该区间最初用于网页文本，跨域泛化须谨慎。",
           f"- A1 拟合固定 Min–Max；A2/A3 分布外值预设策略 `{config['out_of_reference']}`，截断数量逐指标记录在模型元数据中，不改动原始值。",
           "- 熵权、Spearman 冗余修正和 Q=100·Σwz 使用同一批 A1 完整案例。权重表示统计差异与冗余，不是训练效果的因果重要性。", "",
           "## 领域统计", "",
           "|视图|领域|n|平均 Q|中位数|标准差|95% 均值区间|参考范围内 n|", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in domains:
        lines.append(f"|{r['view']}|{r['domain']}|{r['n']}|{r['mean_Q']:.3f}|{r['median_Q']:.3f}|{r['std_Q']:.3f}|[{r['ci95_low_Q']:.3f}, {r['ci95_high_Q']:.3f}]|{r['in_reference_n']}|")
    lines.extend(["", "95% 区间采用固定的 1.96×样本标准误正态近似，只描述当前样本在独立同分布假设下的均值不确定性，不证明 A1 抽样代表全部来源。", "",
                  "## 扩展集对照", "", "|领域|A1 n|扩展 n|重叠 n|重叠比例|A1 平均 Q|扩展平均 Q|排除重叠后 n / 均值 Q|均值差|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in comparisons:
        lines.append(f"|{r['domain']}|{r['a1_n']}|{r['extension_n']}|{r['overlap_n']}|{r['overlap_fraction']:.4f}|{r['a1_mean_Q']:.3f}|{r['extension_mean_Q']:.3f}|{r['extension_nonoverlap_n']} / {r['extension_nonoverlap_mean_Q']:.3f}|{r['mean_difference_Q']:.3f}|")
    lines.extend(["", "指标均值与标准差的分布对照见 `indicator_domain_comparison.csv`。分布外截断的逐指标数量和比例见 `boundary_clipping_by_indicator.csv`；截断样本数及剔除后的均值见 `boundary_sensitivity.csv`。后者仅为边界敏感性对照，不改变主结果。", "",
                  "## 稳健性", "", "|方案|样本数|Pearson|Spearman|主方案均值减对照均值|", "|---|---:|---:|---:|---:|"])
    for r in robustness:
        lines.append(f"|{r['comparison']}|{r['n']}|{r['pearson']:.4f}|{r['spearman']:.4f}|{r['mean_difference']:.4f}|")
    lines.extend(["", "TOPSIS 贴近度按固定权重及 [0,1] 正负理想解保存到样本输出，仅供对照，不替代主分。", "",
                  "## 限制和复现", "", "- `question1-1(1).md` 只给出整体公式，没有逐指标方向。本实现用原始发布方说明、QuRating 论文、公开 Gopher 规则及显式模型假设补齐；详情见配置。",
                  "- `rps_*` 的比例字段在本地附件中呈百分数尺度，且少数高于 100；P0 已将可疑记录隔离。没有凭字段名强制改写原值。",
                  "- `qurater` 采用教育价值维度会舍弃同一字段的其他三维信息；四维均值对照仅评估这种取舍，不能证明哪一种是普适真值。",
                  "- 正态近似区间、领域间均值差和统计权重均不支持来源真实性或因果解释。",
                  "- 所有逐样本结果保存原始文件、行号、ID、记录哈希及 P0 状态；没有补全、生成或插值原始记录。", ""])
    path.write_text("\n".join(lines),encoding="utf-8")
