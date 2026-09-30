# 问题二 q2-fusion-rerun-v1 详细实验报告

**实验版本：**`q2-fusion-rerun-v1`　**随机种子：**7　**状态：**`COMPLETE_WITH_WARNINGS`。本报告由 P0–P7 已保存的结果表生成；生成报告时未重新拟合模型。N、D 分别以十亿参数和十亿 tokens 为单位，除非表中另有说明。

## 摘要

B1 的经典标度律在 8 个规模留一验证中取得很低的误差；B6 半合成数据内加入 Q_B 明显改善组级预测，B7-new 的新 Q 水平上也改善误差。B8 的组内数值方向却与 B6 相反，说明该质量效应不能直接跨附件合并。第一问配比模型只以冻结接口导入；附件 B 没有逐实验配比，因此广义模型中的 λ_p 是情景量。固定算力最优仅针对 C≈6ND 的训练计算近似。

|主要结果|数值|证据等级|
|---|---|---|
|B1 规模留一宏 RMSE|0.000146888|B1 组留出|
|B6 M0/MQ 组 CV 宏 RMSE|0.128762 / 0.0760865|半合成校准|
|B7-new M0/MQ RMSE|0.0589578 / 0.0490942|半合成留出|
|B8 正向 Q–Loss 组数|150/150|半合成压力测试|
|中档算力、Q_B=0.5 解析最优 N/D|2.2612 / 54.397|条件优化情景|

## 46.1 数据合同、审计与证据边界

附件 B1–B12 共对应 19 个实际文件，其中 B3 占 8 个插值轨迹文件。以下是原始行数，不把同源文件或重叠行视为独立实验数。`valid` 仅表示通过当前结构与数值规则，不能证明来源真实性。附件编号、字段和 SHA-256 见[文件清单](../audit/b_attachment_inventory.csv)、[字段合同](../audit/field_contract.yaml)和[数据说明](../../../docs/数据说明.pdf)。

|附件|原始行|来源性质|预定角色|
|---|---|---|---|
|B1|1176|reported_observation|classic_scaling_fit|
|B2|1029|semi_synthetic|cross_source_validation|
|B3|4000|interpolated|interpolated_trajectory_validation|
|B4|57|reported_observation|cross_family_validation|
|B5|44|mixed_reported_sources|literature_scale_validation|
|B6|360|semi_synthetic|quality_scaling_fit|
|B7|450|semi_synthetic|quality_holdout_extension|
|B8|1704|semi_synthetic_and_extrapolated|quality_direction_stress_test|
|B9|132|metadata_only|large_model_metadata|
|B10|128|estimated_reference|estimated_large_scale_reference|
|B11|18|metadata_only|model_family_metadata|
|B12|1386|metadata_only|checkpoint_metadata|

审计识别 4 条非法数值记录，均来自 B9 的 D≤0；B9 可用 N,D 元数据为 128 行。单文件完全重复条目 0；B6/B7 跨附件同 ID 且 N、D、Q、Loss 全一致的重叠 360 行，B7 新增 90 行，冲突 0 行。排除指针见[非法行表](../audit/invalid_rows.csv)，重叠关系见[B6/B7 对照](../audit/b6_b7_overlap.csv)。所有原始文件的运行前后 SHA-256 一致。

B1 的 `N_params_B`、`D_tokens_B` 已分别是十亿参数、十亿 tokens，不重复除以 10⁹。B1 有 8 条规模轨迹，每条 147 个 checkpoint；B12 没有精确 N 和 token，无法可靠建立模型级对齐。B2、B6–B8 为半合成，B3 为插值，B10 为已拟合标度律估算。B1/B4/B5 的 Loss 虽标记为公开或观测数据，跨来源评测集、tokenizer、vocab 和协议未充分给出，因此跨来源绝对误差只作描述。详见[Loss 语义表](../audit/loss_semantics.csv)。

文档差异：任务指定的 `问题二_完整建模思路_扩展建模(1).md` 不在仓库，使用现有 `问题二_完整建模思路_扩展建模.md`。赛题正文讨论质量处理和注意力成本；本次 P6 按总控任务的 C≈6ND 近似实施，结论限定在固定 Q_B 与配比条件下的训练计算分配。

## 46.2 B1 经典 Scaling Law：拟合、验证与不确定性

模型为 $L=E+AN^{-\alpha}+BD^{-\beta}$，约束 $E\ge0$、$A,B,\alpha,\beta>0$。对正参数取 log 参数化，采用四组初值最小二乘拟合；按完整模型规模留一验证，最后用全部 B1 重拟合。每次起点、收敛状态、目标函数和迭代次数见[多起点记录](../b1_baseline/multistart_runs.csv)。

四个起点收敛 4/4；最终目标函数范围 [2.52383e-05, 2.52383e-05]。B1 有 1176 条有效建模行。参数区间由 60 次按规模重抽样得到，并非逐 checkpoint 独立重抽样。

|参数|全量估计|组 bootstrap 2.5%|组 bootstrap 97.5%|
|---|---|---|---|
|E|1.6898|1.68969|1.68992|
|A|0.35398|0.353863|0.35411|
|B|1.24031|1.2402|1.24041|
|alpha|0.339977|0.339804|0.340092|
|beta|0.279878|0.279827|0.279922|

|留出规模 N(B)|行数|RMSE|MAE|R²|Spearman|
|---|---|---|---|---|---|
|0.070542|147|0.00022023|0.000168755|0.999999|1|
|0.162405|147|0.000181818|0.000137142|0.999999|1|
|0.409009|147|0.00013511|0.000107562|1|1|
|1.04087|147|0.000154598|0.000108657|1|1|
|1.41618|147|0.000131917|9.99692e-05|1|1|
|11.9658|147|0.000101975|7.58691e-05|1|1|
|2.78283|147|0.000114415|8.19147e-05|1|1|
|6.86104|147|0.000135042|8.31561e-05|1|1|
|宏平均|1176|0.000146888|0.000107878|1|1|

全量残差平均 -5.81228e-15，最大绝对值 0.000920378，全部小于 0.001。高拟合度不能单独证明数据来源真实性；数据说明把 B1 标为真实轨迹，但现有材料不足以独立核验数值生成过程。残差源表见[residuals.csv](../b1_baseline/residuals.csv)，按 log N、log D、预测值和规模的四幅诊断图见[图表清单](figure_manifest.csv)。

## 46.3 B2/B3/B4/B5 冻结参数迁移

以下全部使用 B1 冻结参数，没有重新调参。B2 是半合成跨来源轨迹；B3 是检查点插值轨迹，只检验曲线一致性；B4 是跨族点；B5 是文献汇编。因 Loss 口径未证实一致，RMSE/R² 和偏置为数值描述，不能当作严格同任务预测精度；Pearson/Spearman 也只能作条件性趋势比较。

|附件|行数|RMSE|MAE|R²|Pearson|Spearman|预测−原值均偏|
|---|---|---|---|---|---|---|---|
|B2|1029|1.24308|1.17844|-5.0728|0.668738|0.788102|-1.17844|
|B3|4000|0.00382128|0.00289295|0.999958|0.99999|0.99999|-0.00196832|
|B4|57|0.292667|0.227263|0.604505|0.975076|0.982982|-0.221009|
|B5|44|0.197595|0.170722|0.730559|0.907626|0.958808|-0.0205699|

B2 的系统偏差尤其大；B3 的近乎重合属于插值一致性，不能充当 4000 次独立实验。B4/B5 内误差较大的分层如下；分层样本量可能很小，不能仅凭排名判断模型族优劣。

|附件|分层|行数|RMSE|Spearman|
|---|---|---|---|---|
|B4|Pythia|8|0.543368|1|
|B4|OPT|7|0.316817|1|
|B4|BLOOM|4|0.30294|1|
|B5|Chowdhery et al. 2022\|PaLM|3|0.330385|1|
|B5|Touvron et al. 2023\|LLaMA-2|3|0.243104|1|
|B5|Zhang et al. 2022\|OPT|5|0.236465|1|

完整的模型族、轨迹与文献分层见[迁移验证矩阵](../transfer_validation/validation_matrix.csv)。

## 46.4 B6 质量 Scaling Law 主实验

B6 主比较使用同一批 360 行和同一组级折分：$M_0=E_6+A_6N^{-\alpha_6}+B_6D^{-\beta_6}$；$M_Q=E_6+A_6N^{-\alpha_6}+B_6D^{-\beta_6}\exp[\gamma_Q(Q_{0,B}-Q_B)]$。$Q_{0,B}=0.5$ 是 B6 中位数，建模前固定。45 个 N,D 基础组各含多个 Q，GroupKFold 保证同组不跨训练/验证。

|模型|折数|总验证行|宏 RMSE|宏 MAE|宏 R²|宏 Spearman|
|---|---|---|---|---|---|---|
|M0|5|360|0.128762|0.108862|0.816938|0.887007|
|MQ|5|360|0.0760865|0.0613199|0.932464|0.976712|
|MQ2|5|360|0.0756543|0.0608639|0.933332|0.976989|

全量 B6 拟合的 MQ 参数为 E=0.909046、A=0.540173、B=1.79454、α=0.279054、β=0.0995977、γ_Q=0.317893。按基础组 bootstrap 60 次，γ_Q 的 95% 分位区间为 [0.193464, 0.42905]。在这个模型和编码下，γ_Q>0 意味 Q_B 数值升高、预测 Loss 降低；Q_B 的操作性“越高越好”定义仍未独立确认。

二次候选 MQ2 的宏 RMSE=0.0756543，按预设至少 2% 改善规则未升级为主模型。候选参数和选择记录见[B6 参数文件](../quality_model/b6_parameters.json)，所有折和预测见[CV 指标](../quality_model/b6_cv_metrics.csv)、[CV 预测](../quality_model/b6_cv_predictions.csv)。

## 46.5 B7-new 最终留出

B7 共 450 行，扣除与 B6 完全一致的 360 行后，B7-new 为 90 行。它的 45/45 个 N,D 基础组已在 B6 出现，因此只是新 Q 水平留出，不是新规模或新 token 配置留出。模型形式、Q 参考点与参数均在查看该表指标前固定。

|模型|行数|RMSE|MAE|R²|Pearson|Spearman|
|---|---|---|---|---|---|---|
|M0|90|0.0589578|0.0487065|0.966424|0.985536|0.983995|
|MQ|90|0.0490942|0.0370802|0.976719|0.988425|0.987361|

MQ−M0 的 RMSE 点差为 -0.00986361；按 N,D 基础组重抽样的 95% 分位区间 [-0.0190309, -0.00344218]。这是半合成体系内部的留出证据，不能外推为真实新模型族上的增益。逐行留出预测见[MQ 预测](../quality_model/b7_new_predictions_MQ.csv)，区间样本见[bootstrap 差值](../quality_model/b7_new_bootstrap_delta.csv)。

## 46.6 B1-anchored 质量参数敏感性

固定 B1 的 E、A、B、α、β，另允许来源偏置后，在 B6 上估得 γ_Q=0.869619、来源偏置=0.166609、训练 RMSE=0.0854608。B6 自由拟合主模型 γ_Q=0.317893，两者差异表明跨来源结构约束显著改变质量参数；该 anchored 结果仅作敏感性分析，不替代 B6 主估计。[原表](../quality_model/anchor_sensitivity.csv)。

## 46.7 B8 分层压力测试

|B8 类型|N,D 组|Spearman>0 组|组内 Spearman 中位数|数值非降组|
|---|---|---|---|---|
|calibrated|90|90|0.99825|89|
|extrapolated|60|60|0.982362|59|

B8 的 calibrated 和 extrapolated 均独立列示，没有进入 B6 拟合或模型选择。所有 150 个分层组的 Q_B–Loss 数值相关为正，而 B6 的 MQ 方向为负。不能为了消除冲突而翻转 Q 编码；也不能在没有生成协议的情况下断定 B8 错误。

|B8 类型|冻结模型|行数|RMSE|R²|Spearman|均偏|
|---|---|---|---|---|---|---|
|calibrated|M0|984|1.32236|-2.46889|0.313437|1.13656|
|calibrated|MQ|984|1.37161|-2.7321|-0.012381|1.12015|
|extrapolated|M0|720|1.23691|-2.77413|0.139268|1.05858|
|extrapolated|MQ|720|1.28677|-3.08452|-0.307417|1.0441|

组内方向完整结果见[B8 审计表](../audit/b8_groupwise_q_loss.csv)，冻结模型压力结果见[B8 stress test](../quality_model/b8_stress_test.csv)。

## 46.8 Q_A 与 Q_B 的语义区分

Q_A 是问题一对 22 个指标所建的分层综合评分，设计上越高越好；Q_B 是 B6–B8 的 `Q_score`，其操作性定义与方向未在附件中充分说明。Q_A 与 Q_B 的来源、数值范围及计算口径不能证明一致，因此数据合同保持 `Q_A != Q_B`。

|变量来源|列|实测最小|实测最大|方向/等价性|
|---|---|---|---|---|
|B6|Q_score|0.1|1|uncertain; vs B6: self|
|B7|Q_score|0.1|1|uncertain; vs B6: uncertain|
|B8|Q_score|0.05|1|uncertain; vs B6: uncertain|
|Q_A|Q_hierarchical_balanced|13.6052|87.8704|yes_by_Q1_design; vs B6: unproven|

比较表见[quality_semantics.csv](../audit/quality_semantics.csv)；两类 Q 未在拟合数据中合并。

## 46.9 第一问冻结配比接口

第一问冻结的 17 维配比到 13 维 Loss 模型定义 $h_{p,v}(p)=\log[\widehat L_v^{mix}(p)/\widehat L_v^{mix}(p_0)]$。正式接口先将 A4 的 512 份训练配方逐行归一化，再取均值得到 $p_0$；未改动 A4。所有用于求 log 的预测 Loss 经正值检查，$h_{p,v}(p_0)=0$。

导入模型为 LightGBM，当前文件 SHA-256 `39c5d5dc13b6f0fbf68699bb45d2d8d18e28f81f0f46f5475311e6aa66bd1c33`；参考配比行和为 1。配比支持阈值为训练配方留一最近邻欧氏距离 95% 分位 0.260035。正式接口清单为 `results/q1_revision_v2_1/q2_interface/manifest.json`。

|验证目标|h 最小|h 中位|h 最大|
|---|---|---|---|
|metric/the_pile_arxiv_val_loss|-0.137317|0.160721|0.611705|
|metric/the_pile_freelaw_val_loss|-0.144969|0.15664|0.37329|
|metric/the_pile_pubmed_central_val_loss|-0.11926|0.130599|0.44714|
|metric/the_pile_wikipedia_en_val_loss|-0.110991|0.0753688|0.256668|
|metric/the_pile_dm_mathematics_val_loss|-0.144175|0.604795|1.01944|
|metric/the_pile_github_val_loss|-0.183351|0.126708|0.563696|
|metric/the_pile_stackexchange_val_loss|-0.134208|0.0909728|0.431185|
|metric/the_pile_gutenberg_pg_19_val_loss|-0.0714079|0.0606445|0.233819|
|metric/the_pile_pile_cc_val_loss|-0.0648754|0.0438414|0.165594|
|metric/the_pile_ubuntu_irc_val_loss|-0.133817|0.219805|0.396244|
|metric/the_pile_hackernews_val_loss|-0.0467216|0.0726388|0.216128|
|metric/the_pile_pubmed_abstracts_val_loss|-0.131232|0.0621341|0.314683|
|metric/the_pile_uspto_backgrounds_val_loss|-0.129117|0.114777|0.277521|

13 目标逐一保留；h_agg 只在明确需要单个情景因子时定义为 13 个无量纲对数比的等权平均。A4 上 h_agg 范围 [0.0134887, 0.333013]，不把 B 标量 Loss 与任一 A 目标天然对应。逐配方结果见[h_by_target.csv](../q1_interface/h_by_target.csv)，输入/输出顺序见[接口元数据](../q1_interface/interface_metadata.json)。

## 46.10 广义模型与参数证据等级

仅 B6 体系的 $E,A,B,\alpha,\beta,\gamma_Q$ 是 estimated；13 个 $h_{p,v}$ 及约定的 $h_{agg}$ 是 imported_Q1；$\lambda_p$ 是 scenario_assumption。Form A：$L_A=E+AN^{-\alpha}+BD^{-\beta}\exp[g_Q(Q_B)+\lambda_ph(p)]$。Form B：$L_B=E+[AN^{-\alpha}+BD^{-\beta}\exp g_Q(Q_B)]\exp[\lambda_ph(p)]$。两式在 $\lambda_p=0$ 且 $Q_B=Q_{0,B}$ 时退化为 B6 经典形式。

附件 B 不含 p，也没有 (N,D,Q_B,p,L) 联合实验；故 λ_p 无法被 B 数据识别。B 的标量 Loss 与 Q1 的 13 个验证域 Loss 尺度/语义未证明相同，h_p 的迁移属于结构假设。Form A 假设配比作用于数据项；Form B 是形式敏感性，均不能用于宣称真实机制已识别。

## 46.11 λ_p 固定情景

λ_p 预先固定为 0、0.5、1、1.5，分别称 no-transfer、attenuated-transfer、unit-transfer、amplified-transfer；没有从附件 B 回归估计。工作点取 B6 实际 N×D 排序的 10%、50%、90% 位置，Q_B 取 B6 的 25%、50%、75% 分位，配比取 p0 和两份 A4 配方。

|工作点|N(B)|D(B tokens)|
|---|---|---|
|0|0.7|10|
|1|0.41|300|
|2|6.9|300|

共 216 个 Form×λ×Q×配比×工作点情景值，均按同一模型计算；其中支持区域标记为 {'within_support': 216}。结果见[scenario_grid.csv](../generalized_law/scenario_grid.csv)，不能视为新的训练观测。

## 46.12 Form A/B 结构敏感性

下表固定 B6 中位工作点、Q_B=0.5、同一 A4 配方，仅改变 λ 和模型形式，展示结构选择带来的预测差异。

|λ_p|h_agg|Form A Loss|Form B Loss|
|---|---|---|---|
|0|0.0925197|2.61862|2.61862|
|0.5|0.0925197|2.66676|2.69956|
|1|0.0925197|2.71718|2.78434|
|1.5|0.0925197|2.76999|2.87312|

λ=0 时两形式完全一致；非零 λ 下差异来自结构假设，不能由附件 B 选择真实形式。全部差值见[form_sensitivity.csv](../generalized_law/form_sensitivity.csv)。

## 46.13 N、D、Q_B 边际效应与弹性

对可约 Loss $R=L-E$，在每个工作点计算 $M_N=-\partial L/\partial N$、$M_D=-\partial L/\partial D$、$M_Q=-\partial L/\partial Q_B$，以及 $\varepsilon_N=\partial\ln R/\partial\ln N$、$\varepsilon_D=\partial\ln R/\partial\ln D$。Q_B 零点不具可靠比例语义，报告 $\partial\ln R/\partial Q_B$ 半弹性。下表为 p0、λ=0、Form A、Q_B=0.5 的条件结果。

|工作点|N|D|M_N|M_D|M_Q|ε_N|ε_D|Q 半弹性|
|---|---|---|---|---|---|---|---|---|
|0|0.7|10|0.237875|0.0142104|0.453562|-0.0822902|-0.0702273|-0.22415|
|1|0.41|300|0.47151|0.000337572|0.323235|-0.11308|-0.0592379|-0.189074|
|2|6.9|300|0.0127435|0.000337572|0.323235|-0.0660182|-0.076035|-0.242686|

导数单位随 N、D 的十亿单位变化；不能跨不同单位直接比较 M_N 与 M_D 大小。全情景值见[marginal_effects.csv](../generalized_law/marginal_effects.csv)、[elasticities.csv](../generalized_law/elasticities.csv)，参数 bootstrap 区间见[local_effect_intervals.csv](../generalized_law/local_effect_intervals.csv)。

## 46.14 质量与规模的局部等损失替代

固定 D、p、当前 Loss 与工作点，由隐函数求 $dN/dQ_B|_{L,D,p}$ 和 $d\ln N/dQ_B|_{L,D,p}$。这只是在已拟合函数附近的 local iso-loss substitution；Q_B 改善成本未知，不能推出成本最优替代率。

|工作点|N|D|dN/dQ_B|dlnN/dQ_B|
|---|---|---|---|---|
|0|0.7|10|-1.90672|-2.72389|
|1|0.41|300|-0.685532|-1.67203|
|2|6.9|300|-25.3648|-3.67606|

局部值及其组 bootstrap 区间分别见[q_n_substitution.csv](../generalized_law/q_n_substitution.csv)和[local_effect_intervals.csv](../generalized_law/local_effect_intervals.csv)。

## 46.15 配比份额转移与路径条件组合响应

每次扰动采用 $p(\delta)=p+\delta(e_d-e_k)$，严格检查非负与行和。双领域组合使用同一 donor、同一基准和同一份额扣减路径，响应为四角有限差分 $\Delta_{jk}-\Delta_j-\Delta_k$。

计算 24 条可行路径，23 条全部四角位于第一问支持阈值内，1 条标为低置信外推。最大绝对响应的若干路径如下；符号随基准和 δ 变化，不归纳为固有或因果协同。

|基准|donor|领域 j|领域 k|δ|h_agg 四角响应|支持状态|
|---|---|---|---|---|---|---|
|A4:170|train_the_pile_stackexchange|train_the_pile_pile_cc|train_the_pile_arxiv|0.03|-0.00392435|within_support|
|A4:170|train_the_pile_stackexchange|train_the_pile_pile_cc|train_the_pile_pubmed_central|0.03|-0.00329078|within_support|
|A4:170|train_the_pile_stackexchange|train_the_pile_pubmed_central|train_the_pile_arxiv|0.03|-0.00270518|within_support|
|A4:261|train_the_pile_stackexchange|train_the_pile_pile_cc|train_the_pile_arxiv|0.01|-0.00246686|within_support|
|A4:261|train_the_pile_stackexchange|train_the_pile_pubmed_central|train_the_pile_arxiv|0.01|-0.00241276|within_support|

全量计算见[path_conditioned_pair_response.csv](../generalized_law/path_conditioned_pair_response.csv)。

## 46.16 固定算力 C≈6ND 的条件优化

采用稠密 Transformer 训练计算近似 $C\approx6ND$，N、D 为十亿单位时表中 C 单位为 $10^{18}$ FLOPs。固定 C 后 $D=C/(6N)$。对 Form A、p0、λ=0，解析内点满足 $\alpha P=\beta T$；并在 B6 观测 N∈[0.07,11.97]、D∈[10,600] 的预设矩形边界内数值优化。矩形边界不等于稠密实测支持区域。

|预算档|C(10¹⁸ FLOPs)|Q_B|解析 N*|解析 D*|带界 N*|带界 D*|解析在边界内|
|---|---|---|---|---|---|---|---|
|low|42|0.275|0.880785|7.94746|0.7|10|否|
|low|42|0.5|1.06391|6.57948|0.7|10|否|
|low|42|0.825|1.39767|5.00834|0.7|10|否|
|medium|738|0.275|1.87195|65.7068|1.87195|65.7068|是|
|medium|738|0.5|2.26116|54.3969|2.26116|54.3969|是|
|medium|738|0.825|2.97049|41.4073|2.97049|41.4073|是|
|high|12420|0.275|3.93359|526.236|3.93359|526.236|是|
|high|12420|0.5|4.75145|435.656|4.75145|435.656|是|
|high|12420|0.825|6.242|331.625|6.242|331.625|是|

预算由 B6 的 6ND 10%、50%、90% 分位事先确定；3 个低档解析点的 D* 小于 B6 最小 D=10，带界解贴边。解析一阶条件的最大绝对残差 5.55112e-17，无边界解析/数值优化最大相对差 4.53665e-09。

Q_B=0.5 时按 B6 基础组重抽样的解析最优 95% 分位区间：

|预算档|N* 2.5%|N* 97.5%|D* 2.5%|D* 97.5%|
|---|---|---|---|---|
|low|0.811032|1.5025|4.65903|8.63167|
|medium|1.91012|2.71339|45.3308|64.3982|
|high|3.69573|6.08725|340.095|560.111|

λ 的优化敏感性只有在 h_agg≠0 的配比情景中显现。以下固定中档预算、Q_B=Q0 和同一 A4 配方；Form B 的配比因子同时乘在两项可约 Loss 上，故其内点 N*/D* 对 λ 不变，这源于所选函数形式。

|形式|λ_p|h_agg|解析 N*|解析 D*|
|---|---|---|---|---|
|A|0|0.0925197|2.26116|54.3969|
|A|0.5|0.0925197|2.00112|61.4655|
|A|1|0.0925197|1.77099|69.4527|
|A|1.5|0.0925197|1.56732|78.4779|
|B|0|0.0925197|2.26116|54.3969|
|B|0.5|0.0925197|2.26116|54.3969|
|B|1|0.0925197|2.26116|54.3969|
|B|1.5|0.0925197|2.26116|54.3969|

详见[analytic_optima.csv](../compute_opt/analytic_optima.csv)、[bounded_optima.csv](../compute_opt/bounded_optima.csv)、[bootstrap_optima_intervals.csv](../compute_opt/bootstrap_optima_intervals.csv)与[mixture_scenario_optima.csv](../compute_opt/mixture_scenario_optima.csv)。质量提升成本、注意力及上下文开销未进入 C≈6ND，因此不是完整质量/配比联合资源最优。

## 46.17 B9/B10 大尺度外推一致性

B9 有 132 行模型元数据，其中 4 行 D≤0 被隔离，保留 128 个有效 N,D 点。B10 有 128 行估算 Loss，未进入 B1/B6 拟合、调参或独立验证。以下仅是与附件既有估算序列的数值一致性。

|冻结曲线|估算行|RMSE|MAE|平均相对误差|Spearman|平均 log 外推距离|
|---|---|---|---|---|---|---|
|B1_M0|128|0.00109383|0.000614049|0.000258908|0.999994|4.05371|
|B6_MQ_at_Q0|128|0.156491|0.0856961|0.0385064|0.997897|4.05371|

B1 曲线与 B10 极接近，但数据说明明确 B10 是用已拟合标度律参数估算；高度一致存在循环生成风险，绝不能表述成百亿参数以上的真实独立验证。逐点外推距离与预测见[B10 一致性表](../extrapolation/b10_estimated_consistency.csv)和[逐点预测](../extrapolation/b10_predictions_B1_M0.csv)。

## 46.18 结论等级、局限与复现

|证据等级|结果条目|在本报告中的解释|
|---|---|---|
|direct_fit|5|附件 B1 的参数拟合|
|held_out_validation|4|组留出或 B7-new；仍须看数据来源|
|semi_synthetic_calibration|7|B2/B6–B8 半合成或 B3 插值检查|
|imported_Q1|2|冻结的第一问配比接口与 Q_A 口径|
|scenario_assumption|4|λ、模型形式、条件最优与边际|
|extrapolation_reference|2|B9/B10 大尺度参考|

主要限制：①Q_A/Q_B 不能证明等价，B8 方向冲突；②B6–B8 为半合成，B7-new 与 B6 共享全部基础组；③附件 B 不含配比，λ_p 不可识别；④B 的单一 Loss 与 Q1 的 13 目标没有天然映射；⑤C≈6ND 忽略质量处理和注意力成本，上下文长度未优化；⑥极大尺度是外推或既有估算；⑦配比跨问题迁移依赖结构假设；⑧B1 的近舍入量级残差与 Q1 历史哈希缺口仍需来源核验。

本报告所述最可信结论限定为：B1 在自身记录的规模留出下符合所拟合曲线；B6/B7 半合成体系内 Q_B 提供额外预测信息；B8 无法支持同一质量方向；条件算力最优可由给定模型解析和数值复核。其余配比和 λ 结论均为透明情景，而非独立实证识别。

### 复现记录

|阶段|状态|测试退出码|有效/关键记录|
|---|---|---|---|
|P0|PASS_WITH_WARNINGS|0|{'B1': 1176, 'B2': 1029, 'B3': 4000, 'B4': 57, 'B5': 44, 'B6': 360, 'B7': 450, 'B8': 1704, 'B9': 132, 'B10': 128, 'B11': 18, 'B12': 1386}|
|P1|PASS|0|{'valid': 1176, 'isolated': 0, 'bootstrap_success': 60}|
|P2|PASS_WITH_WARNINGS|0|{'B2': 1029, 'B3_interpolated': 4000, 'B4': 57, 'B5': 44}|
|P3|PASS_WITH_WARNINGS|0|{'B6': 360, 'B7_new': 90, 'B7_new_group_overlap': 45, 'B8': 1704, 'bootstrap_success': 60}|
|P4|PASS_WITH_WARNINGS|0|{'A4_recipes': 512, 'targets': 13}|
|P5|PASS_WITH_WARNINGS|0|{'scenario_points': 216, 'pair_paths': 24}|
|P6|PASS_WITH_WARNINGS|0|{'budgets': 3, 'quality_levels': 3, 'bootstrap_optima': 540}|
|P7|PASS_WITH_WARNINGS|0|{'B9_raw': 132, 'B9_valid': 128, 'B10_estimated': 128}|

运行命令：`PYTHONPATH=src .venv/bin/python -m src.q2_fusion_rerun --stage all`，阶段测试由该入口顺序执行；独立完整测试命令为 `PYTHONPATH=src .venv/bin/python -m pytest -q tests/q2_v2 tests/q1_revision_v2_1`。版本、全部 B 输入 SHA-256、配置 SHA-256、Q1 模型 SHA-256 与阶段元数据见[reproducibility_summary.json](reproducibility_summary.json)；主要结果见[key_results_table.csv](key_results_table.csv)，逐结论证据等级见[evidence_grade_table.csv](evidence_grade_table.csv)，图表来源见[figure_manifest.csv](figure_manifest.csv)。

配置 SHA-256 `07a646cfd39da0a60f6f4631b22c5c2fd3038255ffe87296615a09aad4f19fa6`；Q1 冻结模型 SHA-256 `39c5d5dc13b6f0fbf68699bb45d2d8d18e28f81f0f46f5475311e6aa66bd1c33`。所有原始附件只读，未新增依赖、未下载外部数据。

**Q2 FINAL STATUS: COMPLETE_WITH_WARNINGS**
