# P3 B6/B7/B8 质量标度律

B6 360 行、45 个 N,D 基础组；同组 Q 不跨折。Q0 为预先固定 B6 中位数 0.5。B6 及 B7 的 Q_score 未自动翻转。

B6 组 CV 宏平均 RMSE：M0=0.128762，MQ=0.0760865，二次候选=0.0756543；二次候选仅按 B6 规则保留敏感性=False，主模型仍为 MQ。

B7-new 90 行只作最终留出。新增 ID 涉及 45 个 N,D 基础组，与 B6 重叠 45 组；因此检验新 Q 水平，不检验全新 N,D 组。M0/MQ 指标见 `b7_new_metrics.csv`，组 bootstrap 的 MQ-M0 RMSE 差 95% 区间 [-0.0190309,-0.00344218]。MQ gamma=0.317893，B6 基础组 bootstrap 95% 区间 [0.193464,0.42905]；B1 固定参数敏感性见 `anchor_sensitivity.csv`。

B8 分层组内 Q-Loss Spearman 摘要：{'calibrated': {'count': 90, 'min': 0.9929823694376384, 'median': 0.9982502173821436, 'max': 1.0}, 'extrapolated': {'count': 60, 'min': 0.9644013324700096, 'median': 0.9823619317924353, 'max': 0.9929823694376384}}。方向冲突完整保留；B8 没有进入拟合。Q_B 高值语义未由数据说明独立确认。

Q2-P3 STATUS: PASS_WITH_WARNINGS
