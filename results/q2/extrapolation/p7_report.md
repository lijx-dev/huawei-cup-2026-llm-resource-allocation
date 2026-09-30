# P7 B9/B10 超大尺度参考

B9 共 132 行，D 非正隔离 4 行，可用元数据 128 行。B10 共 128 行，Loss 是既有标度律估算，未参与任何参数拟合。

B1 曲线与 B10 估算序列 RMSE=0.00109383；数值一致不构成独立真实验证，且提示同源生成/循环验证风险。完整比较见 `b10_estimated_consistency.csv`。外推距离相对 B1 训练上界以 log N/log D 计算。

Q2-P7 STATUS: PASS_WITH_WARNINGS
