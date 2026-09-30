# P5 广义标度律情景

B6 直接估计 E,A,B,alpha,beta,gamma；第一问导入 h_v 和等权 h_agg；lambda_p 仅取 [0, 0.5, 1, 1.5]，从未由 B 拟合。Form A 调节有效数据项，Form B 调节全部可约 Loss；均为结构假设。

工作点取 B6 N*D 的 10/50/90% 对应实际行，Q 取 B6 25/50/75 分位，配比取 p0 和两份 A4 配方。输出 216 个情景点、24 条路径条件组合响应。M_N/M_D/M_Q 为 Loss 负偏导；弹性以 R=L-E 为分母，Q 报半弹性。局部等损失替代固定 D、p、Loss 和工作点。

配比路径响应不是因果协同；低支持区域按距离标记。B 的标量 Loss 与 A 的 13 目标没有天然一致口径。

Q2-P5 STATUS: PASS_WITH_WARNINGS
