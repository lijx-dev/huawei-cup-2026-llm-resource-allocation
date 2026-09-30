# B10 生成与循环验证风险

数据说明第 10 页明确 B10 `supplementary_large_baseline.csv` 的 Loss 是使用已拟合标度律参数估算，并非观测。文件自身没有 formula/source/estimated/fitted/synthetic 等生成过程列，仅有 family、N_params_B、D_tokens_B、val_loss、is_converged。若后续用 B1 标度律预测 B10 并称其为独立验证，会有循环验证风险；是否确实使用同一套 B1 拟合参数，现有材料不能判定。B10 只可作为 estimated consistency reference。
