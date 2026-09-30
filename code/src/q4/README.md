# 问题四融合方案复现

使用仓库根目录的 `团队交付包_问题四融合方案.zip` 和只读的 B/C 附件，在当前环境运行：

```bash
.venv/bin/python src/q4/audit_inputs.py
.venv/bin/python src/q4/run_fusion_package.py
.venv/bin/python src/q4/compare_outputs.py
.venv/bin/python -m pytest tests/q4/test_fusion_runner.py -q
```

输出位于 `results/q4/fusion_reproduction/`。`run_fusion_package.py` 在执行每个脚本前核验包内 SHA-256 清单，从压缩包读取原脚本，在内存中仅转换路径和 `COMPATIBILITY_PATCHES` 列出的兼容/计算修正。原始压缩包及 `data/real_attachments/` 不修改。完整输入 SHA-256 见 `input_inventory.json`，单脚本控制台输出见 `logs/`，机读结果见 `outputs/`。

C8 多 JSON 目录固定选文件名时间戳最早的一份；选中截断文件时整目录隔离。此规则会影响逐任务结果，不能与任意文件系统遍历顺序的结果混用。修正的三处是：P1-E 缺失季度比较、三腿重估 24 个月区间误取 12 个月端点、旧三腿诊断日志区间上下限反序。其余统计公式沿用交付包。`compare_outputs.py` 的退出码 1 表示仍有包内数值差异，详情见 `comparison.json`；不应把它当作整个实验未运行。
