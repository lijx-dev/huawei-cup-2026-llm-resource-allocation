"""问题二融合方案修订重跑入口。

用法：PYTHONPATH=src .venv/bin/python -m src.q2_fusion_rerun --stage all
"""
from __future__ import annotations

import os

from src.q1_revision_v2_1.export_q2_interface import export


def main() -> int:
    manifest = export()
    if manifest["status"] != "PASS":
        raise RuntimeError("Q1→Q2 正式接口未通过校验")
    os.environ["Q2_OUTPUT_DIR"] = "results/q2_fusion_rerun_v1"
    os.environ["Q2_CONFIG_PATH"] = "configs/q2_fusion_rerun_v1.json"
    os.environ["Q2_EXPERIMENT_VERSION"] = "q2-fusion-rerun-v1"
    os.environ["Q1_INTERFACE_DIR"] = "results/q1_revision_v2_1/q2_interface"
    from src.q2_v2.cli import main as run_q2

    return run_q2()


if __name__ == "__main__":
    raise SystemExit(main())
