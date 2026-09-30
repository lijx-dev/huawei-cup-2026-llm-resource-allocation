"""按 P0–P9 顺序运行；每阶段通过相应测试后才继续。"""
from __future__ import annotations

import argparse
import subprocess
import sys

from . import audit, stages
from .common import ROOT


RUNNERS = [audit.run, stages.p1, stages.p2, stages.p3, stages.p4, stages.p5, stages.p6, stages.p7, stages.p8, stages.p9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", type=int, default=0)
    parser.add_argument("--to-stage", type=int, default=9)
    args = parser.parse_args()
    if not (0 <= args.from_stage <= args.to_stage <= 9): parser.error("阶段须为 0–9 且起点不大于终点")
    for stage in range(args.from_stage, args.to_stage + 1):
        status = RUNNERS[stage]()
        if status == "BLOCKED":
            print(f"Q2_V3 FINAL STATUS: INCOMPLETE; P{stage} BLOCKED")
            return 2
        cmd = [sys.executable, "-m", "pytest", "-q", "tests/q2_v3/test_audit.py", "tests/q2_v3/test_model.py", "tests/q2_v3/test_stage_outputs.py", "-k", f"test_p{stage}"]
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode:
            print(f"Q2_V3 FINAL STATUS: INCOMPLETE; P{stage} tests failed")
            return result.returncode
    print("Q2_V3 FINAL STATUS: COMPLETE_WITH_WARNINGS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
