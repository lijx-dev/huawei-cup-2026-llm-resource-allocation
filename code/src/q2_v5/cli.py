"""Q2 V5 阶段命令入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import run_p00


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二 V5 独立流水线")
    parser.add_argument("stage", choices=["p00"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.stage == "p00":
        result = run_p00(args.root, args.root / "configs/q2_v5/config.yaml")
        print(json.dumps({"stage": result["stage"], "status": result["status"],
                          "input_file_count": result["input_file_count"]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
