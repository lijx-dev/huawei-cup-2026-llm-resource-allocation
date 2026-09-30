"""阶段顺序入口：python -m src.q2_v2.cli --stage all。"""
import argparse
import json
import subprocess
import sys

from src.q2_v2 import stages


FUNCTIONS=[stages.p0,stages.p1,stages.p2,stages.p3,stages.p4,stages.p5,stages.p6,stages.p7,stages.p8]
FOLDERS=["audit","b1_baseline","transfer_validation","quality_model","q1_interface","generalized_law","compute_opt","extrapolation","report"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all"]+[f"P{i}" for i in range(9)],default="all")
    args=parser.parse_args()
    selected=range(9) if args.stage=="all" else [int(args.stage[1:])]
    for i in selected:
        meta=FUNCTIONS[i]()
        name=f"P{i}"
        cmd=[sys.executable,"-m","pytest","-q","tests/q2_v2/test_pipeline.py","-k",f"test_p{i}"]
        result=subprocess.run(cmd,cwd=stages.ROOT,check=False)
        metapath=stages.OUT/FOLDERS[i]/f"p{i}_metadata.json"
        meta["test_command"]=" ".join(cmd)
        meta["test_exit_code"]=result.returncode
        metapath.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        print(f"Q2-{name} STATUS: {meta['status']}; tests_exit={result.returncode}",flush=True)
        if meta["status"]=="BLOCKED" or result.returncode:
            return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
