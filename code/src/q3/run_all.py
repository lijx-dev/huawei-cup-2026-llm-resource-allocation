"""按方案顺序重算问题三；每步日志写入 results/q3/logs。"""
import subprocess
import sys
from pathlib import Path

from q3_paths import OUTPUT_DIR, ROOT

STEPS = [
    "q3_model_core.py",
    "q3_constrained_vs_free.py",
    "q3_fused_grid.py",
    "q3_p_subproblem_fix.py",
    "q3_team_claim_check.py",
    "q3_kkt_verify.py",
    "q3_derived_quantities.py",
    "q3_bootstrap_transitions.py",
    "finalize.py",
]
LOG_NAMES = {
    "q3_model_core.py": "core.log",
    "q3_constrained_vs_free.py": "constrained_vs_free.log",
    "q3_fused_grid.py": "fused_grid.log",
    "q3_p_subproblem_fix.py": "p_subproblem.log",
    "q3_team_claim_check.py": "team_claim_check.log",
    "q3_kkt_verify.py": "kkt_verify.log",
    "q3_derived_quantities.py": "derived_quantities.log",
    "q3_bootstrap_transitions.py": "bootstrap.log",
    "finalize.py": "finalize.log",
}


def main():
    logs = OUTPUT_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for script in STEPS:
        path = Path(__file__).resolve().parent / script
        log = logs / LOG_NAMES[script]
        print(f"运行 {script}，日志：{log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", str(path)], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
