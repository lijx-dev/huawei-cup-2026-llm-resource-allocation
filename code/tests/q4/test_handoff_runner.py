"""Q4 新交接包隔离运行器的人工微型/静态测试，不进入正式结果。"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/q4"))
from run_handoff_package import (  # noqa: E402
    PACKAGE, PATCHES, SEED_FILES, STEPS, patched_source, verify_package,
)


def test_handoff_manifest_matches_extracted_package():
    status = verify_package()
    assert status == {"verified_files": 190, "failures": 0}


def test_every_seed_input_exists_in_verified_package():
    for source in SEED_FILES:
        assert (PACKAGE / source).is_file(), source


def test_path_patches_are_exact_and_remove_author_absolute_path():
    for name in STEPS:
        source, applied = patched_source(name)
        assert len(applied) == len(PATCHES.get(name, []))
        assert "/Users/lucasliao/" not in source


def test_downstream_steps_consume_fresh_rerun_outputs():
    robust, _ = patched_source("run_robust_ensemble.py")
    assume, _ = patched_source("run_assumption_scenarios.py")
    figures, _ = patched_source("generate_complete_assumption_solution.py")
    assert "outputs/第四问版本时间统一模型_2026-09-26" in robust
    assert "robust_ensemble_handoff_rerun_20260926" in assume
    assert "outputs/第四问版本时间统一模型_2026-09-26/summary.json" in assume
    assert "第四问C5主桥接复算_2026-09-26" in figures
