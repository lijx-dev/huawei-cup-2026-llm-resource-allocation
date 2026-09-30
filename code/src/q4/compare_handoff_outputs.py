"""比较 Q4 隔离复跑表格与交接包冻结快照，输出结构化验收报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/q4_handoff_rerun_20260926"
RUNTIME = RUN / "runtime"
PACKAGE = RUN / "package_source/第四问完整交接包_2026-09-26"
ATOL = RTOL = 1e-7

DIRECTORY_PAIRS = [
    (
        RUNTIME / "outputs/第四问C5主桥接复算_2026-09-26",
        PACKAGE / "04_关键结果/方案重跑/C5_C6桥接",
    ),
    (
        RUNTIME / "outputs/第四问版本时间统一模型_2026-09-26",
        PACKAGE / "04_关键结果/方案重跑/版本时间统一模型",
    ),
    (
        RUNTIME / "results/Q4/experiments/robust_ensemble_handoff_rerun_20260926/metrics",
        PACKAGE / "04_关键结果/实验结果/robust_ensemble_20260926_165250/metrics",
    ),
    (
        RUNTIME / "results/Q4/experiments/robust_ensemble_handoff_rerun_20260926/tables",
        PACKAGE / "04_关键结果/实验结果/robust_ensemble_20260926_165250/tables",
    ),
    (
        RUNTIME / "results/Q4/experiments/assumption_scenarios_20260926/tables",
        PACKAGE / "04_关键结果/实验结果/assumption_scenarios_20260926/tables",
    ),
]


def compare_csv(actual_path: Path, expected_path: Path) -> dict[str, object]:
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)
    record: dict[str, object] = {
        "file": actual_path.name,
        "actual": str(actual_path.relative_to(ROOT)),
        "expected": str(expected_path.relative_to(ROOT)),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "columns_equal": list(actual.columns) == list(expected.columns),
        "numeric_mismatches": None,
        "string_mismatches": None,
        "max_abs_numeric_difference": None,
    }
    if actual.shape != expected.shape or not record["columns_equal"]:
        record["status"] = "structural_mismatch"
        return record

    numeric_mismatches = 0
    string_mismatches = 0
    max_difference = 0.0
    for column in actual.columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            x, y = left.to_numpy(float), right.to_numpy(float)
            close = np.isclose(x, y, atol=ATOL, rtol=RTOL, equal_nan=True)
            numeric_mismatches += int((~close).sum())
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                max_difference = max(max_difference, float(np.max(np.abs(x[finite] - y[finite]))))
        else:
            x = left.fillna("<NA>").astype(str)
            y = right.fillna("<NA>").astype(str)
            string_mismatches += int((x != y).sum())
    record.update({
        "numeric_mismatches": numeric_mismatches,
        "string_mismatches": string_mismatches,
        "max_abs_numeric_difference": max_difference,
        "status": "equivalent" if numeric_mismatches == 0 and string_mismatches == 0 else "value_mismatch",
    })
    return record


def main() -> int:
    comparisons = []
    for actual_dir, expected_dir in DIRECTORY_PAIRS:
        for actual in sorted(actual_dir.glob("*.csv")):
            expected = expected_dir / actual.name
            if expected.is_file():
                comparisons.append(compare_csv(actual, expected))

    figures = RUNTIME / "results/Q4/experiments/complete_assumption_solution_20260926/figures"
    figure_files = sorted(p.name for p in figures.glob("*") if p.is_file() and p.stat().st_size > 0)
    expected_figure_files = sorted(
        p.name for p in (PACKAGE / "02_图件").glob("fig*.*")
        if p.suffix in {".png", ".svg"}
    )
    equivalent = sum(row["status"] == "equivalent" for row in comparisons)
    report = {
        "schema_version": 1,
        "tolerance": {"absolute": ATOL, "relative": RTOL},
        "table_count": len(comparisons),
        "equivalent": equivalent,
        "mismatch": len(comparisons) - equivalent,
        "max_abs_numeric_difference": max(
            (float(row["max_abs_numeric_difference"] or 0.0) for row in comparisons), default=0.0
        ),
        "tables": comparisons,
        "figures": {
            "generated_nonempty_count": len(figure_files),
            "expected_count": len(expected_figure_files),
            "names_equal": figure_files == expected_figure_files,
            "files": figure_files,
            "note": "图像受 Matplotlib/字体元数据影响，不要求字节级一致；其上游 25 张数值表按容差逐单元比较。",
        },
        "expected_metadata_differences": [
            "运行时间戳与 elapsed_seconds",
            "隔离运行目录路径",
            "经审计路径重定向后的脚本 SHA-256",
            "稳健集成运行 ID",
        ],
    }
    target = RUN / "audit/frozen_snapshot_comparison.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "table_count": report["table_count"],
        "equivalent": report["equivalent"],
        "mismatch": report["mismatch"],
        "max_abs_numeric_difference": report["max_abs_numeric_difference"],
        "figures": report["figures"]["generated_nonempty_count"],
    }, ensure_ascii=False))
    return 0 if report["mismatch"] == 0 and report["figures"]["names_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
