"""在隔离目录中复跑“第四问完整交接包（2026-09-26）”的主链。

交接包内脚本保留原始 SHA-256；本运行器只在 ``results`` 下生成带路径
重定向的运行副本。后续步骤必须读取本轮上一步产物，不能回读仓库旧 Q4 结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "第四问完整交接包_2026-09-26.zip"
RUN = ROOT / "results/q4_handoff_rerun_20260926"
PACKAGE = RUN / "package_source/第四问完整交接包_2026-09-26"
RUNTIME = RUN / "runtime"
CODE_DIR = RUNTIME / "code/Q4"
LOGS = RUN / "logs"
AUDIT = RUN / "audit"

STEPS = [
    "refit_C5_C6_loss_benchmark.py",
    "run_version_aligned_model.py",
    "run_robust_ensemble.py",
    "run_assumption_scenarios.py",
    "generate_complete_assumption_solution.py",
]

# 这些文件是交接包保存的输入/审计证据，不是本轮模型输出。
SEED_FILES = {
    "05_输入数据/C_efficiency_evolution/loss_benchmark_bridge.csv":
        "inputs/C_efficiency_evolution/loss_benchmark_bridge.csv",
    "05_输入数据/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv":
        "inputs/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv",
    "05_输入数据/第四问复跑附件子集.zip":
        "outputs/第四问交接包_联合校准版_2026-09-26/data/第四问复跑附件子集.zip",
    "04_关键结果/口径与日期证据/第四问自定义开源口径_2026-09-26/O口径主样本.csv":
        "outputs/第四问自定义开源口径_2026-09-26/O口径主样本.csv",
    "04_关键结果/口径与日期证据/q4_c2_official_sha_rowlink.json":
        "outputs/第四问继续建模_2026-09-26/q4_c2_official_sha_rowlink.json",
    "04_关键结果/口径与日期证据/audit.json":
        "outputs/第四问版本日期补证_2026-09-26/audit.json",
    "04_关键结果/口径与日期证据/auxiliary_audit.json":
        "outputs/第四问版本日期补证_2026-09-26/auxiliary_audit.json",
    "04_关键结果/完整方案复跑/audit/Q23_to_C6_stage1_diagnostic.csv":
        "outputs/第四问修订方案执行_2026-09-26/Q23_to_C6_stage1_diagnostic.csv",
    "04_关键结果/完整方案复跑/audit/q3_score_conversion_gate.csv":
        "outputs/第四问完整方案复跑_20260926_133810/audit/q3_score_conversion_gate.csv",
    "04_关键结果/完整方案复跑/audit/c8_task_aggregation.csv":
        "inputs/c8_task_aggregation.csv",
}

PATCHES = {
    "refit_C5_C6_loss_benchmark.py": [
        (
            "RAW=Path('/Users/lucasliao/Desktop/F题/real_attachments/C_efficiency_evolution')",
            "RAW=ROOT/'inputs/C_efficiency_evolution'",
        ),
    ],
    "run_robust_ensemble.py": [
        (
            "OUTDIR = ROOT / 'outputs/第四问方案重跑_20260926_162328/版本时间统一模型'",
            "OUTDIR = ROOT / 'outputs/第四问版本时间统一模型_2026-09-26'",
        ),
        (
            "runid='robust_ensemble_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')",
            "runid='robust_ensemble_handoff_rerun_20260926'",
        ),
    ],
    "run_assumption_scenarios.py": [
        (
            "RAW=Path('/Users/lucasliao/Desktop/F题/real_attachments/C_efficiency_evolution')",
            "RAW=ROOT/'inputs/C_efficiency_evolution'",
        ),
        (
            "ROBUST=ROOT/'results/Q4/experiments/robust_ensemble_20260926_165250'",
            "ROBUST=ROOT/'results/Q4/experiments/robust_ensemble_handoff_rerun_20260926'",
        ),
        (
            "DYNAMIC=ROOT/'outputs/第四问方案重跑_20260926_162328/版本时间统一模型/summary.json'",
            "DYNAMIC=ROOT/'outputs/第四问版本时间统一模型_2026-09-26/summary.json'",
        ),
    ],
    "generate_complete_assumption_solution.py": [
        (
            'ROBUST = ROOT / "results/Q4/experiments/robust_ensemble_20260926_165250"',
            'ROBUST = ROOT / "results/Q4/experiments/robust_ensemble_handoff_rerun_20260926"',
        ),
        (
            'DYNAMIC = ROOT / "outputs/第四问方案重跑_20260926_162328/版本时间统一模型"',
            'DYNAMIC = ROOT / "outputs/第四问版本时间统一模型_2026-09-26"',
        ),
        (
            'BRIDGE = ROOT / "outputs/第四问方案重跑_20260926_162328/C5_C6桥接"',
            'BRIDGE = ROOT / "outputs/第四问C5主桥接复算_2026-09-26"',
        ),
        (
            'C8 = ROOT / "outputs/第四问完整方案复跑_20260926_133810/audit/c8_task_aggregation.csv"',
            'C8 = ROOT / "inputs/c8_task_aggregation.csv"',
        ),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: Path) -> dict[str, object]:
    files = []
    if path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            files.append((item.relative_to(path).as_posix(), item.stat().st_size, sha256(item)))
    encoded = json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "path": str(path.relative_to(ROOT)),
        "file_count": len(files),
        "sha256_inventory": hashlib.sha256(encoded).hexdigest(),
    }


def verify_outer_archive() -> dict[str, object]:
    with ZipFile(ARCHIVE) as archive:
        unsafe = [
            name for name in archive.namelist()
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ValueError(f"外层 ZIP 存在非法路径: {unsafe[:3]}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"外层 ZIP CRC 失败: {bad}")
        member_count = len(archive.infolist())
    return {"sha256": sha256(ARCHIVE), "members": member_count, "crc_ok": True}


def verify_package() -> dict[str, object]:
    manifest_path = PACKAGE / "package_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in payload["files"]:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            failures.append(f"unsafe:{row['path']}")
            continue
        path = PACKAGE / Path(*rel.parts)
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
        elif path.stat().st_size != row["bytes"]:
            failures.append(f"size:{row['path']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"sha256:{row['path']}")
    if failures:
        raise ValueError("交接包清单校验失败: " + ", ".join(failures[:8]))
    return {"verified_files": len(payload["files"]), "failures": 0}


def patched_source(name: str) -> tuple[str, list[dict[str, object]]]:
    source_path = PACKAGE / "03_核心代码" / name
    source = source_path.read_text(encoding="utf-8")
    applied = []
    for old, new in PATCHES.get(name, []):
        count = source.count(old)
        if count != 1:
            raise ValueError(f"{name} 路径补丁匹配数应为 1，实际为 {count}: {old}")
        source = source.replace(old, new)
        applied.append({"old": old, "new": new, "count": count})
    if "/Users/lucasliao/" in source:
        raise ValueError(f"{name} 仍包含原作者绝对路径")
    return source, applied


def prepare_runtime() -> dict[str, object]:
    if RUNTIME.exists() and any(RUNTIME.iterdir()):
        raise FileExistsError(
            f"隔离运行目录已非空，拒绝混入旧产物: {RUNTIME.relative_to(ROOT)}"
        )
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    seed_inventory = []
    for source_rel, target_rel in SEED_FILES.items():
        source = PACKAGE / source_rel
        target = RUNTIME / target_rel
        if not source.is_file():
            raise FileNotFoundError(f"交接包缺少复跑输入: {source_rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        seed_inventory.append({
            "source": source_rel,
            "target": target_rel,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    nested_zip = RUNTIME / SEED_FILES["05_输入数据/第四问复跑附件子集.zip"]
    with ZipFile(nested_zip) as archive:
        unsafe = [
            name for name in archive.namelist()
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ValueError(f"内层复跑 ZIP 存在非法路径: {unsafe[:3]}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"内层复跑 ZIP CRC 失败: {bad}")
        nested_zip_status = {"members": len(archive.infolist()), "crc_ok": True}

    scripts = []
    for name in STEPS:
        source_path = PACKAGE / "03_核心代码" / name
        source, applied = patched_source(name)
        target = CODE_DIR / name
        target.write_text(source, encoding="utf-8")
        scripts.append({
            "script": name,
            "original_sha256": sha256(source_path),
            "runtime_sha256": sha256(target),
            "patches": applied,
        })
    return {
        "seed_files": seed_inventory,
        "nested_zip": nested_zip_status,
        "runtime_scripts": scripts,
    }


def run_steps() -> tuple[dict[str, int], str | None]:
    status: dict[str, int] = {}
    robust_run = None
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["MPLCONFIGDIR"] = str(RUN / "mpl_cache")
    (RUN / "mpl_cache").mkdir(parents=True, exist_ok=True)
    for name in STEPS:
        log_path = LOGS / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(CODE_DIR / name)],
                cwd=RUNTIME,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        status[name] = result.returncode
        print(f"{name}: exit={result.returncode}", flush=True)
        if result.returncode != 0:
            break
    robust = RUNTIME / "results/Q4/experiments/robust_ensemble_handoff_rerun_20260926"
    if robust.is_dir():
        robust_run = str(robust.relative_to(ROOT))
    return status, robust_run


def output_inventory() -> list[dict[str, object]]:
    rows = []
    excluded = {"code", "inputs"}
    for path in sorted(p for p in RUNTIME.rglob("*") if p.is_file()):
        rel = path.relative_to(RUNTIME)
        if rel.parts[0] in excluded:
            continue
        # 复制进 runtime/outputs 的固定证据只作为输入，不列作新结果。
        if rel.as_posix() in set(SEED_FILES.values()):
            continue
        rows.append({
            "path": rel.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(RUN / "mpl_cache"))
    old_before = tree_digest(ROOT / "results/q4")
    archive_status = verify_outer_archive()
    package_status = verify_package()
    preparation = prepare_runtime()
    status: dict[str, int] = {}
    robust_run = None
    if not args.prepare_only:
        status, robust_run = run_steps()
    old_after = tree_digest(ROOT / "results/q4")
    outputs = output_inventory()
    manifest = {
        "schema_version": 1,
        "purpose": "Q4 handoff-package isolated rerun",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_status": archive_status,
        "package_status": package_status,
        "runtime": str(RUNTIME.relative_to(ROOT)),
        "legacy_results_before": old_before,
        "legacy_results_after": old_after,
        "legacy_results_unchanged": old_before == old_after,
        "preparation": preparation,
        "step_status": status,
        "robust_run": robust_run,
        "environment": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "outputs": outputs,
        "output_file_count": len(outputs),
    }
    (AUDIT / "execution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        return 0
    complete = len(status) == len(STEPS) and all(code == 0 for code in status.values())
    isolated = old_before == old_after
    return 0 if complete and isolated else 1


if __name__ == "__main__":
    raise SystemExit(main())
