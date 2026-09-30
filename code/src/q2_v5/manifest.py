"""P00：生成可复核的 clean-room 输入和运行环境清单。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "numpy", "scipy", "pandas", "scikit-learn", "lightgbm",
    "matplotlib", "joblib", "pytest",
)
PERMITTED_INPUT_ROOTS = {
    "data/real_attachments/B_scaling_laws",
    "docs/算力约束下提升大语言模型能力的资源配置建模.docx",
    "docs/数据说明.pdf",
}
REQUIRED_FORBIDDEN_PATHS = {
    "data/real_attachments/A_data_value", "src/q2_v2", "src/q2_v3",
    "src/q2_v4", "results/q2_scaling_v1", "results/q2_scaling_v2",
    "results/q2_v3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    # JSON 是 YAML 的合法子集；仅使用标准库，避免 P00 新增依赖。
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_version") != "q2_v5" or config.get("stage") != "P00":
        raise ValueError("配置必须明确指定 q2_v5 / P00")
    if config.get("seed") != 7:
        raise ValueError("P00 统一 seed 必须为 7")
    if set(config.get("input_roots", [])) != PERMITTED_INPUT_ROOTS:
        raise ValueError("P00 输入白名单与预先固定的 B 附件及文档不一致")
    if not REQUIRED_FORBIDDEN_PATHS.issubset(set(config.get("forbidden_paths", []))):
        raise ValueError("P00 禁读路径策略不完整")
    if config.get("output_root") != "results/q2_v5/00_manifest":
        raise ValueError("P00 输出目录不符合约定")
    return config


def assert_allowed_input(path: Path, root: Path, allowed_roots: list[Path]) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"输入不在仓库内: {path}")
    if not any(
        resolved_path == allowed or resolved_path.is_relative_to(allowed)
        for allowed in allowed_roots
    ):
        raise ValueError(f"输入未获白名单授权: {path}")


def discover_inputs(root: Path, config: dict[str, Any]) -> list[Path]:
    allowed = [root / name for name in config["input_roots"]]
    paths: list[Path] = []
    for entry in allowed:
        if not entry.exists():
            raise FileNotFoundError(f"必要输入缺失: {entry}")
        if entry.is_dir():
            paths.extend(path for path in entry.rglob("*") if path.is_file())
        elif entry.is_file():
            paths.append(entry)
        else:
            raise ValueError(f"输入类型不支持: {entry}")
    for path in paths:
        assert_allowed_input(path, root, allowed)
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short", "--branch", "--untracked-files=normal")
    except FileNotFoundError:
        return {"available": False, "commit": None, "status": None,
                "reason": "git executable unavailable"}
    if commit.returncode or status.returncode:
        return {"available": False, "commit": None, "status": None,
                "reason": (commit.stderr or status.stderr).strip()}
    return {"available": True, "commit": commit.stdout.strip(),
            "status": status.stdout.strip(), "reason": None}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_p00(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    if not config_path.is_relative_to(root / "configs" / "q2_v5"):
        raise ValueError("配置文件必须位于 configs/q2_v5")
    config = load_config(config_path)
    output = root / config["output_root"]
    if not output.resolve().is_relative_to(root / "results" / "q2_v5"):
        raise ValueError("P00 输出必须位于 results/q2_v5")
    inputs = discover_inputs(root, config)
    input_hashes = {
        path.relative_to(root).as_posix(): {"size_bytes": path.stat().st_size,
                                            "sha256": sha256_file(path)}
        for path in inputs
    }
    source = Path(__file__).resolve()
    cli = source.with_name("cli.py")
    source_name = source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
    cli_name = cli.relative_to(root).as_posix() if cli.is_relative_to(root) else str(cli)
    manifest = {
        "experiment_version": "q2_v5",
        "stage": "P00",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "git": git_metadata(root),
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "code_sha256": {
            source_name: sha256_file(source),
            cli_name: sha256_file(cli),
        },
        "input_roots": config["input_roots"],
        "input_file_count": len(inputs),
        "input_hashes_file": "input_hashes.json",
        "forbidden_path_policy": {
            "paths": config["forbidden_paths"],
            "old_q2_read_allowed_before_freeze": False,
            "old_q2_import_allowed": False,
            "a_raw_read_allowed": False,
            "enforcement": "P00 输入白名单校验；后续阶段仍须各自强制校验",
            "verification_limit": "普通文件系统无法证明此前不存在其他进程读取旧文件",
        },
        "data_audit": "未运行；P01 执行",
        "valid_record_count": None,
        "quarantined_record_count": None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_bytes(config_path.read_bytes())
    write_json(output / "input_hashes.json", input_hashes)
    write_json(output / "manifest.json", manifest)
    git_note = ("可用" if manifest["git"]["available"] else
                f"不可用：{manifest['git']['reason']}")
    report = (
        "# Q2 V5 P00 Clean-room manifest\n\n"
        f"- 阶段结果：PASS（输入清单与哈希已生成；P01 未运行）\n"
        f"- 输入文件数：{len(inputs)}\n"
        f"- Git：{git_note}\n"
        "- 环境：沿用仓库现有 Python 环境，未创建隔离环境。\n"
        "- 旧 Q2：本阶段未读取旧实现或结果；新代码仅扫描配置白名单。"
        "旧文件是否曾被其他进程读取，普通文件系统不可证。\n"
        "- 数据记录数、隔离数：未知；P01 审计。\n"
        "- 证据边界：哈希证明本次输入字节版本，不证明数据来源真实性。\n"
    )
    (output / "P00_report.md").write_text(report, encoding="utf-8")
    artifact_paths = ("config_snapshot.yaml", "input_hashes.json", "manifest.json", "P00_report.md")
    write_json(output / "artifact_hashes.json",
               {name: sha256_file(output / name) for name in artifact_paths})
    return manifest
