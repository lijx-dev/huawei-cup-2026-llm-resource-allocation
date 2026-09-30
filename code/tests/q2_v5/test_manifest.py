"""P00 测试仅使用测试目录中的人工微型文件。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from q2_v5.manifest import (REQUIRED_FORBIDDEN_PATHS, assert_allowed_input,
                            discover_inputs, run_p00, sha256_file)


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path
    (root / "data/real_attachments/B_scaling_laws").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "configs/q2_v5").mkdir(parents=True)
    (root / "data/real_attachments/B_scaling_laws/tiny.csv").write_text("x\n1\n")
    (root / "docs/算力约束下提升大语言模型能力的资源配置建模.docx").write_bytes(b"test-only")
    (root / "docs/数据说明.pdf").write_bytes(b"test-only")
    config = {
        "experiment_version": "q2_v5", "stage": "P00", "seed": 7,
        "input_roots": ["data/real_attachments/B_scaling_laws",
                        "docs/算力约束下提升大语言模型能力的资源配置建模.docx",
                        "docs/数据说明.pdf"],
        "forbidden_paths": sorted(REQUIRED_FORBIDDEN_PATHS),
        "output_root": "results/q2_v5/00_manifest",
    }
    path = root / "configs/q2_v5/config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return root, path


def test_sha256_known_bytes(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "input"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_discovery_rejects_outside_whitelist(tmp_path: Path) -> None:
    root, path = make_fixture(tmp_path)
    config = json.loads(path.read_text())
    assert len(discover_inputs(root, config)) == 3
    forbidden = root / "data/real_attachments/A_data_value"
    forbidden.mkdir()
    (forbidden / "bad.csv").write_text("secret")
    with pytest.raises(ValueError, match="白名单"):
        assert_allowed_input(forbidden / "bad.csv", root,
                             [root / name for name in config["input_roots"]])


def test_missing_required_input_fails(tmp_path: Path) -> None:
    root, path = make_fixture(tmp_path)
    (root / "docs/数据说明.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="必要输入缺失"):
        run_p00(root, path)
    assert not (root / "results/q2_v5/00_manifest").exists()


def test_manifest_hashes_and_snapshot(tmp_path: Path) -> None:
    root, path = make_fixture(tmp_path)
    manifest = run_p00(root, path)
    output = root / "results/q2_v5/00_manifest"
    hashes = json.loads((output / "input_hashes.json").read_text())
    artifacts = json.loads((output / "artifact_hashes.json").read_text())
    assert manifest["input_file_count"] == 3
    assert manifest["valid_record_count"] is None
    assert manifest["forbidden_path_policy"]["old_q2_read_allowed_before_freeze"] is False
    assert (output / "config_snapshot.yaml").read_bytes() == path.read_bytes()
    for name, value in hashes.items():
        assert value["sha256"] == sha256_file(root / name)
    for name, digest in artifacts.items():
        assert digest == sha256_file(output / name)


def test_no_old_q2_module_imports() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src/q2_v5"
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name.startswith(("q2_v2", "q2_v3", "q2_v4")) for name in imports)
