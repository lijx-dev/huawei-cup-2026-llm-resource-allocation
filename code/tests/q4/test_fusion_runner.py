"""人工构造的微型样例，仅检验复现运行器，不进入正式实验。"""

import ast
import json
from pathlib import Path
import sys
from zipfile import ZipFile


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/q4"))
from compare_outputs import compare_value  # noqa: E402
from run_fusion_package import (  # noqa: E402
    ARCHIVE, INPUTS, OUTPUTS, ROOT, STEPS, map_windows_path, package_root,
    verify_package,
)


def test_every_package_script_path_is_mapped():
    with ZipFile(ARCHIVE) as archive:
        prefix = package_root(archive)
        verify_package(archive, prefix)
        for filename in STEPS:
            source = archive.read(f"{prefix}/02_脚本/{filename}").decode("utf-8-sig")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith("d:\\F题\\"):
                        mapped = Path(map_windows_path(node.value))
                        assert mapped.is_relative_to(ROOT)
                        assert mapped.is_relative_to(ROOT / "data") or mapped.is_relative_to(INPUTS) or mapped.is_relative_to(OUTPUTS)


def test_unrecognized_package_path_fails():
    try:
        map_windows_path("d:\\F题\\unexpected\\data.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("未识别路径必须拒绝")


def test_comparison_detects_changed_statistic():
    changes = []
    compare_value({"s24": 76.71, "ok": True}, {"s24": 82.04, "ok": True},
                  "result", changes)
    assert changes and "s24" in changes[0]


def test_reported_24_month_interval_covers_24_month_point():
    result = json.loads((OUTPUTS / "q4_legs_reweight_results.json").read_text())
    for name in ("old", "main", "sens_uniform"):
        item = result[name]
        assert item["lo"] < item["s24"] < item["hi"]
