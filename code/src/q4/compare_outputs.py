"""将本地 Q4 实验输出与交付包中冻结的机读结果逐项对账。"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

from run_fusion_package import ARCHIVE, OUTPUTS, RUN, package_root, verify_package


def compare_value(expected, actual, location: str, differences: list, limit: int = 8) -> None:
    if len(differences) >= limit:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                differences.append(f"{location}.{key}: 键缺失")
            else:
                compare_value(expected[key], actual[key], f"{location}.{key}", differences)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(f"{location}: 长度 {len(expected)} != {len(actual)}")
            return
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_value(left, right, f"{location}[{i}]", differences)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if math.isnan(expected) and math.isnan(actual):
            return
        if math.isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-6):
            return
    elif expected == actual:
        return
    differences.append(f"{location}: {expected!r} != {actual!r}")


def parse_csv(data: bytes) -> list:
    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def compare_csv(expected: list, actual: list, differences: list) -> None:
    if len(expected) != len(actual):
        differences.append(f"行数 {len(expected)} != {len(actual)}")
        return
    for row_idx, (left_row, right_row) in enumerate(zip(expected, actual)):
        if len(left_row) != len(right_row):
            differences.append(f"第 {row_idx} 行列数不同")
            continue
        for col_idx, (left, right) in enumerate(zip(left_row, right_row)):
            try:
                l, r = float(left), float(right)
            except ValueError:
                if left != right:
                    differences.append(f"[{row_idx},{col_idx}]: {left!r} != {right!r}")
            else:
                if not (math.isnan(l) and math.isnan(r)) and not math.isclose(l, r, abs_tol=1e-6, rel_tol=1e-6):
                    differences.append(f"[{row_idx},{col_idx}]: {left!r} != {right!r}")
            if len(differences) >= 8:
                return


def main() -> int:
    results = []
    with ZipFile(ARCHIVE) as archive:
        prefix = package_root(archive)
        verify_package(archive, prefix)
        frozen = sorted(name for name in archive.namelist()
                        if name.startswith(f"{prefix}/03_结果数据/") and not name.endswith("/"))
        for name in frozen:
            basename = Path(name).name
            target = OUTPUTS / basename
            record = {"file": basename, "status": "missing", "differences": []}
            if target.exists():
                expected_data = archive.read(name)
                actual_data = target.read_bytes()
                if basename.endswith(".json"):
                    left = json.loads(expected_data)
                    right = json.loads(actual_data)
                    if basename == "q4_p2c_backtest_results.json":
                        # 协议文件绝对路径随平台变化；协议哈希仍逐字段比较。
                        for obj in (left, right):
                            obj["protocol"]["file"] = obj["protocol"]["file"].replace("\\", "/").split("/")[-1]
                    compare_value(left, right,
                                  basename, record["differences"])
                elif basename.endswith(".csv"):
                    left, right = parse_csv(expected_data), parse_csv(actual_data)
                    if basename == "q4_p2a_clusters.csv":
                        # 发布者聚类按 n 降序；并列 n 的顺序由 pandas 版本决定。
                        left = [left[0]] + sorted(left[1:], key=lambda row: row[0])
                        right = [right[0]] + sorted(right[1:], key=lambda row: row[0])
                    compare_csv(left, right,
                                record["differences"])
                record["status"] = "match" if not record["differences"] else "mismatch"
            results.append(record)
    report = {"total": len(results),
              "match": sum(r["status"] == "match" for r in results),
              "mismatch": sum(r["status"] == "mismatch" for r in results),
              "missing": sum(r["status"] == "missing" for r in results),
              "files": results}
    path = RUN / "comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total", "match", "mismatch", "missing")}, ensure_ascii=False))
    for row in results:
        if row["status"] != "match":
            print(row["file"], row["status"], row["differences"][:3])
    return 0 if report["mismatch"] == 0 and report["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
