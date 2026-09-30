"""审计 Q4 原始输入的完整性及 C8 JSON 选择状态。"""

import json
from pathlib import Path

try:
    from .run_fusion_package import ROOT, RUN
except ImportError:  # 兼容直接执行 ``python src/q4/audit_inputs.py``。
    from run_fusion_package import ROOT, RUN


def main() -> int:
    base = ROOT / "data/real_attachments"
    source_manifest = json.loads((base / "source_manifest.json").read_text())
    size_differences = []
    manifest_files = 0
    for item in source_manifest:
        if item.get("problem") not in ("B", "C"):
            continue
        manifest_files += 1
        path = base / item["file"]
        if path.is_file():
            actual = path.stat().st_size
        elif path.is_dir():
            actual = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        else:
            actual = None
        if actual != item.get("bytes"):
            size_differences.append({"path": item["file"], "expected": item.get("bytes"),
                                     "actual": actual})

    detail = base / "C_efficiency_evolution/detailed_results"
    folders = sorted(p for p in detail.iterdir() if p.is_dir())
    physical_bad = []
    selected_bad = []
    multi_file = 0
    json_count = 0
    selected_valid = 0
    for folder in folders:
        files = sorted(folder.glob("*.json"))
        json_count += len(files)
        multi_file += len(files) > 1
        if not files:
            selected_bad.append({"directory": folder.name, "reason": "no_json"})
            continue
        for file in files:
            try:
                json.loads(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                physical_bad.append({"path": str(file.relative_to(base)), "error": type(exc).__name__})
                if file == files[0]:
                    selected_bad.append({"directory": folder.name,
                                         "path": str(file.relative_to(base)),
                                         "error": type(exc).__name__})
        if not any(row["directory"] == folder.name for row in selected_bad):
            selected_valid += 1
    result = {
        "source_manifest_BC_files": manifest_files,
        "source_manifest_size_mismatches": size_differences,
        "c8": {"directories": len(folders), "json_files": json_count,
               "multiple_json_directories": multi_file,
               "physical_invalid_count": len(physical_bad),
               "physical_invalid": physical_bad,
               "selected_rule": "文件名排序后首份 JSON；若该文件不可解析，则隔离该目录",
               "selected_valid_count": selected_valid,
               "selected_invalid_count": len(selected_bad),
               "selected_invalid": selected_bad},
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"manifest_files": manifest_files,
                      "manifest_size_mismatches": len(size_differences),
                      "c8": {key: value for key, value in result["c8"].items()
                             if key.endswith("count") or key in ("directories", "json_files",
                                                              "multiple_json_directories")}},
                     ensure_ascii=False))
    return 0 if not size_differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
