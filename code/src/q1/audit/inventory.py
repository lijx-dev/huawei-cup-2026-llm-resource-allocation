"""附件定位、文件摘要与来源清单对照。"""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(root):
    base = root / "data/real_attachments/A_data_value"
    files = {"A1": [base / "slimpajama_quality_signal_sample.jsonl.xz"],
             "A2": sorted((base / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz")),
             "A3": sorted((base / "slimpajama_quality_extended").glob("github_*.jsonl.xz"))}
    names = ["train_mixture_1m", "train_pile_loss_1m", "test_mixture_1m",
             "test_pile_loss_1m", "test_mixture_60m", "test_pile_loss_60m",
             "test_mixture_1B", "test_pile_loss_1B", "est_mixture_10b",
             "est_pile_loss_10b", "est_mixture_70b", "est_pile_loss_70b"]
    for number, name in enumerate(names, 4):
        files[f"A{number}"] = [base / "regmix_tables" / f"{name}.csv"]
    files["A16"] = [base / "domain_mapping_guide.csv"]
    return files


def manifest_entries(root):
    path = root / "data/real_attachments/source_manifest.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["file"]: entry for entry in entries if entry.get("problem") == "A"}


def file_inventory(root, files, manifest):
    rows = []
    for attachment, paths in files.items():
        if not paths:
            rows.append(dict(attachment=attachment, source_file="", exists=False, bytes="",
                             sha256="", manifest_bytes="", manifest_match="", read_status="missing"))
        for path in paths:
            rel = path.relative_to(root).as_posix()
            key = path.relative_to(root / "data/real_attachments").as_posix()
            entry = manifest.get(key, {})
            exists = path.is_file()
            size = path.stat().st_size if exists else ""
            rows.append(dict(attachment=attachment, source_file=rel, exists=exists,
                             bytes=size, sha256=sha256_file(path) if exists else "",
                             manifest_bytes=entry.get("bytes", ""),
                             manifest_match=(size == entry["bytes"]) if exists and "bytes" in entry else "",
                             read_status="pending" if exists else "missing"))
    return rows
