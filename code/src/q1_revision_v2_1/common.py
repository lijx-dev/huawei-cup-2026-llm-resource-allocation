"""路径、种子、哈希和版本元数据。"""
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np


def paths(root):
    root = Path(root).resolve()
    return root, root / "results/q1_revision_v2_1", root / "results/q1_revision_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(root):
    return json.loads((Path(root) / "configs/q1_revision_v2_1/config.json").read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def file_manifest(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def verify_file_manifest(folder, reference):
    current = file_manifest(folder)
    if current != reference:
        lost = sorted(set(reference) - set(current))
        new = sorted(set(current) - set(reference))
        changed = sorted(k for k in set(current) & set(reference) if current[k] != reference[k])
        raise RuntimeError(f"v2 历史结果已变化: missing={lost}, new={new}, changed={changed}")


def metadata(root, config, raw_paths, inherited_hashes):
    import pandas
    import sklearn
    import lightgbm
    import scipy
    source = sorted((Path(root) / "src/q1_revision_v2_1").glob("*.py"))
    source += [Path(root) / "src" / name for name in (
        "q1_revision_v2/quality.py", "q1_revision_v2/mixture.py",
        "q1/mixture/dataset.py", "q1/mixture/evaluation.py",
        "q1/mixture/perturbation.py", "q1/audit/inventory.py")]
    code = {str(p.relative_to(root)): sha256(p) for p in source}
    digest = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    return {"experiment_version": config["experiment_version"], "random_seed": config["seed"],
            "config_sha256": sha256(Path(root) / "configs/q1_revision_v2_1/config.json"),
            "input_file_sha256": {str(Path(p).relative_to(root)): sha256(p) for p in raw_paths},
            "read_v2_result_sha256": inherited_hashes,
            "code_sha256": code, "code_commit_or_hash": digest,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "pandas_version": pandas.__version__, "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__}
