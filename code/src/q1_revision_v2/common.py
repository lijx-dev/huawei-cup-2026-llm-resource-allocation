"""版本、路径、校验与固定随机性。"""
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np


def paths(root):
    root = Path(root).resolve()
    return root, root / "results/q1_revision_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(root):
    return json.loads((Path(root) / "configs/q1_revision_v2/config.json").read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metadata(root, config, raw_paths):
    import pandas
    import scipy
    import sklearn
    import lightgbm
    files = sorted((Path(root) / "src/q1_revision_v2").glob("*.py"))
    files += [Path(root) / "src/q1" / rel for rel in (
        "audit/inventory.py", "audit/tables.py", "mixture/dataset.py",
        "mixture/perturbation.py", "mixture/evaluation.py")]
    code_hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    code_digest = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()
    return {"experiment_version": config["experiment_version"], "random_seed": config["seed"],
            "input_file_sha256": {str(Path(p).relative_to(root)): sha256(p) for p in raw_paths},
            "config_sha256": sha256(Path(root) / "configs/q1_revision_v2/config.json"),
            "design_sha256": sha256(Path(root) / "第一问方案修订稿.md"),
            "requirements_sha256": sha256(Path(root) / "requirements.txt"),
            "source_manifest_sha256": sha256(Path(root) / "data/real_attachments/source_manifest.json"),
            "code_sha256": code_hashes, "code_commit_or_hash": code_digest,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "pandas_version": pandas.__version__, "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__}
