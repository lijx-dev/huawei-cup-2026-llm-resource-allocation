"""在仓库中原样执行问题四交付包脚本，并记录输入与复现差异。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "团队交付包_问题四融合方案.zip"
_run = Path(os.environ.get("Q4_RUN_DIR", "results/q4/fusion_reproduction"))
RUN = _run if _run.is_absolute() else ROOT / _run
INPUTS = RUN / "inputs"
OUTPUTS = RUN / "outputs"
LOGS = RUN / "logs"
LATEST_MODE = os.environ.get("Q4_LATEST_MODE") == "1"
STEPS = [
    "q4_evidence.py", "q4_extra_probe.py", "q4_c8_p90.py", "q4_caliber2.py",
    "q4_gN_trace.py", "q4_closure.py", "q4_closure2.py", "q4_ref_repro.py",
    "q4_legs.py", "q4_legs_reweight.py", "q4_bounded_forecast.py",
    "q4_p1_chinchilla_rho.py", "q4_p1b_dynamics.py", "q4_p1c_schaeffer.py",
    "q4_p1d_channels.py", "q4_p1e_musr_check.py", "q4_p2a_cluster.py",
    "q4_p2b_eligibility.py", "q4_p2c_backtest.py", "_q4_c1c8_probe.py",
]
DEPENDENCIES = {
    "q4_p1e_musr_check.py": ["q4_p1c_schaeffer.py"],
    "q4_p2c_backtest.py": ["q4_legs_reweight.py"],
}
COMPATIBILITY_PATCHES = {
    # 95 个 C8 目录有多个评测 JSON；选文件系统返回的第一个文件会随系统变化。
    # 固定取文件名时间戳最早的一份，截断的选中文件仍整份隔离。
    "c8_first_evaluation": (
        "fn = [f for f in os.listdir(dp) if f.endswith('.json')]",
        "fn = sorted(f for f in os.listdir(dp) if f.endswith('.json'))",
    ),
    # pandas 3 的季度列可同时含字符串和浮点缺失值；原脚本对两者排序报错。
    "p1e_missing_quarter": (
        "M = M[M.q.isin(sorted([q for q in M.q.unique() if q != 'NaT']))].copy()",
        "M = M[M.q.notna() & M.q.ne('NaT')].copy()",
    ),
    # 原脚本把 logistic24 返回的 s12 赋给标注为 24M 区间的 lo/hi。
    "reweight_24m_interval_lo": (
        "lo, _, _ = logistic24(g_comb - 1.645 * sd_comb)",
        "_, lo, _ = logistic24(g_comb - 1.645 * sd_comb)",
    ),
    "reweight_24m_interval_hi": (
        "hi, _, _ = logistic24(g_comb + 1.645 * sd_comb)",
        "_, hi, _ = logistic24(g_comb + 1.645 * sd_comb)",
    ),
    "legs_logit_interval_order": (
        "[{100/(1+np.exp(-hi_l)):6.2f},{100/(1+np.exp(-lo_l)):6.2f}]",
        "[{100/(1+np.exp(-lo_l)):6.2f},{100/(1+np.exp(-hi_l)):6.2f}]",
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_root(archive: ZipFile) -> str:
    roots = {name.split("/")[0] for name in archive.namelist() if name}
    if len(roots) != 1:
        raise ValueError(f"交付包根目录不唯一: {roots}")
    return roots.pop()


def verify_package(archive: ZipFile, prefix: str) -> dict:
    manifest = json.loads(archive.read(f"{prefix}/文件清单_含SHA256.json"))
    for item in manifest["files"]:
        rel = item["path"]
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(f"非法清单路径: {rel}")
        actual = sha256_bytes(archive.read(f"{prefix}/{rel}"))
        if actual.lower() != item["sha256"].lower():
            raise ValueError(f"包内 SHA-256 不符: {rel}")
    return manifest


def extract_inputs(archive: ZipFile, prefix: str) -> list[dict]:
    files = []
    for name in archive.namelist():
        base = f"{prefix}/04_接口数据/"
        if not name.startswith(base) or name.endswith("/"):
            continue
        rel = Path(name.removeprefix(base))
        if ".." in rel.parts:
            raise ValueError(f"非法输入路径: {rel}")
        data = archive.read(name)
        target = INPUTS / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        is_latest_override = LATEST_MODE and rel.name in {"Q1_M2响应接口.csv.gz", "Q1_M2接口_manifest.json"}
        if target.exists() and target.read_bytes() != data and not is_latest_override:
            raise ValueError(f"已提取接口文件被改动: {target}")
        target.write_bytes(data)
        files.append({"path": str(target.relative_to(ROOT)), "bytes": len(data),
                      "sha256": sha256_bytes(data)})
    if LATEST_MODE:
        # 旧脚本只读取 h_p_eq；由正式 LightGBM 接口生成同义兼容视图，不再使用旧二次响应表。
        formal_dir = ROOT / "results/q1_revision_v2_1/q2_interface"
        manifest = json.loads((formal_dir / "manifest.json").read_text())
        if manifest["status"] != "PASS":
            raise RuntimeError("Q1 正式接口未通过")
        frame = __import__("pandas").read_csv(ROOT / manifest["mixture_response_path"])
        compat = frame[["split", "index", "h_agg", "support_status", "nearest_train_distance"]].copy()
        compat = compat.rename(columns={"index": "recipe_id", "h_agg": "h_p_eq"})
        compat = __import__("pandas").concat([__import__("pandas").DataFrame([{
            "split": "p0 (formal reference)", "recipe_id": 0, "h_p_eq": 0.0,
            "support_status": "within_support", "nearest_train_distance": 0.0,
        }]), compat], ignore_index=True)
        target = INPUTS / "Q1_M2响应接口.csv.gz"
        compat.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
        latest_manifest = {
            "version": "q4-latest-q1-compat-v1", "source_manifest": manifest,
            "compatibility_columns": list(compat.columns), "rows": len(compat),
            "policy": "h_p_eq is the formal LightGBM h_agg; old quadratic coefficients are not used",
        }
        (INPUTS / "Q1_M2接口_manifest.json").write_text(
            json.dumps(latest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in (target, INPUTS / "Q1_M2接口_manifest.json"):
            record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": sha256_bytes(path.read_bytes()), "latest_override": True}
            files = [item for item in files if item["path"] != record["path"]]
            files.append(record)
    return files


def apply_latest_patches(source: str, name: str) -> str:
    if not LATEST_MODE:
        return source
    params = json.loads((ROOT / "results/q2_fusion_rerun_v1/quality_model/b6_parameters.json").read_text())
    mq, q0 = params["MQ"], params["q0"]
    b_eff = mq["B"] * np.exp(mq["gamma"] * q0)
    mechanism = json.loads((ROOT / "results/q3_latest_20260926/mechanism_elasticity.json").read_text())
    eps_c = float(mechanism["epsilon_C"])
    k_br, g_c, sig_br = -2.7353, 1.2875, .5825
    g_mech = k_br * eps_c * g_c
    sd_mech = float(np.sqrt((abs(eps_c)*g_c*sig_br)**2 + (abs(k_br)*g_c*.02)**2))
    if name == "q4_p1b_dynamics.py":
        old = ("SET_A = dict(E=1.71120, A=0.65190, alpha=0.27830, B=1.42070, beta=0.28340,\n"
               "             rhoN=0.35550, rhoD=0.14280, E1=0.10270)")
        new = (f"SET_A = dict(E={mq['E']!r}, A={mq['A']!r}, alpha={mq['alpha']!r}, "
               f"B={b_eff!r}, beta={mq['beta']!r},\n"
               f"             rhoN=0.0, rhoD={mq['gamma']!r}, E1=0.0)")
        if old not in source:
            raise ValueError("找不到 P1-B 旧 SET_A")
        source = source.replace(old, new)
        source = source.replace("span_support_constrained=0.159", "span_support_constrained=0.333013")
        source = source.replace("支持约束下修正后的极差为 **0.159**", "正式接口候选极差为 **0.333013**")
        source = source.replace("|ΔlnL| ≲ 0.159", "|ΔlnL| ≲ 0.333013")
    if name == "q4_p1d_channels.py":
        source = source.replace("hp_span_support = 0.159", "hp_span_support = 0.333013")
        source = source.replace("支持约束下极差（问题三修正）", "正式接口候选极差（问题三最新版）")
    if name == "q4_evidence.py":
        source = source.replace("E_P2 = 1.71120", f"E_P2 = {mq['E']!r}")
    if name == "q4_legs.py":
        source = source.replace("eps_C = -0.1404134235356952", f"eps_C = {eps_c!r}")
    if name == "q4_legs_reweight.py":
        source = source.replace("('腿1 机制', 0.49449377814887835, 0.1266895063459456, 'model')",
                                f"('腿1 机制', {g_mech!r}, {sd_mech!r}, 'model')")
        source = source.replace("s24_robust_lower=r_all['s24'],",
            "s24_robust_lower=min(r_main['s24'], r_all['s24']),\n"
            "                             s24_robust_upper=max(r_main['s24'], r_all['s24']),")
    if name == "q4_bounded_forecast.py":
        source = source.replace("g_mech, sd_mech = 0.4945, 0.1267",
                                f"g_mech, sd_mech = {g_mech!r}, {sd_mech!r}")
    return source


def map_windows_path(value: str) -> str:
    base = "d:\\F题\\"
    if not value.startswith(base):
        return value
    rel = value[len(base):].split("\\")
    if rel[:2] == ["F题", "real_attachments"]:
        return str(ROOT / "data/real_attachments" / Path(*rel[2:]))
    if rel[:1] == ["q1_quality_results"]:
        return str(INPUTS / Path(*rel[1:]))
    if rel[:4] == ["_team4_handoff", "outputs", "第四问交接包", "结果表"]:
        return str(INPUTS / "队友交接输入" / Path(*rel[4:]))
    if rel == ["q4_p2c_protocol.md"]:
        return str(INPUTS / "q4_p2c_protocol.md")
    if len(rel) == 1 and rel[0].startswith("q4_"):
        return str(OUTPUTS / rel[0])
    raise ValueError(f"未映射的交付包路径: {value}")


class PortablePaths(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(map_windows_path(node.value)), node)
        return node


def run_step(name: str) -> None:
    if name not in STEPS:
        raise ValueError(f"未知脚本: {name}")
    with ZipFile(ARCHIVE) as archive:
        prefix = package_root(archive)
        verify_package(archive, prefix)
        source = archive.read(f"{prefix}/02_脚本/{name}").decode("utf-8-sig")
    source = apply_latest_patches(source, name)
    first, replacement = COMPATIBILITY_PATCHES["c8_first_evaluation"]
    if first in source:
        source = source.replace(first, replacement)
    if name == "q4_p1e_musr_check.py":
        first, replacement = COMPATIBILITY_PATCHES["p1e_missing_quarter"]
        if source.count(first) != 1:
            raise ValueError("P1-E 兼容修正的原始代码位置已改变")
        source = source.replace(first, replacement)
    if name == "q4_legs_reweight.py":
        for key in ("reweight_24m_interval_lo", "reweight_24m_interval_hi"):
            first, replacement = COMPATIBILITY_PATCHES[key]
            if source.count(first) != 1:
                raise ValueError(f"三腿重估 {key} 的原始代码位置已改变")
            source = source.replace(first, replacement)
    if name == "q4_legs.py":
        first, replacement = COMPATIBILITY_PATCHES["legs_logit_interval_order"]
        if source.count(first) != 1:
            raise ValueError("三腿原始 logit 区间的代码位置已改变")
        source = source.replace(first, replacement)
    tree = ast.fix_missing_locations(PortablePaths().visit(ast.parse(source, filename=name)))
    code = compile(tree, f"{ARCHIVE.name}!/02_脚本/{name}", "exec")
    exec(code, {"__name__": "__main__", "__file__": name})


def input_inventory() -> dict:
    files = []
    base = ROOT / "data/real_attachments"
    for folder in ["B_scaling_laws", "C_efficiency_evolution"]:
        for path in sorted((base / folder).rglob("*")):
            if not path.is_file():
                continue
            # 对模型评测详情保存整树的汇总校验；不把原始 JSON 复制到结果区。
            data_hash = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    data_hash.update(block)
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": data_hash.hexdigest()})
    return {"files": files, "count": len(files),
            "total_bytes": sum(f["bytes"] for f in files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=STEPS)
    args = parser.parse_args()
    if args.script:
        run_step(args.script)
        return 0
    for folder in [RUN, INPUTS, OUTPUTS, LOGS]:
        folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        prefix = package_root(archive)
        manifest = verify_package(archive, prefix)
        interface_files = extract_inputs(archive, prefix)
    inventory = input_inventory()
    (RUN / "input_inventory.json").write_text(json.dumps({
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256_bytes(ARCHIVE.read_bytes()),
        "package_manifest_file_count": len(manifest["files"]),
        "interface_files": interface_files,
        "attachments": inventory,
        "python": sys.version, "numpy": __import__("numpy").__version__,
        "pandas": __import__("pandas").__version__,
        "scipy": __import__("scipy").__version__,
        "compatibility_patches": COMPATIBILITY_PATCHES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {}
    for name in STEPS:
        if any(status.get(dep) != 0 for dep in DEPENDENCIES.get(name, [])):
            status[name] = "skipped_dependency"
            print(f"SKIP {name}: 依赖失败", flush=True)
            continue
        print(f"RUN {name}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", "0")
        with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, __file__, "--script", name],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, check=False)
        status[name] = result.returncode
        print(f"DONE {name}: exit={result.returncode}", flush=True)
        (RUN / "execution_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(x == 0 for x in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
