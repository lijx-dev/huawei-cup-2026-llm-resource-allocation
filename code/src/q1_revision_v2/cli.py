"""第一问 revision-v2 的单入口全量重算。"""
import argparse
from pathlib import Path

from q1.audit.inventory import expected_files

from . import audit, quality, conflict, mixture, integration, postprocess, report
from .common import load_config, metadata, paths, save_json, seed_all


def main():
    parser = argparse.ArgumentParser(description="从 A1–A16 原件完整重算第一问 q1-revision-v2")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--finalize-only", action="store_true", help="基于已完成的 v2 结果更新元数据与报告")
    args = parser.parse_args()
    root, out = paths(args.root)
    config = load_config(root)
    seed_all(config["seed"])
    if not args.finalize_only:
        from . import figures
        audit_result = audit.run(root, out, config)
        print("P0 完成", flush=True)
        quality.run(root, config)
        print("P1 完成", flush=True)
        conflict.run(root, config)
        print("P2 完成", flush=True)
        mixture.run(root, config, audit_result)
        postprocess.run(root, config)
        print("P3 完成", flush=True)
        integration.run(root, config, audit_result)
        print("P4 完成", flush=True)
        figures.run(root, config)
    raw = [one[0] for one in expected_files(root).values()]
    save_json(out / "metadata.json", metadata(root, config, raw))
    result = report.run(root, config)
    print(f"报告：{result}", flush=True)


if __name__ == "__main__":
    main()
