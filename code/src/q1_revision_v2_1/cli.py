"""v2.1 全量复现实验入口；历史 v2 始终只读。"""
import argparse
import json
from pathlib import Path

from q1.audit.inventory import expected_files
from q1_revision_v2.common import load_config as load_v2_config

from . import audit, quality, conflict, mixture_validation, q_mapping, report
from .common import load_config, paths, seed_all, metadata, save_json, verify_file_manifest


def main():
    parser=argparse.ArgumentParser(description="第一问 q1-revision-v2.1 完整稳健性验证")
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--finalize-only",action="store_true",help="基于已有 v2.1 产物更新元数据和报告")
    args=parser.parse_args()
    root,out,old=paths(args.root)
    config=load_config(root)
    old_config=load_v2_config(root)
    seed_all(config["seed"])
    if not args.finalize_only:
        audit.run(root,config,old_config)
        print("Phase 0 字段审计完成",flush=True)
        quality.run(root,config,old_config)
        print("Phase 1–2 分层质量评分完成",flush=True)
        conflict.run(root,config)
        print("Phase 3–4 三种指标分歧完成",flush=True)
        mixture_validation.run(root,config,old_config)
        print("Phase 5–7 冻结模型补充验证完成",flush=True)
        q_mapping.run(root,config,old_config)
        print("Phase 8–10 coverage 与 1000 次 placebo 完成",flush=True)
        from . import figures
        figures.run(root,config)
        print("Phase 11 图表完成",flush=True)
    old_hashes=json.loads((out / "audit/v2_result_hashes_before.json").read_text())
    verify_file_manifest(old,old_hashes)
    raw=[one[0] for one in expected_files(root).values()]
    save_json(out / "metadata.json",metadata(root,config,raw,old_hashes))
    result=report.run(root,config)
    print(f"报告：{result}",flush=True)


if __name__=="__main__":
    main()
