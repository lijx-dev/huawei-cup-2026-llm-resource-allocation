"""用冻结的 1M LightGBM 在训练配方支持区域内搜索预测最优配比。"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from q1.mixture.dataset import fields, load_pair, sha256
from q1.mixture.models import predict
from q1.mixture.perturbation import check_simplex, pair, single


SEARCH_VERSION = "q1-m3-mixture-search-v1"


def _context(root):
    root = Path(root).resolve()
    meta_path = root / "results/q1/mixture/mixture_model_metadata.json"
    model_path = root / "results/q1/mixture/frozen_models.joblib"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if sha256(model_path) != metadata["frozen_model_sha256"]:
        raise RuntimeError("冻结模型校验值与 P3 元数据不一致")
    saved = joblib.load(model_path)
    mixture_fields, loss_fields = fields(root)
    if saved["mixture_fields"] != mixture_fields or saved["loss_fields"] != loss_fields:
        raise RuntimeError("冻结模型的领域顺序与审计模式不一致")
    cfg = json.loads((root / "configs/q1/mixture.json").read_text(encoding="utf-8"))
    ids, train_x, train_y, audit = load_pair(root, "train_1m", mixture_fields, loss_fields, cfg["sum_tolerance"])
    if audit["mixture_sha256"] != metadata["input_sha256"]["train_1m"]["mixture"]:
        raise RuntimeError("训练配比与 P3 冻结时使用的输入不一致")
    return root, metadata, saved, cfg, ids, train_x, train_y


def _weights(weights, n_targets):
    if weights is None:
        return np.full(n_targets, 1 / n_targets)
    result = np.asarray(weights, dtype=float)
    if result.shape != (n_targets,) or not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
        raise ValueError(f"目标权重必须是 {n_targets} 个有限非负数，且至少一个为正")
    return result / result.sum()


def support_mask(candidates, train_x, threshold, batch_size=4096):
    """沿用 P3 的最近邻阈值与训练观察到的各分量上界。"""
    candidate = np.asarray(candidates, dtype=float)
    check_simplex(candidate)
    if not 0 < threshold or train_x.shape[1] != candidate.shape[1]:
        raise ValueError("支持区域输入维度或阈值非法")
    max_share = train_x.max(axis=0)
    within_max = np.all(candidate <= max_share + 1e-12, axis=1)
    nearest = np.empty(len(candidate))
    for start in range(0, len(candidate), batch_size):
        stop = min(start + batch_size, len(candidate))
        nearest[start:stop] = cdist(candidate[start:stop], train_x).min(axis=1)
    return within_max & (nearest <= threshold), nearest


def transfer(p, target, donor, amount):
    """从 donor 向 target 精确转移份额，保持其余 15 个领域不变。"""
    vector = np.asarray(p, dtype=float)
    if vector.ndim != 1 or target == donor or not 0 <= target < len(vector) or not 0 <= donor < len(vector):
        raise ValueError("领域索引非法")
    check_simplex(vector[None, :])
    if amount <= 0 or vector[donor] + 1e-12 < amount:
        raise ValueError("供给领域份额不足或幅度非法")
    changed = vector.copy()
    changed[target] += amount
    changed[donor] -= amount
    changed[donor] = max(changed[donor], 0.0)
    check_simplex(changed[None, :])
    return changed


def predict_loss(p, root=".", weights=None, require_support=True):
    """返回给定 17 维配比的 13 项 Loss 预测和加权综合预测。"""
    _, metadata, saved, _, _, train_x, _ = _context(root)
    mixture_fields, loss_fields = saved["mixture_fields"], saved["loss_fields"]
    vector = np.asarray(p, dtype=float)
    if vector.shape != (len(mixture_fields),):
        raise ValueError(f"P 必须严格按模型顺序给出 {len(mixture_fields)} 个份额")
    check_simplex(vector[None, :])
    weights_array = _weights(weights, len(loss_fields))
    supported, distance = support_mask(vector[None, :], train_x, metadata["support_threshold"])
    if require_support and not supported[0]:
        raise ValueError("P 超出 P3 定义的训练支持区域；可设 require_support=False 查看探索性预测")
    predicted = predict(saved["models"], "lightgbm", vector[None, :], saved["reference_index"])[0]
    return {"model_version": metadata["model_version"], "prediction_role": "model_prediction",
            "supported": bool(supported[0]), "nearest_train_distance": float(distance[0]),
            "support_threshold": float(metadata["support_threshold"]),
            "P": dict(zip((name.removeprefix("train_the_pile_") for name in mixture_fields), map(float, vector))),
            "predicted_loss_by_domain": dict(zip(loss_fields, map(float, predicted))),
            "predicted_weighted_loss": float(predicted @ weights_array),
            "target_weights": dict(zip(loss_fields, map(float, weights_array)))}


def find_best_mixture(root=".", weights=None, single_deltas=(0.01, 0.03, 0.05), pair_delta=0.03,
                      transfer_steps=(0.001, 0.005, 0.01, 0.03, 0.05), local_rounds=5):
    """枚举训练配方及支持内扰动，再对当前最优点做确定性坐标转移搜索。

    返回本次有限候选搜索的最优预测点；不提供全局最优保证。
    """
    root, metadata, saved, cfg, ids, train_x, train_y = _context(root)
    mixture_fields, loss_fields = saved["mixture_fields"], saved["loss_fields"]
    objective_weights = _weights(weights, len(loss_fields))
    if local_rounds < 0 or not isinstance(local_rounds, int):
        raise ValueError("local_rounds 必须为非负整数")
    if pair_delta <= 0 or any(d <= 0 for d in single_deltas) or any(d <= 0 for d in transfer_steps):
        raise ValueError("搜索扰动幅度必须为正")
    best = None
    counts = {"candidate_rows": 0, "supported_rows": 0, "prediction_rows": 0}
    threshold = float(metadata["support_threshold"])

    def consider(candidates, origin):
        nonlocal best
        if not len(candidates):
            return
        support, dist = support_mask(candidates, train_x, threshold)
        counts["candidate_rows"] += len(candidates)
        counts["supported_rows"] += int(support.sum())
        valid = candidates[support]
        if not len(valid):
            return
        predictions = predict(saved["models"], "lightgbm", valid, saved["reference_index"])
        scores = predictions @ objective_weights
        counts["prediction_rows"] += len(valid)
        winner = int(np.argmin(scores))
        if best is None or scores[winner] < best["score"] - 1e-12:
            best = {"P": valid[winner].copy(), "score": float(scores[winner]),
                    "predicted_loss": predictions[winner].copy(),
                    "nearest_train_distance": float(dist[support][winner]), "origin": origin}

    # 只从 A4/A5 出发；检验集与外推估算表均不进入候选、调参或目标值。
    consider(train_x, "observed_training_recipe")
    best_train = best.copy()
    for delta in single_deltas:
        for j, domain in enumerate(mixture_fields):
            candidates, feasible = single(train_x, j, delta)
            consider(candidates[feasible], f"single:{domain}:{delta}")
    for j in range(len(mixture_fields)):
        for k in range(j + 1, len(mixture_fields)):
            candidates, feasible = pair(train_x, j, k, pair_delta, pair_delta)
            consider(candidates[feasible], f"pair:{mixture_fields[j]}:{mixture_fields[k]}:{pair_delta}")
    for round_index in range(local_rounds):
        before = best["score"]
        proposals = []
        for j in range(len(mixture_fields)):
            for k in range(len(mixture_fields)):
                if j == k:
                    continue
                for amount in transfer_steps:
                    if best["P"][k] >= amount:
                        proposals.append(transfer(best["P"], j, k, amount))
        if proposals:
            consider(np.asarray(proposals), f"local_transfer_round_{round_index + 1}")
        if best["score"] >= before - 1e-12:
            break
    names = [name.removeprefix("train_the_pile_") for name in mixture_fields]
    result = {"search_version": SEARCH_VERSION, "model_version": metadata["model_version"],
              "prediction_role": "model_prediction", "objective": "weighted_mean_13_validation_loss",
              "objective_weights": dict(zip(loss_fields, map(float, objective_weights))),
              "P": dict(zip(names, map(float, best["P"]))),
              "predicted_loss_by_domain": dict(zip(loss_fields, map(float, best["predicted_loss"]))),
              "predicted_weighted_loss": best["score"], "P_sum": float(best["P"].sum()),
              "search_best_origin": best["origin"], "nearest_train_distance": best["nearest_train_distance"],
              "support_threshold": threshold, "support_rule": metadata["support_rule"],
              "candidate_counts": counts,
              "best_training_recipe_predicted_weighted_loss": best_train["score"],
              "best_training_recipe_P": dict(zip(names, map(float, best_train["P"]))),
              "model_sha256": metadata["frozen_model_sha256"],
              "training_mixture_sha256": metadata["input_sha256"]["train_1m"]["mixture"],
              "optimizer_sha256": sha256(Path(__file__)),
              "search_settings": {"single_deltas": list(single_deltas), "pair_delta": pair_delta,
                                  "transfer_steps": list(transfer_steps), "local_rounds": local_rounds},
              "limitation": "有限候选中预测 Loss 最低；树模型非凸且分段常值，不保证全局最优；新配方未经过真实训练验证"}
    return result


def write_result(root=".", **kwargs):
    result = find_best_mixture(root, **kwargs)
    out = Path(root) / "results/q1/mixture"
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimal_mixture.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{"domain": domain, "share": share} for domain, share in result["P"].items()]).to_csv(out / "optimal_mixture.csv", index=False, float_format="%.17g")
    p_rows = "\n".join(f"|{domain}|{share:.12f}|{share * 100:.6f}%|" for domain, share in result["P"].items())
    loss_rows = "\n".join(f"|{name.removeprefix('metric/the_pile_').removesuffix('_val_loss')}|{value:.6f}|"
                          for name, value in result["predicted_loss_by_domain"].items())
    report = f"""# 冻结 LightGBM 的配比候选搜索

目标：13 个验证域的**预测 Loss 等权平均最小**。只读取 A4/A5 训练配方和 P3 冻结模型；A6–A15 不参与候选搜索或选参。模型版本 `{result['model_version']}`，搜索版本 `{SEARCH_VERSION}`。

候选包括 512 条原训练配方、从其出发的单域与双域可行扰动、以及最优候选附近的坐标份额转移。每个候选必须非负、和为 1，并同时满足训练最近邻距离阈值和各领域训练观察最大份额。总候选 {result['candidate_counts']['candidate_rows']} 条，其中 {result['candidate_counts']['supported_rows']} 条通过支持规则；搜索不保证树模型的全局最优。

## 搜索到的 P

|训练领域|P（完整结果见 CSV/JSON）|百分比|
|---|---:|---:|
{p_rows}

P 总和：{result['P_sum']:.12f}；与最近训练配方的欧氏距离：{result['nearest_train_distance']:.6f}，支持阈值：{result['support_threshold']:.6f}。最低预测综合 Loss：**{result['predicted_weighted_loss']:.6f}**。最佳原训练配方的预测综合 Loss：{result['best_training_recipe_predicted_weighted_loss']:.6f}。最优点来自 `{result['search_best_origin']}`。

## 13 个验证域的预测 Loss

|验证域|预测 Loss|
|---|---:|
{loss_rows}

这些值均为冻结 1M 模型的**预测值**。新配方没有对应的真实训练实验，不应将 `{result['predicted_weighted_loss']:.6f}` 写成观测 Loss。树模型在有限候选上的最低值也不构成全局最优证明。若改成特定验证域权重，需显式给出 13 维 `weights` 并重新调用 `find_best_mixture`。

函数入口：`q1.mixture.optimize.find_best_mixture(root='.', weights=None)`；给定 P 的预测：`q1.mixture.optimize.predict_loss(p, root='.')`。原始精度及模型 SHA-256 见 `optimal_mixture.json`。
"""
    (out / "optimal_mixture_report.md").write_text(report, encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结 LightGBM 的训练支持区域配比搜索")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = write_result(args.root)
    print(json.dumps({key: result[key] for key in ("predicted_weighted_loss", "P_sum", "search_best_origin", "candidate_counts", "P")}, ensure_ascii=False, indent=2))
