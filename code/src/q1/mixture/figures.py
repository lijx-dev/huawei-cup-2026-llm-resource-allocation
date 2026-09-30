"""从已保存的 P3/M3 结构化结果绘图；每张图可追溯到 CSV。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate(root):
    root = Path(root)
    mix, integ, figs = root / "results/q1/mixture", root / "results/q1/integration", root / "results/q1/figures"
    figs.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(name, source):
        plt.tight_layout()
        plt.savefig(figs / f"m3_{name}.svg", bbox_inches="tight")
        plt.close()
        manifest.append({"figure": f"m3_{name}.svg", "source_file": source})

    m = pd.read_csv(mix / "test_1m_metrics.csv")
    avg = m[m.target == "average_loss"]
    plt.figure(figsize=(6, 4)); plt.bar(avg.model, avg.rmse); plt.ylabel("1M held-out average Loss RMSE"); save("model_rmse", "test_1m_metrics.csv")
    domain = m[(m.model.isin(["ridge", "lightgbm"])) & (m.target != "average_loss")]
    pivot = domain.pivot(index="target", columns="model", values="rmse")
    pivot.plot(kind="bar", figsize=(11, 5)); plt.ylabel("1M held-out RMSE"); plt.xticks(rotation=75, ha="right"); save("target_rmse", "test_1m_metrics.csv")
    pred = pd.read_csv(mix / "predictions_test_1m_lightgbm.csv")
    plt.figure(figsize=(5, 5)); plt.scatter(pred.observed_average_loss, pred.predicted_average_loss, s=10, alpha=.65)
    lo = min(pred.observed_average_loss.min(), pred.predicted_average_loss.min()); hi = max(pred.observed_average_loss.max(), pred.predicted_average_loss.max())
    plt.plot([lo, hi], [lo, hi], color="black", lw=1); plt.xlabel("Observed 1M average Loss"); plt.ylabel("Predicted average Loss"); save("heldout_scatter", "predictions_test_1m_lightgbm.csv")
    effect = pd.read_csv(mix / "marginal_effects.csv").query('target == "average_loss"')
    mid = effect[effect.delta == 0.03].sort_values("mean_delta_loss_supported")
    plt.figure(figsize=(8, 6)); plt.barh(mid.domain, mid.mean_delta_loss_supported); plt.xlabel("Supported mean predicted Loss change, delta=0.03"); save("marginal_effects", "marginal_effects.csv")
    effect.pivot(index="domain", columns="delta", values="mean_delta_loss_supported").T.plot(figsize=(9, 6), legend=False)
    plt.xlabel("Mixture share increase"); plt.ylabel("Supported mean predicted average Loss change"); save("response_curves", "marginal_effects.csv")
    inter = pd.read_csv(mix / "interaction_effects.csv").query('target == "average_loss"')
    domains = mid.domain.tolist(); matrix = np.full((len(domains), len(domains)), np.nan)
    for row in inter.itertuples():
        j, k = domains.index(row.domain_j), domains.index(row.domain_k)
        matrix[j, k] = matrix[k, j] = row.mean_interaction_supported
    plt.figure(figsize=(9, 7)); plt.imshow(matrix, cmap="coolwarm"); plt.colorbar(label="Predicted interaction, supported")
    plt.xticks(range(len(domains)), domains, rotation=90, fontsize=7); plt.yticks(range(len(domains)), domains, fontsize=7); save("interaction_matrix", "interaction_effects.csv")
    surface = pd.read_csv(mix / "representative_response_surface.csv")
    grid = surface.pivot(index="delta_k", columns="delta_j", values="mean_predicted_average_loss")
    plt.figure(figsize=(6, 5)); plt.imshow(grid, origin="lower", aspect="auto"); plt.colorbar(label="Predicted average Loss")
    plt.xticks(range(len(grid.columns)), [f"{v:.2f}" for v in grid.columns]); plt.yticks(range(len(grid.index)), [f"{v:.2f}" for v in grid.index])
    plt.xlabel(surface.domain_j.iloc[0] + " increase"); plt.ylabel(surface.domain_k.iloc[0] + " increase"); save("response_surface", "representative_response_surface.csv")
    sim = pd.read_csv(integ / "domain_similarity_matrix.csv", index_col=0)
    plt.figure(figsize=(8, 7)); plt.imshow(sim, vmin=0, vmax=1); plt.colorbar(label="exp(-Euclidean distance)")
    plt.xticks(range(len(sim)), sim.columns, rotation=90, fontsize=7); plt.yticks(range(len(sim)), sim.index, fontsize=7); save("a17_similarity", "domain_similarity_matrix.csv")
    q = pd.read_csv(integ / "projected_domain_quality.csv")
    plt.figure(figsize=(8, 6)); plt.barh(q.mixture_domain, q.quality_proxy_main, color=["steelblue" if t != "inferred" else "orange" for t in q.a16_type])
    plt.xlabel("Projected quality proxy (A17 inferred=orange)"); save("projected_quality", "projected_domain_quality.csv")
    plt.figure(figsize=(5, 5)); plt.scatter(q.quality_proxy_main, q.quality_proxy_equal)
    for row in q.itertuples(): plt.annotate(row.mixture_domain, (row.quality_proxy_main, row.quality_proxy_equal), fontsize=6)
    plt.xlabel("P1 primary quality"); plt.ylabel("P1 equal-weight quality"); save("score_sensitivity", "projected_domain_quality.csv")
    proxy = pd.read_csv(integ / "mixture_quality_proxy.csv").query('scenario == "a17_assisted_main" and split == "train_1m"')
    plt.figure(figsize=(6, 4)); plt.scatter(proxy.quality_proxy, proxy.predicted_average_loss, s=10, alpha=.6)
    plt.xlabel("Mixture quality proxy"); plt.ylabel("Model-predicted average Loss"); save("quality_loss_proxy", "mixture_quality_proxy.csv")
    sens = pd.read_csv(integ / "quality_loss_association.csv")
    sens = sens[(sens.split == "train_1m") & (sens.loss_role == "model_prediction")]
    plt.figure(figsize=(8, 4)); plt.bar(sens.scenario, sens.spearman); plt.ylabel("Spearman with predicted Loss"); plt.xticks(rotation=50, ha="right"); save("mapping_sensitivity", "quality_loss_association.csv")
    for split, label in (("test_60m", "60M observed"), ("test_1b", "1B observed")):
        p = pd.read_csv(mix / f"predictions_{split}_lightgbm.csv")
        obs = p.observed_average_loss.rank().to_numpy(); fitted = p.predicted_average_loss.rank().to_numpy()
        plt.figure(figsize=(5, 5)); plt.scatter(obs, fitted, s=12, alpha=.7)
        plt.xlabel(label + " rank"); plt.ylabel("Frozen 1M model rank"); save(f"rank_{split}", f"predictions_{split}_lightgbm.csv")
    for split in ("est_10b", "est_70b"):
        p = pd.read_csv(mix / f"predictions_{split}_lightgbm.csv")
        plt.figure(figsize=(5, 5)); plt.scatter(p.estimated_reference_average_loss.rank(), p.predicted_average_loss.rank(), s=12)
        plt.xlabel(split + " estimated reference rank"); plt.ylabel("Frozen 1M model rank"); save(f"estimated_rank_{split}", f"predictions_{split}_lightgbm.csv")
    pd.DataFrame(manifest).to_csv(figs / "m3_figure_manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    generate(Path.cwd())
