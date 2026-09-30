"""P14：从冻结表生成可追溯图件。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, sha, stage_dir


def create_figures(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    manifests, source_rows = [], []

    def save(fig, name: str, sources: list[Path], title: str) -> None:
        path = figdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifests.append({"figure": name, "title": title, "path": path.relative_to(out).as_posix(),
                          "sha256": sha(path)})
        for source in sources:
            source_rows.append({"figure": name, "source_file": source.relative_to(ROOT).as_posix(),
                                "source_sha256": sha(source)})

    p02 = stage_dir("P02")
    resid = pd.read_csv(p02 / "residuals.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(resid.val_loss, resid.fitted, s=6, alpha=.5)
    span = [min(resid.val_loss.min(), resid.fitted.min()), max(resid.val_loss.max(), resid.fitted.max())]
    axes[0].plot(span, span, color="black", lw=1)
    axes[0].set(xlabel="B1 observed loss", ylabel="B1 fitted loss")
    axes[1].scatter(resid.log_D, resid.residual, s=6, alpha=.5)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(xlabel="log D (billion tokens)", ylabel="observed - fitted")
    save(fig, "01_b1_fit_residual.png", [p02 / "residuals.csv"], "B1 fitted and residual")

    p04 = stage_dir("P04")
    candidates = pd.read_csv(p04 / "candidate_model_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(candidates.model, candidates.macro_fold_rmse, yerr=candidates.fold_rmse_sem, capsize=3)
    ax.set(ylabel="Grouped CV macro fold RMSE", title="B6 quality model selection")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "02_b6_candidates.png", [p04 / "candidate_model_metrics.csv"], "B6 grouped CV")

    p05 = stage_dir("P05")
    b7 = pd.read_csv(p05 / "b7_new_predictions.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(b7.val_loss, b7.selected_prediction, s=15, alpha=.7, label="selected")
    ax.scatter(b7.val_loss, b7.m0_prediction, s=10, alpha=.35, label="M0")
    span = [b7.val_loss.min(), b7.val_loss.max()]
    ax.plot(span, span, color="black", lw=1)
    ax.set(xlabel="B7-new observed loss", ylabel="frozen prediction")
    ax.legend()
    save(fig, "03_b7_locked.png", [p05 / "b7_new_predictions.csv"], "B7-new locked validation")

    p06 = stage_dir("P06")
    b8 = pd.read_csv(p06 / "b8_predictions.csv")
    mid_group = b8.groupby(["N_params_B", "D_tokens_B"]).size().sort_values(ascending=False).index[0]
    sub = b8.loc[(b8.N_params_B == mid_group[0]) & (b8.D_tokens_B == mid_group[1])].sort_values("Q_score")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub.Q_score, sub.val_loss, "o-", label="B8 stress observed")
    ax.plot(sub.Q_score, sub.prediction, "s--", label="B6+B7 selected model")
    ax.set(xlabel="Q_B (original sign)", ylabel="validation loss",
           title=f"B8 fixed N={mid_group[0]:g}, D={mid_group[1]:g}")
    ax.legend()
    save(fig, "04_b8_quality_direction.png", [p06 / "b8_predictions.csv"], "B8 quality direction stress")

    p09 = stage_dir("P09")
    marginal = pd.read_csv(p09 / "marginal_effects.csv")
    m = marginal.loc[(marginal.p_scenario == "p0") & (marginal.lambda_p == 0)].set_index("workpoint").loc[["low", "mid", "high"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m.index, m.M_N, "o-", label="M_N")
    ax.plot(m.index, m.M_D, "s-", label="M_D")
    ax.plot(m.index, m.M_Q, "^-", label="M_Q")
    ax.set_yscale("symlog")
    ax.set(ylabel="Marginal loss reduction (native units)", title="Representative workpoints")
    ax.legend()
    save(fig, "05_marginal_effects.png", [p09 / "marginal_effects.csv"], "Marginal effects")

    nonlin = pd.read_csv(p09 / "substitution_nonlinearity.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (point, variable), part in nonlin.groupby(["workpoint", "variable"]):
        ax.plot(part.delta_Q, part.relative_loss_error, marker="o", label=f"{point}-{variable}")
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Delta Q", ylabel="relative loss error", title="First-order substitution error")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "06_substitution_error.png", [p09 / "substitution_nonlinearity.csv"], "Local substitution error")

    p11 = stage_dir("P11")
    delta = pd.read_csv(p11 / "delta_sensitivity.csv")
    subset = delta.loc[np.isclose(delta.delta, .02)]
    matrix = subset.pivot(index="donor", columns="receiver", values="aggregate_loss_change")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.to_numpy(float), cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(17), [x.replace("train_the_pile_", "") for x in matrix.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(17), [x.replace("train_the_pile_", "") for x in matrix.index], fontsize=7)
    ax.set(xlabel="receiver", ylabel="donor", title="Pairwise transfer, delta=0.02")
    fig.colorbar(im, ax=ax, label="aggregate predicted loss change")
    save(fig, "07_pairwise_transfer.png", [p11 / "delta_sensitivity.csv"], "Simplex pairwise transfer")

    p08 = stage_dir("P08")
    scenarios = pd.read_csv(p08 / "scenario_predictions.csv")
    sub = scenarios.loc[scenarios.domain == "arxiv"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (mapping, form), part in sub.groupby(["mapping", "form"]):
        ax.plot(part.lambda_p, part.predicted_loss, marker="o", label=f"{mapping}/{form}")
    ax.set(xlabel="lambda scenario", ylabel="predicted loss", title="QA-QB, lambda and Form sensitivity")
    ax.legend(fontsize=7)
    save(fig, "08_structural_scenarios.png", [p08 / "scenario_predictions.csv"], "Scenario sensitivity")

    return pd.DataFrame(manifests), pd.DataFrame(source_rows)
