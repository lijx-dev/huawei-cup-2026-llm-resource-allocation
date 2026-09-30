"""从结构化输出生成可溯源的第一问图表。"""
import json
import os
from pathlib import Path

_mpl_cache = Path(__file__).resolve().parents[2] / "results/q1_revision_v2/.matplotlib_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import paths, save_json, sha256


def run(root, config):
    _, out = paths(root)
    dest = out / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    fields = config["quality_fields"]
    rho_file = out / "quality/spearman_signed.csv"
    rho = pd.read_csv(rho_file, index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(rho, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(fields)), fields, rotation=90, fontsize=7)
    ax.set_yticks(range(len(fields)), fields, fontsize=7)
    ax.set_title("Signed Spearman correlation of 22 normalized indicators")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(dest / "indicator_spearman.png", dpi=180)
    plt.close(fig)
    pair_file = out / "conflict/indicator_pair_contributions.csv"
    pair = pd.read_csv(pair_file)
    data = pair[pair.attachment == "A1"]
    matrix = np.zeros((len(fields), len(fields)))
    index = {field: i for i, field in enumerate(fields)}
    for row in data.itertuples():
        a,b = index[row.field_j], index[row.field_k]
        matrix[a,b] = matrix[b,a] = row.H_jk
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(range(len(fields)), fields, rotation=90, fontsize=7)
    ax.set_yticks(range(len(fields)), fields, fontsize=7)
    ax.set_title("A1 high-disagreement pair contribution H(j,k)")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(dest / "A1_disagreement_pairs.png", dpi=180)
    plt.close(fig)
    mixture_summary = json.loads((out / "mixture/mixture_summary.json").read_text())
    mix_fields, loss_fields = mixture_summary["mixture_fields"], mixture_summary["loss_fields"]
    train_loss = pd.read_csv(root / "data/real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv")
    std = train_loss[loss_fields].std(ddof=1).to_numpy()
    interaction_file = out / "mixture/pair_interactions.csv"
    interactions = pd.read_csv(interaction_file)
    main = interactions[interactions.model == mixture_summary["selection"]["main_model"]].copy()
    target_index = {name: i for i,name in enumerate(loss_fields)}
    main["standardized_interaction"] = [r.interaction/std[target_index[r.target]] for r in main.itertuples()]
    grouped = main.groupby(["domain_j", "domain_k"], as_index=False).agg(J_interaction=("standardized_interaction", "mean"), n_supported=("n_supported", "min"))
    matrix = np.full((len(mix_fields), len(mix_fields)), np.nan)
    index = {name:i for i,name in enumerate(mix_fields)}
    for row in grouped.itertuples():
        a,b = index[row.domain_j], index[row.domain_k]
        matrix[a,b] = matrix[b,a] = row.J_interaction if row.n_supported > 0 else np.nan
    names = [f.removeprefix("train_the_pile_") for f in mix_fields]
    fig, ax = plt.subplots(figsize=(11, 9))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1
    im = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title("Path-conditional pair response in standardized J; unsupported cells blank")
    fig.colorbar(im, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(dest / "pair_response_J.png", dpi=180)
    plt.close(fig)
    grouped.to_csv(dest / "pair_response_J_source.csv", index=False)
    save_json(dest / "figure_sources.json", {"experiment_version": config["experiment_version"],
              "random_seed": config["seed"], "inputs_sha256": {str(p.relative_to(out)): sha256(p)
              for p in (rho_file, pair_file, interaction_file)},
              "figure_names": ["indicator_spearman.png", "A1_disagreement_pairs.png", "pair_response_J.png"]})
