# %% [markdown]
# # Study 1. The empirical targets: FC, FS, FS_S, FS_A
#
# What do the targets of the GEC fit look like, how similar are they between
# subjects, and how reliable are they within a subject? FS is split into its
# symmetric part FS_S = (FS + FS')/2 and antisymmetric part
# FS_A = (FS - FS')/2. Only FS_A carries the temporal direction (the
# irreversibility) a GEC can pick up, so the key question is whether FS_A is a
# reliable signal or estimation noise.
#
# Needs: `python -m gectargets targets CONFIG`. Run as a script
# (`python studies/study1_empirical_targets.py CONFIG`) or cell by cell.
# Figures and tables go to `<results_root>/studies/study1/`.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gectargets.analysis import (TARGETS, edge_vectors, target_matrices,
                                 offdiag, between_subject, corr_matrix,
                                 identifiability, loo_group_corr,
                                 signflip_null)
from gectargets.config import load_config, study_dir, target_variants, \
    variant_name
from gectargets.plotting import TARGET_COLORS, GRAY, show_matrix, save
from gectargets.results import load_targets

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/demo.json"
GROUPS = None        # e.g. ["HC"] to restrict every analysis to one group
N_PERM = 1000        # sign flips for the FS_A null

cfg = load_config(CONFIG)
out = study_dir(cfg, "study1")
data = {}
for dataset, prep in target_variants(cfg):
    variant = variant_name(cfg["datasets"][dataset], prep)
    name = f"{dataset} | {variant}"
    d = load_targets(cfg, dataset, variant)
    keep = np.ones(len(d["groups"]), bool) if GROUPS is None else \
        np.isin(d["groups"], GROUPS)
    data[name] = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) ==
                      len(keep) else v) for k, v in d.items()}
    data[name]["subjects"] = list(np.array(d["subjects"])[keep])
    print(f"{name}: {keep.sum()} subjects, {d['fc'].shape[-1]} nodes, "
          f"tau = {d['meta']['tau']} samples ({d['meta']['tau_s']:.3g} s)")

# %% [markdown]
# ## 1. Group-mean matrices
# One row per dataset. FC and FS share the diverging scale of their row; FS_A
# has its own (it is much smaller).

# %%
fig, axes = plt.subplots(len(data), 4, figsize=(13, 3.2 * len(data)),
                         squeeze=False)
for row, (name, d) in zip(axes, data.items()):
    mats = target_matrices(d["fc"].mean(0), d["fs"].mean(0))
    vmax = max(np.nanmax(np.abs(offdiag(mats[k]))) for k in ("FC", "FS"))
    for ax, key in zip(row, TARGETS):
        show_matrix(ax, mats[key], f"{name}\n{key}",
                    None if key == "FS_A" else vmax)
fig.tight_layout()
save(fig, out / "1_group_mean_matrices.png")

# %% [markdown]
# ## 2. How big is the antisymmetric part?
# Per subject: ||FS_A|| / ||FS|| over the off-diagonal, and the node
# autocorrelation at tau (diagonal of FS), which sets the network model's
# theta.

# %%
rows = []
for name, d in data.items():
    fs = d["fs"]
    ratio = (np.linalg.norm(offdiag(target_matrices(d["fc"], fs)["FS_A"]),
                            axis=1) / np.linalg.norm(offdiag(fs), axis=1))
    acf = np.diagonal(fs, axis1=1, axis2=2).mean(1)
    rows += [{"data": name, "subject": s, "group": g, "fs_a_ratio": r,
              "acf_tau": a, "n_epochs": n}
             for s, g, r, a, n in zip(d["subjects"], d["groups"], ratio, acf,
                                      d["n_epochs"])]
sizes = pd.DataFrame(rows)
sizes.to_csv(out / "2_fs_a_size.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 + 0.5 * len(data)))
for ax, col, label in zip(axes, ["fs_a_ratio", "acf_tau"],
                          ["||FS_A|| / ||FS||", "mean acf(tau)"]):
    groups = [sizes.loc[sizes.data == n, col] for n in data]
    ax.boxplot(groups, vert=False, tick_labels=list(data), widths=0.5,
               medianprops={"color": TARGET_COLORS["FS_A"]})
    ax.set(xlabel=label)
axes[1].set_yticklabels([])
fig.tight_layout()
save(fig, out / "2_fs_a_size.png")

# %% [markdown]
# ## 3. Between-subject similarity
# Subject x subject correlation of each target (subjects sorted by group).
# High similarity of FC / FS_S is expected (shared anatomy, leakage); the
# question is FS_A.

# %%
fig, axes = plt.subplots(len(data), 4, figsize=(13, 3.2 * len(data)),
                         squeeze=False)
summary = []
for row, (name, d) in zip(axes, data.items()):
    order = np.argsort(d["groups"], kind="stable")
    edges = edge_vectors(d["fc"][order], d["fs"][order])
    for ax, key in zip(row, TARGETS):
        show_matrix(ax, corr_matrix(edges[key]), f"{name}\n{key}", vmax=1)
        summary.append({"data": name, "target": key,
                        "between_r_mean": between_subject(edges[key]).mean()})
fig.tight_layout()
save(fig, out / "3_between_subject_similarity.png")

# %% [markdown]
# ## 4. Shared pattern: each subject vs the mean of the others
# Leave-one-out correlation r_s = corr(subject s, mean of the rest). For FS_A
# a sign-flip null tests whether the shared direction is more than noise
# (under the null every subject's FS_A is as likely as its time reverse).

# %%
fig, axes = plt.subplots(1, len(data), figsize=(4.5 * len(data), 3.2),
                         squeeze=False)
for ax, (name, d) in zip(axes[0], data.items()):
    edges = edge_vectors(d["fc"], d["fs"])
    for key in TARGETS:
        r = loo_group_corr(edges[key])
        ax.hist(r, bins=25, range=(-1, 1), histtype="step",
                color=TARGET_COLORS[key], label=key)
        row = next(x for x in summary if x["data"] == name and
                   x["target"] == key)
        row["loo_r_mean"] = r.mean()
        if key == "FS_A":
            null = signflip_null(edges[key], N_PERM)
            row["loo_null_95"] = np.quantile(null, 0.95)
            row["loo_p_signflip"] = (1 + (null >= r.mean()).sum()) / \
                (N_PERM + 1)
            ax.axvspan(*np.quantile(null, [0.025, 0.975]), color=GRAY,
                       alpha=0.25, label="FS_A sign-flip null")
    ax.set(title=name, xlabel="corr(subject, mean of the others)",
           ylabel="subjects")
    ax.legend(fontsize=7)
fig.tight_layout()
save(fig, out / "4_loo_shared_pattern.png")

# %% [markdown]
# ## 5. Split-half reliability and identifiability
# Each subject's targets from odd vs even epochs (one-epoch fMRI: first vs
# second half of the run). Within-subject correlations above the
# between-subject ones mean an individual, reliable pattern; i_diff = mean
# within - mean between, accuracy = how often a half finds its own subject.

# %%
fig, axes = plt.subplots(len(data), 4, figsize=(13, 2.6 * len(data)),
                         squeeze=False)
for row, (name, d) in zip(axes, data.items()):
    odd = edge_vectors(d["fc_odd"], d["fs_odd"])
    even = edge_vectors(d["fc_even"], d["fs_even"])
    for ax, key in zip(row, TARGETS):
        ident = identifiability(odd[key], even[key])
        bins = np.linspace(-1, 1, 41)
        ax.hist(ident["between"], bins=bins, density=True, color=GRAY,
                alpha=0.6, label="between subjects")
        ax.hist(ident["within"], bins=bins, density=True, alpha=0.7,
                color=TARGET_COLORS[key], label="within subject")
        ax.set(title=f"{name}: {key}\ni_diff {ident['i_diff']:.2f}, "
                     f"accuracy {ident['accuracy']:.2f}",
               xlabel="corr(odd half, even half)")
        row_s = next(x for x in summary if x["data"] == name and
                     x["target"] == key)
        row_s.update({k: ident[k] for k in
                      ("i_self", "i_others", "i_diff", "accuracy")})
    row[0].legend(fontsize=7)
fig.tight_layout()
save(fig, out / "5_split_half_identifiability.png")

# %% [markdown]
# ## 6. Summary across datasets

# %%
summary = pd.DataFrame(summary)
summary.to_csv(out / "6_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(summary.to_string(index=False))
