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
# Needs: `python -m gectargets targets CONFIG` (section 8 also reads the raw
# data, to recompute FS at other lags). Run as a script
# (`python studies/study1_empirical_targets.py CONFIG`) or cell by cell.
# Figures and tables go to `<results_root>/studies/study1/`.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gectargets.analysis import (TARGETS, SNR_METRICS, edge_vectors,
                                 target_matrices, offdiag, between_subject,
                                 corr_matrix, identifiability, loo_group_corr,
                                 signflip_null, snr_metric, snr_split_half,
                                 snr_epochs, epoch_tstats, fdr_fraction,
                                 epoch_signflip)
from gectargets.config import load_config, study_dir, target_variants, \
    variant_name
from gectargets.plotting import TARGET_COLORS, GRAY, show_matrix, save
from gectargets.results import load_targets, load_tau_scan
from gectargets.targets import compute_tau_scan

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/fdt_datasets.json"
GROUPS = None        # e.g. ["HC"] to restrict every analysis to one group
N_PERM = 1000        # sign flips for the FS_A null
TAUS = list(range(1, 16))       # lags (samples) of the tau scan, section 8
# any of analysis.SNR_METRICS (listed when section 8 runs)
TAU_METRICS = ["snr", "snr_epochs", "epoch_signflip", "between_r", "i_diff"]

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
    data[name].update(dataset=dataset, prep=prep, variant=variant, keep=keep)
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
# ## 6. Signal to noise of each subject's targets
# SNR = signal power / noise power of the subject's full-data estimate, with
# power the mean square over edges (uncentred: for FS_A, 0 is the reversible
# reference) and signal = total - noise (< 0: no detectable signal).
# * split half (every dataset): (odd - even) / 2 holds only noise, and its
#   mean square is the noise power of (odd + even) / 2.
# * epochs (subjects with more than one epoch, so not fMRI): noise variance
#   of each edge's epoch mean = var over epochs / K. For FS_A also a t-test
#   of every edge (fraction of edges significant, Benjamini-Hochberg 5 %) and
#   a sign-flip test of the whole matrix: time-reversing an epoch flips its
#   FS_A, so under reversibility each epoch's sign is arbitrary.
# Both assume independent noise between halves / epochs.

# %%
snr_rows, tmaps = [], {}
for name, d in data.items():
    has_epochs = "fs_epochs" in d
    if not has_epochs and d["n_epochs"].max() > 1:
        print(f"{name}: no per-epoch targets (computed before they were "
              f"stored); recompute the targets for the epoch SNR")
    odd = edge_vectors(d["fc_odd"], d["fs_odd"])
    even = edge_vectors(d["fc_even"], d["fs_even"])
    split = {key: snr_split_half(odd[key], even[key]) for key in TARGETS}
    for i, (s, g) in enumerate(zip(d["subjects"], d["groups"])):
        base = {"data": name, "subject": s, "group": g,
                "n_epochs": d["n_epochs"][i]}
        for key in TARGETS:
            snr_rows.append({**base, "target": key, "method": "split_half",
                             **{k: v[i] for k, v in split[key].items()}})
        if not has_epochs or d["n_epochs"][i] < 2:
            continue
        epochs = edge_vectors(d["fc_epochs"][i], d["fs_epochs"][i])
        for key in TARGETS:
            row = {**base, "target": key, "method": "epochs",
                   **snr_epochs(epochs[key])}
            if key == "FS_A":
                t, p = epoch_tstats(epochs[key])
                row.update(frac_edges_fdr=fdr_fraction(p),
                           p_signflip=epoch_signflip(epochs[key], N_PERM))
                tmaps.setdefault(name, {})[s] = t
            snr_rows.append(row)
snr = pd.DataFrame(snr_rows)
snr.to_csv(out / "6_snr.csv", index=False)

for row in summary:
    s = snr[(snr.data == row["data"]) & (snr.target == row["target"])]
    for method in ("split_half", "epochs"):
        row[f"snr_{method}_median"] = s.loc[s.method == method, "snr"].median()
    e = s[s.method == "epochs"]
    # Single-epoch data (fMRI) has no epoch rows; with no multi-epoch dataset
    # at all, snr does not even have these columns
    if row["target"] == "FS_A" and not e.empty:
        row["frac_subjects_signflip_05"] = (e.p_signflip < 0.05).mean()
        row["frac_edges_fdr_median"] = e.frac_edges_fdr.median()

# SNR per target: one row per dataset, split half | epochs
fig, axes = plt.subplots(len(data), 2, figsize=(10, 0.8 + 1.8 * len(data)),
                         squeeze=False)
for row, name in zip(axes, data):
    for ax, method in zip(row, ("split_half", "epochs")):
        ax.set_title(f"{name}: {method.replace('_', ' ')}")
        s = snr[(snr.data == name) & (snr.method == method)]
        if s.empty:
            ax.text(0.5, 0.5, "single epoch: split half only", color=GRAY,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set(xticks=[], yticks=[])
            continue
        bp = ax.boxplot([s.loc[s.target == key, "snr"] for key in TARGETS],
                        vert=False, tick_labels=TARGETS, widths=0.5,
                        patch_artist=True, medianprops={"color": "#52514e"})
        for box, key in zip(bp["boxes"], TARGETS):
            box.set(facecolor=TARGET_COLORS[key], alpha=0.6)
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.invert_yaxis()
        ax.set(xlabel="SNR (signal / noise power)")
fig.tight_layout()
save(fig, out / "6_snr.png")

multi = [n for n in data if ((snr.data == n) & (snr.method == "epochs")).any()]
if multi:
    # FS_A: epoch vs split-half SNR (cross-check), and the per-edge t of the
    # subject with the median epoch SNR
    fig, axes = plt.subplots(2, len(multi), figsize=(4.2 * len(multi), 7),
                             squeeze=False)
    for col, name in zip(axes.T, multi):
        fs_a = snr[(snr.data == name) & (snr.target == "FS_A")]
        w = fs_a.pivot(index="subject", columns="method",
                       values="snr").dropna()
        col[0].scatter(w.split_half, w.epochs, s=16, alpha=0.7,
                       color=TARGET_COLORS["FS_A"])
        lim = [min(w.min().min(), 0), w.max().max()]
        col[0].plot(lim, lim, color=GRAY, linewidth=1, linestyle="--")
        col[0].set(title=name, xlabel="FS_A SNR, split half",
                   ylabel="FS_A SNR, epochs")

        e = fs_a[fs_a.method == "epochs"].sort_values("snr")
        ex = e.iloc[len(e) // 2]
        n = data[name]["fc"].shape[-1]
        T = np.zeros((n, n))
        T[np.triu_indices(n, k=1)] = tmaps[name][ex.subject]
        show_matrix(col[1], T - T.T,
                    f"{ex.subject} (median SNR {ex.snr:.2f}): FS_A t\n"
                    f"{ex.frac_edges_fdr:.0%} of edges FDR 5 %, "
                    f"sign-flip p {ex.p_signflip:.3f}")
    fig.tight_layout()
    save(fig, out / "6_snr_fs_a_epochs.png")

# %% [markdown]
# ## 7. Summary across datasets

# %%
summary = pd.DataFrame(summary)
summary.to_csv(out / "7_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(summary.to_string(index=False))

# %% [markdown]
# ## 8. Signal to noise as a function of tau
# FS (all epochs, the two halves and every epoch) is recomputed at every lag
# in TAUS, cached next to the targets (only new lags are computed; needs the
# raw data), and each metric in TAU_METRICS is followed over tau. Lines are
# the median over subjects (or subject pairs), bands the interquartile range
# where the metric has one value per subject. Epoch metrics are skipped for
# single-epoch data (fMRI). FC does not depend on tau (flat reference); the
# dashed line is the tau of the targets.

# %%
for metric, text in SNR_METRICS.items():
    print(f"{metric:>14}: {text}")

tau_rows = []
for name, d in data.items():
    compute_tau_scan(cfg, d["dataset"], d["prep"], TAUS,
                     cfg.get("n_jobs", 1))
    n = len(d["subjects"])
    fc_ep = d.get("fc_epochs", np.full(n, None))
    for tau in TAUS:
        scan = load_tau_scan(cfg, d["dataset"], d["variant"], tau)
        all_, odd, even = (edge_vectors(d[f"fc{h}"], scan[f"fs{h}"][d["keep"]])
                           for h in ("", "_odd", "_even"))
        fs_ep = scan["fs_epochs"][d["keep"]] if "fs_epochs" in scan else \
            np.full(n, None)
        ep = [None if fc is None or fs is None else edge_vectors(fc, fs)
              for fc, fs in zip(fc_ep, fs_ep)]
        if all(e is None for e in ep):
            if d["n_epochs"].max() > 1:
                print(f"{name}, tau {tau}: no per-epoch FS in the targets or "
                      f"the tau scan (computed before they were stored); "
                      f"delete targets/tau_scan/tau-{tau} to recompute it")
            ep = None
        for key in TARGETS:
            E = None if ep is None else \
                [None if e is None else e[key] for e in ep]
            for metric in TAU_METRICS:
                v = np.atleast_1d(snr_metric(metric, all_[key], odd[key],
                                             even[key], key, E, N_PERM))
                if np.isnan(v).all():
                    continue
                tau_rows.append({"data": name, "tau": tau,
                                 "tau_s": tau / d["meta"]["sfreq"],
                                 "target": key, "metric": metric,
                                 "value": np.nanmedian(v),
                                 "q25": np.nanquantile(v, 0.25),
                                 "q75": np.nanquantile(v, 0.75)})
tau_scan = pd.DataFrame(tau_rows)
tau_scan.to_csv(out / "8_tau_scan.csv", index=False)

fig, axes = plt.subplots(len(TAU_METRICS), len(data),
                         figsize=(4.5 * len(data), 3 * len(TAU_METRICS)),
                         squeeze=False)
for row, metric in zip(axes, TAU_METRICS):
    for ax, (name, d) in zip(row, data.items()):
        ax.set(title=name, xlabel="tau (s)", ylabel=metric)
        rows = tau_scan[(tau_scan.data == name) & (tau_scan.metric == metric)]
        if rows.value.isna().all():
            ax.text(0.5, 0.5, "not available (single epoch)", color=GRAY,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set(xticks=[], yticks=[])
            continue
        for key in TARGETS:
            s = rows[rows.target == key]
            if s.value.isna().all():
                continue
            ax.plot(s.tau_s, s.value, color=TARGET_COLORS[key], label=key,
                    marker="o", markersize=3)
            ax.fill_between(s.tau_s, s.q25, s.q75, color=TARGET_COLORS[key],
                            alpha=0.2, linewidth=0)
        ax.axvline(d["meta"]["tau_s"], color=GRAY, linestyle="--",
                   linewidth=1)
        if metric in ("snr", "snr_epochs", "within_snr"):
            ax.axhline(0, color=GRAY, linewidth=1)
            ax.set_yscale("symlog", linthresh=0.1)
    row[0].legend(fontsize=7)
fig.tight_layout()
save(fig, out / "8_tau_scan.png")
