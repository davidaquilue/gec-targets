# %% [markdown]
# # Study 2. Training dynamics of the GEC fit
#
# Every fit logs, every `log_every` iterations, the correlation between the
# empirical and simulated FC, FS, FS_S and FS_A. Here we look at
# 1. the final fit quality across hyperparameters (e.g. eps_fs at fixed
#    eps_fc and the reverse),
# 2. the trajectories of the four correlations during training,
# 3. whether FS_S and FS_A have converged when the default rule (on FC and
#    FS only) stops the fit,
# 4. how similar the fitted GECs are between subjects and within a subject.
#
# Needs: the targets and the fits of the config's sweep. Run as a script
# (`python studies/study2_training_dynamics.py CONFIG`) or cell by cell.
# Outputs in `<results_root>/studies/study2/`.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gectargets.analysis import TARGETS, gec_reliability
from gectargets.config import expand_tasks, load_config, study_dir
from gectargets.fitting import START_ERROR, load_matrix
from gectargets.plotting import TARGET_COLORS, GRAY, color_for, save
from gectargets.results import load_fits

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/fdt_datasets.json"
X_PARAM, LINE_PARAM = "eps_fs", "eps_fc"  # section 1: x axis, one line each
TRAJECTORY_TASKS = None   # task ids for section 2; None: the "none" runs
GAIN_FRACTION = 0.95      # section 3: "converged" = this fraction of the gain

cfg = load_config(CONFIG)
out = study_dir(cfg, "study2")
KEYS = {"FC": "fc", "FS": "fs", "FS_S": "fs_s", "FS_A": "fs_a"}

tasks, fits = [], {}
for t in expand_tasks(cfg):
    if t["half"] != "all":
        continue
    f = load_fits(cfg, t)
    if f is None:
        print(f"task {t['id']}: no fits yet, skipped")
        continue
    tasks.append(t)
    fits[t["id"]] = f
print(f"{len(tasks)} tasks with results")


def label(t):
    return f"{t['dataset']} | {t['variant']} | {t['model']}"


rows = []
for t in tasks:
    f = fits[t["id"]]
    rows.append({"id": t["id"], "data_model": label(t),
                 **{k: t["fit"][k] for k in ("eps_fc", "eps_fs",
                                              "convergence")},
                 **t["model_params"], "n_subjects": len(f["subjects"]),
                 "n_iter_median": float(np.median(f["n_iter"])),
                 "stopped_by_rule": float(np.mean(f["stop"] != "max_iter")),
                 **{f"r_{k}": float(np.nanmean(f[f"r_{v}"]))
                    for k, v in KEYS.items()}})
table = pd.DataFrame(rows)
table.to_csv(out / "1_final_fit_quality.csv", index=False)
with pd.option_context("display.width", 220, "display.precision", 3):
    print(table.to_string(index=False))

# %% [markdown]
# ## 1. Final fit quality across hyperparameters
# Mean over subjects of corr(empirical, simulated) at the end of the fit.
# One row per dataset and model, one panel per target; x = `X_PARAM`, one
# line per value of `LINE_PARAM` (the other settings held at their default).

# %%
stopped = table[table.convergence != "none"]
groups = [g for g in stopped.groupby("data_model")
          if g[1][X_PARAM].nunique() > 1]
if groups:
    fig, axes = plt.subplots(len(groups), 4, figsize=(13, 2.8 * len(groups)),
                             squeeze=False, sharex="row")
    for row, (name, g) in zip(axes, groups):
        colors = color_for(sorted(g[LINE_PARAM].unique()))
        for ax, key in zip(row, TARGETS):
            for value, line in g.groupby(LINE_PARAM):
                line = line.sort_values(X_PARAM)
                ax.plot(line[X_PARAM], line[f"r_{key}"], marker="o",
                        color=colors[value], label=f"{LINE_PARAM} = {value:g}")
            ax.set(xscale="log", title=f"{name}\n{key}", xlabel=X_PARAM,
                   ylabel="mean corr(emp, sim)")
        row[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out / "1_final_fit_quality.png")


# %% [markdown]
# ## 2. Trajectories during training
# Mean (line) and interquartile range (band) over subjects of the four
# correlations. A subject that has stopped keeps its last value. The dashed
# line is the median iteration where the default rule stops (computed from
# the logged legacy error, also for runs with `convergence: "none"`).

# %%
def history_matrix(f, key):
    """(n_subjects, n_logs) with the last value carried forward."""
    hs = [np.asarray(h, float) for h in f[f"history_{key}"]]
    n = max(len(h) for h in hs)
    return np.array([np.pad(h, (0, n - len(h)), mode="edge") for h in hs])


def legacy_stop(iters, errors, tol):
    """Iteration where the default rule (error on FC + FS) stops."""
    old = START_ERROR
    for it, err in zip(iters, errors):
        if np.isnan(err):
            continue
        if (old - err) / err < tol:
            return it
        old = err
    return iters[-1]


def stop_iterations(t, f):
    if t["fit"]["convergence"] == "legacy":
        return np.asarray(f["n_iter"], float)
    if t["fit"]["convergence"] == "none":
        return np.array([legacy_stop(i, e, t["fit"]["tol"]) for i, e in
                         zip(f["history_iter"], f["history_error"])], float)
    return np.full(len(f["subjects"]), np.nan)  # other rules: not comparable


selected = TRAJECTORY_TASKS or [t["id"] for t in tasks
                                if t["fit"]["convergence"] == "none"] \
    or [t["id"] for t in tasks[:4]]
sel_tasks = [t for t in tasks if t["id"] in selected]
fig, axes = plt.subplots(len(sel_tasks), 4, figsize=(13, 2.6 * len(sel_tasks)),
                         squeeze=False)
for row, t in zip(axes, sel_tasks):
    f = fits[t["id"]]
    iters = history_matrix(f, "iter").max(axis=0)
    stops = stop_iterations(t, f)
    for ax, key in zip(row, TARGETS):
        H = history_matrix(f, f"r_{KEYS[key]}")
        ax.fill_between(iters, *np.nanpercentile(H, [25, 75], axis=0),
                        color=TARGET_COLORS[key], alpha=0.2, linewidth=0)
        ax.plot(iters, np.nanmean(H, axis=0), color=TARGET_COLORS[key])
        if np.isfinite(stops).any():
            ax.axvline(np.nanmedian(stops), color=GRAY, linestyle="--",
                       linewidth=1)
        ax.set(title=f"task {t['id']}: {t['model']} {t['swept']}\n{key}",
               xlabel="iteration", ylabel="corr(emp, sim)")
fig.tight_layout()
save(fig, out / "2_trajectories.png")

# %% [markdown]
# ## 3. Have FS_S and FS_A converged when the default rule stops?
# Per subject and target: the iteration where the correlation reaches
# `GAIN_FRACTION` of its total gain, the iteration of its maximum, and the
# gain still to come after the default stop (only measurable on runs with
# `convergence: "none"`). A target that converges after the stop, or keeps
# gaining, is not followed by the default rule.

# %%
def gain_iteration(iters, r, fraction):
    # The initial couplings are symmetric, so the first FS_A correlation can
    # be undefined: start from the first finite value
    iters, r = iters[np.isfinite(r)], r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    gain = r[-1] - r[0]
    if not np.isfinite(gain) or gain <= 0:
        return np.nan
    return iters[np.argmax(r >= r[0] + fraction * gain)]


conv_rows = []
for t in tasks:
    f = fits[t["id"]]
    stops = stop_iterations(t, f)
    for key in TARGETS:
        reach, peak, after = [], [], []
        for s in range(len(f["subjects"])):
            iters = np.asarray(f["history_iter"][s], float)
            r = np.asarray(f[f"history_r_{KEYS[key]}"][s], float)
            reach.append(gain_iteration(iters, r, GAIN_FRACTION))
            peak.append(iters[np.nanargmax(r)] if np.isfinite(r).any()
                        else np.nan)
            if t["fit"]["convergence"] == "none" and np.isfinite(stops[s]):
                after.append(r[-1] - r[np.searchsorted(iters, stops[s])])
        conv_rows.append({
            "id": t["id"], "data_model": label(t), **t["swept"],
            "convergence": t["fit"]["convergence"], "target": key,
            "stop_iter_median": np.nanmedian(stops),
            f"iter_{int(GAIN_FRACTION * 100)}pct_median": np.nanmedian(reach),
            "iter_peak_median": np.nanmedian(peak),
            "gain_after_stop_mean": np.mean(after) if after else np.nan})
conv = pd.DataFrame(conv_rows)
conv.to_csv(out / "3_convergence_by_target.csv", index=False)
with pd.option_context("display.width", 220, "display.precision", 3):
    print(conv.to_string(index=False))

# %% [markdown]
# ## 4. GEC reliability
# Correlation of the fitted couplings between subjects (`between_r`) and
# with the SC they started from (`r_to_sc`), over the SC support only. With
# `split_half` on, `i_diff` / `accuracy` compare each subject's odd- and
# even-epoch fits: within minus between subjects, which cancels the part all
# subjects share through the SC.

# %%
rel_rows = []
for t in tasks:
    sc_path = cfg["datasets"][t["dataset"]].get("sc_path")
    rel = gec_reliability(
        fits[t["id"]], None if sc_path is None else load_matrix(sc_path),
        load_fits(cfg, {**t, "half": "odd"}),
        load_fits(cfg, {**t, "half": "even"}))
    rel_rows.append({"id": t["id"], "data_model": label(t), **t["swept"],
                     "convergence": t["fit"]["convergence"], **rel})
reliability = pd.DataFrame(rel_rows)
reliability.to_csv(out / "4_gec_reliability.csv", index=False)
with pd.option_context("display.width", 220, "display.precision", 3):
    print(reliability.to_string(index=False))
