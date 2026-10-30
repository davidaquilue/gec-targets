# %% [markdown]
# # Study 2. Training dynamics of the GEC fit
#
# Every fit logs, every `log_every` iterations, the correlation between the
# empirical and simulated FC, FS, FS_S and FS_A. Here we look at
# 1. the final fit quality across hyperparameters (e.g. eps_fs at fixed
#    eps_fc and the reverse), and (1b) the fit of FS and FS_A on the edges
#    with negative FC, which the models cannot reproduce,
# 2. the trajectories of the four correlations during training,
# 3. when the fit stops (its error, FC + FS, no longer improves), whether
#    FC, FS, FS_S and FS_A have each stopped improving too,
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

from gectargets.analysis import (TARGETS, asym, corr, gec_reliability,
                                 offdiag, triu)
from gectargets.config import expand_tasks, load_config, study_dir
from gectargets.fitting import START_ERROR, load_matrix
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.patches import Rectangle

from gectargets.plotting import (TARGET_COLORS, GRAY, DIVERGING, SEQUENTIAL,
                                 save)
from gectargets.results import load_fits, load_targets

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/fdt_datasets.json"
X_PARAM, Y_PARAM = "eps_fs", "eps_fc"     # section 1: heatmap columns, rows
MIN_EDGES = 10            # section 1b: fewest edges of a sign to correlate
TRAJECTORY_TASKS = None   # task ids for section 2; None: the "none" runs

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
# Mean over subjects of corr(empirical, simulated) at the end of the fit, over
# the grid of `X_PARAM` (columns) x `Y_PARAM` (rows). One row of panels per
# dataset, model and any other swept setting (e.g. maxC); only tasks whose
# sweep entry varies both parameters, and panels with at least two fits. A
# panel whose correlations have both signs uses a diverging scale centred on
# 0 (FS_A can be fitted with the wrong sign), any other a sequential scale
# over its own range; every cell is annotated and the outlined cell is the
# best of the panel. Scales differ between panels: compare values, not
# colours. The last panel is the median number of iterations: cells at the
# first check (2 x check_every) stopped as soon as they could, usually
# because the error went up.

# %%
def heat_panel(ax, grid, title, norm, cmap, fmt, best=False):
    """grid: DataFrame (Y_PARAM rows x X_PARAM columns), log-spaced values
    drawn as equal cells, each annotated."""
    Z = np.ma.masked_invalid(grid.to_numpy(float))
    im = ax.pcolormesh(Z, cmap=cmap, norm=norm, edgecolors="#fcfcfb",
                       linewidth=1.5)
    for (i, j), v in np.ndenumerate(Z.filled(np.nan)):
        if np.isfinite(v):
            r, g, b = np.ravel(cmap(norm(v)))[:3]
            ink = "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 \
                else "#0b0b0b"
            text = fmt(v) if callable(fmt) else format(v, fmt)
            ax.text(j + 0.5, i + 0.5, text, ha="center",
                    va="center", fontsize=6, color=ink)
    if best and Z.count():
        i, j = np.unravel_index(np.ma.argmax(Z), Z.shape)
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="#0b0b0b",
                               linewidth=1.5))
    ax.set(title=title, xlabel=X_PARAM, ylabel=Y_PARAM,
           xticks=np.arange(grid.shape[1]) + 0.5,
           yticks=np.arange(grid.shape[0]) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in grid.columns], rotation=90)
    ax.set_yticklabels([f"{v:g}" for v in grid.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if isinstance(norm, LogNorm):
        ticks = [v for v in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                 if norm.vmin <= v <= norm.vmax] or [norm.vmin, norm.vmax]
        cbar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cbar.minorticks_off()
    elif isinstance(norm, SymLogNorm):
        top = int(np.floor(np.log10(norm.vmax)))
        ticks = sorted({0} | {s * 10 ** k for k in range(top + 1)
                              for s in (-1, 1)})
        cbar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cbar.minorticks_off()


def corr_norm(values):
    """Diverging scale centred on 0 when the panel has both signs (FS_A can
    be fitted with the wrong sign), else sequential over the panel's range,
    so that a panel of positive correlations keeps its contrast."""
    lo, hi = np.nanmin(values, initial=np.inf), np.nanmax(values, initial=-np.inf)
    if not np.isfinite([lo, hi]).all():
        return Normalize(-1, 1), DIVERGING
    if lo < 0 < hi:
        vmax = max(-lo, hi)
        return Normalize(-vmax, vmax), DIVERGING
    return Normalize(lo, hi), SEQUENTIAL


def panel_name(t):
    """data_model plus the swept settings other than the two grid axes (the
    preprocessing options are already in the variant)."""
    extra = {k: v for k, v in t["swept"].items()
             if k not in (X_PARAM, Y_PARAM, "envelope", "orthogonalize")}
    return " | ".join([label(t)] + [f"{k} {v:g}" if isinstance(v, float)
                                    else f"{k} {v}" for k, v in extra.items()])


on_grid = {t["id"]: panel_name(t) for t in tasks
           if X_PARAM in t["swept"] and Y_PARAM in t["swept"]
           and t["fit"]["convergence"] != "none"}
grid_table = table[table.id.isin(on_grid)].assign(
    panel=lambda df: df.id.map(on_grid))
# A panel needs at least two fitted cells (a sweep still running fills in)
panels = [(name, g) for name, g in grid_table.groupby("panel", sort=False)
          if len(g) > 1]
if panels:
    fig, axes = plt.subplots(len(panels), len(TARGETS) + 1,
                             figsize=(3.4 * (len(TARGETS) + 1),
                                      3.0 * len(panels)), squeeze=False)
    for row, (name, g) in zip(axes, panels):
        for ax, key in zip(row, TARGETS):
            grid = g.pivot(index=Y_PARAM, columns=X_PARAM, values=f"r_{key}")
            heat_panel(ax, grid, f"{name}\n{key}: mean corr(emp, sim)",
                       *corr_norm(grid.to_numpy(float)), ".2f", best=True)
        grid = g.pivot(index=Y_PARAM, columns=X_PARAM, values="n_iter_median")
        heat_panel(row[-1], grid, f"{name}\nmedian iterations",
                   LogNorm(max(np.nanmin(grid.to_numpy(float)), 1),
                           np.nanmax(grid.to_numpy(float))),
                   SEQUENTIAL, ".0f")
    fig.tight_layout()
    save(fig, out / "1_final_fit_quality.png")


# %% [markdown]
# ## 1b. FS and FS_A on the edges with negative FC
# The fitted couplings are non-negative, so the models cannot produce
# negative correlations: on an edge whose empirical FC is negative, the
# simulated FC and FS cannot match the sign of the data. Do FS and its
# antisymmetric part still follow the data there?
#
# Per subject, corr(empirical, simulated) over the edges whose empirical FC
# (the subject's own) is negative, and for reference over those with
# positive FC: FS over both directed entries (i, j) and (j, i) of each such
# pair, FS_A over the upper triangle. Mean over subjects, on the grid of
# section 1; a subject with fewer than MIN_EDGES edges of a sign is left out
# of that cell (the table gives how many subjects are left). FC < 0 and
# FC > 0 share one scale per target, so the two columns compare directly.
# The first title gives the median share of edges with negative FC.

# %%
SIGN_COLUMNS = {f"{key} | FC {op} 0": (key, op)
                for key in ("FS", "FS_A") for op in ("<", ">")}


def corr_on(emp, sim, select):
    """corr over the selected entries; nan if fewer than MIN_EDGES."""
    return corr(emp[select], sim[select]) if select.sum() >= MIN_EDGES \
        else np.nan


def mean_finite(values):
    v = np.asarray(values, float)
    return v[np.isfinite(v)].mean() if np.isfinite(v).any() else np.nan


targets_by_data, sign_rows = {}, []
for t in tasks:
    if t["id"] not in on_grid:
        continue
    key = (t["dataset"], t["variant"])
    if key not in targets_by_data:
        d = load_targets(cfg, *key)
        targets_by_data[key] = ({s: i for i, s in enumerate(d["subjects"])},
                                d["fc"], d["fs"])
    index, fc_emp, fs_emp = targets_by_data[key]
    f = fits[t["id"]]
    values, neg_share = {c: [] for c in SIGN_COLUMNS}, []
    for s, subject in enumerate(f["subjects"]):
        fc, fs, fs_sim = fc_emp[index[subject]], fs_emp[index[subject]], \
            f["fs_sim"][s]
        edges = {"FS": (offdiag(fc), offdiag(fs), offdiag(fs_sim)),
                 "FS_A": (triu(fc), triu(asym(fs)), triu(asym(fs_sim)))}
        neg_share.append(np.mean(triu(fc) < 0))
        for column, (target, op) in SIGN_COLUMNS.items():
            fc_edges, emp, sim = edges[target]
            select = fc_edges < 0 if op == "<" else fc_edges > 0
            values[column].append(corr_on(emp, sim, select))
    sign_rows.append({"id": t["id"], "panel": on_grid[t["id"]],
                      X_PARAM: t["swept"][X_PARAM],
                      Y_PARAM: t["swept"][Y_PARAM],
                      "neg_fc_share": np.median(neg_share),
                      **{c: mean_finite(v) for c, v in values.items()},
                      **{f"n_subjects {c}": int(np.isfinite(v).sum())
                         for c, v in values.items()}})
sign_table = pd.DataFrame(sign_rows)
sign_table.to_csv(out / "1b_negative_fc_edges.csv", index=False)

sign_panels = [(name, g) for name, g in sign_table.groupby("panel", sort=False)
               if len(g) > 1] if len(sign_table) else []
if sign_panels:
    fig, axes = plt.subplots(len(sign_panels), len(SIGN_COLUMNS),
                             figsize=(3.4 * len(SIGN_COLUMNS),
                                      3.0 * len(sign_panels)), squeeze=False)
    for row, (name, g) in zip(axes, sign_panels):
        grids = {c: g.pivot(index=Y_PARAM, columns=X_PARAM, values=c)
                 for c in SIGN_COLUMNS}
        for target in ("FS", "FS_A"):
            pair = [c for c, (k, _) in SIGN_COLUMNS.items() if k == target]
            norm, cmap = corr_norm(np.concatenate(
                [grids[c].to_numpy(float).ravel() for c in pair]))
            for c in pair:
                ax = row[list(SIGN_COLUMNS).index(c)]
                title = f"{name}\n{c}: mean corr(emp, sim)"
                if c == next(iter(SIGN_COLUMNS)):
                    title = (f"{name}\n{c} ({g.neg_fc_share.median():.0%} "
                             f"of edges): mean corr")
                heat_panel(ax, grids[c], title, norm, cmap, ".2f", best=True)
    fig.tight_layout()
    save(fig, out / "1b_negative_fc_edges.png")


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
# ## 3. When the fit stops, have FC, FS, FS_S and FS_A stopped improving?
# The fit stops when its error E (mse of FC + mse of FS) stops improving: its
# relative improvement over the last `check_every` iterations,
# (E_before - E_now) / E_now, falls below `tol`, or is negative (E went up).
# E mixes FC and FS, so this does not say whether each part has settled.
# Here the same relative improvement, over the same last interval, is
# computed for the mse of each part (per entry, off-diagonal), in units of
# `tol`:
#
#     ratio = (mse_before - mse_now) / mse_now / tol
#
# Being relative, it does not depend on the size of the matrices or of the
# target (FS_A is much smaller than FC), so all parts are read on the scale
# of the rule:
# * ratio <= 1: the part had settled as much as the rule asks of E;
# * ratio >> 1: it was still improving when the fit stopped;
# * ratio < 0: it was getting worse (common when the fit stops because E
#   went up).
#
# "rule" is E itself (<= 1 by construction, unless the fit ran to
# `max_iter`). Runs with `convergence: "none"` are read at the iteration
# where the default rule would have stopped. Heatmap: median over subjects,
# one shared scale per row; table: also the fraction of subjects whose part
# was still improving (ratio > 1).

# %%
PARTS = {"rule": "error", **{k: f"mse_{v}" for k, v in KEYS.items()}}


def stop_point(t, f, s):
    """Iteration where subject s stopped (or would have, for "none")."""
    if t["fit"]["convergence"] == "none":
        return legacy_stop(np.asarray(f["history_iter"][s], float),
                           np.asarray(f["history_error"][s], float),
                           t["fit"]["tol"])
    return float(f["n_iter"][s])


def improvement_at(iters, values, stop, window):
    """(v_before - v_now) / v_now from stop - window to stop; nan if either
    iteration was not logged."""
    now = np.flatnonzero(iters == stop)
    before = np.flatnonzero(iters == stop - window)
    if not (len(now) and len(before)) or not values[now[0]] > 0:
        return np.nan
    return (values[before[0]] - values[now[0]]) / values[now[0]]


def summary(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return (np.median(v), np.mean(v > 1)) if len(v) else (np.nan, np.nan)


stop_rows = []
for t in tasks:
    f = fits[t["id"]]
    window, tol = t["fit"]["check_every"], t["fit"]["tol"]
    stops = [stop_point(t, f, s) for s in range(len(f["subjects"]))]
    row = {"id": t["id"], "data_model": label(t), **t["swept"],
           "convergence": t["fit"]["convergence"],
           "stop_iter_median": np.median(stops)}
    for part, key in PARTS.items():
        ratios = [improvement_at(np.asarray(f["history_iter"][s], float),
                                 np.asarray(f[f"history_{key}"][s], float),
                                 stop, window) / tol
                  for s, stop in enumerate(stops)]
        row[f"ratio_{part}"], row[f"still_improving_{part}"] = \
            summary(ratios)
    stop_rows.append(row)
stop_table = pd.DataFrame(stop_rows)
stop_table.to_csv(out / "3_improvement_at_stop.csv", index=False)
with pd.option_context("display.width", 220, "display.precision", 3):
    print(stop_table.to_string(index=False))


def compact(v):
    """Short cell label; below 0.05 x tol the part did not change: "0"."""
    a = abs(v)
    return "0" if a < 0.05 else f"{v:.1f}" if a < 10 else \
        f"{v:.0f}" if a < 1e3 else f"{v / 1e3:.0f}k" if a < 1e6 else \
        f"{v:.0e}"


grid_stop = stop_table[stop_table.id.isin(on_grid)].assign(
    panel=lambda df: df.id.map(on_grid))
stop_panels = [(name, g) for name, g in grid_stop.groupby("panel", sort=False)
               if len(g) > 1]
if stop_panels:
    fig, axes = plt.subplots(len(stop_panels), len(PARTS),
                             figsize=(3.4 * len(PARTS), 3.0 * len(stop_panels)),
                             squeeze=False)
    for row, (name, g) in zip(axes, stop_panels):
        grids = {p: g.pivot(index=Y_PARAM, columns=X_PARAM,
                            values=f"ratio_{p}") for p in PARTS}
        vmax = max([10] + [np.nanmax(np.abs(x.to_numpy(float)), initial=0)
                           for x in grids.values()])
        norm = SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax)
        for ax, (part, grid) in zip(row, grids.items()):
            heat_panel(ax, grid, f"{name}\n{part}: improvement at stop / tol",
                       norm, DIVERGING, compact)
    fig.tight_layout()
    save(fig, out / "3_improvement_at_stop.png")

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
        load_fits(cfg, {**t , "half": "odd"}),
        load_fits(cfg, {**t, "half": "even"}))
    rel_rows.append({"id": t["id"], "data_model": label(t), **t["swept"],
                     "convergence": t["fit"]["convergence"], **rel})
reliability = pd.DataFrame(rel_rows)
reliability.to_csv(out / "4_gec_reliability.csv", index=False)
with pd.option_context("display.width", 220, "display.precision", 3):
    print(reliability.to_string(index=False))
