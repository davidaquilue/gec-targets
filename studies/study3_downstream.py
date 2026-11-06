# %% [markdown]
# # Study 3. Do the hyperparameters change the downstream measures?
#
# Study 2 asks how the learning rates change the fit; here, how they change
# what is reported from a GEC: the FDT violation of each node and the entropy
# production rate (EPR, fdt_eeg's `compute_entropy_production`; legacy and
# network models only, not linear_hopf). The FDT measures shown are the
# config's `fdt_metrics`: "integral" (fdt_eeg's
# `pipeline_analytical_fdt_integral`), "onsager" (Deco's Onsager formalism,
# `gectargets/onsager_fdt.py`) and / or "ivfdt" (the lab's updated HCP
# pipeline: stationary, variance-normalised I_tot, summarised per subject as
# its RMS over nodes, plus the order parameter rho_brk;
# `gectargets/ivfdt.py`); each is a measure of its own in every section. See gectargets/downstream.py for how they are called. On
# the eps_fc x eps_fs grid of each dataset, variant and model:
# 1. how much the mean EPR and the mean FDT violation move across the grid,
# 2. whether different learning rates order the subjects the same way,
# 3. whether the measures follow the fit (of FS_A) or the data (the empirical
#    FS_A),
# 4. the paired contrasts between conditions (HCP working memory vs rest,
#    sleep N3 vs wake), cell by cell.
#
# Two cells are outlined in every grid figure: the lab's default learning
# rates (`study3.lab_default` in the config, solid black) and the best fit of
# FS_A (highest mean corr(emp, sim) of FS_A among the cells with most
# subjects fitted, dashed amber). Both are listed in `selected_cells.csv`
# for later analyses.
#
# Needs: the fits of the config's sweep and their downstream measures
# (`python -m gectargets downstream CONFIG --all`, or
# hpc/downstream_array.sbatch); nothing is refitted. Run as a script
# (`python studies/study3_downstream.py CONFIG`) or cell by cell. Outputs in
# `<results_root>/studies/study3/<model>/`, one folder per model (figures
# and tables of that model only). A model fitted with another procedure
# (config.PROCEDURES) is a model of its own, e.g. `network_updated`, without
# a lab default.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import (false_discovery_control, rankdata, spearmanr,
                         ttest_rel, wilcoxon)

from gectargets.analysis import asym, triu
from gectargets.config import (expand_tasks, load_config, model_dir,
                               model_name, study_dir)
from gectargets.downstream import (FDT_METRICS, FDT_SCALARS, fdt_metrics,
                                   fdt_summary, load_downstream, panel_key,
                                   selected_cells)
from gectargets.plotting import (CORR, DIVERGING, GRAY, corr_norm,
                                 dataset_label, figure_title, heat_panel,
                                 model_label, outline_legend, save)
from gectargets.results import load_fits, load_targets, save_per_model

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/electrophysiology.json"
X_PARAM, Y_PARAM = "eps_fs", "eps_fc"     # heatmap columns, rows
# Section 4: paired contrasts (a, b) between conditions, shown as a - b
CONTRASTS = [("HCP_WMLR", "HCP_RESTLR"), ("Sleep_N3", "Sleep_W")]
ALPHA = 0.05                              # section 4: BH level over the cells
CELLS = {"lab_default": {"color": "#0b0b0b", "linestyle": "-",
                         "label": "lab default"},
         "best_fs_a": {"color": "#eda100", "linestyle": "--",
                       "label": "best FS_A fit"}}
# Names of the measures (keys of the downstream files): short (node
# patterns) and of the value per subject
SHORT = {"fdt": "FDT", "fdt_onsager": "Onsager FDT",
         "fdt_ivfdt": "IV-FDT I_tot", "rho_brk": "rho_brk", "epr": "EPR"}
LABEL = {"fdt": "mean FDT violation", "fdt_onsager": "mean Onsager FDT violation",
         "fdt_ivfdt": "IV-FDT I_tot^RMS", "rho_brk": "IV-FDT rho_brk",
         "epr": "EPR"}

cfg = load_config(CONFIG)
out = study_dir(cfg, "study3")
# FDT measures shown: the config's fdt_metrics, node measures (FDT) and
# scalars (rho_brk); no EPR for linear_hopf
FDT = [FDT_METRICS[m] for m in fdt_metrics(cfg)]
SCALARS = [FDT_SCALARS[m] for m in fdt_metrics(cfg) if m in FDT_SCALARS]
MEASURES = {m: LABEL[m] for m in (*FDT, *SCALARS, "epr")}
MEAN = {m: LABEL[m].replace(" violation", "") for m in MEASURES}

tasks, fits, down = [], {}, {}
for t in expand_tasks(cfg):
    if (t["half"] != "all" or X_PARAM not in t["swept"]
            or Y_PARAM not in t["swept"] or t["fit"]["convergence"] == "none"):
        continue
    f = load_fits(cfg, t, keys=("r_fs_a",))   # the matrices are not needed
    d = None if f is None else load_downstream(cfg, t)
    if d is None:
        print(f"task {t['id']}: no {'fits' if f is None else 'downstream'} "
              f"yet, skipped")
        continue
    tasks.append(t)
    fits[t["id"]], down[t["id"]] = f, d
print(f"{len(tasks)} grid tasks with downstream measures")


def settings(extra):
    return [f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in extra]


def panel_name(key):
    """Name of a panel in the tables."""
    dataset, variant, model, extra = key
    return " | ".join([dataset, variant, model] + settings(extra))


def panel_title(key):
    """Title of a panel in the figures; the model is the figure's title."""
    dataset, variant, model, extra = key
    return " | ".join([dataset_label(dataset), variant] + settings(extra))


def write(table, name):
    save_per_model(table, out, name)


def by_model(keys):
    """Panel keys grouped by model, {model: [keys]}: one figure per model."""
    groups = {}
    for key in keys:
        groups.setdefault(key[2], []).append(key)
    return groups


# One row per task and subject; node FDT per task and FDT measure
# (subjects x nodes)
frames, node_fdt = [], {}
for t in tasks:
    f, d = fits[t["id"]], down[t["id"]]
    frames.append(pd.DataFrame({
        "id": t["id"], "panel": panel_name(panel_key(t)),
        "dataset": t["dataset"], "variant": t["variant"],
        "model": model_name(t),
        X_PARAM: t["swept"][X_PARAM], Y_PARAM: t["swept"][Y_PARAM],
        "subject": d["subjects"], "stable": d["stable"].astype(bool),
        "epr": d["epr"].astype(float),
        **{m: fdt_summary(m, d[m]) for m in FDT},
        **{m: d[m].astype(float) for m in SCALARS},
        "r_fs_a": pd.Series(f["r_fs_a"], index=f["subjects"])
        .reindex(d["subjects"]).to_numpy(float)}))
    node_fdt[t["id"]] = {m: pd.DataFrame(d[m], index=d["subjects"])
                         for m in FDT}
data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
write(data, "downstream_per_subject.csv")
n_unstable = int((~data.stable).sum()) if len(data) else 0
if n_unstable:
    print(f"{n_unstable} unstable fits left out (no downstream measures)")

by_panel = {}
for t in tasks:
    by_panel.setdefault(panel_key(t), []).append(t)
cells = selected_cells(cfg, tasks, fits, X_PARAM, Y_PARAM)
sel_rows = [{"panel": panel_name(key), "model": key[2], "cell": name,
             "id": None if t is None else t["id"],
             X_PARAM: None if t is None else t["swept"][X_PARAM],
             Y_PARAM: None if t is None else t["swept"][Y_PARAM],
             "r_fs_a_mean": None if t is None
             else float(np.nanmean(fits[t["id"]]["r_fs_a"]))}
            for key, chosen in cells.items() for name, t in chosen.items()]
selected = pd.DataFrame(sel_rows)
write(selected, "selected_cells.csv")
with pd.option_context("display.width", 200, "display.precision", 3):
    print(selected.to_string(index=False))


def outlines(key):
    return [{"cell": (t["swept"][Y_PARAM], t["swept"][X_PARAM]),
             **CELLS[name]} for name, t in cells[key].items() if t is not None]


def measures_of(d):
    """Measures with values in d (no EPR for linear_hopf)."""
    return [m for m in MEASURES if d[m].notna().any()]


def grid_of(d, value):
    return d.pivot(index=Y_PARAM, columns=X_PARAM, values=value)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 3 else np.nan


def mean_row_corr(X, Y):
    """Mean over subjects (rows) of corr(X_s, Y_s) (node patterns); rows
    with NaN and constant rows are left out."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[ok] - X[ok].mean(axis=1, keepdims=True)
    Y = Y[ok] - Y[ok].mean(axis=1, keepdims=True)
    den = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    r = (X * Y).sum(axis=1)[den > 0] / den[den > 0]
    return float(r.mean()) if len(r) else np.nan


def figure(n_rows, n_cols):
    n_cols = max(n_cols, 1)   # no columns: no stable fits, an empty figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols,
                                                      3.0 * n_rows),
                             squeeze=False)
    return fig, axes


def model_measures(model):
    """Measures with values for the model (no EPR for linear_hopf): the
    columns of its figures."""
    return measures_of(data[(data.model == model) & data.stable])


def model_cells(keys):
    """The outlined cells on at least one of the panels."""
    return [name for name in CELLS
            if any(cells.get(k, {}).get(name) is not None for k in keys)]


def finish(fig, axes, model, keys, name):
    """Hide the empty panels, title the figure with the model and save it in
    the model's folder."""
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)
    fig.tight_layout()
    legend = outline_legend(fig, [CELLS[c] for c in model_cells(keys)])
    figure_title(fig, model_label(model), above=legend)
    save(fig, model_dir(out, model) / name)


# %% [markdown]
# ## 1. How much do the mean EPR and the mean FDT violation move?
# Median over subjects of each measure (FDT: mean over nodes first), on the
# grid. The spread ratio in the title is the SD of the cell medians over the
# grid divided by the median, over cells, of the SD between subjects: above
# 1, the learning rates move the measure more than subjects differ.

# %%
level_rows, spread_rows = [], []
for key in by_panel:
    d = data[(data.panel == panel_name(key)) & data.stable]
    for m in measures_of(d):
        stats = d.groupby([Y_PARAM, X_PARAM])[m].agg(
            median="median", sd="std", n="count").reset_index()
        level_rows += [{"panel": panel_name(key), "model": key[2],
                        "measure": m, **r} for r in stats.to_dict("records")]
        at = {name: d[d.id == t["id"]][m].median()
              for name, t in cells[key].items() if t is not None}
        spread_rows.append({
            "panel": panel_name(key), "model": key[2], "measure": m,
            "n_cells": len(stats),
            "median_min": stats["median"].min(),
            "median_max": stats["median"].max(),
            "spread_ratio": stats["median"].std() / stats["sd"].median(),
            **{f"median_{name}": v for name, v in at.items()}})
level = pd.DataFrame(level_rows)
write(level, "1_level_per_cell.csv")
spread = pd.DataFrame(spread_rows)
write(spread, "1_spread.csv")
with pd.option_context("display.width", 200, "display.precision", 3):
    print(spread.to_string(index=False))

if len(level):
    for model, keys in by_model(by_panel).items():
        ms = model_measures(model)
        fig, axes = figure(len(keys), len(ms))
        for row, key in zip(axes, keys):
            for ax, m in zip(row, ms):
                lv = level[(level.panel == panel_name(key))
                           & (level.measure == m)]
                if not len(lv):
                    continue
                ratio = spread[(spread.panel == panel_name(key))
                               & (spread.measure == m)].spread_ratio.iloc[0]
                grid = grid_of(lv, "median")
                heat_panel(ax, grid, f"{panel_title(key)}\n{MEASURES[m]}: "
                           f"median (spread ratio {ratio:.2g})",
                           *corr_norm(grid.to_numpy(float)), ".2g",
                           outlines=outlines(key))
        finish(fig, axes, model, keys, "1_level.png")


# %% [markdown]
# ## 2. Do the learning rates order the subjects the same way?
# Against each outlined cell (the reference), for every cell of the grid:
# the Spearman correlation over subjects of EPR and of the mean FDT, and the
# mean over subjects of the correlation between the node patterns of FDT
# (the reference cell itself is 1). A change of level that keeps the order
# is harmless for between-subject analyses.

# %%
def node_pattern(ref_id, cell_id, m):
    a, b = node_fdt[ref_id][m], node_fdt[cell_id][m]
    common = a.index.intersection(b.index)
    return mean_row_corr(a.loc[common].to_numpy(), b.loc[common].to_numpy())


# Column -> (measure, title): each measure's order, and the node pattern of
# each FDT measure
ORDER = {}
for m in MEASURES:
    ORDER[m] = (m, f"{MEAN[m]} order")
    if m in FDT:
        ORDER[f"{m}_nodes"] = (m, f"{SHORT[m]} node pattern")
order_rows = []
for key, group in by_panel.items():
    d = data[(data.panel == panel_name(key)) & data.stable]
    # subjects x cells, one table per measure
    wide = {m: d.pivot(index="subject", columns="id", values=m)
            for m in measures_of(d)}
    for ref, rt in cells[key].items():
        if rt is None:
            continue
        for t in group:
            row = {"panel": panel_name(key), "model": key[2],
                   "reference": ref, "id": t["id"],
                   X_PARAM: t["swept"][X_PARAM], Y_PARAM: t["swept"][Y_PARAM]}
            for m, w in wide.items():
                row[m] = (spearman(w[rt["id"]], w[t["id"]])
                          if rt["id"] in w and t["id"] in w else np.nan)
            for m in FDT:
                row[f"{m}_nodes"] = node_pattern(rt["id"], t["id"], m)
            order_rows.append(row)
order = pd.DataFrame(order_rows)
write(order, "2_subject_order.csv")

if len(order):
    for model, keys in by_model(by_panel).items():
        cols = [(ref, o) for ref in model_cells(keys) for o in ORDER
                if ORDER[o][0] in model_measures(model)]
        fig, axes = figure(len(keys), len(cols))
        for row, key in zip(axes, keys):
            for ax, (ref, o) in zip(row, cols):
                g = order[(order.panel == panel_name(key))
                          & (order.reference == ref)]
                if not len(g) or o not in g or g[o].isna().all():
                    continue
                heat_panel(ax, grid_of(g, o), f"{panel_title(key)}\n"
                           f"{ORDER[o][1]} vs {CELLS[ref]['label']}", *CORR,
                           ".2f", outlines=outlines(key))
        finish(fig, axes, model, keys, "2_subject_order.png")

# Every cell against every cell: Spearman over subjects, cells ordered by
# eps_fc then eps_fs; the outlined diagonal entries are the two cells.
matrix_rows = []
for model, keys in by_model(by_panel).items():
    ms = model_measures(model)
    fig, axes = figure(len(keys), len(ms))
    for row, key in zip(axes, keys):
        d = data[(data.panel == panel_name(key)) & data.stable]
        group = sorted(by_panel[key], key=lambda t: (t["swept"][Y_PARAM],
                                                     t["swept"][X_PARAM]))
        ids = [t["id"] for t in group]
        for ax, m in zip(row, ms):
            if m not in measures_of(d):
                continue
            R = d.pivot(index="subject", columns="id", values=m)[ids].corr(
                method="spearman")
            matrix_rows += [{"panel": panel_name(key), "model": model,
                             "measure": m, "id_a": a, "id_b": b,
                             "spearman": R.loc[a, b]}
                            for a in ids for b in ids]
            im = ax.imshow(R.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1,
                           interpolation="nearest")
            labels = [f"{t['swept'][Y_PARAM]:g} / {t['swept'][X_PARAM]:g}"
                      for t in group]
            ax.set_xticks(range(len(ids)), labels, rotation=90, fontsize=4)
            ax.set_yticks(range(len(ids)), labels, fontsize=4)
            for name, t in cells[key].items():
                if t is not None:
                    k = ids.index(t["id"])
                    ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1,
                                           fill=False, linewidth=1.5,
                                           **{a: CELLS[name][a] for a in
                                              ("color", "linestyle")}))
            ax.set(title=f"{panel_title(key)}\n{MEASURES[m]}: Spearman "
                         f"between cells", xlabel=f"{Y_PARAM} / {X_PARAM}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    finish(fig, axes, model, keys, "2_cell_by_cell.png")
write(pd.DataFrame(matrix_rows), "2_cell_by_cell.csv")


# %% [markdown]
# ## 3. Do the measures follow the fit or the data?
# The data: the irreversibility of each subject's empirical targets, the
# mean square of FS_A over the upper triangle, and per node sum_j FS_A_ij^2.
# The fit: corr(empirical, simulated) of FS_A. Within each cell, over
# subjects: Spearman(measure, fit) next to Spearman(measure, data), and the
# mean over subjects of corr(node FDT, node empirical asymmetry). A measure
# that follows the fit more than the data reflects the learning rates more
# than the recordings. Below: the same over cells, mean fit vs median
# measure.

# %%
empirical = {}
for dataset, variant in dict.fromkeys((t["dataset"], t["variant"])
                                      for t in tasks):
    targets = load_targets(cfg, dataset, variant, keys=("fs",))
    fs_a = asym(targets["fs"])
    empirical[dataset, variant] = (
        # Mean total FS squared -> below named power
        pd.Series(np.mean(triu(fs_a) ** 2, axis=-1), index=targets["subjects"]),
        # Regional FS squared (sum over one axis in the matrix) -> below named nodes
        pd.DataFrame((fs_a ** 2).sum(axis=-1), index=targets["subjects"])
    )

follow_rows = []
for key, group in by_panel.items():  # key = data type; group = list of hyperparam comb
    d = data[(data.panel == panel_name(key)) & data.stable] # data for panel
    power, nodes = empirical[key[0], key[1]]
    for t in group:
        dt = d[d.id == t["id"]].set_index("subject")  # data of task
        row = {"panel": panel_name(key), "model": key[2], "id": t["id"],
               X_PARAM: t["swept"][X_PARAM], Y_PARAM: t["swept"][Y_PARAM],
               "r_fs_a_mean": dt.r_fs_a.mean()}
        for m in measures_of(d):
            row[f"{m}_median"] = dt[m].median()
            row[f"{m}_vs_fit"] = spearman(dt[m], dt.r_fs_a)  # corr metric with fit fs_A
            row[f"{m}_vs_data"] = spearman(dt[m], power.loc[dt.index])
        for m in FDT:
            sim = node_fdt[t["id"]][m].loc[dt.index]  # Set task index of FDT
            row[f"{m}_nodes_vs_data"] = mean_row_corr(  # Corr fdt with fs_A**2 per nodes
                sim.to_numpy(), nodes.loc[dt.index].to_numpy()
            )
        follow_rows.append(row)
follow = pd.DataFrame(follow_rows)
write(follow, "3_fit_vs_data.csv")

# Column -> (measure, title)
FOLLOW = {}
for m in MEASURES:
    FOLLOW[f"{m}_vs_fit"] = (m, f"{MEAN[m]} vs fit")
    FOLLOW[f"{m}_vs_data"] = (m, f"{MEAN[m]} vs data")
    if m in FDT:
        FOLLOW[f"{m}_nodes_vs_data"] = (m, f"{SHORT[m]} node pattern vs data")
for model, keys in by_model(by_panel).items():
    ms = model_measures(model)
    cols = [c for c, (m, _) in FOLLOW.items() if m in ms]
    fig, axes = figure(len(keys), len(cols))
    for row, key in zip(axes, keys):
        g = follow[follow.panel == panel_name(key)]
        for ax, col in zip(row, cols):
            if col not in g or g[col].isna().all():
                continue
            heat_panel(ax, grid_of(g, col), f"{panel_title(key)}\n"
                       f"{FOLLOW[col][1]}", *CORR, ".2f",
                       outlines=outlines(key))
    finish(fig, axes, model, keys, "3_fit_vs_data.png")

    fig, axes = figure(len(keys), len(ms))
    for row, key in zip(axes, keys):
        g = follow[follow.panel == panel_name(key)]
        for ax, m in zip(row, ms):
            if f"{m}_median" not in g or g[f"{m}_median"].isna().all():
                continue
            ax.scatter(g.r_fs_a_mean, g[f"{m}_median"], s=14, color=GRAY)
            for name, t in cells[key].items():
                if t is not None:
                    c = g[g.id == t["id"]]
                    ax.scatter(c.r_fs_a_mean, c[f"{m}_median"], s=60,
                               facecolor="none", linewidth=1.5,
                               edgecolor=CELLS[name]["color"],
                               linestyle=CELLS[name]["linestyle"])
            rho = spearman(g.r_fs_a_mean, g[f"{m}_median"])
            ax.set(title=f"{panel_title(key)}\n{MEASURES[m]}, over cells: "
                         f"Spearman {rho:.2f}",
                   xlabel="mean corr(emp, sim) FS_A",
                   ylabel=f"median {MEASURES[m]}")
    finish(fig, axes, model, keys, "3_fit_vs_measure_over_cells.png")


# %% [markdown]
# ## 4. Paired contrasts between conditions
# For every contrast (a, b) of CONTRASTS (HCP working memory vs rest, sleep
# N3 vs wake): the two datasets are fitted on the same grid; cells are
# matched by their settings, subjects by their id (those with a stable fit
# in both). In every cell, a - b: paired t-test with its effect size Cohen's
# d_z = mean(diff) / SD(diff), and, as a check without normality, the
# Wilcoxon signed-rank test with the matched-pairs rank-biserial r. The
# heatmap shows d_z; * marks the cells whose paired t-test passes
# Benjamini-Hochberg at ALPHA over the cells of the grid. Outlined: the lab
# default and the best FS_A fit averaged over the two conditions. Note: tau
# is fitted per dataset, so the two conditions may be fitted at different
# lags.

# %%
def rank_biserial(diff):
    diff = diff[diff != 0]
    ranks = rankdata(np.abs(diff))
    plus, minus = ranks[diff > 0].sum(), ranks[diff < 0].sum()
    return (plus - minus) / (plus + minus) if len(diff) else np.nan


# Each hyperparameter set fitted is a "task" (t, u, as everywhere in this
# study); the datasets compared are a and b. A contrast key is (a, panel key
# of b): the panel key carries b, the variant, the model and the settings.
pairs = {}
for a_ds, b_ds in CONTRASTS:
    for t in tasks:
        if t["dataset"] != b_ds:
            continue
        # The task of a with the same settings
        u = next((u for u in tasks if u["dataset"] == a_ds
                  and (u["variant"], u["model"], u["tag"])
                  == (t["variant"], t["model"], t["tag"])), None)
        if u is not None:
            pairs.setdefault((a_ds, panel_key(t)), []).append((u, t))
    if not any(k[0] == a_ds and k[1][0] == b_ds for k in pairs):
        print(f"Section 4: no matching fits of {a_ds} and {b_ds}, skipped")


def contrast_name(ckey):
    a_ds, key = ckey
    return f"{a_ds} vs {panel_name(key)}"


def contrast_title(ckey):
    a_ds, key = ckey
    return " | ".join([f"{dataset_label(a_ds)} vs {dataset_label(key[0])}",
                       key[1]] + settings(key[3]))


def contrast_file(ckey):
    return f"{ckey[0]}_vs_{ckey[1][0]}"


contrast_rows = []
for ckey, group in pairs.items():
    name = contrast_name(ckey)
    for u, t in group:
        a = data[(data.id == u["id"]) & data.stable].set_index("subject")
        b = data[(data.id == t["id"]) & data.stable].set_index("subject")
        common = a.index.intersection(b.index)  # same subjects, same order
        row = {"panel": name, "model": ckey[1][2], "dataset_a": u["dataset"],
               "dataset_b": t["dataset"], "id_a": u["id"], "id_b": t["id"],
               X_PARAM: t["swept"][X_PARAM], Y_PARAM: t["swept"][Y_PARAM],
               "n_subjects": len(common),
               "r_fs_a_mean": (a.r_fs_a.mean() + b.r_fs_a.mean()) / 2}
        for m in measures_of(data[data.id.isin([u["id"], t["id"]])]):
            x, y = a.loc[common, m].to_numpy(), b.loc[common, m].to_numpy()
            diff = x - y
            if len(diff) < 3:  # Cases with very few data
                continue
            row.update({f"{m}_a_mean": x.mean(), f"{m}_b_mean": y.mean(),
                        f"{m}_d_z": diff.mean() / diff.std(ddof=1),
                        f"{m}_p_t": ttest_rel(x, y).pvalue,
                        f"{m}_rank_biserial": rank_biserial(diff),
                        f"{m}_p_wilcoxon": wilcoxon(x, y).pvalue})
        contrast_rows.append(row)
contrast = pd.DataFrame(contrast_rows)
for m in MEASURES:
    if f"{m}_p_t" not in contrast:
        continue
    contrast[f"{m}_q_t"] = np.nan
    for panel, g in contrast.groupby("panel"):
        ok = g[f"{m}_p_t"].notna()
        if ok.any():
            contrast.loc[g.index[ok], f"{m}_q_t"] = false_discovery_control(
                g.loc[ok, f"{m}_p_t"])
if len(contrast):
    write(contrast, "4_contrast.csv")
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(contrast.to_string(index=False))


def contrast_outlines(ckey, g):
    """Lab default (same learning rates for both conditions) and the best
    FS_A fit averaged over the two conditions."""
    marks = []
    lab = cells.get(ckey[1], {}).get("lab_default")
    if lab is not None:
        marks.append({"cell": (lab["swept"][Y_PARAM], lab["swept"][X_PARAM]),
                      **CELLS["lab_default"]})
    best = g.loc[g.r_fs_a_mean.idxmax()]
    marks.append({"cell": (best[Y_PARAM], best[X_PARAM]),
                  **CELLS["best_fs_a"]})
    return marks


if len(contrast):
    # One figure per contrast and model
    figures = {}
    for ckey in pairs:
        figures.setdefault((contrast_file(ckey), ckey[1][2]), []).append(ckey)
    for (file, model), ckeys in figures.items():
        a_label, b_label = (dataset_label(ckeys[0][0]),
                            dataset_label(ckeys[0][1][0]))
        ms = model_measures(model)
        fig, axes = figure(len(ckeys), len(ms))
        for row, ckey in zip(axes, ckeys):
            g = contrast[contrast.panel == contrast_name(ckey)]
            marks = contrast_outlines(ckey, g)
            for ax, m in zip(row, ms):
                if f"{m}_d_z" not in g or g[f"{m}_d_z"].isna().all():
                    continue
                grid = grid_of(g, f"{m}_d_z")
                vmax = np.nanmax(np.abs(grid.to_numpy(float)))
                heat_panel(ax, grid, f"{contrast_title(ckey)}\n{MEASURES[m]}: "
                           f"d_z ({a_label} - {b_label})",
                           Normalize(-vmax, vmax), DIVERGING, ".2f",
                           outlines=marks)
                q = grid_of(g, f"{m}_q_t").to_numpy(float)
                for i, j in zip(*np.nonzero(q < ALPHA)):
                    ax.text(j + 0.88, i + 0.8, "*", ha="center", va="center",
                            fontsize=9, color="#0b0b0b")
        finish(fig, axes, model, [k[1] for k in ckeys],
               f"4_contrast_{file}.png")

    # Per subject, b -> a, at the two outlined cells
    for ckey in pairs:
        model, variant = ckey[1][2], ckey[1][1]
        g = contrast[contrast.panel == contrast_name(ckey)]
        marks = contrast_outlines(ckey, g)
        has = [m for m in MEASURES if f"{m}_d_z" in g
               and g[f"{m}_d_z"].notna().any()]
        fig, axes = figure(len(has), len(marks))
        for row, m in zip(axes, has):
            for ax, mark in zip(row, marks):
                c = g[np.isclose(g[Y_PARAM], mark["cell"][0])
                      & np.isclose(g[X_PARAM], mark["cell"][1])].iloc[0]
                a = data[(data.id == c.id_a) & data.stable].set_index(
                    "subject")[m]
                b = data[(data.id == c.id_b) & data.stable].set_index(
                    "subject")[m]
                both = pd.concat([b, a], axis=1, join="inner").to_numpy()
                ax.plot([0, 1], both.T, color=GRAY, linewidth=0.6, alpha=0.5)
                ax.plot([0, 1], both.mean(axis=0), color=mark["color"],
                        linestyle=mark["linestyle"], linewidth=2.5)
                ax.set(xticks=[0, 1], xticklabels=[dataset_label(c.dataset_b),
                                                   dataset_label(c.dataset_a)],
                       ylabel=MEASURES[m], title=f"{mark['label']}: "
                       f"d_z {c[f'{m}_d_z']:.2f}, q {c[f'{m}_q_t']:.2g}")
        fig.tight_layout()
        figure_title(fig, f"{model_label(model)}\n{contrast_title(ckey)}")
        save(fig, model_dir(out, model)
             / f"4_paired_{contrast_file(ckey)}_{variant}.png")
