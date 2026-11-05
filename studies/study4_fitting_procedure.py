# %% [markdown]
# # Study 4. Does the updated fitting procedure fit FS_A better?
#
# Studies 2 and 3 show that FS_A, the only part of the targets that carries
# the direction of time, is the one the GEC fits worst. The updated
# procedure (`config.PROCEDURES["updated"]`, see gectargets/fitting.py)
# changes the fit in two ways:
# * **weighted step** (`step_norm: "rms"`): in the learning step the errors
#   of FS_S and FS_A are scaled by rms(FC) / rms(target) of the subject's own
#   empirical targets, so that the three targets pull equally hard on C (in
#   the original step FS_A, much smaller than FC, barely moves it);
# * **FC + FS_A stopping rule** (`convergence: "fc_fs_a"`): the fit stops
#   when the normalised error of FC and FS_A (sum of mse / mean square of the
#   target) stops improving; FS_S is left out.
#
# Studies 2 and 3 show the updated fits over the grid as models of their own
# (`<model>_updated`). Here, for each dataset, variant and model, three fits
# are compared subject by subject:
# * **lab default**: original procedure, the lab's learning rates
#   (`study3.lab_default` in the config; left out if not set);
# * **best FS_A**: original procedure, the cell with the best mean fit of
#   FS_A;
# * **updated**: updated procedure, its cell with the best mean fit of FS_A.
#
# The best cells are those of Study 3 (`downstream.selected_cells`: among the
# cells fitted for most subjects). Sections:
# 1. the fit of FC, FS_S and FS_A,
# 2. the downstream measures: their level, and whether the fits order the
#    subjects the same way,
# 3. whether the measures follow the data (the empirical FS_A) or the fit,
# 4. the paired contrasts between conditions (HCP working memory vs rest,
#    sleep N3 vs wake), at each of the fits.
#
# Needs: the fits of the original and the updated grids (the config's sweep)
# and, for sections 2-4, the downstream measures of the three fits (all of
# them are computed for Study 3). Run as a script
# (`python studies/study4_fitting_procedure.py CONFIG`) or cell by cell.
# Outputs in `<results_root>/studies/study4/<model>/`, one folder per model.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from gectargets.analysis import asym, triu
from gectargets.config import (PROCEDURES, expand_tasks, load_config,
                               model_dir, procedure, study_dir)
from gectargets.downstream import (load_downstream, panel_key,
                                   selected_cells, well_fitted)
from gectargets.plotting import (GRAY, dataset_label, figure_title,
                                 model_label, save)
from gectargets.results import load_fits, load_targets, save_per_model

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/electrophysiology.json"
X_PARAM, Y_PARAM = "eps_fs", "eps_fc"     # the grid the cells are taken from
UPDATED = "updated"                       # the procedure (config.PROCEDURES)
# Section 4: paired contrasts (a, b) between conditions, shown as a - b
CONTRASTS = [("HCP_WMLR", "HCP_RESTLR"), ("Sleep_N3", "Sleep_W")]
# The fits compared, in the order of the figures: one colour and marker each
FITS = {"lab_default": {"label": "lab default", "color": "#2a78d6",
                        "marker": "o"},
        "best_fs_a": {"label": "best FS_A", "color": "#eda100",
                      "marker": "s"},
        "updated": {"label": "updated, best FS_A", "color": "#1baf7a",
                    "marker": "D"}}
# Paired comparisons (a - b) of the tables
PAIRS = [("updated", "best_fs_a"), ("updated", "lab_default"),
         ("best_fs_a", "lab_default")]
FIT_TARGETS = {"FC": "fc", "FS_S": "fs_s", "FS_A": "fs_a"}
MEASURES = {"fdt": "mean FDT violation", "epr": "EPR"}  # no EPR: linear_hopf
KEYS = ("r_fc", "r_fs_s", "r_fs_a", "n_iter", "stop", "step_weights")

cfg = load_config(CONFIG)
out = study_dir(cfg, "study4")


def role(t):
    """"original" (original step and stopping rule), UPDATED, or None (any
    other settings, e.g. convergence "none")."""
    name = procedure(t)
    if name == UPDATED:
        return name
    if (name == "original" and "step_norm" not in t["fit"]
            and t["fit"].get("convergence", "legacy") == "legacy"):
        return name
    return None


def comparison_key(t):
    """Dataset, variant, model and the other swept settings: the same for
    the original and the updated grid of one model."""
    _, _, _, extra = panel_key(t, X_PARAM, Y_PARAM)
    skip = PROCEDURES[UPDATED]
    return (t["dataset"], t["variant"], t["model"],
            tuple((k, v) for k, v in extra if k not in skip))


def settings(extra):
    return [f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}"
            for k, v in extra]


def comparison_name(key):
    """Name of a comparison in the tables."""
    dataset, variant, model, extra = key
    return " | ".join([dataset, variant, model] + settings(extra))


def comparison_title(key):
    """Title of a panel in the figures; the model is the figure's title."""
    dataset, variant, _, extra = key
    return " | ".join([dataset_label(dataset), variant] + settings(extra))


tasks, fits = [], {}
for t in expand_tasks(cfg):
    if (t["half"] != "all" or X_PARAM not in t["swept"]
            or Y_PARAM not in t["swept"] or role(t) is None):
        continue
    f = load_fits(cfg, t, keys=KEYS)   # the matrices are not needed
    if f is None:
        print(f"task {t['id']}: no fits yet, skipped")
        continue
    tasks.append(t)
    fits[t["id"]] = f

# The three fits of every comparison, from the cells of each grid
grids = {}
for t in tasks:
    grids.setdefault(panel_key(t, X_PARAM, Y_PARAM), []).append(t)
cells = selected_cells(cfg, tasks, fits, X_PARAM, Y_PARAM)
comparisons = {}
for key, group in grids.items():
    chosen = comparisons.setdefault(comparison_key(group[0]), {})
    if role(group[0]) == UPDATED:
        chosen["updated"] = cells[key]["best_fs_a"]
    else:
        chosen["lab_default"] = cells[key]["lab_default"]
        chosen["best_fs_a"] = cells[key]["best_fs_a"]
for key in list(comparisons):
    chosen = {n: comparisons[key].get(n) for n in FITS}
    comparisons[key] = {n: t for n, t in chosen.items() if t is not None}
    if chosen["updated"] is None or chosen["best_fs_a"] is None:
        print(f"{comparison_name(key)}: needs both the original and the "
              f"updated grid, skipped")
        del comparisons[key]
if not comparisons:
    sys.exit("No model fitted on both the original and the updated grid: "
             "run the fits of the config's updated sweep entries first")
print(f"{len(comparisons)} comparisons")


def by_model(keys):
    """Comparison keys grouped by model, {model: [keys]}."""
    groups = {}
    for key in keys:
        groups.setdefault(key[2], []).append(key)
    return groups


def figure(n_rows, n_cols):
    n_cols = max(n_cols, 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols,
                                                      3.0 * n_rows),
                             squeeze=False)
    return fig, axes


def finish(fig, axes, model, names, name):
    """Hide the empty panels; legend of the fits *names* and the model as
    the title, above the panels; save in the model's folder."""
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)
    fig.tight_layout()
    handles = [Line2D([], [], linestyle="none", markersize=7,
                      label=FITS[n]["label"], color=FITS[n]["color"],
                      marker=FITS[n]["marker"]) for n in FITS if n in names]
    legend = fig.legend(handles=handles, loc="lower center",
                        ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
                        fontsize=8)
    figure_title(fig, model_label(model), above=legend)
    save(fig, model_dir(out, model) / name)


def fits_of(keys):
    """The fits present in at least one of the comparisons."""
    return [n for n in FITS if any(n in comparisons[k] for k in keys)]


def paired_panel(ax, values, title, ylabel):
    """values: DataFrame subjects x fits. One gray line per subject across
    the fits, the mean of each fit as its marker."""
    names = [n for n in FITS if n in values]
    x = np.arange(len(names))
    ax.plot(x, values[names].to_numpy(float).T, color=GRAY, linewidth=0.6,
            alpha=0.4)
    for i, n in enumerate(names):
        ax.scatter(i, values[n].mean(), s=70, zorder=3, color=FITS[n]["color"],
                   marker=FITS[n]["marker"], edgecolor="#fcfcfb",
                   linewidth=1.5)
    ax.set(title=title, ylabel=ylabel, xticks=x,
           xticklabels=[FITS[n]["label"] for n in names],
           xlim=(-0.4, len(names) - 0.6))
    ax.tick_params(axis="x", labelrotation=20)


def paired(a, b):
    """a - b over the subjects of both (Series by subject): n, median,
    Wilcoxon signed-rank p."""
    d = (a - b).dropna()
    if len(d) < 3 or not np.any(d != 0):
        return len(d), (float(np.median(d)) if len(d) else np.nan), np.nan
    return len(d), float(np.median(d)), float(wilcoxon(d).pvalue)


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


# One row per fit; one per fit and subject
rows, frames = [], []
for key, chosen in comparisons.items():
    for name, t in chosen.items():
        f = fits[t["id"]]
        # Fits made before step_norm existed have no weights saved: 1
        w = np.asarray(f.get("step_weights",
                             np.ones((len(f["subjects"]), 3))))
        rows.append({"comparison": comparison_name(key), "model": key[2],
                     "fit": name, "id": t["id"],
                     X_PARAM: t["swept"][X_PARAM],
                     Y_PARAM: t["swept"][Y_PARAM],
                     "n_subjects": len(f["subjects"]),
                     "n_iter_median": float(np.median(f["n_iter"])),
                     "max_iter_share": float(np.mean(f["stop"] == "max_iter")),
                     **{f"r_{v}": float(np.nanmean(f[f"r_{v}"]))
                        for v in FIT_TARGETS.values()},
                     "w_fs_s_median": float(np.median(w[:, 1])),
                     "w_fs_a_median": float(np.median(w[:, 2]))})
        frames.append(pd.DataFrame({"comparison": comparison_name(key),
                                    "fit": name, "subject": f["subjects"],
                                    **{f"r_{v}": f[f"r_{v}"]
                                       for v in FIT_TARGETS.values()}}))
fits_table = pd.DataFrame(rows)
per_subject = pd.concat(frames, ignore_index=True)
save_per_model(fits_table, out, "0_fits_compared.csv")
with pd.option_context("display.width", 220, "display.precision", 3):
    print(fits_table.to_string(index=False))


def subjects_by_fit(key, column, table=per_subject):
    """DataFrame subjects x fits of one column, for one comparison."""
    d = table[table.comparison == comparison_name(key)]
    return d.pivot(index="subject", columns="fit", values=column)


# %% [markdown]
# ## 1. The fit of FC, FS_S and FS_A
# corr(empirical, simulated) of each subject (gray lines, paired across the
# fits) and its mean over subjects (marker). The learning rates, iterations
# and step weights of the three fits are in `0_fits_compared.csv`.
#
# Table `1_fit_paired.csv`: for every pair of fits (a - b), over the
# subjects fitted in both, the median difference and the Wilcoxon
# signed-rank p, per target.

# %%
pair_rows = []
for key, chosen in comparisons.items():
    for a, b in PAIRS:
        if a not in chosen or b not in chosen:
            continue
        row = {"comparison": comparison_name(key), "model": key[2],
               "a": a, "b": b}
        for target, v in FIT_TARGETS.items():
            values = subjects_by_fit(key, f"r_{v}")
            row["n_subjects"], row[f"d_r_{v}_median"], row[f"p_r_{v}"] = \
                paired(values[a], values[b])
        pair_rows.append(row)
fit_pairs = pd.DataFrame(pair_rows)
save_per_model(fit_pairs, out, "1_fit_paired.csv")
with pd.option_context("display.width", 220, "display.precision", 3):
    print(fit_pairs.to_string(index=False))

for model, keys in by_model(comparisons).items():
    fig, axes = figure(len(keys), len(FIT_TARGETS))
    for row, key in zip(axes, keys):
        for ax, (target, v) in zip(row, FIT_TARGETS.items()):
            paired_panel(ax, subjects_by_fit(key, f"r_{v}"),
                         f"{comparison_title(key)}\n{target}",
                         "corr(emp, sim)")
    finish(fig, axes, model, fits_of(keys), "1_fit.png")


# %% [markdown]
# ## 2. The downstream measures: level and order of the subjects
# Each subject's mean FDT violation and EPR (stable fits only) at the three
# fits, as in section 1. Does fitting FS_A better change what is reported?
#
# Table `2_downstream_paired.csv`: for every pair of fits, the median
# difference and its Wilcoxon p (the level), the Spearman correlation over
# subjects (the order of the subjects) and, for FDT, the mean over subjects
# of the correlation between the node patterns. Fits without downstream
# measures are listed and left out.

# %%
down, missing = {}, []
for key, chosen in comparisons.items():
    for name, t in chosen.items():
        d = load_downstream(cfg, t)
        if d is None:
            missing.append(t["id"])
            continue
        stable = d["stable"].astype(bool)
        subjects = np.asarray(d["subjects"])[stable]
        down[key, name] = (
            pd.DataFrame({"fdt": d["fdt"][stable].mean(axis=1),
                          "epr": d["epr"][stable].astype(float)},
                         index=subjects),
            pd.DataFrame(d["fdt"][stable], index=subjects))
if missing:
    print(f"No downstream measures yet for tasks {missing}: python -m "
          f"gectargets downstream {CONFIG} --task ID")
down_keys = [k for k in comparisons if any((k, n) in down for n in FITS)]
measured = pd.concat(
    [m.assign(comparison=comparison_name(k), fit=n).rename_axis("subject")
     .reset_index() for (k, n), (m, _) in down.items()], ignore_index=True) \
    if down else pd.DataFrame()


def measures_of(key):
    """Measures with values for a comparison (no EPR for linear_hopf)."""
    d = measured[measured.comparison == comparison_name(key)]
    return [m for m in MEASURES if d[m].notna().any()]


down_rows = []
for key in down_keys:
    for a, b in PAIRS:
        if (key, a) not in down or (key, b) not in down:
            continue
        (ma, na), (mb, nb) = down[key, a], down[key, b]
        common = ma.index.intersection(mb.index)
        row = {"comparison": comparison_name(key), "model": key[2],
               "a": a, "b": b, "n_subjects": len(common)}
        for m in measures_of(key):
            _, row[f"{m}_d_median"], row[f"{m}_p"] = paired(ma[m], mb[m])
            row[f"{m}_order"] = spearman(ma.loc[common, m],
                                         mb.loc[common, m])
        row["fdt_nodes"] = mean_row_corr(na.loc[common].to_numpy(),
                                         nb.loc[common].to_numpy())
        down_rows.append(row)
down_pairs = pd.DataFrame(down_rows)
save_per_model(down_pairs, out, "2_downstream_paired.csv")
if len(down_pairs):
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(down_pairs.to_string(index=False))

for model, keys in by_model(down_keys).items():
    ms = [m for m in MEASURES if any(m in measures_of(k) for k in keys)]
    fig, axes = figure(len(keys), len(ms))
    for row, key in zip(axes, keys):
        for ax, m in zip(row, ms):
            if m in measures_of(key):
                paired_panel(ax, subjects_by_fit(key, m, measured),
                             f"{comparison_title(key)}\n{MEASURES[m]}",
                             MEASURES[m])
    finish(fig, axes, model, fits_of(keys), "2_downstream.png")


# %% [markdown]
# ## 3. Do the measures follow the data or the fit?
# At each fit, over subjects (as Study 3, section 3):
# * vs data: Spearman(measure, the empirical irreversibility, mean square of
#   the subject's FS_A over the upper triangle);
# * vs fit: Spearman(measure, the subject's corr(emp, sim) of FS_A);
# * FDT node pattern vs data: mean over subjects of corr(node FDT, node
#   empirical asymmetry sum_j FS_A_ij^2).
# If fitting FS_A better makes the measures reflect the recordings, "vs
# data" is higher for the fit with the better FS_A (section 1).

# %%
empirical = {}
for dataset, variant, _, _ in down_keys:
    if (dataset, variant) not in empirical:
        targets = load_targets(cfg, dataset, variant, keys=("fs",))
        fs_a = asym(targets["fs"])
        empirical[dataset, variant] = (
            pd.Series(np.mean(triu(fs_a) ** 2, axis=-1),
                      index=targets["subjects"]),
            pd.DataFrame((fs_a ** 2).sum(axis=-1), index=targets["subjects"]))

RELATIONS = {"vs_data": "vs data", "vs_fit": "vs fit",
             "nodes_vs_data": "node pattern vs data"}
follow_rows = []
for (key, name), (m_table, nodes_fdt) in down.items():
    power, nodes = empirical[key[0], key[1]]
    fit = subjects_by_fit(key, "r_fs_a")[name]
    idx = m_table.index
    row = {"comparison": comparison_name(key), "model": key[2], "fit": name,
           "id": comparisons[key][name]["id"], "n_stable": len(idx)}
    for m in measures_of(key):
        row[f"{m}_vs_data"] = spearman(m_table[m], power.loc[idx])
        row[f"{m}_vs_fit"] = spearman(m_table[m], fit.reindex(idx))
    row["fdt_nodes_vs_data"] = mean_row_corr(nodes_fdt.to_numpy(),
                                             nodes.loc[idx].to_numpy())
    follow_rows.append(row)
follow = pd.DataFrame(follow_rows)
save_per_model(follow, out, "3_fit_vs_data.csv")

for model, keys in by_model(down_keys).items():
    ms = [m for m in MEASURES if any(m in measures_of(k) for k in keys)]
    fig, axes = figure(len(keys), len(ms))
    for row, key in zip(axes, keys):
        g = follow[follow.comparison == comparison_name(key)]
        for ax, m in zip(row, ms):
            if m not in measures_of(key):
                continue
            relations = [r for r in RELATIONS
                         if r != "nodes_vs_data" or m == "fdt"]
            columns = [f"fdt_{r}" if r == "nodes_vs_data" else f"{m}_{r}"
                       for r in relations]
            ax.axvline(0, color=GRAY, linewidth=1)
            for _, p in g.iterrows():
                # Each fit a little apart, so equal values stay visible
                dodge = 0.15 * (list(FITS).index(p.fit) - 1)
                ax.scatter(p[columns].to_numpy(float),
                           np.arange(len(relations)) + dodge, s=40, zorder=3,
                           color=FITS[p.fit]["color"],
                           marker=FITS[p.fit]["marker"])
            ax.set(title=f"{comparison_title(key)}\n{MEASURES[m]}: Spearman",
                   xlim=(-1.05, 1.05), yticks=range(len(relations)),
                   yticklabels=[RELATIONS[r] for r in relations])
            ax.invert_yaxis()
    finish(fig, axes, model, fits_of(keys), "3_fit_vs_data.png")


# %% [markdown]
# ## 4. Paired contrasts between conditions, at each fit
# As Study 3, section 4 (the contrasts (a, b) of CONTRASTS), but at the
# three fits only; a contrast needs both datasets fitted on the original and
# the updated grid. Both conditions are fit with the same learning rates:
# the lab default, and for the two best FS_A fits the cell with the best
# mean fit of FS_A averaged over the two conditions (among the cells fitted
# for most subjects in both). Subjects with a stable fit in both conditions;
# a - b: Cohen's d_z = mean(diff) / SD(diff), the paired t-test and, without
# normality, the Wilcoxon signed-rank test (three tests per measure, not
# corrected).

# %%
def matched_best(group_a, group_b):
    """(a, b) fits of the cell, matched by its settings (tag), with the best
    mean corr FS_A over the two conditions."""
    b_by_tag = {t["tag"]: t for t in well_fitted(group_b, fits)}
    both = [(u, b_by_tag[u["tag"]]) for u in well_fitted(group_a, fits)
            if u["tag"] in b_by_tag]
    if not both:
        return None
    return max(both, key=lambda p: np.nanmean(fits[p[0]["id"]]["r_fs_a"])
               + np.nanmean(fits[p[1]["id"]]["r_fs_a"]))


contrast_rows = []
for a_ds, b_ds in CONTRASTS:
    n_rows = len(contrast_rows)
    for key in comparisons:
        if key[0] != b_ds:
            continue
        key_a = (a_ds, *key[1:])
        if key_a not in comparisons:
            continue
        matched = {}
        lab_a = comparisons[key_a].get("lab_default")
        lab_b = comparisons[key].get("lab_default")
        if lab_a is not None and lab_b is not None:
            matched["lab_default"] = (lab_a, lab_b)
        for name, procedure_name in (("best_fs_a", "original"),
                                     ("updated", UPDATED)):
            groups = [g for k, g in grids.items()
                      if comparison_key(g[0]) in (key, key_a)
                      and role(g[0]) == procedure_name]
            g_a = [t for g in groups for t in g if t["dataset"] == a_ds]
            g_b = [t for g in groups for t in g if t["dataset"] == b_ds]
            pair = matched_best(g_a, g_b) if g_a and g_b else None
            if pair is not None:
                matched[name] = pair
        for name, (u, t) in matched.items():
            a, b = load_downstream(cfg, u), load_downstream(cfg, t)
            if a is None or b is None:
                print(f"Section 4: no downstream measures yet for tasks "
                      f"{[v['id'] for v, d in ((u, a), (t, b)) if d is None]}")
                continue
            frames = [pd.DataFrame({"fdt": d["fdt"].mean(axis=1),
                                    "epr": d["epr"].astype(float)},
                                   index=d["subjects"])[
                                       d["stable"].astype(bool)]
                      for d in (a, b)]
            common = frames[0].index.intersection(frames[1].index)
            row = {"comparison": f"{a_ds} vs {comparison_name(key)}",
                   "model": key[2], "fit": name, "dataset_a": a_ds,
                   "dataset_b": b_ds, "id_a": u["id"], "id_b": t["id"],
                   X_PARAM: t["swept"][X_PARAM],
                   Y_PARAM: t["swept"][Y_PARAM], "n_subjects": len(common)}
            for m in MEASURES:
                x = frames[0].loc[common, m].to_numpy(float)
                y = frames[1].loc[common, m].to_numpy(float)
                if len(common) < 3 or not np.isfinite(x - y).all():
                    continue
                diff = x - y
                row.update({f"{m}_a_mean": x.mean(), f"{m}_b_mean": y.mean(),
                            f"{m}_d_z": diff.mean() / diff.std(ddof=1),
                            f"{m}_p_t": ttest_rel(x, y).pvalue,
                            f"{m}_p_wilcoxon": wilcoxon(x, y).pvalue})
            contrast_rows.append(row)
    if len(contrast_rows) == n_rows:
        print(f"Section 4: no matching fits of {a_ds} and {b_ds} on both "
              f"grids, skipped")
contrast = pd.DataFrame(contrast_rows)
if len(contrast):
    save_per_model(contrast, out, "4_contrast.csv")
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(contrast.to_string(index=False))

# d_z of each fit, its paired t-test p next to it
for model, g_model in (contrast.groupby("model") if len(contrast) else []):
    names = list(dict.fromkeys(g_model.comparison))
    ms = [m for m in MEASURES if f"{m}_d_z" in g_model
          and g_model[f"{m}_d_z"].notna().any()]
    fig, axes = figure(len(names), len(ms))
    for row, name in zip(axes, names):
        g = g_model[g_model.comparison == name]
        a_label = dataset_label(g.dataset_a.iloc[0])
        b_label = dataset_label(g.dataset_b.iloc[0])
        for ax, m in zip(row, ms):
            fit_names = [n for n in FITS if n in set(g.fit)]
            ax.axhline(0, color=GRAY, linewidth=1)
            for i, n in enumerate(fit_names):
                c = g[g.fit == n].iloc[0]
                ax.scatter(i, c[f"{m}_d_z"], s=70, zorder=3,
                           color=FITS[n]["color"], marker=FITS[n]["marker"])
                ax.annotate(f"p {c[f'{m}_p_t']:.2g}", (i, c[f"{m}_d_z"]),
                            xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=7, color="#52514e")
            ax.set(title=f"{a_label} vs {b_label} | {name.split(' | ')[1]}"
                         f"\n{MEASURES[m]}: d_z ({a_label} - {b_label})",
                   xticks=range(len(fit_names)),
                   xticklabels=[FITS[n]["label"] for n in fit_names],
                   xlim=(-0.4, len(fit_names) - 0.4), ylabel="d_z")
            ax.tick_params(axis="x", labelrotation=20)
    finish(fig, axes, model, set(g_model.fit), "4_contrast.png")
