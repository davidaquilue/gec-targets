# %% [markdown]
# # Study 3. Do different GEC models give different answers?
#
# For each dataset, one fit per model (the best hyperparameters by default)
# is compared on
# 1. fit quality (FC, FS, FS_S, FS_A),
# 2. agreement of the fitted couplings between models, subject by subject,
# 3. a downstream measure: the entropy production rate of the fitted linear
#    system, and whether models agree on it and on its group differences,
# 4. reliability of the couplings (between subjects, split-half).
#
# Needs: the fits of the models to compare. Run as a script
# (`python studies/study3_model_comparison.py CONFIG`) or cell by cell.
# Outputs in `<results_root>/studies/study3/`.

# %% Setup
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from gectargets.analysis import (TARGETS, corr, entropy_production,
                                 gec_reliability)
from gectargets.config import expand_tasks, load_config, study_dir
from gectargets.fitting import load_matrix
from gectargets.plotting import color_for, save
from gectargets.results import load_fits, load_meta

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/fdt_datasets.json"
SELECT_BY = "r_fs"   # pick, per dataset and model, the task with the best mean
CHOSEN = {}          # or fix them: {("UCM_AAL", "network"): task_id, ...}
KEYS = {"FC": "fc", "FS": "fs", "FS_S": "fs_s", "FS_A": "fs_a"}

cfg = load_config(CONFIG)
out = study_dir(cfg, "study3")

candidates = {}
for t in expand_tasks(cfg):
    if t["half"] != "all":
        continue
    f = load_fits(cfg, t)
    if f is not None:
        key = (t["dataset"], t["variant"], t["model"])
        candidates.setdefault(key, []).append((t, f))

chosen = {}
for key, options in candidates.items():
    fixed = CHOSEN.get(key[::2])
    t, f = (next(o for o in options if o[0]["id"] == fixed) if fixed is not None
            else max(options, key=lambda o: np.nanmean(o[1][SELECT_BY])))
    chosen[key] = (t, f)
    print(f"{key}: task {t['id']} {t['swept']}")

by_data = {}
for (dataset, variant, model), tf in chosen.items():
    by_data.setdefault((dataset, variant), {})[model] = tf
model_colors = color_for(sorted({m for d in by_data.values() for m in d}))

# %% [markdown]
# ## 1. Fit quality per model
# Per-subject corr(empirical, simulated); one panel per dataset and target.

# %%
fig, axes = plt.subplots(len(by_data), 4, figsize=(13, 2.8 * len(by_data)),
                         squeeze=False)
for row, ((dataset, variant), models) in zip(axes, by_data.items()):
    for ax, key in zip(row, TARGETS):
        values = [models[m][1][f"r_{KEYS[key]}"] for m in models]
        box = ax.boxplot(values, tick_labels=list(models), widths=0.5,
                         patch_artist=True)
        for patch, m in zip(box["boxes"], models):
            patch.set(facecolor=model_colors[m], alpha=0.5)
        ax.set(title=f"{dataset} | {variant}\n{key}",
               ylabel="corr(emp, sim)")
fig.tight_layout()
save(fig, out / "1_fit_quality.png")


# %% [markdown]
# ## 2. Do the models find the same couplings?
# Per subject, the correlation between two models' couplings (off-diagonal,
# on the support both models update) and between their simulated FS_A (the
# directed part). Couplings of different models live on different scales and
# parameterisations, so only patterns are compared.

# %%
def per_subject(f, key, subjects):
    idx = [f["subjects"].index(s) for s in subjects]
    return f[key][idx]


agree_rows = []
for (dataset, variant), models in by_data.items():
    for m1, m2 in itertools.combinations(models, 2):
        f1, f2 = models[m1][1], models[m2][1]
        common = sorted(set(f1["subjects"]) & set(f2["subjects"]))
        mask = f1["mask"][0] & f2["mask"][0]
        C1, C2 = (per_subject(f, "C", common) for f in (f1, f2))
        A1, A2 = (per_subject(f, "fs_sim", common) for f in (f1, f2))
        r_c = [corr(a[mask], b[mask]) for a, b in zip(C1, C2)]
        r_a = [corr((a - a.T)[mask], (b - b.T)[mask]) for a, b in zip(A1, A2)]
        agree_rows.append({"dataset": dataset, "variant": variant,
                           "model_1": m1, "model_2": m2,
                           "n_subjects": len(common),
                           "r_coupling_mean": np.nanmean(r_c),
                           "r_fs_a_sim_mean": np.nanmean(r_a)})
agreement = pd.DataFrame(agree_rows)
agreement.to_csv(out / "2_model_agreement.csv", index=False)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(agreement.to_string(index=False))

# %% [markdown]
# ## 3. Downstream: entropy production rate
# Phi of the fitted system dx/dt = A x + noise (0 if reversible). Its scale
# depends on the model's timescale, so models are compared by rank: Spearman
# between models over subjects, and the group difference of each model
# (Mann-Whitney between the first two groups, rank-biserial effect size).

# %%
epr_rows = []
for (dataset, variant), models in by_data.items():
    meta = load_meta(cfg, dataset, variant)
    groups = dict(zip(meta["subjects"], meta["groups"]))
    for m, (t, f) in models.items():
        for s, A, sigma in zip(f["subjects"], f["jacobian"], f["sigma"]):
            epr_rows.append({"dataset": dataset, "variant": variant,
                             "model": m, "subject": s, "group": groups[s],
                             "epr": entropy_production(A, float(sigma))})
epr = pd.DataFrame(epr_rows)
epr.to_csv(out / "3_entropy_production.csv", index=False)

stats_rows = []
fig, axes = plt.subplots(1, len(by_data), figsize=(4.5 * len(by_data), 3.2),
                         squeeze=False)
for ax, ((dataset, variant), models) in zip(axes[0], by_data.items()):
    d = epr[(epr.dataset == dataset) & (epr.variant == variant)]
    wide = d.pivot(index="subject", columns="model", values="epr")
    for m1, m2 in itertools.combinations(models, 2):
        rho = spearmanr(wide[m1], wide[m2], nan_policy="omit").statistic
        stats_rows.append({"dataset": dataset, "variant": variant,
                           "comparison": f"{m1} vs {m2}", "spearman": rho})
    group_names = sorted(d.group.unique())
    for i, m in enumerate(models):
        dm = d[d.model == m]
        # Rank within model, so all models share one axis
        ranks = dm.epr.rank(pct=True)
        for j, g in enumerate(group_names):
            y = ranks[dm.group == g]
            ax.scatter(np.full(len(y), i + (j - 0.5) * 0.3)
                       + np.random.default_rng(0).uniform(-0.05, 0.05, len(y)),
                       y, s=12, color=model_colors[m],
                       alpha=1.0 if j == 0 else 0.45,
                       label=g if i == 0 else None)
        if len(group_names) >= 2:
            a = dm.epr[dm.group == group_names[0]]
            b = dm.epr[dm.group == group_names[1]]
            test = mannwhitneyu(a, b)
            stats_rows.append({
                "dataset": dataset, "variant": variant, "comparison":
                f"{m}: {group_names[0]} vs {group_names[1]}",
                "p_mannwhitney": test.pvalue,
                "rank_biserial": 1 - 2 * test.statistic / (len(a) * len(b))})
    ax.set(xticks=range(len(models)), xticklabels=list(models),
           ylabel="EPR percentile within model",
           title=f"{dataset} | {variant}\n(opaque: {group_names[0]})")
    ax.legend(fontsize=7)
fig.tight_layout()
save(fig, out / "3_entropy_production.png")
stats = pd.DataFrame(stats_rows)
stats.to_csv(out / "3_entropy_production_stats.csv", index=False)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(stats.to_string(index=False))

# %% [markdown]
# ## 4. Reliability of the couplings per model
# As in Study 2, section 4, for the chosen fit of each model.

# %%
rel_rows = []
for (dataset, variant, model), (t, f) in chosen.items():
    sc_path = cfg["datasets"][dataset].get("sc_path")
    rel_rows.append({"dataset": dataset, "variant": variant, "model": model,
                     **gec_reliability(
                         f, None if sc_path is None else load_matrix(sc_path),
                         load_fits(cfg, {**t, "half": "odd"}),
                         load_fits(cfg, {**t, "half": "even"}))})
reliability = pd.DataFrame(rel_rows)
reliability.to_csv(out / "4_gec_reliability.csv", index=False)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(reliability.to_string(index=False))
