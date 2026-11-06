# %% [markdown]
# # Study 5. Onsager-formalism FDT violation vs the analytic integral violation
#
# Gustavo Deco's original FDT violation (the Onsager / chi-ratio formalism:
# `chi = |2*COV/sigma^2 + inv(A)|`, `chij = mean_i(chi) / mean_i(chi2)` per
# region, from the user-supplied `Compute_FDT_Violation_Onsager.m`, ported in
# `gectargets/onsager_fdt.py`) is computed on the Sleep_W / Sleep_N3 (wake vs
# N3) `linear_hopf` fits already on disk, and compared with the analytic
# integral violation already computed by `gectargets downstream` (Study 3's
# "fdt" measure, `gectargets/fdt_eeg/analytic_fdt.py`):
# 1. per-cell agreement (Spearman over subjects) between the two measures,
# 2. the paired Sleep N3 vs wake contrast (Cohen's d_z, paired t-test), side
#    by side for both measures, over the eps_fc x eps_fs grid.
#
# Cheap: both measures are read from the fits and downstream measures
# already on disk (no refitting); the Onsager measure costs one inv() and
# one Lyapunov solve per subject per cell (2N = 180 for AAL90).
#
# Needs: the fits and downstream measures of CONFIG's Sleep_W and Sleep_N3
# tasks (`python -m gectargets fit/downstream CONFIG --all`). Run as a
# script (`python studies/study5_onsager_fdt.py [CONFIG]`, default
# configs/fmri.json) or cell by cell. Outputs in
# `<results_root>/studies/study5_onsager/linear_hopf/`.

# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from gectargets.config import expand_tasks, load_config, model_dir, study_dir
from gectargets.downstream import load_downstream
from gectargets.onsager_fdt import fdt_violation_onsager
from gectargets.plotting import GRAY, dataset_label, figure_title, save
from gectargets.results import load_fits, save_per_model

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/fmri.json"
MODEL = "linear_hopf"
CONTRAST = ("Sleep_N3", "Sleep_W")   # a - b: sleep N3 vs wake

cfg = load_config(CONFIG)
out = study_dir(cfg, "study5_onsager")

tasks = [t for t in expand_tasks(cfg)
         if t["dataset"] in CONTRAST and t["model"] == MODEL
         and t["half"] == "all" and "eps_fc" in t["swept"]
         and "eps_fs" in t["swept"]]
print(f"{len(tasks)} {'/'.join(CONTRAST)} {MODEL} grid tasks")

# %% [markdown]
# ## 1. Compute the Onsager FDT violation of every subject and cell
# Read straight from the fit (`jacobian`, `sigma`), no refitting; only
# subjects with a stable downstream fit (Study 3's "stable" flag) are kept.

# %%
rows = []
for t in tasks:
    down = load_downstream(cfg, t)
    if down is None:
        print(f"task {t['id']}: no downstream measures yet, skipped (run "
              f"`python -m gectargets downstream {CONFIG} --all` first)")
        continue
    fits = load_fits(cfg, t, keys=("jacobian", "sigma"))
    if fits is None:
        continue
    by_subject = dict(zip(fits["subjects"], zip(fits["jacobian"], fits["sigma"])))
    n = down["fdt"].shape[1]
    for subject, stable, fdt_node in zip(down["subjects"], down["stable"],
                                         down["fdt"]):
        if not stable or subject not in by_subject:
            continue
        A, sigma = by_subject[subject]
        try:
            _, fdt_onsager = fdt_violation_onsager(A, float(sigma), n)
        except np.linalg.LinAlgError:
            continue
        rows.append({
            "model": MODEL, "id": t["id"], "dataset": t["dataset"],
            "variant": t["variant"], "tag": t["tag"],
            "eps_fc": t["swept"]["eps_fc"], "eps_fs": t["swept"]["eps_fs"],
            "subject": subject, "fdt_integral": float(np.mean(fdt_node)),
            "fdt_onsager": fdt_onsager})
data = pd.DataFrame(rows)
if not len(data):
    sys.exit(f"No stable {'/'.join(CONTRAST)} {MODEL} fits with downstream "
             f"measures found for {CONFIG}: nothing to compare")
save_per_model(data, out, "1_onsager_vs_integral_per_subject.csv")

# %% [markdown]
# ## 2. Do the two measures agree within a cell?
# Spearman correlation over subjects, FDT_integral vs FDT_onsager, at every
# grid cell of each dataset.

# %%
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 3 else np.nan


agreement = pd.DataFrame([
    {"model": MODEL, "id": id_, "dataset": g.dataset.iloc[0],
     "eps_fc": g.eps_fc.iloc[0], "eps_fs": g.eps_fs.iloc[0],
     "n_subjects": len(g),
     "spearman_integral_vs_onsager": spearman(g.fdt_integral, g.fdt_onsager)}
    for id_, g in data.groupby("id")])
save_per_model(agreement, out, "2_agreement_per_cell.csv")
with pd.option_context("display.width", 160, "display.precision", 3):
    print(agreement.to_string(index=False))

# %% [markdown]
# ## 3. Sleep N3 vs wake: does each measure agree on direction?
# Cells are matched between the two datasets by `tag` (same hyperparameters,
# same preprocessing settings), subjects by id (those with a stable fit in
# both). Cohen's d_z and the paired t-test, for both measures, at every
# matched cell.

# %%
a_ds, b_ds = CONTRAST
a_tasks = {t["tag"]: t for t in tasks if t["dataset"] == a_ds}
b_tasks = {t["tag"]: t for t in tasks if t["dataset"] == b_ds}
common_tags = sorted(set(a_tasks) & set(b_tasks))
if not common_tags:
    print(f"No matching {a_ds}/{b_ds} cells (same tag): nothing to contrast")

contrast_rows = []
for tag in common_tags:
    u, t = a_tasks[tag], b_tasks[tag]
    a = data[data.id == u["id"]].set_index("subject")
    b = data[data.id == t["id"]].set_index("subject")
    common = a.index.intersection(b.index)
    if len(common) < 3:
        continue
    row = {"model": MODEL, "tag": tag, "eps_fc": t["swept"]["eps_fc"],
           "eps_fs": t["swept"]["eps_fs"], "n_subjects": len(common)}
    for m in ("fdt_integral", "fdt_onsager"):
        x, y = a.loc[common, m].to_numpy(), b.loc[common, m].to_numpy()
        diff = x - y
        row.update({f"{m}_a_mean": x.mean(), f"{m}_b_mean": y.mean(),
                    f"{m}_d_z": diff.mean() / diff.std(ddof=1),
                    f"{m}_p_t": ttest_rel(x, y).pvalue})
    contrast_rows.append(row)
contrast = pd.DataFrame(contrast_rows)
save_per_model(contrast, out, "3_sleep_n3_vs_w_contrast.csv")
with pd.option_context("display.width", 200, "display.precision", 3):
    print(contrast.to_string(index=False))

# %% [markdown]
# ## 4. Figure: do the two measures' effect sizes agree in sign?
# Each point is one grid cell; the diagonal quadrants (both positive or both
# negative) are where the two FDT measures point the same way.

# %%
if len(contrast):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.scatter(contrast.fdt_integral_d_z, contrast.fdt_onsager_d_z, s=24,
               color=GRAY, edgecolor="black", linewidth=0.4)
    lim = 1.1 * np.nanmax(np.abs(contrast[["fdt_integral_d_z",
                                           "fdt_onsager_d_z"]].to_numpy()))
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, linewidth=0.6,
            linestyle=":")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
           xlabel=f"d_z, analytic integral FDT ({dataset_label(a_ds)} - "
                 f"{dataset_label(b_ds)})",
           ylabel=f"d_z, Onsager FDT ({dataset_label(a_ds)} - "
                 f"{dataset_label(b_ds)})")
    fig.tight_layout()
    figure_title(fig, f"{MODEL}\nSleep N3 vs wake: do the two FDT measures "
                      f"agree?")
    save(fig, model_dir(out, MODEL) / "4_agreement_d_z.png")
