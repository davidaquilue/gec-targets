# gec-targets

What do whole-brain linear models (GEC, generative effective connectivity)
actually fit? This repository studies the **targets** of the GEC fit, the
functional connectivity (FC) and the lagged covariance (FS), and their
**fits**, across datasets and modalities (EEG, MEG, fMRI).

FS is split into a symmetric part `FS_S = (FS + FS')/2` and an antisymmetric
part `FS_A = (FS - FS')/2`. Only FS_A carries the direction of time, so it is
what a GEC needs to estimate irreversibility (entropy production, FDT
violation). The three studies ask:

| Study | Question | Script |
|---|---|---|
| 1 | What do FC, FS, FS_S and FS_A look like, how similar are they across subjects, and how reliable within a subject? | `studies/study1_empirical_targets.py` |
| 2 | How does the fit evolve during training, and do FS_S / FS_A converge when the default rule stops? Which learning rates fit what? | `studies/study2_training_dynamics.py` |
| 3 | Do different GEC models (legacy, network, linear Hopf) give different couplings and different downstream results? | `studies/study3_model_comparison.py` |

The methods (preprocessing, models, learning rule, metrics) are written out in
[docs/methods.md](docs/methods.md).

## Install

```bash
git clone <this repo> && cd gec-targets
conda create -n gectargets python=3.12 -y && conda activate gectargets
pip install -e .
```

## Quick start

The paths in `configs/fdt_datasets.json` start with `$DATA_DIR`: set it in a
`.env` file at the repository root (e.g. `DATA_DIR=D:\`) or in the
environment.

```bash
python -m gectargets tasks   configs/fdt_datasets.json           # what will be fitted
python -m gectargets targets configs/fdt_datasets.json           # 1. empirical targets
python -m gectargets fit     configs/fdt_datasets.json --all     # 2. GEC fits
python studies/study1_empirical_targets.py configs/fdt_datasets.json   # 3. analyses
python studies/study2_training_dynamics.py configs/fdt_datasets.json
python studies/study3_model_comparison.py  configs/fdt_datasets.json
```

Figures and tables are written to `<results_root>/studies/study{1,2,3}/`. The
study scripts are split into `# %%` cells: open them in PyCharm, VS Code or
Jupyter (jupytext) to run them step by step.

## The pipeline

1. **Define** datasets, models and the hyperparameter sweep in a JSON config.
2. **Fit**: `targets` computes FC / FS per subject; `fit` runs the GEC fit of
   one task (one dataset x model x hyperparameter set) for all its subjects
   in parallel.
3. **Analyse**: the study scripts read the targets and fits back.

Every step is restartable: existing targets and subject fits are skipped.

### The config

See `configs/fdt_datasets.json` (UCM MEG, UniRoma EEG, ADNI fMRI). Keys
starting with `_` are comments. Paths are
relative to the config file and may contain `$ENV_VARS`.

| Block | Content |
|---|---|
| `results_root` | where everything is written |
| `datasets` | one entry per dataset: loader, paths, `band` (`null` = use as stored, e.g. fMRI), `sc_path` (`null` = seeded random initialisation, e.g. channel-space EEG), optional `n_epochs` (fixed number of random epochs per subject) |
| `bands` | band name -> `[f_low, f_high]` Hz |
| `preprocessing` | `envelope`, `orthogonalize` (booleans), `tau` (`"FIT"` or samples) |
| `models` | default parameters of each model, including its learning rates `eps_fc`, `eps_fs` |
| `fitting` | `max_iter`, `tol`, `check_every`, `convergence` (`legacy` / `normalized` / `none`), `log_every`, `n_init`, `seed`, `split_half` |
| `sweep` | list of entries `{datasets, models, grid}`; each is the product datasets x models x grid. `grid` keys can be any model parameter, fitting setting or preprocessing option. `split_half: true` in an entry also fits odd and even epochs (3x the cost) |

### Outputs

```
<results_root>/<dataset>/<variant>/targets/meta.json, <subject>.npz
<results_root>/<dataset>/<variant>/fits/<model>/<tag>/<half>/task.json, <subject>.npz
<results_root>/studies/study{1,2,3}/...
```

`variant` is the band plus `_env` / `_orth` (e.g. `alpha_env`, `nofilter`);
`tag` lists the hyperparameters plus a short hash of all settings; `half` is
`all`, `odd` or `even`.

## High-performance computing

Each array task is one hyperparameter set; within it, subjects are fitted in
parallel over the task's CPUs (one BLAS thread each).

```bash
python -m gectargets tasks CONFIG        # check the task list and its size
bash hpc/submit.sh CONFIG [CONDA_ENV]    # targets job, then the fit array
```

`submit.sh` submits `hpc/targets.sbatch`, then `hpc/fit_array.sbatch` as an
array of the right size that starts when the targets are done. Adjust the
`#SBATCH` lines (CPUs, memory, time, partition) to your cluster. Locally,
`python -m gectargets fit CONFIG --task 3 -j 8` runs one task.

Rough cost: one iteration is one Lyapunov solve and one matrix exponential,
O(N^3). Up to 90 nodes a subject takes seconds to a few minutes; at 400 nodes
(Schaefer400) expect several minutes per subject at the default 5000
iterations.

## Using your own data

See [docs/adding_your_data.md](docs/adding_your_data.md): either store your
epochs in the expected layout (no code), or write a ~20-line loader.

## Origin

Ported from the `fdt_eeg` project (MATLAB `GEC_fit_linHopf.m` /
`GEC_pipeline.m` and `studies/0_other_studies/gec_diagnostics`). Differences
from that pipeline are listed at the end of docs/methods.md.
