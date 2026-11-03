"""
Downstream measures of the fitted GECs: the analytical integral FDT violation
and the entropy production rate (EPR), computed from the saved fits (no
refitting) with fdt_eeg's own functions (gectargets/fdt_eeg, verbatim
copies), called as in fdt_eeg's
studies/4c_analytical_GEC_features/step1_compute_gec_features.py:

    B   = -jacobian                     friction, dx/dt = -B x
    fdt = pipeline_analytical_fdt_integral(B, sigma * ones, normalizations=False)
    S   = compute_theoretical_covariance(B, sigma)
    epr = compute_entropy_production(B, S, sigma)

fdt is the integral violation of each node with the defaults of fdt_eeg (tmax
100 s, ts 0, Gaussian V0); ``normalizations=False`` returns only that one, so
the normalised variants are not computed. sigma is the fit's own (``sigma``
of the fit file). For legacy and network -jacobian is the saved friction. For
linear_hopf B is the full 2N x 2N system and the fdt of the N observed (x)
nodes is kept; its EPR is not computed.

Results: <fit_dir>/downstream/<subject>.npz, one per fitted subject.
"""
import numpy as np
from joblib import Parallel, delayed, parallel_config

from .config import fit_dir, grid_settings, model_name
from .fdt_eeg.analytic_fdt import pipeline_analytical_fdt_integral
from .fdt_eeg.entropy import (compute_entropy_production,
                              compute_theoretical_covariance)
from .results import is_complete, read_npz_files, save_npz

EPR_MODELS = ("legacy", "network")
# A best cell needs at least this share of the grid's most subjects fitted:
# a mean over the few subjects that survive a too large step is no best
MIN_FITTED = 0.9


def downstream_dir(cfg, task):
    return fit_dir(cfg, task) / "downstream"


def subject_downstream(fit_file, model, out_file):
    """fdt (and EPR) of one subject's fit; saves one .npz."""
    with np.load(fit_file) as z:
        A, sigma, n = z["jacobian"], float(z["sigma"]), len(z["fc_sim"])
    B = -A
    stable = bool(np.all(np.linalg.eigvals(A).real < 0))
    fdt, epr = np.full(n, np.nan), np.nan
    if stable:
        fdt = pipeline_analytical_fdt_integral(
            B, sigma * np.ones(len(B)), normalizations=False)[:n]
        if model in EPR_MODELS:
            S = compute_theoretical_covariance(B, sigma)
            epr = compute_entropy_production(B, S, sigma)
    save_npz(out_file, fdt=fdt, epr=epr, stable=stable)
    return fit_file.stem, stable


def run_downstream(cfg, task, n_jobs=1):
    """Downstream measures of every fitted subject of one task, in parallel;
    finished subjects are skipped."""
    f_dir = fit_dir(cfg, task)
    fits = [p for p in sorted(f_dir.glob("*.npz")) if is_complete(p)]
    out_dir = downstream_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in fits if not is_complete(out_dir / p.name)]
    print(f"Task {task['id']}: {task['dataset']} / {task['variant']} / "
          f"{task['model']} / {task['tag']} / {task['half']}: "
          f"{len(todo)} of {len(fits)} fitted subjects to do, {n_jobs} jobs")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(subject_downstream)(p, task["model"], out_dir / p.name)
            for p in todo)
    unstable = [s for s, ok in results if not ok]
    if unstable:
        print(f"Unstable fits (no downstream measures): {unstable}")
    return out_dir


def load_downstream(cfg, task):
    """Downstream measures of one task: "fdt" (n_subjects, N), "epr",
    "stable" (n_subjects,) and "subjects"; None if there are none."""
    files = sorted(downstream_dir(cfg, task).glob("*.npz"))
    loaded = [(p, z) for p, z in zip(files, read_npz_files(files))
              if z is not None]
    if not loaded:
        return None
    out = {k: np.array([z[k] for _, z in loaded])
           for k in ("fdt", "epr", "stable")}
    out["subjects"] = [p.stem for p, _ in loaded]
    return out


# ----------------------------------------------- cells singled out on a grid
def panel_key(task, x="eps_fs", y="eps_fc"):
    """Dataset, variant, model (config.model_name: with the procedure, e.g.
    "network_updated") and the other swept settings (config.grid_settings):
    the tasks of one eps_fc x eps_fs grid."""
    extra = tuple(sorted(grid_settings(task, (x, y)).items()))
    return task["dataset"], task["variant"], model_name(task), extra


def well_fitted(group, fits, min_fitted=MIN_FITTED):
    """The tasks of a grid fitted for at least min_fitted of the subjects of
    its most complete cell."""
    n = max(len(fits[t["id"]]["subjects"]) for t in group)
    return [t for t in group
            if len(fits[t["id"]]["subjects"]) >= min_fitted * n]


def selected_cells(cfg, tasks, fits, x="eps_fs", y="eps_fc"):
    """The two cells of each eps grid that the studies single out:

    lab_default  the learning rates of cfg["study3"]["lab_default"][model]
                 (None if not set or not on the grid; only for the original
                 procedure: the key is the model_name)
    best_fs_a    the highest mean corr(empirical, simulated) FS_A over
                 subjects, among the well_fitted cells

    tasks: the grid tasks; fits: task id -> load_fits result. Returns
    {panel_key: {"lab_default": task or None, "best_fs_a": task}}.
    """
    lab = cfg.get("study3", {}).get("lab_default", {})
    panels = {}
    for t in tasks:
        panels.setdefault(panel_key(t, x, y), []).append(t)
    out = {}
    for key, group in panels.items():
        best = max(well_fitted(group, fits),
                   key=lambda t: np.nanmean(fits[t["id"]]["r_fs_a"]))
        rates = lab.get(key[2]) or {}
        default = None
        if rates.get(x) is not None and rates.get(y) is not None:
            default = next((t for t in group
                            if np.isclose(t["swept"][x], rates[x])
                            and np.isclose(t["swept"][y], rates[y])), None)
            if default is None:
                print(f"{key}: lab default {rates} is not on the grid")
        out[key] = {"lab_default": default, "best_fs_a": best}
    return out
