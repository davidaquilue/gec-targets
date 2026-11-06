"""
Downstream measures of the fitted GECs: the FDT violation of each node and
the entropy production rate (EPR), computed from the saved fits (no
refitting).

Two FDT measures, chosen with ``fdt_metrics`` in the config (a list, default
``["integral"]``); each is a key of the downstream files:

``integral`` -> "fdt"
    the analytical integral violation, with fdt_eeg's own functions
    (gectargets/fdt_eeg, verbatim copies), called as in fdt_eeg's
    studies/4c_analytical_GEC_features/step1_compute_gec_features.py:

        B   = -jacobian                     friction, dx/dt = -B x
        fdt = pipeline_analytical_fdt_integral(B, sigma * ones,
                                               normalizations=False)

    with the defaults of fdt_eeg (tmax 100 s, ts 0, Gaussian V0);
    ``normalizations=False`` returns only the integral violation.
``onsager`` -> "fdt_onsager"
    Deco's Onsager-formalism violation, chij of
    gectargets/onsager_fdt.fdt_violation_onsager(jacobian, sigma, N): the
    chi ratio from inv(jacobian) and the stationary covariance.

and, for legacy and network,

    S   = compute_theoretical_covariance(B, sigma)
    epr = compute_entropy_production(B, S, sigma)

sigma is the fit's own (``sigma`` of the fit file). For legacy and network
-jacobian is the saved friction. For linear_hopf the full 2N x 2N system is
used and the FDT of the N observed (x) nodes is kept; its EPR is not
computed.

Results: <fit_dir>/downstream/<subject>.npz, one per fitted subject. A file
that lacks a requested FDT measure gets it added, the measures already in it
are kept: adding "onsager" to a config whose integral FDT is done only
computes the Onsager one (cheap).
"""
import numpy as np
from joblib import Parallel, delayed, parallel_config

from .config import fit_dir, grid_settings, model_name
from .fdt_eeg.analytic_fdt import pipeline_analytical_fdt_integral
from .fdt_eeg.entropy import (compute_entropy_production,
                              compute_theoretical_covariance)
from .onsager_fdt import fdt_violation_onsager
from .results import is_complete, read_npz_files, save_npz

EPR_MODELS = ("legacy", "network")
# FDT measure (config "fdt_metrics") -> its key in the downstream files
FDT_METRICS = {"integral": "fdt", "onsager": "fdt_onsager"}
# A best cell needs at least this share of the grid's most subjects fitted:
# a mean over the few subjects that survive a too large step is no best
MIN_FITTED = 0.9


def fdt_metrics(cfg):
    """The config's FDT measures ("fdt_metrics", default ["integral"])."""
    metrics = list(cfg.get("fdt_metrics", ["integral"]))
    unknown = [m for m in metrics if m not in FDT_METRICS]
    if unknown or not metrics:
        raise ValueError(f"fdt_metrics {metrics}: give one or more of "
                         f"{list(FDT_METRICS)}")
    return metrics


def downstream_dir(cfg, task):
    return fit_dir(cfg, task) / "downstream"


def _saved_keys(path):
    """Arrays in a downstream file; None if it is missing or truncated."""
    if not is_complete(path):
        return None
    with np.load(path) as z:
        return set(z.files)


def subject_downstream(fit_file, model, out_file, metrics):
    """The FDT measures *metrics* (and EPR) of one subject's fit; saves one
    .npz. Measures already in out_file are kept, not recomputed."""
    with np.load(fit_file) as z:
        A, sigma, n = z["jacobian"], float(z["sigma"]), len(z["fc_sim"])
    out = {}
    if _saved_keys(out_file) is not None:
        with np.load(out_file) as z:
            out = {k: z[k] for k in z.files}
    B = -A
    stable = bool(np.all(np.linalg.eigvals(A).real < 0))
    out["stable"] = stable
    if "epr" not in out:
        out["epr"] = np.nan
        if stable and model in EPR_MODELS:
            S = compute_theoretical_covariance(B, sigma)
            out["epr"] = compute_entropy_production(B, S, sigma)
    for metric in metrics:
        key = FDT_METRICS[metric]
        if key in out:
            continue
        out[key] = np.full(n, np.nan)
        if not stable:
            continue
        if metric == "integral":
            out[key] = pipeline_analytical_fdt_integral(
                B, sigma * np.ones(len(B)), normalizations=False)[:n]
        else:
            out[key] = fdt_violation_onsager(A, sigma, n)[0]
    save_npz(out_file, **out)
    return fit_file.stem, stable


def run_downstream(cfg, task, n_jobs=1):
    """Downstream measures of every fitted subject of one task, in parallel;
    subjects that have every measure of the config are skipped."""
    metrics = fdt_metrics(cfg)
    keys = {FDT_METRICS[m] for m in metrics}
    f_dir = fit_dir(cfg, task)
    fits = [p for p in sorted(f_dir.glob("*.npz")) if is_complete(p)]
    out_dir = downstream_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for p in fits:
        saved = _saved_keys(out_dir / p.name)
        if saved is None or not keys <= saved:
            todo.append(p)
    print(f"Task {task['id']}: {task['dataset']} / {task['variant']} / "
          f"{task['model']} / {task['tag']} / {task['half']}: "
          f"{len(todo)} of {len(fits)} fitted subjects to do "
          f"(FDT: {', '.join(metrics)}), {n_jobs} jobs")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(subject_downstream)(p, task["model"], out_dir / p.name,
                                        metrics)
            for p in todo)
    unstable = [s for s, ok in results if not ok]
    if unstable:
        print(f"Unstable fits (no downstream measures): {unstable}")
    return out_dir


def load_downstream(cfg, task):
    """Downstream measures of one task: every FDT key of FDT_METRICS ("fdt",
    "fdt_onsager"; (n_subjects, N), NaN for the subjects whose file lacks
    it), "epr", "stable" (n_subjects,) and "subjects"; None if there are
    none."""
    files = sorted(downstream_dir(cfg, task).glob("*.npz"))
    loaded = [(p, z) for p, z in zip(files, read_npz_files(files))
              if z is not None]
    if not loaded:
        return None
    n = next(len(z[k]) for _, z in loaded for k in FDT_METRICS.values()
             if k in z)
    out = {k: np.array([z[k] if k in z else np.full(n, np.nan)
                        for _, z in loaded])
           for k in FDT_METRICS.values()}
    out.update({k: np.array([z[k] for _, z in loaded])
                for k in ("epr", "stable")})
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
