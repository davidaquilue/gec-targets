"""
GEC fitting: Python port of fdt_eeg's GEC_fit_linHopf.m / GEC_pipeline.m,
subject level only, with the fit trajectory logged.

Learning rule (every iteration, on the entries that are non-zero in the
initial matrix, i.e. the SC support):

    C += eps_fc (FC_emp - FC_sim) + eps_fs (FS_emp - FS_sim)
    C  = max(C, 0), then the model's normalisation

``step_norm`` weighs the parts of the error in the step:
    "none"        as above (original)
    "rms"         C += eps_fc dFC + eps_fs (w_S dFS_S + w_A dFS_A), with
                  w_k = rms(FC_emp) / rms(target_k) (off-diagonal, each
                  subject's own targets): FS_S and FS_A pull as hard as FC,
                  whatever their size, and eps_fc keeps its meaning.

Convergence is checked every ``check_every`` iterations on an error E:
    "legacy"      E = mse(FC) + mse(FS) over the full matrices (original);
                  stop when (E_old - E) / E < tol (includes E increasing)
    "normalized"  E = sum over FC, FS_S, FS_A of mse / mean square of the
                  target (off-diagonal), so the small antisymmetric part
                  weighs as much as the rest. Same stopping rule.
    "fc_fs_a"     as "normalized", over FC and FS_A only.
    "none"        run all ``max_iter`` iterations (to study the dynamics)
Since mse(FS) = mse(FS_S) + mse(FS_A) exactly, the legacy error already
contains FS_A, but with the weight of its (small) size.

Initial matrix: the dataset SC (``sc_path``), or for channel-space data
without SC a seeded random |randn| matrix. With random initialisation the fit
is repeated ``n_init`` times (seed = config seed, subject id, init index) and
C, FC_sim and FS_sim are averaged over the successful runs, as NSIMFIT did;
the iteration count, stop reason and history saved are those of the first
run.
"""
import json
import time
import zlib

import numpy as np
from joblib import Parallel, delayed, parallel_config

from .analysis import sym, asym, triu, offdiag, corr
from .config import fit_dir, targets_dir
from .models import build_model
from .results import is_complete, save_npz

HISTORY_KEYS = ("iter", "error", "r_fc", "r_fs", "r_fs_s", "r_fs_a",
                "mse_fc", "mse_fs", "mse_fs_s", "mse_fs_a")
START_ERROR = 1e5  # SETOLDERROR in the MATLAB code
# Parts of the error E of the normalised rules; "legacy" and "none" are not
# normalised
NORMALIZED_RULES = {"normalized": ("fc", "fs_s", "fs_a"),
                    "fc_fs_a": ("fc", "fs_a")}
CONVERGENCE_RULES = ("legacy", "none", *NORMALIZED_RULES)
STEP_NORMS = ("none", "rms")


def edge_parts(fc, fs):
    """Upper triangle of FC, FS_S and FS_A."""
    return {"fc": triu(fc), "fs_s": triu(sym(fs)), "fs_a": triu(asym(fs))}


def fit_metrics(fc_emp, fs_emp, fc_sim, fs_sim):
    """Correlation and mse of FC, FS, FS_S and FS_A (off-diagonal / upper
    triangle, as in analysis.edge_vectors)."""
    pairs = {
        "fc": (triu(fc_emp), triu(fc_sim)),
        "fs": (offdiag(fs_emp), offdiag(fs_sim)),
        "fs_s": (triu(sym(fs_emp)), triu(sym(fs_sim))),
        "fs_a": (triu(asym(fs_emp)), triu(asym(fs_sim))),
    }
    out = {}
    for name, (e, s) in pairs.items():
        out[f"r_{name}"] = corr(e, s)
        out[f"mse_{name}"] = float(np.mean((e - s) ** 2))
    return out


def rule_error(rule, fc_emp, fs_emp, fc_sim, fs_sim):
    if rule in NORMALIZED_RULES:
        emp, sim = edge_parts(fc_emp, fs_emp), edge_parts(fc_sim, fs_sim)
        return float(sum(np.mean((emp[k] - sim[k]) ** 2) / np.mean(emp[k] ** 2)
                         for k in NORMALIZED_RULES[rule]))
    return float(np.mean((fc_emp - fc_sim) ** 2)
                 + np.mean((fs_emp - fs_sim) ** 2))


def step_weights(step_norm, fc_emp, fs_emp):
    """Weights of the FC, FS_S and FS_A errors in the learning step."""
    if step_norm == "none":
        return np.ones(3)
    rms = {k: np.sqrt(np.mean(v ** 2))
           for k, v in edge_parts(fc_emp, fs_emp).items()}
    return np.array([1.0, rms["fc"] / rms["fs_s"], rms["fc"] / rms["fs_a"]])


def fit_gec(fc_emp, fs_emp, model, C0, tau_s, eps_fc, eps_fs,
            max_iter=5000, tol=1e-5, check_every=10, convergence="legacy",
            step_norm="none", log_every=10, **_):
    """Fit one subject. Returns coupling C, simulated FC / FS, history."""
    weights = step_weights(step_norm, fc_emp, fs_emp)
    C = model.prepare(C0)
    mask = (C > 0) & ~np.eye(len(C), dtype=bool)
    history = {k: [] for k in HISTORY_KEYS}
    old_error, stop = START_ERROR, "max_iter"

    for it in range(1, max_iter + 1):
        fc_sim, fs_sim = model.observables(C, tau_s)
        if not (np.all(np.isfinite(fc_sim)) and np.all(np.isfinite(fs_sim))):
            raise FloatingPointError(f"non-finite observables at iter {it}")
        error = None
        if it % check_every == 0:
            error = rule_error(convergence, fc_emp, fs_emp, fc_sim, fs_sim)
        if log_every and (it == 1 or it % log_every == 0):
            for k, v in {"iter": it, "error": np.nan if error is None
                         else error, **fit_metrics(fc_emp, fs_emp, fc_sim,
                                                   fs_sim)}.items():
                history[k].append(v)
        if error is not None and convergence != "none":
            if (old_error - error) / error < tol:
                stop = "error_increase" if error > old_error else "tol"
                break
            old_error = error

        if step_norm == "none":   # as written, to keep the original fits
            step = eps_fc * (fc_emp - fc_sim) + eps_fs * (fs_emp - fs_sim)
        else:
            d_fs = fs_emp - fs_sim
            step = (eps_fc * weights[0] * (fc_emp - fc_sim)
                    + eps_fs * (weights[1] * sym(d_fs)
                                + weights[2] * asym(d_fs)))
        C = C + np.where(mask, step, 0.0)
        C = model.normalize(np.where(mask & (C < 0), 0.0, C))

    # Observables of the returned C (the loop's last ones may lag one step)
    fc_sim, fs_sim = model.observables(C, tau_s)
    return {"C": C, "fc_sim": fc_sim, "fs_sim": fs_sim, "mask": mask,
            "n_iter": it, "stop": stop, "step_weights": weights,
            "history": {k: np.asarray(v) for k, v in history.items()}}


def load_matrix(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)


def initial_matrices(sc, n_nodes, subject, n_init, seed):
    """SC (single deterministic run) or n_init seeded random matrices."""
    if sc is not None:
        if sc.shape != (n_nodes, n_nodes):
            raise ValueError(f"SC is {sc.shape}, data have {n_nodes} nodes")
        return [sc]
    return [np.abs(np.random.default_rng(
        [seed, zlib.crc32(subject.encode()), k]).standard_normal(
        (n_nodes, n_nodes))) for k in range(n_init)]


def fit_subject(task, meta, target_file, sc, out_file):
    """All initialisations of one subject; saves one .npz."""
    half = "" if task["half"] == "all" else f"_{task['half']}"
    with np.load(target_file) as t:
        fc_emp, fs_emp = t[f"fc{half}"], t[f"fs{half}"]
    subject = out_file.stem
    model = build_model(task["model"], task["model_params"], meta, subject)
    fit = task["fit"]
    start = time.time()
    runs = []
    inits = initial_matrices(sc, len(fc_emp), subject, fit.get("n_init", 1),
                             fit.get("seed", 0))
    for C0 in inits:
        try:
            runs.append(fit_gec(fc_emp, fs_emp, model, C0, meta["tau_s"],
                                **fit))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as err:
            print(f"{subject}: fit failed ({err})")
    if not runs:
        return subject, False

    C = np.mean([r["C"] for r in runs], axis=0)
    fc_sim = np.mean([r["fc_sim"] for r in runs], axis=0)
    fs_sim = np.mean([r["fs_sim"] for r in runs], axis=0)
    first = runs[0]
    save_npz(
        out_file, C=C, friction=model.friction(C), jacobian=model.jacobian(C),
        fc_sim=fc_sim, fs_sim=fs_sim, mask=first["mask"],
        n_iter=first["n_iter"], stop=first["stop"],
        step_weights=first["step_weights"],
        n_success=len(runs), n_init=len(inits),
        runtime_s=time.time() - start, sigma=model.sigma,
        **fit_metrics(fc_emp, fs_emp, fc_sim, fs_sim),
        **{f"history_{k}": v for k, v in first["history"].items()},
    )
    return subject, True


def run_task(cfg, task, n_jobs=1):
    """Fit every subject of one task in parallel; finished subjects are
    skipped, so an interrupted task can simply be resubmitted."""
    # Checked here: inside the fit a ValueError is a failed subject
    for key, allowed in (("convergence", CONVERGENCE_RULES),
                         ("step_norm", STEP_NORMS)):
        if task["fit"].get(key, allowed[0]) not in allowed:
            raise ValueError(f"{key} '{task['fit'][key]}' is not one of "
                             f"{allowed}")
    t_dir = targets_dir(cfg, task["dataset"], task["variant"])
    if not (t_dir / "meta.json").exists():
        raise FileNotFoundError(f"No targets in {t_dir}: run the 'targets' "
                                f"step first")
    with open(t_dir / "meta.json") as f:
        meta = json.load(f)
    out_dir = fit_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "task.json", "w") as f:
        json.dump(task, f, indent=1, default=str)

    sc_path = cfg["datasets"][task["dataset"]].get("sc_path")
    sc = None if sc_path is None else load_matrix(sc_path)
    # A truncated file (job killed while writing, before writes were made
    # atomic) is fitted again
    todo = [s for s in meta["subjects"]
            if not is_complete(out_dir / f"{s}.npz")]
    print(f"Task {task['id']}: {task['dataset']} / {task['variant']} / "
          f"{task['model']} / {task['tag']} / {task['half']}: "
          f"{len(todo)} of {len(meta['subjects'])} subjects to fit, "
          f"{n_jobs} jobs")
    # One BLAS thread per worker: the parallelism is over subjects
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(fit_subject)(task, meta, t_dir / f"{s}.npz", sc,
                                 out_dir / f"{s}.npz") for s in todo)
    failed = [s for s, ok in results if not ok]
    if failed:
        print(f"Failed subjects: {failed}")
    return out_dir
