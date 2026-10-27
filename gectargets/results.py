"""Read targets and fits back as stacked arrays for the studies."""
import json

import numpy as np
import pandas as pd

from .config import expand_tasks, fit_dir, tau_scan_dir, targets_dir

EPOCH_KEYS = ("fc_epochs", "fs_epochs")


def load_meta(cfg, dataset, variant):
    """meta.json of the targets: tau, theta, sfreq, subjects, groups, ..."""
    with open(targets_dir(cfg, dataset, variant) / "meta.json") as f:
        return json.load(f)


def _stack_subjects(directory, subjects, dtype=None):
    """Every array of the subjects' .npz files, stacked over subjects.

    "fc_epochs", "fs_epochs" (if any subject has them): object arrays of one
    (n_epochs, N, N) array per subject, None for single-epoch subjects."""
    data, epochs = {}, {k: [] for k in EPOCH_KEYS}
    for s in subjects:
        with np.load(directory / f"{s}.npz") as z:
            for k in z.files:
                if k not in EPOCH_KEYS:
                    data.setdefault(k, []).append(z[k])
            for k in EPOCH_KEYS:
                epochs[k].append(np.asarray(z[k], dtype=dtype)
                                 if k in z.files else None)
    out = {k: np.array(v, dtype=dtype) for k, v in data.items()}
    for k, v in epochs.items():
        if any(e is not None for e in v):
            out[k] = np.empty(len(v), dtype=object)
            for i, e in enumerate(v):
                out[k][i] = e
    return out


def load_targets(cfg, dataset, variant):
    """Targets of all subjects: dict of (n_subjects, N, N) arrays ("fc",
    "fs", "fc_odd", ...), the per-epoch "fc_epochs", "fs_epochs" (see
    _stack_subjects), plus "n_epochs", "subjects", "groups", "meta"."""
    meta = load_meta(cfg, dataset, variant)
    out = _stack_subjects(targets_dir(cfg, dataset, variant),
                          meta["subjects"])
    out.update(subjects=meta["subjects"], groups=np.array(meta["groups"]),
               meta=meta)
    return out


def load_tau_scan(cfg, dataset, variant, tau):
    """FS at lag *tau* from the tau scan (targets.compute_tau_scan): dict of
    (n_subjects, N, N) arrays "fs", "fs_odd", "fs_even" and the per-epoch
    "fs_epochs" (see _stack_subjects), subjects in the order of the targets'
    meta.json."""
    return _stack_subjects(tau_scan_dir(cfg, dataset, variant, tau),
                           load_meta(cfg, dataset, variant)["subjects"],
                           dtype=float)


def load_fits(cfg, task):
    """Fits of one task, stacked over the subjects that have a result.

    Keys: every array saved by fitting.fit_subject; "history_*" entries are
    lists (their length depends on when each subject stopped).
    """
    f_dir = fit_dir(cfg, task)
    files = sorted(f_dir.glob("*.npz"))
    data, subjects = {}, []
    for path in files:
        with np.load(path) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
        subjects.append(path.stem)
    if not subjects:
        return None
    out = {k: (v if k.startswith("history_") else np.array(v))
           for k, v in data.items()}
    out["subjects"] = subjects
    return out


def task_table(cfg):
    """One row per task: id, dataset, variant, model, half, tag, the swept
    parameters and whether results exist."""
    rows = []
    for t in expand_tasks(cfg):
        f_dir = fit_dir(cfg, t)
        rows.append({"id": t["id"], "dataset": t["dataset"],
                     "variant": t["variant"], "model": t["model"],
                     "half": t["half"],
                     **{k: t["fit"][k] for k in ("eps_fc", "eps_fs",
                                                 "convergence")},
                     **t["swept"], "tag": t["tag"],
                     "n_fitted": len(list(f_dir.glob("*.npz")))
                     if f_dir.exists() else 0})
    return pd.DataFrame(rows)
