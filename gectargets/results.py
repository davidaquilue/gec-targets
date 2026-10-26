"""Read targets and fits back as stacked arrays for the studies."""
import json

import numpy as np
import pandas as pd

from .config import expand_tasks, fit_dir, targets_dir


def load_meta(cfg, dataset, variant):
    """meta.json of the targets: tau, theta, sfreq, subjects, groups, ..."""
    with open(targets_dir(cfg, dataset, variant) / "meta.json") as f:
        return json.load(f)


def load_targets(cfg, dataset, variant):
    """Targets of all subjects: dict of (n_subjects, N, N) arrays ("fc",
    "fs", "fc_odd", ...), plus "n_epochs", "subjects", "groups", "meta"."""
    t_dir = targets_dir(cfg, dataset, variant)
    meta = load_meta(cfg, dataset, variant)
    data = {}
    for s in meta["subjects"]:
        with np.load(t_dir / f"{s}.npz") as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    out = {k: np.array(v) for k, v in data.items()}
    out.update(subjects=meta["subjects"], groups=np.array(meta["groups"]),
               meta=meta)
    return out


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
