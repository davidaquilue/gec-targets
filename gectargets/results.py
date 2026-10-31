"""Read targets and fits back as stacked arrays for the studies."""
import json
import os
import zipfile

import numpy as np
import pandas as pd

from .config import expand_tasks, fit_dir, tau_scan_dir, targets_dir

EPOCH_KEYS = ("fc_epochs", "fs_epochs")


def save_npz(path, **arrays):
    """np.savez through a temporary file renamed at the end: a job killed
    while writing leaves no truncated .npz, which would count as done."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def is_complete(path):
    """The .npz exists and is readable (not truncated)."""
    return path.exists() and zipfile.is_zipfile(path)


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
        path = directory / f"{s}.npz"
        if not is_complete(path):
            raise FileNotFoundError(
                f"{path} is missing or unreadable (truncated, e.g. by an "
                f"interrupted copy): copy it again, or delete it and compute "
                f"it again. `python -m gectargets check CONFIG` lists all such "
                f"files.")
        with np.load(path) as z:
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
    lists (their length depends on when each subject stopped). Unreadable
    (truncated) files are reported and left out; the fit step refits them.
    """
    f_dir = fit_dir(cfg, task)
    files = sorted(f_dir.glob("*.npz"))
    broken = [p for p in files if not zipfile.is_zipfile(p)]
    if broken:
        print(f"{f_dir}: {len(broken)} unreadable fit files left out "
              f"(refitted by the fit step): {[p.stem for p in broken]}")
    data, subjects = {}, []
    for path in (p for p in files if p not in broken):
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


def check_results(cfg):
    """Are the targets and fits of a config complete and readable? (e.g.
    after copying results from a cluster). Read-only.

    One row per targets folder and per task: subjects expected (the targets'
    meta.json), done, missing and unreadable (truncated .npz). Returns the
    table and the list of unreadable files."""
    tasks = expand_tasks(cfg)
    rows, unreadable = [], []

    def count(kind, directory, subjects, **info):
        files = [directory / f"{s}.npz" for s in subjects]
        present = [p for p in files if p.exists()]
        bad = [p for p in present if not zipfile.is_zipfile(p)]
        unreadable.extend(bad)
        rows.append({"kind": kind, **info, "expected": len(files),
                     "done": len(present) - len(bad),
                     "missing": len(files) - len(present),
                     "unreadable": len(bad)})

    subjects = {}
    for dataset, variant in dict.fromkeys((t["dataset"], t["variant"])
                                          for t in tasks):
        t_dir = targets_dir(cfg, dataset, variant)
        info = {"dataset": dataset, "variant": variant}
        if not (t_dir / "meta.json").exists():
            rows.append({"kind": "targets", **info, "expected": np.nan})
            continue
        subjects[dataset, variant] = load_meta(cfg, dataset, variant)[
            "subjects"]
        count("targets", t_dir, subjects[dataset, variant], **info)
    for t in tasks:
        info = {"id": t["id"], "dataset": t["dataset"],
                "variant": t["variant"], "model": t["model"],
                "half": t["half"], "tag": t["tag"]}
        if (t["dataset"], t["variant"]) not in subjects:
            rows.append({"kind": "fit", **info, "expected": np.nan})
            continue
        count("fit", fit_dir(cfg, t), subjects[t["dataset"], t["variant"]],
              **info)
    return pd.DataFrame(rows), unreadable


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
