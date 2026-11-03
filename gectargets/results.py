"""Read targets and fits back as stacked arrays for the studies."""
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .config import (expand_tasks, fit_dir, model_dir, tau_scan_dir,
                     targets_dir)

EPOCH_KEYS = ("fc_epochs", "fs_epochs")
# Files are read in parallel: with one small file per subject, the time is
# mostly the latency of opening each file (USB / network drives), which
# threads overlap
READ_THREADS = 16


def save_npz(path, **arrays):
    """np.savez through a temporary file renamed at the end: a job killed
    while writing leaves no truncated .npz, which would count as done."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_per_model(table, study, name):
    """A study's table as one CSV per model (its "model" column), in the
    model's folder of the study; nothing for a table without that column
    (no rows)."""
    if "model" in table:
        for model, rows in table.groupby("model"):
            rows.to_csv(model_dir(study, model) / name, index=False)


def is_complete(path):
    """The .npz exists and is readable (not truncated)."""
    return path.exists() and zipfile.is_zipfile(path)


def _read_npz(path, keys=None):
    """Arrays of one .npz, only *keys* if given (the others are not read);
    None if the file is missing or unreadable (truncated)."""
    if not is_complete(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files if keys is None or k in keys}


def read_npz_files(paths, keys=None):
    """_read_npz of every path, in parallel, in the order of *paths*."""
    with ThreadPoolExecutor(READ_THREADS) as ex:
        return list(ex.map(lambda p: _read_npz(p, keys), paths))


def load_meta(cfg, dataset, variant):
    """meta.json of the targets: tau, theta, sfreq, subjects, groups, ..."""
    with open(targets_dir(cfg, dataset, variant) / "meta.json") as f:
        return json.load(f)


def _stack_subjects(directory, subjects, dtype=None, keys=None):
    """Every array (or only *keys*) of the subjects' .npz files, stacked
    over subjects.

    "fc_epochs", "fs_epochs" (if any subject has them): object arrays of one
    (n_epochs, N, N) array per subject, None for single-epoch subjects."""
    paths = [directory / f"{s}.npz" for s in subjects]
    loaded = read_npz_files(paths, keys)
    for path, z in zip(paths, loaded):
        if z is None:
            raise FileNotFoundError(
                f"{path} is missing or unreadable (truncated, e.g. by an "
                f"interrupted copy): copy it again, or delete it and compute "
                f"it again. `python -m gectargets check CONFIG` lists all such "
                f"files.")
    data, epochs = {}, {k: [] for k in EPOCH_KEYS}
    for z in loaded:
        for k, v in z.items():
            if k not in EPOCH_KEYS:
                data.setdefault(k, []).append(v)
        for k in EPOCH_KEYS:
            epochs[k].append(np.asarray(z[k], dtype=dtype) if k in z
                             else None)
    out = {k: np.array(v, dtype=dtype) for k, v in data.items()}
    for k, v in epochs.items():
        if any(e is not None for e in v):
            out[k] = np.empty(len(v), dtype=object)
            for i, e in enumerate(v):
                out[k][i] = e
    return out


def load_targets(cfg, dataset, variant, keys=None):
    """Targets of all subjects: dict of (n_subjects, N, N) arrays ("fc",
    "fs", "fc_odd", ...), the per-epoch "fc_epochs", "fs_epochs" (see
    _stack_subjects), plus "n_epochs", "subjects", "groups", "meta". keys:
    read only these arrays (e.g. ("fs",): the epochs are the bulk)."""
    meta = load_meta(cfg, dataset, variant)
    out = _stack_subjects(targets_dir(cfg, dataset, variant),
                          meta["subjects"], keys=keys)
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


def load_fits(cfg, task, keys=None):
    """Fits of one task, stacked over the subjects that have a result.

    Keys: every array saved by fitting.fit_subject, or only *keys* (much
    faster when the matrices are not needed, e.g. ("r_fs_a",)); "history_*"
    entries are lists (their length depends on when each subject stopped).
    Unreadable (truncated) files are reported and left out; the fit step
    refits them.
    """
    f_dir = fit_dir(cfg, task)
    files = sorted(f_dir.glob("*.npz"))
    loaded = read_npz_files(files, keys)
    broken = [p for p, z in zip(files, loaded) if z is None]
    if broken:
        print(f"{f_dir}: {len(broken)} unreadable fit files left out "
              f"(refitted by the fit step): {[p.stem for p in broken]}")
    data, subjects = {}, []
    for path, z in zip(files, loaded):
        if z is None:
            continue
        for k, v in z.items():
            data.setdefault(k, []).append(v)
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
