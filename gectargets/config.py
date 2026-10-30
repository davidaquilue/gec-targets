"""
Configuration: read the JSON file, expand the sweep into tasks, build paths.

A *task* is one GEC fit of every subject of one dataset with one model and one
set of hyperparameters (and one data half when ``split_half`` is on). Tasks are
numbered in a fixed order, so task ``i`` is SLURM array task ``i``.

Keys starting with ``_`` anywhere in the JSON are comments and are ignored.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PREPROCESSING_KEYS = ("envelope", "orthogonalize")
# Settings of the fitting loop (everything else in a model block is a model
# parameter, passed to the model class)
FIT_KEYS = ("eps_fc", "eps_fs", "max_iter", "tol",
            "check_every", "convergence", "log_every", "n_init", "seed")
HALVES_SPLIT = ("all", "odd", "even")


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _resolve(path, base):
    """Expand ``$VARS`` and ``~``; relative paths are taken from *base*."""
    if path is None:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(path))))
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path):
    """Read the config JSON and resolve its paths.

    Relative paths are relative to the config file, and environment variables
    are expanded, so one file can serve a laptop and a cluster.
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = _strip_comments(json.load(f))
    base = path.parent
    cfg["results_root"] = _resolve(cfg["results_root"], base)
    for ds in cfg["datasets"].values():
        for key in ("root", "sc_path"):
            ds[key] = _resolve(ds.get(key), base)
        if ds.get("freqs_path"):
            ds["freqs_path"] = str(_resolve(ds["freqs_path"], base))
    for params in cfg["models"].values():
        if params.get("freqs_path"):
            # Keep "{subject}" placeholders: resolve the rest of the path
            params["freqs_path"] = str(_resolve(params["freqs_path"], base))
    cfg.setdefault("bands", {})
    cfg.setdefault("preprocessing", {"envelope": False, "orthogonalize": False})
    return cfg


# ------------------------------------------------------------------- naming
def variant_name(dataset_cfg, prep):
    """Folder name of a signal variant, e.g. "alpha", "alpha_env_orth".

    Datasets with ``band: null`` (fMRI) are used as stored: "nofilter".
    """
    name = dataset_cfg.get("band") or "nofilter"
    if prep["envelope"]:
        name += "_env"
    if prep["orthogonalize"]:
        name += "_orth"
    return name


def _fmt(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def run_tag(model_params, fit):
    """Folder name of one hyperparameter set: readable part + short hash.

    The readable part lists the model parameters and learning rates; the hash
    covers every setting, so two runs differing only in, e.g., ``max_iter``
    never share a folder.
    """
    shown = {k: v for k, v in model_params.items() if not k.endswith("_path")}
    shown.update({k: fit[k] for k in ("eps_fc", "eps_fs")})
    text = "_".join(f"{k}-{_fmt(v)}" for k, v in sorted(shown.items()))
    digest = hashlib.md5(json.dumps(
        {"model": model_params, "fit": fit}, sort_keys=True, default=str
    ).encode()).hexdigest()[:6]
    return f"{text}_{digest}"


# -------------------------------------------------------------------- tasks
def _make_task(cfg, dataset, model, swept, half):
    if dataset not in cfg["datasets"]:
        raise KeyError(f"Dataset '{dataset}' is not defined in 'datasets'")
    if model not in cfg["models"]:
        raise KeyError(f"Model '{model}' is not defined in 'models'")
    defaults = {**cfg["fitting"], **cfg["models"][model]}
    unknown = set(swept) - set(defaults) - set(PREPROCESSING_KEYS)
    if unknown:
        raise KeyError(f"Swept keys {sorted(unknown)} are neither fitting "
                       f"settings, preprocessing options nor parameters of "
                       f"model '{model}'")
    merged = {**defaults, **swept}
    prep = {k: swept.get(k, cfg["preprocessing"][k])
            for k in PREPROCESSING_KEYS}
    fit = {k: merged[k] for k in FIT_KEYS if k in merged}
    model_params = {k: v for k, v in merged.items()
                    if k not in FIT_KEYS and k not in cfg["fitting"]
                    and k not in PREPROCESSING_KEYS}
    ds_cfg = cfg["datasets"][dataset]
    # Node frequencies (linear_hopf) set per dataset override the model's
    # default, unless swept
    if ("freqs_path" in model_params and ds_cfg.get("freqs_path")
            and "freqs_path" not in swept):
        model_params["freqs_path"] = ds_cfg["freqs_path"]
    return {
        "dataset": dataset,
        "variant": variant_name(ds_cfg, prep),
        "prep": prep,
        "model": model,
        "model_params": model_params,
        "fit": fit,
        "swept": swept,
        "half": half,
        "tag": run_tag(model_params, fit),
    }


def expand_tasks(cfg):
    """All tasks of the sweep, in a fixed order (index = array task id).

    Each sweep entry is the Cartesian product datasets x models x grid. With
    ``split_half`` (fitting default, or per entry) every hyperparameter set
    is fitted three times: on all epochs, on the odd and on the even ones.
    A task that repeats an earlier one (same output folder) is dropped, so
    two array tasks never write the same files.
    """
    tasks, seen = [], set()
    for entry in cfg["sweep"]:
        grid = entry.get("grid", {})
        keys = list(grid)
        split = entry.get("split_half", cfg["fitting"].get("split_half", False))
        halves = HALVES_SPLIT if split else ("all",)
        for dataset in entry["datasets"]:
            for model in entry["models"]:
                for values in itertools.product(*(grid[k] for k in keys)):
                    swept = dict(zip(keys, values))
                    for half in halves:
                        task = _make_task(cfg, dataset, model, swept, half)
                        key = (dataset, task["variant"], model, task["tag"],
                               half)
                        if key not in seen:
                            seen.add(key)
                            tasks.append(task)
    for i, task in enumerate(tasks):
        task["id"] = i
    return tasks


def target_variants(cfg):
    """Unique (dataset, prep) pairs used by the sweep, for the targets step."""
    seen = {}
    for task in expand_tasks(cfg):
        seen[(task["dataset"], task["variant"])] = task["prep"]
    return [(ds, prep) for (ds, _), prep in seen.items()]


# -------------------------------------------------------------------- paths
def targets_dir(cfg, dataset, variant):
    return cfg["results_root"] / dataset / variant / "targets"


def tau_scan_dir(cfg, dataset, variant, tau):
    return targets_dir(cfg, dataset, variant) / "tau_scan" / f"tau-{tau}"


def fit_dir(cfg, task):
    return (cfg["results_root"] / task["dataset"] / task["variant"] / "fits"
            / task["model"] / task["tag"] / task["half"])


def study_dir(cfg, name):
    path = cfg["results_root"] / "studies" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
