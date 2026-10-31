"""
Command line:

    python -m gectargets tasks   CONFIG            list the tasks (array ids)
    python -m gectargets targets CONFIG [-j N]     empirical FC / FS targets
    python -m gectargets fit     CONFIG [--task I | --all] [-j N]
    python -m gectargets check   CONFIG            are targets and fits complete
                                                   and readable (after a copy)

``--task`` defaults to $SLURM_ARRAY_TASK_ID and ``-j`` to
$SLURM_CPUS_PER_TASK (else the config's n_jobs, else 1).
"""
import argparse
import os
import sys

import pandas as pd

from .config import expand_tasks, load_config, target_variants
from .fitting import run_task
from .results import check_results, task_table
from .targets import compute_targets


def check(cfg):
    """Print what is incomplete; exit status 1 if any file is unreadable."""
    table, unreadable = check_results(cfg)
    todo = table[(table.missing > 0) | (table.unreadable > 0)
                 | table.expected.isna()].astype({"id": "Int64"})
    for kind, part in table.groupby("kind", sort=False):
        print(f"{kind}: {int(part.done.sum())} of {int(part.expected.sum())} "
              f"subject files done, {int(part.missing.sum())} missing, "
              f"{int(part.unreadable.sum())} unreadable; "
              f"{int(part.expected.isna().sum())} without targets")
    if len(todo):
        with pd.option_context("display.width", 200, "display.max_rows", None,
                               "display.max_colwidth", 40):
            print(todo.drop(columns="tag", errors="ignore")
                  .to_string(index=False))
    if unreadable:
        print("\nUnreadable (truncated) files: copy them again, or let the "
              "targets / fit step recompute them:")
        print("\n".join(str(p) for p in unreadable))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="gectargets")
    parser.add_argument("command", choices=["tasks", "targets", "fit",
                                            "check"])
    parser.add_argument("config")
    parser.add_argument("--task", type=int,
                        default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--all", action="store_true",
                        help="fit every task, one after the other")
    parser.add_argument("-j", "--n-jobs", type=int,
                        default=os.environ.get("SLURM_CPUS_PER_TASK"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_jobs = int(args.n_jobs or cfg.get("n_jobs", 1))

    if args.command == "tasks":
        with pd.option_context("display.width", 200, "display.max_rows", None,
                               "display.max_colwidth", 40):
            print(task_table(cfg).drop(columns="tag").to_string(index=False))
    elif args.command == "check":
        check(cfg)
    elif args.command == "targets":
        for dataset, prep in target_variants(cfg):
            compute_targets(cfg, dataset, prep, n_jobs)
    else:
        tasks = expand_tasks(cfg)
        if args.all:
            selected = tasks
        elif args.task is not None:
            selected = [tasks[int(args.task)]]
        else:
            parser.error("give --task ID or --all")
        for task in selected:
            run_task(cfg, task, n_jobs)


if __name__ == "__main__":
    main()
