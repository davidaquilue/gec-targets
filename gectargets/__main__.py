"""
Command line:

    python -m gectargets tasks   CONFIG            list the tasks (array ids)
    python -m gectargets targets CONFIG [-j N]     empirical FC / FS targets
    python -m gectargets fit     CONFIG [--task I | --all] [-j N]

``--task`` defaults to $SLURM_ARRAY_TASK_ID and ``-j`` to
$SLURM_CPUS_PER_TASK (else the config's n_jobs, else 1).
"""
import argparse
import os

import pandas as pd

from .config import expand_tasks, load_config, target_variants
from .fitting import run_task
from .results import task_table
from .targets import compute_targets


def main():
    parser = argparse.ArgumentParser(prog="gectargets")
    parser.add_argument("command", choices=["tasks", "targets", "fit"])
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
