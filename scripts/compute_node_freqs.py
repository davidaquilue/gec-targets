"""
Node frequencies for the Hopf model (linear_hopf) of one dataset.

    python scripts/compute_node_freqs.py CONFIG DATASET [--fmin 0.008]
                                         [--fmax 0.08] [--out PATH] [-j N]

Per subject, the time series are preprocessed as for the targets (band of the
dataset if any, detrend, z-score) and the PSD of every node is computed; the
PSDs are averaged over subjects and each node's frequency is that of the
maximum within [fmin, fmax] (default: the usual fMRI band, 0.008-0.08 Hz).

Writes to the dataset's targets folder, <results_root>/<dataset>/<variant>/
targets/ (or next to --out):
    node_freqs.csv   one frequency (Hz) per node, in the node order of the data
    node_psd.npz     grid, mean PSD (n_nodes, n_grid), freqs, fmin, fmax,
                     subjects
    node_psd.png     mean PSD of every node with its peak, and the peak
                     distribution

Point the dataset at the file with "freqs_path" in its config block; several
datasets can share one (e.g. HCP tasks using the rest frequencies).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gectargets.config import load_config, targets_dir, variant_name
from gectargets.plotting import GRAY, SEQUENTIAL, save
from gectargets.targets import node_peak_freqs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("config")
    parser.add_argument("dataset")
    parser.add_argument("--fmin", type=float, default=0.008,
                        help="lowest frequency of the peak search (Hz)")
    parser.add_argument("--fmax", type=float, default=0.08,
                        help="highest frequency of the peak search (Hz)")
    parser.add_argument("--out", type=Path,
                        help="frequencies file (default: node_freqs.csv in "
                             "the dataset's targets folder)")
    parser.add_argument("-j", "--n-jobs", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset not in cfg["datasets"]:
        parser.error(f"dataset '{args.dataset}' is not in the config: "
                     f"{list(cfg['datasets'])}")
    n_jobs = int(args.n_jobs or cfg.get("n_jobs", 1))
    res = node_peak_freqs(cfg, args.dataset, args.fmin, args.fmax, n_jobs)

    variant = variant_name(cfg["datasets"][args.dataset],
                           {"envelope": False, "orthogonalize": False})
    out_csv = args.out or targets_dir(cfg, args.dataset, variant) / \
        "node_freqs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, res["freqs"], delimiter=",", fmt="%.6g")
    np.savez(out_csv.with_name("node_psd.npz"), **res)
    freqs = res["freqs"]
    print(f"[{args.dataset}] {len(res['subjects'])} subjects, "
          f"{len(freqs)} nodes: peak {np.median(freqs):.4g} Hz median "
          f"({freqs.min():.4g}-{freqs.max():.4g}) -> {out_csv}")
    at_edge = np.isclose(freqs, res["fmin"], atol=res["grid"][1]) | \
        np.isclose(freqs, res["fmax"], atol=res["grid"][1])
    if at_edge.any():
        print(f"  {at_edge.sum()} nodes peak at an edge of [{res['fmin']}, "
              f"{res['fmax']}] Hz: their PSD may keep rising outside it")

    # Check figure: PSD of every node (each scaled to its maximum in the
    # search range), nodes sorted by peak, and the distribution of the peaks
    grid, psd = res["grid"], res["psd"]
    shown = grid <= min(2 * res["fmax"], grid[-1])
    inside = (grid >= res["fmin"]) & (grid <= res["fmax"])
    order = np.argsort(freqs)
    scaled = psd[order][:, shown] / psd[order][:, inside].max(axis=1,
                                                               keepdims=True)
    fig, (ax_psd, ax_hist) = plt.subplots(1, 2, figsize=(11, 4),
                                          width_ratios=[2, 1])
    im = ax_psd.imshow(scaled, aspect="auto", origin="lower", cmap=SEQUENTIAL,
                       vmin=0, vmax=1, interpolation="nearest",
                       extent=[grid[shown][0], grid[shown][-1], 0, len(freqs)])
    ax_psd.plot(freqs[order], np.arange(len(freqs)) + 0.5, ".", ms=2,
                color="#0b0b0b")
    for edge in (res["fmin"], res["fmax"]):
        ax_psd.axvline(edge, color=GRAY, linestyle="--", linewidth=1)
    ax_psd.set(xlabel="frequency (Hz)", ylabel="node (sorted by peak)",
               title=f"{args.dataset}: mean PSD per node (scaled to its peak)")
    fig.colorbar(im, ax=ax_psd, fraction=0.046, pad=0.03)
    ax_hist.hist(freqs, bins=30, range=(res["fmin"], res["fmax"]),
                 color="#2a78d6")
    ax_hist.set(xlabel="peak frequency (Hz)", ylabel="nodes",
                title="node frequencies")
    fig.tight_layout()
    save(fig, out_csv.with_name("node_psd.png"))


if __name__ == "__main__":
    main()
