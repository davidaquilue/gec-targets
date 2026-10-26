"""
Write a small synthetic dataset in the layout read by the "bids_epochs"
loader, so the whole pipeline can be tried without access to real data.

Each subject is a network OU process dx/dt = -theta (I - gamma W) x + noise
whose W is a subject-specific, slightly asymmetric perturbation of a shared
SC. Group "HC" has a stronger coupling (gamma) than group "AD".

    python examples/make_synthetic_dataset.py [OUT_DIR]   (default demo_data)

Writes OUT_DIR/participants.tsv, OUT_DIR/sc.csv, OUT_DIR/freqs.csv (node
frequencies for the linear_hopf demo) and the epochs under
OUT_DIR/derivatives/epochs_demo/. It is also a template for converting your
own time series (see docs/adding_your_data.md).
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov

N_NODES, N_PER_GROUP = 16, 10
SFREQ, EPOCH_S, N_EPOCHS = 100.0, 4.0, 20
THETA = 10.0                        # 1/s
GAMMA = {"HC": 0.75, "AD": 0.55}
SEED = 0


def save_subject(root, sub, epochs):
    """The three files the loader reads for one subject."""
    ses_dir = root / "derivatives" / "epochs_demo" / sub / "eeg" / "ses-rest"
    ses_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ses_dir / f"{sub}_ses-rest_eegEpochs.h5", "w") as f:
        f.create_dataset("eegData", data=epochs)       # (epochs, nodes, times)
    pd.DataFrame({"Label": [f"node_{i}" for i in range(epochs.shape[1])],
                  "fsample": SFREQ}).to_csv(
        ses_dir / f"{sub}_ses-rest_mtgInfo.csv", index=False)


def simulate(W, gamma, n_samples, rng):
    """Exact discretisation of the OU process at the sampling rate."""
    A = -THETA * (np.eye(len(W)) - gamma * W)
    S = solve_continuous_lyapunov(A, -np.eye(len(W)))
    F = expm(A / SFREQ)
    L = np.linalg.cholesky(S - F @ S @ F.T)
    x = np.linalg.cholesky(S) @ rng.standard_normal(len(W))
    out = np.empty((n_samples, len(W)))
    for t in range(n_samples):
        x = F @ x + L @ rng.standard_normal(len(W))
        out[t] = x
    return out.T


def main(out_dir="demo_data"):
    root = Path(out_dir)
    rng = np.random.default_rng(SEED)
    sc = rng.gamma(2.0, 1.0, (N_NODES, N_NODES)) * \
        (rng.random((N_NODES, N_NODES)) < 0.5)
    sc = np.triu(sc, 1)
    sc = sc + sc.T
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "sc.csv", sc, delimiter=",")
    np.savetxt(root / "freqs.csv", np.full(N_NODES, 1.0), delimiter=",")

    n_times = int(EPOCH_S * SFREQ)
    rows = []
    for group, gamma in GAMMA.items():
        for k in range(N_PER_GROUP):
            sub = f"sub-{group}{k:02d}"
            # Subject network: the SC support with perturbed, asymmetric weights
            W = sc * rng.lognormal(0.0, 0.3, sc.shape)
            W /= np.max(np.abs(np.linalg.eigvals(W)))
            x = simulate(W, gamma, N_EPOCHS * n_times, rng)
            epochs = x.reshape(N_NODES, N_EPOCHS, n_times).transpose(1, 0, 2)
            save_subject(root, sub, epochs)
            rows.append({"participant_id": sub, "classification": group})
            print(f"{sub}: {epochs.shape}")
    pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)
    print(f"Synthetic dataset in {root.resolve()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
