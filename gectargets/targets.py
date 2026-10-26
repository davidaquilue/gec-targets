"""
Empirical targets of the GEC fit: FC and the lagged covariance FS.

Per subject and epoch the signals are

    band-pass (if the dataset has a band) -> detrend -> [orthogonalise]
    -> z-score -> [Hilbert envelope]

and then

    FC_ij = corr(x_i(t), x_j(t))
    FS_ij = <x_i(t + tau) x_j(t)> / sqrt(var_i var_j)     (zero-lag variances)

averaged over epochs. FS is stored in the model orientation (row = later
node), the same as the simulated ``expm(tau A) S``, so FS_ii is the
autocorrelation at tau and the antisymmetric part (FS - FS') / 2 has the same
sign convention in data and model.

tau is either fixed (samples) or "FIT": the first lag at which the
autocorrelation, averaged over subjects, epochs and nodes, drops to 1/e.
theta (network model, "FIT") is the mean over nodes of -ln(acf_i(tau)) / tau,
the decay rate of an isolated OU node with the observed autocorrelation.
"""
import json

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

from .config import targets_dir, variant_name
from .loaders import get_loader

ACF_THRESHOLD = 1 / np.e


# ----------------------------------------------------------- preprocessing
def bandpass(x, sfreq, band):
    """Zero-phase 4th-order Butterworth (MNE's default IIR), last axis."""
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def symmetric_orthogonalize(x):
    """Löwdin orthogonalisation (Colclough et al. 2015) from the covariance
    pooled over epochs; each node keeps the sign of its original signal."""
    n_nodes = x.shape[1]
    pooled = x.transpose(1, 0, 2).reshape(n_nodes, -1)
    pooled = pooled - pooled.mean(axis=1, keepdims=True)
    lam, V = np.linalg.eigh(pooled @ pooled.T / pooled.shape[1])
    inv_sqrt = np.where(lam > np.finfo(float).eps, 1 / np.sqrt(np.abs(lam)), 0)
    y = np.einsum("ij,ejt->eit", (V * inv_sqrt) @ V.T, x)
    sign = np.sign(np.einsum("eit,eit->i", x, y))
    return y * np.where(sign == 0, 1, sign)[None, :, None]


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)


def preprocess(epochs, sfreq, band=None, envelope=False, orthogonalize=False):
    x = np.asarray(epochs, dtype=float)
    if band is not None:
        x = bandpass(x, sfreq, band)
    x = detrend(x, axis=-1)
    if orthogonalize:
        x = symmetric_orthogonalize(x)
    x = zscore(x)
    if envelope:
        x = np.abs(hilbert(x, axis=-1))
    return x


# ----------------------------------------------------------------- targets
def fc_epochs(x):
    """Pearson correlation per epoch: (n_epochs, N, N)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    xc = xc / np.linalg.norm(xc, axis=-1, keepdims=True)
    return xc @ xc.transpose(0, 2, 1)


def fs_epochs(x, tau):
    """Lagged covariance <x_i(t + tau) x_j(t)> per epoch, divided by the
    zero-lag standard deviations (as fdt_eeg's compute_fs_matrix_epochs,
    transposed to the model orientation)."""
    if not 0 < tau < x.shape[-1]:
        raise ValueError(f"tau = {tau} must be in [1, n_times)")
    later = x[..., tau:] - x[..., tau:].mean(axis=-1, keepdims=True)
    earlier = x[..., :-tau] - x[..., :-tau].mean(axis=-1, keepdims=True)
    cov = later @ earlier.transpose(0, 2, 1) / later.shape[-1]
    sd = x.std(axis=-1, ddof=1)
    return cov / (sd[:, :, None] * sd[:, None, :])


def mean_acf(x):
    """Autocorrelation averaged over epochs and nodes, lags 0..n_times-1."""
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(xc, n=nfft, axis=-1)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=-1)[..., :n]
    acf = acf / np.maximum(acf[..., :1], np.finfo(float).eps)
    return acf.mean(axis=(0, 1))


def halves(x):
    """Odd and even epochs; a single epoch (fMRI) is split in time."""
    if len(x) >= 2:
        return x[0::2], x[1::2]
    mid = x.shape[-1] // 2
    return x[..., :mid], x[..., mid:]


def subject_targets(x, tau):
    """FC and FS (epoch means) for all epochs and for the two halves."""
    out = {"n_epochs": len(x)}
    for name, part in zip(("", "_odd", "_even"), (x, *halves(x))):
        out[f"fc{name}"] = fc_epochs(part).mean(axis=0)
        out[f"fs{name}"] = fs_epochs(part, tau).mean(axis=0)
    return out


# ------------------------------------------------------------------ driver
def _load(loader, subject, band, prep):
    epochs = loader.get_epochs(subject)
    if epochs is None:
        return None
    return preprocess(epochs, loader.sfreq, band, **prep)


def _subject_acf(loader, subject, band, prep):
    x = _load(loader, subject, band, prep)
    return None if x is None else mean_acf(x)


def _subject_save(loader, subject, band, prep, tau, out_dir):
    x = _load(loader, subject, band, prep)
    if x is None:
        return None
    np.savez(out_dir / f"{subject}.npz", **subject_targets(x, tau))
    return subject


def compute_targets(cfg, dataset, prep, n_jobs=1):
    """Targets of one dataset and signal variant, one .npz per subject plus
    meta.json (tau, theta, sfreq, subjects and groups). Skipped if done."""
    ds_cfg = cfg["datasets"][dataset]
    out_dir = targets_dir(cfg, dataset, variant_name(ds_cfg, prep))
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        print(f"[{dataset}] targets exist: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = get_loader(ds_cfg)
    groups = loader.subjects()
    band_name = ds_cfg.get("band")
    band = cfg["bands"][band_name] if band_name else None
    sfreq = loader.sfreq
    parallel = Parallel(n_jobs=n_jobs)

    tau = cfg["preprocessing"].get("tau", "FIT")
    acf = None
    if tau == "FIT":
        acfs = parallel(delayed(_subject_acf)(loader, s, band, prep)
                        for s in groups)
        acfs = [a for a in acfs if a is not None]
        n = min(len(a) for a in acfs)
        acf = np.mean([a[:n] for a in acfs], axis=0)
        below = np.flatnonzero(acf <= ACF_THRESHOLD)
        if below.size == 0:
            raise RuntimeError(f"[{dataset}] the mean autocorrelation never "
                               f"drops to 1/e within {n - 1} samples")
        tau = max(1, int(below[0]))
    tau = int(tau)
    print(f"[{dataset}] tau = {tau} samples ({tau / sfreq:.4g} s)")

    done = parallel(delayed(_subject_save)(loader, s, band, prep, tau, out_dir)
                    for s in groups)
    subjects = [s for s in done if s is not None]

    # theta: decay rate of the mean node autocorrelation at tau
    acf_tau = np.mean([np.diag(np.load(out_dir / f"{s}.npz")["fs"])
                       for s in subjects], axis=0)
    valid = (acf_tau > 0) & (acf_tau < 1)
    node_rates = np.full(acf_tau.shape, np.nan)
    node_rates[valid] = -np.log(acf_tau[valid]) / (tau / sfreq)

    meta = {
        "dataset": dataset, "band": band_name, "band_hz": band, **prep,
        "sfreq": sfreq, "tau": tau, "tau_s": tau / sfreq,
        "theta": float(np.nanmean(node_rates)),
        "node_rates": node_rates.tolist(),
        "acf": None if acf is None else acf.tolist(),
        "n_nodes": int(acf_tau.size),
        "subjects": subjects, "groups": [groups[s] for s in subjects],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{dataset}] {len(subjects)} subjects -> {out_dir}")
    return out_dir
