"""
Analysis helpers shared by the studies: decomposition of FS, edge vectors,
similarity / reliability / identifiability, GEC comparison, entropy
production.

Edge vectors (what is correlated between two matrices)
    FC    upper triangle            (symmetric)
    FS    all off-diagonal entries  (asymmetric)
    FS_S  upper triangle of (FS + FS') / 2
    FS_A  upper triangle of (FS - FS') / 2  (holds all of it: antisymmetric)
Functions accept a matrix (N, N) or a stack (..., N, N).
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import ttest_1samp

TARGETS = ("FC", "FS", "FS_S", "FS_A")


# ------------------------------------------------------------ decomposition
def sym(M):
    return (M + np.swapaxes(M, -1, -2)) / 2


def asym(M):
    return (M - np.swapaxes(M, -1, -2)) / 2


def triu(M):
    i, j = np.triu_indices(M.shape[-1], k=1)
    return M[..., i, j]


def offdiag(M):
    return M[..., ~np.eye(M.shape[-1], dtype=bool)]


def corr(a, b):
    """Pearson correlation of two vectors (NaN if one is constant)."""
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else np.nan


def edge_vectors(fc, fs):
    """dict target -> edge vectors, for a matrix or a stack of matrices."""
    return {"FC": triu(fc), "FS": offdiag(fs),
            "FS_S": triu(sym(fs)), "FS_A": triu(asym(fs))}


def target_matrices(fc, fs):
    return {"FC": fc, "FS": fs, "FS_S": sym(fs), "FS_A": asym(fs)}


# ------------------------------------------------- similarity / reliability
def corr_matrix(X, Y=None):
    """R[s, t] = corr(X_s, Y_t) between the rows of X and Y (default X)."""
    def unit(Z):
        Z = Z - Z.mean(axis=1, keepdims=True)
        return Z / np.linalg.norm(Z, axis=1, keepdims=True)
    Xu = unit(X)
    return Xu @ (Xu if Y is None else unit(Y)).T


def between_subject(X):
    """Off-diagonal entries of the subject x subject correlation matrix."""
    R = corr_matrix(X)
    return R[np.triu_indices(len(R), k=1)]


def identifiability(X_a, X_b):
    """Split-half reliability and identifiability (Amico & Goni 2018).

    X_a, X_b: (n_subjects, n_edges) from two halves of the same recordings.
    within: corr(half a of s, half b of s); between: s != t.
    i_diff = mean(within) - mean(between); accuracy: fraction of subjects
    whose half a is most similar to their own half b (both directions).
    """
    R = corr_matrix(X_a, X_b)
    n = len(R)
    within = np.diag(R)
    between = R[~np.eye(n, dtype=bool)]
    acc = ((R.argmax(axis=1) == np.arange(n)).mean()
           + (R.argmax(axis=0) == np.arange(n)).mean()) / 2
    return {"within": within, "between": between,
            "i_self": float(within.mean()), "i_others": float(between.mean()),
            "i_diff": float(within.mean() - between.mean()),
            "accuracy": float(acc)}


def loo_group_corr(X, signs=None):
    """r_s = corr(x_s, mean of the other subjects): how much of each subject's
    pattern is shared with the group."""
    if signs is not None:
        X = X * signs[:, None]
    loo = (X.sum(axis=0)[None] - X) / (len(X) - 1)
    Xc, Lc = X - X.mean(1, keepdims=True), loo - loo.mean(1, keepdims=True)
    return (Xc * Lc).sum(1) / np.sqrt((Xc ** 2).sum(1) * (Lc ** 2).sum(1))


def signflip_null(X, n_perm=1000, seed=0):
    """Mean LOO r under random sign flips of whole subjects.

    For FS_A: with no shared direction each subject's pattern is as likely as
    its time reverse (-FS_A), so flips keep the noise and remove the shared
    part. Not meaningful for FC / FS_S, whose sign is not arbitrary.
    """
    rng = np.random.default_rng(seed)
    return np.array([loo_group_corr(X, rng.choice([-1.0, 1.0], len(X))).mean()
                     for _ in range(n_perm)])


# ----------------------------------------------- signal to noise (power)
# Power = mean square over edges, uncentred: for FS_A, 0 is the reversible
# reference. Signal power = total power of the estimate - its noise power;
# it can come out < 0 when there is no detectable signal.
def _power_snr(estimate, noise):
    signal = np.mean(estimate ** 2, axis=-1) - noise
    return {"signal": signal, "noise": noise, "snr": signal / noise}


def snr_split_half(X_odd, X_even):
    """SNR of (odd + even) / 2, X_*: (..., n_edges).

    Each half = signal + independent noise of equal variance: odd - even is
    noise only, and the noise variance of the mean of the halves is
    mean((odd - even)^2) / 4.
    """
    noise = np.mean((X_odd - X_even) ** 2, axis=-1) / 4
    return _power_snr((X_odd + X_even) / 2, noise)


def snr_epochs(E):
    """SNR of the epoch mean, E: (n_epochs, n_edges), assuming independent
    epochs: the noise variance of the mean of edge k is var_e(E_k) / K."""
    noise = np.mean(E.var(axis=0, ddof=1) / len(E))
    return _power_snr(E.mean(axis=0), noise)


def epoch_tstats(E):
    """One-sample t and two-sided p of every edge's epoch mean against 0."""
    res = ttest_1samp(E, 0.0, axis=0)
    return res.statistic, res.pvalue


def fdr_fraction(p, q=0.05):
    """Fraction of the tests rejected by Benjamini-Hochberg at level q."""
    p = np.sort(p)
    passed = np.flatnonzero(p <= q * np.arange(1, len(p) + 1) / len(p))
    return (passed[-1] + 1) / len(p) if passed.size else 0.0


def epoch_signflip(E, n_perm=1000, seed=0):
    """p-value of the total power of the epoch mean of FS_A under random
    sign flips of whole epochs.

    Time-reversing an epoch flips the sign of its FS_A exactly, so for a
    reversible process with independent epochs every sign pattern is
    equally likely. Not meaningful for FC / FS_S (unchanged by reversal).
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], (n_perm, len(E)))
    null = np.mean((signs @ E / len(E)) ** 2, axis=1)
    observed = np.mean(E.mean(axis=0) ** 2)
    return (1 + (null >= observed).sum()) / (n_perm + 1)


# ------------------------------------------- metrics for scans (e.g. tau)
SNR_METRICS = {
    "snr": "signal / noise power of each subject, split halves "
           "(snr_split_half)",
    "snr_epochs": "signal / noise power of each subject, epochs "
                  "(snr_epochs; multi-epoch data)",
    "fdr_edges": "fraction of edges with epoch mean != 0, BH 5 %, of each "
                 "subject (FS_A; multi-epoch data)",
    "epoch_signflip": "fraction of subjects whose FS_A beats the epoch "
                      "sign-flip null at p < 0.05 (multi-epoch data)",
    "within_r": "corr(odd half, even half) of each subject",
    "within_snr": "signal / noise variance (centred) of the full data "
                  "implied by the mean split-half r: 2 r / (1 - r)",
    "between_r": "corr between every pair of subjects",
    "i_diff": "mean within r - mean between r (split halves)",
    "accuracy": "split-half identification accuracy",
    "loo_r": "corr(subject, mean of the others) of each subject",
    "signflip_z": "mean loo_r against the subject sign-flip null, in null "
                  "SDs (FS_A only)",
}
EPOCH_METRICS = ("snr_epochs", "fdr_edges", "epoch_signflip")


def snr_metric(name, X, X_odd, X_even, target, E=None, n_perm=1000):
    """Metric *name* of SNR_METRICS for one target.

    X, X_odd, X_even: (n_subjects, n_edges) from all epochs and the two
    halves; E: per subject, (n_epochs, n_edges) or None (single epoch),
    needed by EPOCH_METRICS. Returns one value per subject (or subject pair)
    or a float; NaN where the metric does not apply.

    within_snr: with each half = signal + independent noise of equal size,
    r = var_s / (var_s + var_half_noise), and the noise of the full data has
    half the variance of a half's.
    """
    if name in EPOCH_METRICS:
        if E is None or (name != "snr_epochs" and target != "FS_A"):
            return np.nan
        if name == "snr_epochs":
            return np.array([np.nan if e is None else snr_epochs(e)["snr"]
                             for e in E])
        if name == "fdr_edges":
            return np.array([np.nan if e is None else
                             fdr_fraction(epoch_tstats(e)[1]) for e in E])
        p = np.array([epoch_signflip(e, n_perm) for e in E if e is not None])
        return float(np.mean(p < 0.05)) if p.size else np.nan
    if name == "snr":
        return snr_split_half(X_odd, X_even)["snr"]
    if name in ("within_r", "within_snr", "i_diff", "accuracy"):
        ident = identifiability(X_odd, X_even)
        if name == "within_r":
            return ident["within"]
        if name == "within_snr":
            r = ident["i_self"]
            return 2 * r / (1 - r)
        return ident[name]
    if name == "between_r":
        return between_subject(X)
    if name == "loo_r":
        return loo_group_corr(X)
    if name == "signflip_z":
        if target != "FS_A":
            return np.nan
        null = signflip_null(X, n_perm)
        return float((loo_group_corr(X).mean() - null.mean()) / null.std())
    raise KeyError(f"Unknown metric '{name}'. Available: {list(SNR_METRICS)}")


# --------------------------------------------------------------------- GEC
def gec_edges(C, mask):
    """Fitted couplings on the support of the initial matrix (off-diagonal).
    Entries outside the SC support are 0 in every subject and would inflate
    any similarity, so they are left out."""
    return C[..., mask]


def gec_reliability(fits, sc=None, fits_odd=None, fits_even=None):
    """Similarity of the fitted couplings on the support of the initial
    matrix.

    between_r : mean correlation between subjects
    r_to_sc   : mean correlation of each subject with the SC it started from.
                Shared SC weights make every subject alike, so between_r must
                be read against it
    i_diff, accuracy : split-half identifiability (odd vs even fits), which
                cancels what all subjects share, SC included
    """
    mask = fits["mask"][0]
    X = gec_edges(fits["C"], mask)
    out = {"between_r": float(between_subject(X).mean())}
    if sc is not None:
        sc_edges = gec_edges((np.abs(sc) + np.abs(sc).T) / 2, mask)
        out["r_to_sc"] = float(np.mean([corr(x, sc_edges) for x in X]))
    if fits_odd is not None and fits_even is not None:
        common = sorted(set(fits_odd["subjects"]) & set(fits_even["subjects"]))
        pick = lambda f: gec_edges(f["C"][[f["subjects"].index(s)
                                           for s in common]], mask)
        ident = identifiability(pick(fits_odd), pick(fits_even))
        out.update(i_self=ident["i_self"], i_others=ident["i_others"],
                   i_diff=ident["i_diff"], accuracy=ident["accuracy"])
    return out


def entropy_production(A, sigma=1.0):
    """Entropy production rate of dx/dt = A x + sigma xi (white noise).

    With D = sigma^2 / 2 I and S the stationary covariance, A S = -(D + Q)
    with Q antisymmetric (the irreversible part), and
        Phi = tr(Q' D^-1 Q S^-1)   [1/s]  (Godreche & Luck 2019),
    0 for a reversible (equilibrium) system. Independent of sigma.
    """
    n = len(A)
    D = sigma ** 2 / 2 * np.eye(n)
    S = solve_continuous_lyapunov(A, -2 * D)
    S = (S + S.T) / 2
    Q = -(A @ S) - D
    Q = (Q - Q.T) / 2
    return float(np.trace(Q.T @ np.linalg.solve(D, Q) @ np.linalg.inv(S)))
