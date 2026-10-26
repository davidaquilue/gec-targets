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
