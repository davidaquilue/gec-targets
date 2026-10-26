"""
GEC models: linear stochastic systems dx/dt = A x + noise, fitted through a
coupling matrix C.

Every model provides
  prepare(C0)   initial coupling from the SC (or random) matrix
  normalize(C)  constraint re-applied after every learning step
  jacobian(C)   drift A of the full state
  friction(C)   matrix stored as the GEC ("Ceff") for downstream use

and shares ``observables``: the stationary covariance S (A S + S A' =
-sigma^2 I), FC = corr(S) and FS = expm(tau A) S over the observed nodes (the
first N state variables). FS is divided either by the zero-lag variances
("zerolag", as the empirical FS) or by its own diagonal ("lagged", the
original legacy choice, FS_ii = 1).

Models
------
legacy       dx/dt = -C x. C_ii = sum_j |C_ij| (row sum of the off-diagonal),
             C rescaled so that max(C) = maxC. abs() of the covariance
             (use_abs, legacy behaviour). FS normalisation "lagged".
network      dx/dt = -theta (I - gamma W) x. W >= 0, zero diagonal, spectral
             radius 1 (Nartallo-Kaluarachchi et al.). theta (1/s) is the mean
             decay rate over modes; "FIT" takes it from the targets. Any
             gamma < 1 is stable. FS normalisation "zerolag".
linear_hopf  Linearised Hopf (Stuart-Landau) oscillators, x and y per node:
             A = [[a I - L, -diag(w)], [diag(w), a I - L]], L the Laplacian
             of C (normalised as legacy), w = 2 pi f with one frequency per
             node (``freqs_path``). Only x is observed. FS "lagged".
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def stationary_cov(A, sigma):
    """S solving A S + S A' = -sigma^2 I."""
    S = solve_continuous_lyapunov(A, -(sigma ** 2) * np.eye(len(A)))
    return (S + S.T) / 2


def cov2corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


class GECModel:
    fs_norm = "zerolag"
    use_abs = False

    def __init__(self, sigma=0.05, fs_norm=None):
        self.sigma = sigma
        if fs_norm is not None:
            self.fs_norm = fs_norm

    def prepare(self, C0):
        C = np.abs(np.asarray(C0, dtype=float))
        return self.normalize((C + C.T) / 2)

    def normalize(self, C):
        raise NotImplementedError

    def jacobian(self, C):
        raise NotImplementedError

    def friction(self, C):
        return C

    def observables(self, C, tau_s):
        """Simulated FC and FS (N x N) for coupling C and lag tau_s (s)."""
        n = len(C)
        A = self.jacobian(C)
        S = stationary_cov(A, self.sigma)
        if self.use_abs:
            S = np.abs(S)
        FS = (expm(tau_s * A) @ S)[:n, :n]
        S0 = S[:n, :n]
        d = np.diag(FS) if self.fs_norm == "lagged" else np.diag(S0)
        return cov2corr(S0), FS / np.sqrt(np.outer(d, d))


class Legacy(GECModel):
    fs_norm = "lagged"
    use_abs = True

    def __init__(self, maxC=0.2, use_abs=True, sigma=0.05, fs_norm=None):
        super().__init__(sigma, fs_norm)
        self.maxC = maxC
        self.use_abs = use_abs

    def normalize(self, C):
        C = C * (1 - np.eye(len(C)))
        C = C + np.diag(np.abs(C).sum(axis=1))
        return C / C.max() * self.maxC

    def jacobian(self, C):
        return -C


class Network(GECModel):
    def __init__(self, theta=1.0, gamma=0.8, sigma=0.05, fs_norm=None):
        super().__init__(sigma, fs_norm)
        self.theta = theta
        self.gamma = gamma

    def normalize(self, C):
        C = C * (1 - np.eye(len(C)))
        return C / np.max(np.abs(np.linalg.eigvals(C)))

    def jacobian(self, C):
        return -self.friction(C)

    def friction(self, C):
        return self.theta * (np.eye(len(C)) - self.gamma * C)


class LinearHopf(Legacy):
    def __init__(self, freqs, a=-0.02, maxC=0.2, sigma=0.05, fs_norm=None):
        super().__init__(maxC=maxC, use_abs=False, sigma=sigma,
                         fs_norm=fs_norm)
        self.omega = 2 * np.pi * np.asarray(freqs, dtype=float).ravel()
        self.a = a

    def jacobian(self, C):
        n = len(C)
        if self.omega.size != n:
            raise ValueError(f"{self.omega.size} frequencies for {n} nodes")
        # Same as hopf_int.m: a I - diag(rowsum(C)) + C
        block = self.a * np.eye(n) - np.diag(C.sum(axis=1)) + C
        w = np.diag(self.omega)
        return np.block([[block, -w], [w, block]])


MODELS = {"legacy": Legacy, "network": Network, "linear_hopf": LinearHopf}


def load_freqs(path, subject):
    """Node frequencies (Hz) for the Hopf model. ``{subject}`` in the path
    selects one file per subject; .npy or text (.csv / .txt)."""
    path = str(path).format(subject=subject)
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)


def build_model(name, params, meta, subject):
    """Model instance for one subject: resolves theta "FIT" and the Hopf
    frequencies."""
    params = dict(params)
    if name == "network" and params.get("theta") == "FIT":
        params["theta"] = meta["theta"]
    if name == "linear_hopf":
        if not params.get("freqs_path"):
            raise ValueError("linear_hopf needs 'freqs_path' in its config")
        params["freqs"] = load_freqs(params.pop("freqs_path"), subject)
    return MODELS[name](**params)
