# Methods

## 1. Empirical targets (`gectargets/targets.py`)

Per subject, each epoch is

1. band-passed (if the dataset has a `band`): 4th-order Butterworth,
   zero-phase (`sosfiltfilt`);
2. detrended (linear);
3. optionally orthogonalised (`orthogonalize`): Löwdin symmetric
   orthogonalisation (Colclough et al. 2015), from the covariance pooled over
   epochs, each node keeping the sign of its original signal;
4. z-scored;
5. optionally replaced by its Hilbert envelope (`envelope`).

Then, per epoch, and averaged over epochs:

    FC_ij = corr(x_i(t), x_j(t))
    FS_ij = < x_i(t + tau) x_j(t) > / sqrt(var_i var_j)

FS is divided by the **zero-lag** variances, so `FS_ii = acf_i(tau)`. Row =
the later node: this is the orientation of the model's `expm(tau A) S`, so
data and model share the sign of `FS_A`.

**tau** (`preprocessing.tau: "FIT"`): first lag at which the autocorrelation,
averaged over subjects, epochs and nodes, falls to 1/e. One value per dataset
and signal variant.

**Split halves**: the same targets from odd and from even epochs; with a
single epoch (fMRI), from the first and the second half of the run.

## 2. Models (`gectargets/models.py`)

All models are linear, `dx/dt = A x + sigma xi`. With S the stationary
covariance (`A S + S A' = -sigma^2 I`) and the first N state variables
observed:

    FC_sim = corr(S[:N, :N])
    FS_sim = (expm(tau A) S)[:N, :N] / norm

`norm` is `sqrt(S_ii S_jj)` ("zerolag", as the data) or
`sqrt(FS_ii FS_jj)` of the lagged matrix itself ("lagged", so FS_ii = 1).

| Model | Drift A | Constraint after each step | FS norm |
|---|---|---|---|
| `legacy` | `-C`, `C_ii = sum_j abs(C_ij)` | `max(C) = maxC`; `abs(S)` if `use_abs` | lagged |
| `network` | `-theta (I - gamma W)` | `W >= 0`, zero diagonal, spectral radius 1 | zerolag |
| `linear_hopf` | `[[a I - L, -diag(w)], [diag(w), a I - L]]`, `L` Laplacian of C, `w = 2 pi f` | as legacy, no abs | lagged |

* `network`: theta (1/s) is the mean decay rate over modes (`tr(B)/N`);
  `theta: "FIT"` uses the mean over nodes of `-ln(acf_i(tau)) / tau` from the
  targets. gamma in (0, 1) is the coupling relative to the decay.
* `linear_hopf`: one frequency per node from `freqs_path` (Hz; `{subject}` in
  the path for one file per subject). Only x is observed; the lag is applied
  to the full 2N-dimensional state. With the "lagged" normalisation, FS_ii is
  proportional to the autocovariance at tau, which becomes negative when
  `2 pi f tau > pi/2` (alpha at 10 Hz: tau > 25 ms); the fit then fails
  (non-finite FS). Use `"fs_norm": "zerolag"` in that case.
* Any model accepts `sigma` and `fs_norm`. FC and FS do not depend on sigma.

## 3. Learning rule (`gectargets/fitting.py`)

Starting from the SC (`prepare`: absolute value, symmetrised, normalised) or
from `|randn|` (seed from `seed`, the subject id and the initialisation
index), on the entries that are non-zero at the start:

    C += eps_fc (FC - FC_sim) + eps_fs (FS - FS_sim)
    C  = max(C, 0);  C = constraint(C)

Every `check_every` iterations an error E is computed and the fit stops when
`(E_old - E) / E < tol` (this includes E increasing):

* `legacy`: `E = mean((FC - FC_sim)^2) + mean((FS - FS_sim)^2)`;
* `normalized`: `E = sum_k mse_k / mean(target_k^2)` over FC, FS_S, FS_A
  (off-diagonal), so FS_A counts as much as FC;
* `none`: never stop before `max_iter`.

Note `mse(FS) = mse(FS_S) + mse(FS_A)`: the legacy error contains FS_A, but
weighted by its size, which is small.

Every `log_every` iterations the correlation and mse of FC, FS, FS_S, FS_A
are logged (`history_*` in the subject's `.npz`).

## 4. Metrics (`gectargets/analysis.py`)

* **Edge vectors**: FC, FS_S, FS_A upper triangle; FS all off-diagonal.
* **Between-subject similarity**: correlation of edge vectors between
  subjects.
* **Leave-one-out**: corr(subject, mean of the other subjects). For FS_A, a
  sign-flip null (each subject's FS_A multiplied by a random ±1: time
  reversal) tests whether the shared direction exceeds noise.
* **Split-half identifiability** (Amico & Goñi 2018): `i_self` = mean
  corr(odd half of s, even half of s), `i_others` = mean over s != t,
  `i_diff = i_self - i_others`, accuracy = fraction of halves whose best match
  is the same subject.
* **GEC similarity**: on the support of the initial matrix only (the
  entries outside it are 0 for every subject). Reported with `r_to_sc`, the
  similarity of each GEC to the SC it started from, since a shared SC makes
  all subjects alike; split-half identifiability cancels this shared part.
* **Entropy production rate** of the fitted system:
  `Phi = tr(Q' D^-1 Q S^-1)` with `D = sigma^2 / 2 I` and `A S = -(D + Q)`,
  Q antisymmetric (Godrèche & Luck 2019). 0 for a reversible system;
  independent of sigma.

## 5. Differences from fdt_eeg

* Subject fits start from the SC (or seeded random matrix). fdt_eeg started
  each subject from a cohort-level fit; there are no group fits here.
* Band-pass with scipy (same filter design as MNE's default IIR); the cached
  `_eegEpochs_<band>.h5` files of fdt_eeg are not used.
* tau is fitted on the preprocessed signals (after detrending,
  orthogonalisation and z-scoring); fdt_eeg used the filtered (and
  enveloped) epochs.
* FS is stored in the model orientation directly (no transposes).
* The returned FC_sim / FS_sim are those of the returned C (MATLAB returned
  those of the previous iteration when the loop ran to `max_iter`).
* The linear Hopf branch keeps the full 2N covariance (in the MATLAB code
  `expm(tau A)` (2N x 2N) multiplied the N x N block, a dimension error).
* New: the `normalized` / `none` convergence rules, the logged history,
  `split_half`.
* Not ported: `legacy_noabs` (use `legacy` with `"use_abs": false`), the
  leakage operator M (the fit loop can take it later: `M S M'`).
