"""Stationary integral FDT violation (IV-FDT) of the lab's updated HCP
pipeline (neuro_HCP, src/fdt/ivfdt_metrics.py: I_inf_var_matrix,
node_iv_fdt_metrics, global_iv_fdt_metrics; computed per subject in
04_compute_metrics), from the jacobian A of a fit:

    Gamma = -A, Q = sigma^2 I          dx/dt = -Gamma x + noise
    V solves Gamma V + V Gamma' = Q    stationary covariance
    I_inf = (V - inv(Gamma) V Gamma') / 2  =  V - inv(Gamma) Q / 2
    I_var = Delta^-1/2 I_inf Delta^-1/2, Delta = diag(V)

the t -> infinity limit of I(t, s) = C(t,t) - C(t,s) - T int_s^t R, started
from the stationary state, for every pair of variables, variance-normalised.
For linear_hopf the x block (the n observed nodes) is used, as their
``block='x'``. Per node, as their I_tot,

    I_tot(i) = sqrt(mean_j I_var(i,j)^2)       over the n x n block

and globally the order parameter of their main text

    rho_brk = ||I_skew|| / (||I_skew|| + ||I_sym||)

with I_skew / I_sym the antisymmetric / symmetric parts of I_var and ||.||
the RMS over the off-diagonal entries. Their I_tot^RMS (the global measure
of the task vs REST1 results) is the RMS of I_tot over nodes.

The diagonal of I_inf (not normalised) is what fdt_eeg's integral violation
approaches (it starts from a non-stationary V0 and stops at t = 100 s);
I_var differs from it by the normalisation and the cross terms.
"""
import numpy as np

from .models import stationary_cov


def ivfdt_matrix(A, sigma):
    """I_var, the variance-normalised stationary violation matrix (full
    system)."""
    gamma = -A
    V = stationary_cov(A, sigma)          # A V + V A' = -sigma^2 I
    I_inf = 0.5 * (V - np.linalg.solve(gamma, V) @ gamma.T)
    d = np.sqrt(np.diag(V))
    return I_inf / np.outer(d, d)


def ivfdt_stationary(A, sigma, n):
    """I_tot of each of the n observed nodes and rho_brk, from I_var
    restricted to its n x n (x) block."""
    I_var = ivfdt_matrix(A, sigma)[:n, :n]
    i_tot = np.sqrt(np.mean(I_var ** 2, axis=1))
    off = ~np.eye(n, dtype=bool)
    skew = np.sqrt(np.mean((0.5 * (I_var - I_var.T))[off] ** 2))
    sym = np.sqrt(np.mean((0.5 * (I_var + I_var.T))[off] ** 2))
    return i_tot, float(skew / (skew + sym))
