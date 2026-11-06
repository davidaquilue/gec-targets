"""Onsager-formalism FDT violation (Gustavo Deco's original formulation),
ported from the user-supplied ``Compute_FDT_Violation_Onsager.m`` (author:
Tomas Berjaga Buisan), same formula as ``neuronumba``'s
``LinearFdtDeco2023``:

    invA       = inv(A)
    chi(i,j)   = |2 * COV(i,j) / sigma^2 + invA(i,j)|
    chi2(i,j)  = |invA(i,j)|
    chij[j]    = mean_i(chi[:n,:n])[j] / mean_i(chi2[:n,:n])[j]
    FDT_value  = mean_j(chij[j])

where A is the (stable) Jacobian of the full state -- ``model.jacobian(C)``,
already 2N x 2N for linear_hopf (the MATLAB code rebuilds it from Ceff via
hopf_int; here it is read straight from the saved fit, which is the same
matrix, see LinearHopf.jacobian in models.py) -- COV is its stationary
covariance (A COV +
COV A' = -sigma^2 I, ``models.stationary_cov``), and n is the number of
observed nodes: the matrices are restricted to the n x n (x-x) block before
averaging, as in the original, which only ever used the N x N Ceff block.

sigma is the fit's own noise level (the ``sigma`` saved by ``fit_subject``).
"""
import numpy as np

from .models import stationary_cov


def fdt_violation_onsager(A, sigma, n):
    """chij (per observed node, length n) and the scalar FDT_value.

    Raises ``numpy.linalg.LinAlgError`` if A (equivalently inv(A)) is
    singular; callers should only pass a stable A (as ``downstream.py``'s
    "stable" check already requires for the other downstream measures)."""
    cov = stationary_cov(A, sigma)
    inv_a = np.linalg.inv(A)
    chi = np.abs(2 * cov / sigma ** 2 + inv_a)
    chi2 = np.abs(inv_a)
    chij = chi[:n, :n].mean(axis=0) / chi2[:n, :n].mean(axis=0)
    return chij, float(np.mean(chij))
