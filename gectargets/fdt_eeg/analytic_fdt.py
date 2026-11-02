"""This module contains the functions used to compute the analytical variant of FDT
violations. We will adapt the functions from Juan Monti (provided by Melle) for fMRI
data with heterogeneous noise and frequencies fit to each ROI freq.

In this case, we will apply the same methodology as has been applied for the original
implementation of FDT, which just takes into account the GEC and a sigma value (
homogeneous sigma).
"""
import numpy as np

from numba import jit, njit
from scipy.linalg import solve_continuous_lyapunov


def v_0_calculation_v0gauss(v0std, v0bias, ndim):
    I = np.ones(ndim)
    v0bias_array = v0bias * I
    v0std_square_array = v0std**2 * I
    term1 = np.diag(v0std_square_array)
    term2 = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            term2[i,j] = v0bias_array[i] * v0bias_array[j]
    v_0 = term1 + term2
    return v_0

def v_0_calculation_v0fixed(v0):
    v_0 = np.outer(v0,v0)
    return v_0


@jit(nopython=True)
def exp_scaling_squaring(gamma, m=10):
    """
    Compute exp(Gamma) using the Scaling and Squaring Method.

    Parameters:
        gamma (ndarray): Input square matrix (NxN).
        m (int): Degree of the Pade approximant for the series expansion.

    Returns:
        ndarray: Matrix exponential exp(Gamma).
    """
    # Compute the norm of Gamma (1-norm)
    norm_gamma = np.linalg.norm(gamma, ord=1)

    # Scaling step: Find scaling factor 2^k
    k = max(0, int(np.ceil(np.log2(norm_gamma))))  # Scale s.t. norm(Gamma / 2^k) small
    gamma_scaled = gamma / (2 ** k)

    # Compute exp(gamma_scaled) using a series expansion or Pade approximant
    n = gamma.shape[0]
    result = np.eye(n)  # Initialize result as the identity matrix
    term = np.eye(n)  # Initialize current term as the identity matrix

    for i in range(1, m + 1):
        term = np.dot(term, gamma_scaled) / i  # Compute the next term in the series
        result += term  # Add to the result

    # Squaring step: Square the result k times
    for _ in range(k):
        result = np.dot(result, result)

    return result


@njit
def compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared):
    """
    term2 analytic computation.
    Optimized computation of the vector B(t, s) using Numba.
    """
    N = len(eigvals)

    # eigvals = np.real(eigvals)

    # Precompute sigma weight terms
    S = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        for m in range(N):
            S[k, m] = np.sum(sigma_squared * P_inv[k, :] * P_inv[m, :])

    # Precompute exponentials and integration terms
    integral_term = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        eigval_k = eigvals[k]
        for m in range(N):
            eigval_m = eigvals[m]
            denom = eigval_k + eigval_m
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                integral_term[k, m] = (np.exp(-eigval_k * (t - s)) - np.exp(
                    -(eigval_k * t + eigval_m * s))) / denom
            else:
                integral_term[k, m] = s * np.exp(
                    -(eigval_k * t + eigval_m * s))  # Special case when denom is zero

    # Compute B[i]
    term2_of_C = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for k in range(N):
            for m in range(N):
                term_km = (
                        P[i, k] * P[i, m] *
                        integral_term[k, m] *
                        S[k, m]
                )
                term2_of_C[i] += term_km

    return np.real(term2_of_C)  # Return the real part


def compute_term2_of_C_precompute(t, s, Gamma, sigma_squared):
    """
    Wrapper to precompute eigenvalues and eigenvectors for a general matrix Gamma.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    P = eigvecs
    P_inv = np.linalg.inv(P)
    return compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared)


### Compute Cts ###############################################
@jit(forceobj=True)
def Cts_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Exponential matrices using scipy.linalg.expm
    exp_Gamma_t = exp_scaling_squaring(-Gamma * t)
    exp_GammaT_s = exp_scaling_squaring(-Gamma.T * s)
    # First term: (e^{-\Gamma t} V_0 e^{-\Gamma^T s})_{ii}
    term1 = np.diag(exp_Gamma_t @ V_0 @ exp_GammaT_s)
    # Second term:
    term2 = compute_term2_of_C_precompute(t, s, Gamma, sigma**2)
    C = term1 + term2
    return C


### Compute I ###############################################
@jit(forceobj=True)
def Its_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Its = C_tt - C_ts - (sigma**2 / 2) * term3_ii
    return Its

### Compute Inorm1 ###############################################
### Inorm1(t,s) = I(t,s)/(sigma^2 / 2 * int_R(t,s))
@jit(forceobj=True)
def Its_norm1_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Compute int_R
    intRts = np.diag(np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s))
    # Calculate I_i(t, s)
    Inorm1 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (sigma**2 / 2 * intRts + eps)
    return Inorm1

### Compute Inorm2 ###############################################
### Inorm2(t,s) = I(t,s)/(C(t,t) + eps)
@jit(forceobj=True)
def Its_norm2_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Inorm2 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (C_tt + eps)
    return Inorm2

### Compute Inorm3 ###############################################
### Inorm3(t,s) = I(t,s)/(C(t,t) - C(t,s) + eps)
@jit(forceobj=True)
def Its_norm3_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Inorm2 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (C_tt - C_ts + eps)
    return Inorm2




def pipeline_analytical_fdt_integral(Ceff, sigma, normalizations=True,
                                     thermalization=False, ts=0, tmax=100,
                                     v0std=0.05, v0bias=0):
    """Takes a subject's Ceff and returns the integral violation of FDT for each
    region. If normalizations=True, it returns a 3-element tuple with the original
    integral violation, and the Norm1 and Norm2 (unsure about what these mean)."""
    NROIS = Ceff.shape[0]
    # Gamma seems to be the opposite sign to the Jacobian
    gamma = -(-Ceff)  # Since we just use the linearized Langevin

    if thermalization:
        # D should be the diagonal with just sigma in the diagonal
        D = np.diag(sigma ** 2 * np.ones(2 * NROIS))
        V_0 = solve_continuous_lyapunov(gamma, D)
    else:
        ndim = np.shape(gamma)[0]
        V_0 = v_0_calculation_v0gauss(v0std, v0bias, ndim)

    # Compute integral violation
    int_viol_tmax_s0 = Its_Langevin_ND(gamma, sigma, V_0, tmax, ts)[0:NROIS]

    if normalizations:
        int_viol_norm1_tmax_s0 = Its_norm1_Langevin_ND(gamma, sigma, V_0, tmax, ts)[0:NROIS]
        int_viol_norm2_tmax_s0 = Its_norm2_Langevin_ND(gamma, sigma, V_0, tmax, ts)[0:NROIS]
        return int_viol_tmax_s0, int_viol_norm1_tmax_s0, int_viol_norm2_tmax_s0
    else:
        return int_viol_tmax_s0

