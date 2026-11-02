"""This module contains functions to compute asymmetry and entropy production,
from a Jupyter notebook provided by Tomas."""

import warnings
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

def compute_theoretical_covariance(B, sigma):

    eigvals = np.linalg.eigvals(B)

    if np.any(np.real(eigvals) < 0):
      warnings.warn("Eigenvalues with negative real part: possibly unstable.")

    return solve_continuous_lyapunov(B, 2 * sigma * np.eye(B.shape[0]))



def asymmetry(B,S,sigma):
  n = B.shape[0]

  xi = np.dot(B,S)-np.eye(n)*sigma
  xisum = np.sum(xi**2)

  return xi, xisum


def compute_entropy_production(B, S, sigma):
    """
    Compute the drift matrix B, solve the Lyapunov equation for the stationary covariance matrix S,
    and compute entropy production (Phi).

    Parameters:
    - Drift matrix B (N X N)

    - S (numpy.ndarray): Stationary covariance matrix.

    Returns:
    - Phi (float): Entropy production.

    """
    I = np.eye(B.shape[0])

    # Compute Q and Entropy Production (Phi)
    Q = np.dot(B, S) - sigma * I
    Phi = -np.trace(np.dot(B, Q)) / sigma

    return Phi

def compute_entropy_production_node(B, S, sigma):
    """
    Computes a node-based metric of EP. We follow Nartallo-Kaluarachchi's EP work
    2024, from where we derive the Q = BS - D matrix that is an edge-wise measure of
    directionality.

    We then sum over a node's edges, taking either:
     - es_i: \sum_j Q_ij / sigma_i net value of asymmetries (signed)
     - em_i: \sum_j Q_ij ^ 2 / sigma_i ^ 2 magnitude of total asymmetry

    Parameters:
    - Drift matrix B (N X N)
    - S (numpy.ndarray): Stationary covariance matrix.
    - sigma: diffusion noise, can be in vector form

    Returns:
    - es: node-specific signed asymmetry
    - em: node-specific magnitude asymmetry
    """
    Q = np.dot(B, S) - sigma * np.eye(B.shape[0])
    Q = 0.5 * (Q - Q.T)  # Take only the antysymmetric component of Q
    es = np.sum(Q, axis=0) / sigma
    em = np.sum(Q**2, axis=0) / sigma**2
    return es, em


def compute_coco_global(J, sigma):
    # Obtain the diagonal matrix related to standard Wiener process
    D = sigma * np.eye(J.shape[0])  # According to right before eq 26

    # Compute theoretical covariance (K, S)
    theta = compute_theoretical_covariance(J, -sigma)

    # We can compute in straightforward manner the information processing (rate of
    # change in entropy)
    info_processing = np.trace(D @ np.linalg.inv(theta) + J)

    # Directly Compute Energy Consumption with eq. 25 from Gus Deco pre-print
    energy_consumption = np.trace(J.T  @ np.linalg.inv(D) @ J @ theta  + J)

    # We return entropy production and should recover the same results,
    # else something wrong
    ep = energy_consumption + info_processing
    return energy_consumption, info_processing, ep


def compute_coco(J, sigma):
    """Computes a ROI-based version of CoCo following Deco et al. 2025's methods."""
    # Obtain the diagonal matrix related to standard Wiener process
    D = sigma * np.eye(J.shape[0])  # According to right before eq 26

    # Compute theoretical covariance (K, S)
    theta = compute_theoretical_covariance(J, -sigma)

    # We need eigendecomposition of theta: theta @ P = lambda @ P, lambda eigenvalues
    # diag matrix and P eigenvectors in each column
    eig_vals, P = np.linalg.eig(theta)
    X = -J @ P
    T = P.T @ theta @ P
    Z = D @ np.linalg.inv(theta) + J
    W = X @ T @ X.T

    # We can compute in straightforward manner the information processing (rate of
    # change in entropy)
    info_processing = np.diag(Z)

    # Directly Compute Energy Consumption with eq. 25 from Gus Deco pre-print
    energy_consumption = np.diag(W) / np.diag(D) + np.diag(J)

    # We return entropy production and should recover the same results,
    # else something wrong
    ep = energy_consumption + info_processing
    return energy_consumption, info_processing, ep