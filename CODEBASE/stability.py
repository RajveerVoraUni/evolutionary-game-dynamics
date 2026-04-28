"""
stability.py
------------
Analytical linearisation of the replicator equation at a fixed point.

The Jacobian of dx_i/dt at x* is:

    J_ij = delta_ij * (f_i(x*) - f_bar(x*))
         + x_i* * (A_ij - [A x*]_j - [A^T x*]_j)

Because the state lives on the (n-1)-dimensional simplex, J always has
one eigenvalue identically equal to zero (the structural zero from the
constraint sum(x) = 1).  Stability classification is based on the
remaining n-1 eigenvalues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Jacobian ──────────────────────────────────────────────────────────────────

def jacobian(A: np.ndarray, x_star) -> np.ndarray:
    """
    Compute the analytical Jacobian of the replicator equation at x*.

    Parameters
    ----------
    A      : Payoff matrix, shape (n, n).
    x_star : Fixed point (strategy frequencies), shape (n,).

    Returns
    -------
    J : Jacobian matrix, shape (n, n).
    """
    x = np.asarray(x_star, dtype=float)
    n = len(x)

    f = A @ x                           # per-strategy fitness at x*
    f_bar = float(x @ f)                # average fitness at x*
    grad_fbar = (A @ x) + (A.T @ x)    # gradient of x^T A x

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            J[i, j] = (
                delta * (f[i] - f_bar)
                + x[i] * (A[i, j] - grad_fbar[j])
            )
    return J


# ── Classification ─────────────────────────────────────────────────────────────

def classify_fixed_point(A: np.ndarray,
                          x_star,
                          tol: float = 1e-8) -> tuple[np.ndarray, str]:
    """
    Classify a fixed point by the eigenvalues of the Jacobian.

    The eigenvalue closest to zero is treated as the structural zero from
    the simplex constraint and excluded from the stability decision.

    Parameters
    ----------
    A      : Payoff matrix.
    x_star : Fixed point to analyse.
    tol    : Tolerance for distinguishing zero from non-zero real parts.

    Returns
    -------
    eigvals        : All n eigenvalues of J (including structural zero).
    classification : One of
                     'Stable Attractor (ESS)' | 'Neutral Center (cycles)'
                     | 'Unstable' | 'Non-hyperbolic (indeterminate)'
    """
    J = jacobian(A, x_star)
    eigvals = np.linalg.eigvals(J)

    # Remove the structural zero (eigenvalue closest to 0)
    idx_zero = np.argmin(np.abs(eigvals))
    reduced = np.delete(eigvals, idx_zero)

    if len(reduced) == 0:
        return eigvals, "Trivial / Undefined"

    max_re = reduced.real.max()
    max_im = np.abs(reduced.imag).max()

    if max_re < -tol:
        label = "Stable Attractor (ESS)"
    elif max_re > tol:
        label = "Unstable"
    elif max_im > tol:
        label = "Neutral Center (cycles)"
    else:
        label = "Non-hyperbolic (indeterminate)"

    return eigvals, label


# ── Batch summary ──────────────────────────────────────────────────────────────

def stability_table(experiments: list[tuple]) -> pd.DataFrame:
    """
    Run classify_fixed_point over a list of (name, A, x_star) tuples
    and return a formatted summary DataFrame.

    Parameters
    ----------
    experiments : List of (game_name, A, x_star) tuples.

    Returns
    -------
    df : DataFrame with columns Game, Fixed Point, Max Re(λ), Has Complex λ,
         Classification.
    """
    rows = []
    for name, A, x_star in experiments:
        eigvals, classification = classify_fixed_point(A, x_star)
        rows.append({
            "Game":           name,
            "Fixed Point":    str(np.round(x_star, 3)),
            "Max Re(λ)":      round(eigvals.real.max(), 4),
            "Has Complex λ":  bool(np.any(np.abs(eigvals.imag) > 1e-8)),
            "Classification": classification,
        })
    return pd.DataFrame(rows)
