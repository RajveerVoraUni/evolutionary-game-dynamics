"""
dynamics.py
-----------
Deterministic replicator dynamics via scipy.integrate.odeint.

Implements the replicator equation:
    dx_i/dt = x_i * [(Ax)_i - x^T A x]

The probability simplex is forward-invariant: solutions starting on
Delta^{n-1} remain there for all t >= 0.
"""

import numpy as np
from scipy.integrate import odeint


def replicator_rhs(x: np.ndarray, t: float, A: np.ndarray) -> np.ndarray:
    """
    Right-hand side of the replicator ODE.

    Parameters
    ----------
    x : Current state vector (strategy frequencies), shape (n,).
    t : Current time (unused; system is autonomous).
    A : Payoff matrix, shape (n, n).

    Returns
    -------
    dxdt : ndarray, shape (n,).
    """
    x = np.clip(x, 0.0, 1.0)
    s = x.sum()
    if s > 0:
        x = x / s                   # enforce simplex constraint numerically
    f = A @ x                       # per-strategy fitness
    f_bar = x @ f                   # average population fitness
    return x * (f - f_bar)          # replicator equation (vectorized)


def simulate(A: np.ndarray,
             x0,
             t_end: float = 50.0,
             n_steps: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the replicator ODE from initial condition x0.

    Parameters
    ----------
    A       : Payoff matrix, shape (n, n).
    x0      : Initial strategy distribution, shape (n,). Will be normalized.
    t_end   : Integration end time.
    n_steps : Number of output time points.

    Returns
    -------
    t   : Time array, shape (n_steps,).
    sol : Solution array, shape (n_steps, n).
    """
    x0 = np.asarray(x0, dtype=float)
    x0 = np.clip(x0, 1e-8, None)
    x0 /= x0.sum()

    t = np.linspace(0.0, t_end, n_steps)
    sol = odeint(replicator_rhs, x0, t, args=(A,),
                 rtol=1e-8, atol=1e-10, mxstep=10_000)

    # Post-process: keep solution on the simplex
    sol = np.clip(sol, 0.0, 1.0)
    sol /= sol.sum(axis=1, keepdims=True)
    return t, sol
