"""
stochastic.py
-------------
Stochastic replicator dynamics via Euler-Maruyama integration.

Implements the Itô SDE:
    dx_i = x_i(f_i - f_bar) dt  +  sigma * x_i(1 - x_i) dW_t

The diffusion coefficient sigma * x_i * (1 - x_i) vanishes at the
simplex boundaries, preserving the qualitative geometry of the state space.

After each Euler-Maruyama step the state is projected back onto the simplex
by clipping to [0, inf) and renormalizing.
"""

import numpy as np


def stochastic_replicator(A: np.ndarray,
                           x0,
                           t_end: float = 50.0,
                           n_steps: int = 5000,
                           sigma: float = 0.05,
                           seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate stochastic replicator dynamics (Euler-Maruyama scheme).

    Parameters
    ----------
    A       : Payoff matrix, shape (n, n).
    x0      : Initial strategy distribution, shape (n,). Will be normalized.
    t_end   : Simulation end time.
    n_steps : Number of time steps.
    sigma   : Noise intensity (diffusion coefficient scale).
    seed    : Random seed for reproducibility. None → non-deterministic.

    Returns
    -------
    t   : Time array, shape (n_steps,).
    sol : Solution array, shape (n_steps, n). Each row sums to 1.
    """
    rng = np.random.default_rng(seed)

    x0 = np.asarray(x0, dtype=float)
    x0 = np.clip(x0, 1e-6, 1.0 - 1e-6)
    x0 /= x0.sum()

    n = len(x0)
    dt = t_end / n_steps
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0.0, t_end, n_steps)
    sol = np.zeros((n_steps, n))
    sol[0] = x0
    x = x0.copy()

    for k in range(1, n_steps):
        f = A @ x
        f_bar = float(x @ f)

        drift = x * (f - f_bar) * dt
        diffusion = sigma * x * (1.0 - x) * sqrt_dt * rng.standard_normal(n)

        x = x + drift + diffusion

        # Simplex projection: clip negatives, then renormalize
        x = np.clip(x, 1e-10, None)
        x /= x.sum()

        sol[k] = x

    return t, sol
