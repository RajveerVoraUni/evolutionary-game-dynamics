"""
games.py
--------
Canonical payoff matrices for evolutionary game theory experiments.

Each factory function returns a float64 ndarray of shape (n, n) where
entry A[i, j] is the payoff to strategy i when matched against strategy j.
"""

import numpy as np


# ── Payoff matrix factories ────────────────────────────────────────────────────

def prisoners_dilemma(R: float = 3, P: float = 1,
                      T: float = 5, S: float = 0) -> np.ndarray:
    """
    Prisoner's Dilemma.

    Strategies: [Cooperate (C), Defect (D)]
    Ordering:   T > R > P > S

    Parameters
    ----------
    R : Reward for mutual cooperation.
    P : Punishment for mutual defection.
    T : Temptation payoff (defect vs cooperator).
    S : Sucker payoff (cooperate vs defector).
    """
    assert T > R > P > S, "PD requires T > R > P > S"
    return np.array([[R, S],
                     [T, P]], dtype=float)


def hawk_dove(V: float = 4, C: float = 6) -> np.ndarray:
    """
    Hawk-Dove game.

    Strategies: [Hawk (H), Dove (D)]
    Analytical mixed ESS: x*(Hawk) = V / C   (requires C > V)

    Parameters
    ----------
    V : Value of the contested resource.
    C : Cost of injury from fighting (must exceed V for interior ESS).
    """
    assert C > V, "Hawk-Dove interior ESS requires C > V"
    return np.array([[(V - C) / 2, V],
                     [0,           V / 2]], dtype=float)


def rock_paper_scissors(eps: float = 0.0) -> np.ndarray:
    """
    Rock-Paper-Scissors with optional asymmetry parameter.

    Strategies: [Rock (R), Paper (P), Scissors (S)]

    Parameters
    ----------
    eps : Diagonal perturbation.
          eps = 0  → neutral center, closed orbits, H = x1*x2*x3 conserved.
          eps > 0  → trajectories spiral inward (stable center).
          eps < 0  → trajectories spiral outward (unstable, heteroclinic cycle).
    """
    return np.array([[ eps,  -1,   1],
                     [  1,   eps, -1],
                     [ -1,   1,  eps]], dtype=float)


def market_microstructure() -> np.ndarray:
    """
    Financial market microstructure game.

    Strategies: [Momentum (M), Mean-Reversion (MR), Noise (N)]

    Entry A[i, j] = expected return of strategy i when the counterpart uses j.
      A[0,0] = -0.5  crowded-trade effect (momentum vs momentum)
      A[1,0] =  1.0  mean-reversion exploits momentum overshooting
      A[1,1] =  0.2  moderate self-interaction for mean-reversion
      A[2,·] =  0.0  noise traders earn zero expected return
    """
    return np.array([[-0.5,  1.5,  0.5],
                     [ 1.0,  0.2,  0.3],
                     [ 0.0,  0.0,  0.0]], dtype=float)


# ── Fitness helpers ────────────────────────────────────────────────────────────

def fitness(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-strategy fitness vector: f = A @ x."""
    return A @ x


def avg_fitness(A: np.ndarray, x: np.ndarray) -> float:
    """Average population fitness: f_bar = x^T A x."""
    return float(x @ A @ x)
