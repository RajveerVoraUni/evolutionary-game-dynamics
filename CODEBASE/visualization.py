"""
visualization.py
----------------
Plotting utilities for replicator dynamics experiments.

Provides:
  - Ternary simplex drawing (draw_simplex, draw_simplex_clean)
  - Coordinate conversion (to_cartesian)
  - Time-series plots (plot_timeseries_2strategy, plot_timeseries_3strategy)
  - Simplex trajectory plot (ternary_plot)
  - Eigenvalue complex-plane plot (plot_eigenvalues)
  - Market phase portrait helpers (plot_trajectory_clean, plot_equilibrium,
    vector_field)
  - Figure saving (save_fig)

All functions accept an optional `ax` / `axes` argument so they can be
embedded into larger figure layouts.  When called without one, a new figure
is created and returned.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

from dynamics import replicator_rhs     # used by vector_field

# ── Global style ──────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
SIGMA_COLORS = ["#FF9800", "#F44336", "#9C27B0"]   # per stochastic σ level
IC_COLORS = ["#2196F3", "#E53935", "#43A047", "#FB8C00"]

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 2,
    "legend.framealpha": 0.85,
})


# ── Figure persistence ────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: str | Path) -> None:
    """Save figure to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Ternary geometry ──────────────────────────────────────────────────────────

def to_cartesian(x1, x2, x3) -> tuple[np.ndarray, np.ndarray]:
    """
    Map barycentric coordinates (x1, x2, x3) to 2-D Cartesian.

    Vertex mapping:
        x1 → top    (0.5, sqrt(3)/2)
        x2 → left   (0, 0)
        x3 → right  (1, 0)
    """
    s = x1 + x2 + x3
    cx = 0.5 * (2 * x3 + x1) / s
    cy = (np.sqrt(3) / 2) * x1 / s
    return cx, cy


def draw_simplex(ax: plt.Axes,
                 labels: tuple = ("S1", "S2", "S3"),
                 offset: float = 0.06) -> None:
    """Draw a basic equilateral triangle simplex on *ax*."""
    h = np.sqrt(3) / 2
    tri = plt.Polygon([[0.5, h], [0, 0], [1, 0]],
                       fill=False, edgecolor="#333", linewidth=1.5)
    ax.add_patch(tri)
    ax.text(0.5,      h + offset,  labels[0], ha="center", fontsize=12, fontweight="bold")
    ax.text(-offset, -offset,      labels[1], ha="center", fontsize=12, fontweight="bold")
    ax.text(1+offset, -offset,     labels[2], ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.10, 1.00)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_simplex_clean(ax: plt.Axes,
                       labels: tuple = ("S1", "S2", "S3"),
                       label_offset: float = 0.10) -> None:
    """Draw a clean simplex with larger, semibold labels and more padding."""
    h = np.sqrt(3) / 2
    tri = plt.Polygon([[0.5, h], [0, 0], [1, 0]],
                       fill=False, edgecolor="#222222", linewidth=1.8, zorder=1)
    ax.add_patch(tri)
    ax.text(0.50,              h + label_offset, labels[0],
            ha="center", va="bottom", fontsize=13, fontweight="semibold")
    ax.text(0.00 - label_offset, -0.05,           labels[1],
            ha="center", va="top",    fontsize=13, fontweight="semibold")
    ax.text(1.00 + label_offset, -0.05,           labels[2],
            ha="center", va="top",    fontsize=13, fontweight="semibold")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.12, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Time-series plots ─────────────────────────────────────────────────────────

def plot_timeseries_2strategy(results: list[tuple],
                               strategy_names: list[str],
                               title: str,
                               hline: float | None = None,
                               hline_label: str = "",
                               semilogy_panel: bool = False,
                               semilogy_ref: float | None = None) -> plt.Figure:
    """
    Two-panel time-series figure for a 2-strategy game.

    Left panel  : strategy 0 frequency for all initial conditions.
    Right panel : either strategy 0 & 1 for the middle IC, or a
                  semi-log deviation plot if semilogy_panel=True.

    Parameters
    ----------
    results        : List of (t, sol, label) tuples from dynamics.simulate.
    strategy_names : Names of the two strategies.
    title          : Figure suptitle.
    hline          : Optional horizontal reference line on the left panel.
    hline_label    : Legend label for the hline.
    semilogy_panel : If True, right panel shows log-scale deviation from hline.
    semilogy_ref   : Reference value for semi-log deviation (defaults to hline).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for t, sol, lbl in results:
        ax.plot(t, sol[:, 0], label=lbl, alpha=0.9)
    if hline is not None:
        ax.axhline(hline, color="k", linestyle="--", linewidth=2, label=hline_label)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Fraction of {strategy_names[0]}")
    ax.set_title(f"{strategy_names[0]} Frequency")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if semilogy_panel and hline is not None:
        ref = semilogy_ref if semilogy_ref is not None else hline
        for t, sol, lbl in results:
            ax2.semilogy(t, np.abs(sol[:, 0] - ref) + 1e-12, label=lbl, alpha=0.9)
        ax2.set_ylabel(f"|{strategy_names[0]}(t) − x*| (log scale)")
        ax2.set_title("Convergence Rate to ESS")
    else:
        mid = results[len(results) // 2]
        t, sol, _ = mid
        ax2.plot(t, sol[:, 0], color=COLORS[0], label=strategy_names[0])
        ax2.plot(t, sol[:, 1], color=COLORS[1], label=strategy_names[1])
        ax2.set_ylabel("Population share")
        ax2.set_title(f"Strategy Dynamics")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_timeseries_3strategy(results: list[tuple],
                               strategy_names: list[str],
                               title: str) -> plt.Figure:
    """
    2×2 grid of time-series panels, one per initial condition.

    Parameters
    ----------
    results        : List of (t, sol, label) tuples.
    strategy_names : Names of the three strategies.
    title          : Figure suptitle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (t, sol, lbl) in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        for i, sname in enumerate(strategy_names):
            ax.plot(t, sol[:, i], color=COLORS[i], label=sname)
        ax.axhline(1 / 3, color="k", linestyle=":", linewidth=1.5, label="x*=1/3")
        ax.set_title(f"Initial condition: {lbl}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population share")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Simplex trajectory plot ───────────────────────────────────────────────────

def ternary_plot(trajectories: list[np.ndarray],
                 labels: list[str],
                 strategy_names: tuple = ("S1", "S2", "S3"),
                 title: str = "Simplex Trajectories",
                 ax: plt.Axes | None = None,
                 colors: list | None = None) -> plt.Figure | None:
    """
    Plot multiple trajectories on the ternary simplex.

    Parameters
    ----------
    trajectories   : List of (n_steps, 3) arrays.
    labels         : Trajectory labels.
    strategy_names : Vertex labels (top, left, right).
    title          : Axes title.
    ax             : Optional axes to plot on.
    colors         : Optional color list; defaults to COLORS.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    draw_simplex(ax, labels=strategy_names)
    _colors = colors if colors is not None else COLORS

    for idx, (traj, lbl) in enumerate(zip(trajectories, labels)):
        cx, cy = to_cartesian(traj[:, 0], traj[:, 1], traj[:, 2])
        c = _colors[idx % len(_colors)]
        ax.plot(cx, cy, color=c, alpha=0.75, linewidth=1.8, label=lbl)
        ax.plot(cx[0],  cy[0],  "o", color=c, markersize=7, zorder=5)
        ax.plot(cx[-1], cy[-1], "*", color=c, markersize=12, zorder=5)

    ax.set_title(title, pad=18)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    if own_fig:
        fig.tight_layout()
        return fig
    return None


# ── Eigenvalue plot ───────────────────────────────────────────────────────────

def plot_eigenvalues(experiments: list[tuple],
                     classify_fn) -> plt.Figure:
    """
    1×n complex-plane plot of eigenvalues at each fixed point.

    Parameters
    ----------
    experiments  : List of (name, A, x_star) tuples.
    classify_fn  : stability.classify_fixed_point function.
    """
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    theta = np.linspace(0, 2 * np.pi, 200)

    for ax, (name, A, x_star) in zip(axes, experiments):
        eigvals, classification = classify_fn(A, x_star)

        ax.plot(np.cos(theta) * 0.5, np.sin(theta) * 0.5, "gray", alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.scatter(eigvals.real, eigvals.imag,
                   s=120, c=COLORS[:len(eigvals)], zorder=5, edgecolors="k")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(f"{name}\n{classification}", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle(
        "Eigenvalue Analysis at Fixed Points\n"
        "(one eigenvalue is structurally zero — simplex constraint)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Market microstructure phase portrait helpers ──────────────────────────────

def plot_trajectory_clean(ax: plt.Axes,
                           sol: np.ndarray,
                           color: str,
                           label: str | None = None,
                           lw: float = 2.0,
                           alpha: float = 0.88,
                           n_arrows: int = 3) -> None:
    """
    Plot a single simplex trajectory with direction arrows,
    hollow-circle start, and filled-star end.
    """
    cx, cy = to_cartesian(sol[:, 0], sol[:, 1], sol[:, 2])
    ax.plot(cx, cy, color=color, linewidth=lw, alpha=alpha, label=label, zorder=3)

    n = len(cx)
    for idx in np.linspace(int(n * 0.10), int(n * 0.80), n_arrows).astype(int):
        ax.annotate("",
                    xy=(cx[idx + 1], cy[idx + 1]),
                    xytext=(cx[idx], cy[idx]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=1.6, mutation_scale=13),
                    zorder=4)

    # ○ hollow start
    ax.plot(cx[0], cy[0], "o", color="white", markersize=9,
            markeredgecolor=color, markeredgewidth=2.0, zorder=5)
    # ★ filled end
    ax.plot(cx[-1], cy[-1], "*", color=color, markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)


def plot_equilibrium(ax: plt.Axes,
                     eq_pt: np.ndarray,
                     label: str = r"$x^*$",
                     offset: tuple = (0.11, 0.08)) -> None:
    """Plot the equilibrium as a dominant black diamond with a white halo."""
    cx, cy = to_cartesian(*eq_pt)
    ax.plot(cx, cy, "D", color="white",  markersize=14, zorder=6)
    ax.plot(cx, cy, "D", color="black",  markersize=10, zorder=7)
    ax.annotate(label, xy=(cx, cy),
                xytext=(cx + offset[0], cy + offset[1]),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
                zorder=8)


def vector_field(ax: plt.Axes,
                 A: np.ndarray,
                 n_grid: int = 20,
                 alpha: float = 0.25,
                 scale: float = 0.028) -> None:
    """
    Draw a faint replicator vector field over the simplex interior.

    All arrows have the same length (unit direction × scale); only the
    position encodes the flow geometry.  This keeps the field readable as
    a background layer behind trajectory overlays.
    """
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            x = np.array([i, j, k], dtype=float) / n_grid
            if np.any(x < 0.04):
                continue
            dxdt = replicator_rhs(x, 0, A)
            px, py = to_cartesian(*x)
            vx = 0.5 * (2 * dxdt[2] + dxdt[0])
            vy = (np.sqrt(3) / 2) * dxdt[0]
            mag = np.hypot(vx, vy) + 1e-12
            ax.annotate("",
                        xy=(px + vx / mag * scale, py + vy / mag * scale),
                        xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color="#333333",
                                        lw=0.7, mutation_scale=6, alpha=alpha),
                        zorder=2)


def market_phase_portrait(mkt_results: list[tuple],
                           grid_ics: list[np.ndarray],
                           A: np.ndarray,
                           eq_sol: np.ndarray,
                           strategy_names: tuple,
                           simulate_fn) -> plt.Figure:
    """
    Three-panel market microstructure phase portrait:
      Panel 1 — Core strategic flow (4 canonical trajectories).
      Panel 2 — Local flow geometry (vector field + one representative).
      Panel 3 — Basin robustness (18 random interior starts).

    Parameters
    ----------
    mkt_results    : List of (t, sol, label) from simulate.
    grid_ics       : List of 18 interior initial conditions.
    A              : Market payoff matrix.
    eq_sol         : Equilibrium point (converged state).
    strategy_names : Vertex labels for the simplex.
    simulate_fn    : dynamics.simulate function.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # ── Panel 1: Core strategic flow ──────────────────────────────────────────
    draw_simplex_clean(ax1, labels=strategy_names)
    for (t, sol, lbl), color in zip(mkt_results, IC_COLORS):
        plot_trajectory_clean(ax1, sol, color, label=lbl, n_arrows=3)
    plot_equilibrium(ax1, eq_sol,
                     label=r"$x^*=(0.46,\,0.54,\,0)$",
                     offset=(0.12, 0.09))

    legend_handles = (
        [Line2D([0], [0], color=c, lw=2, label=lbl)
         for (_, _, lbl), c in zip(mkt_results, IC_COLORS)]
        + [
            Line2D([0], [0], marker="o", color="gray", lw=0,
                   markerfacecolor="white", markeredgecolor="gray",
                   markersize=8, label="Start"),
            Line2D([0], [0], marker="*", color="gray", lw=0,
                   markersize=11, label="End"),
            Line2D([0], [0], marker="D", color="black", lw=0,
                   markersize=8, label="Equilibrium"),
        ]
    )
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax1.set_title("Core Strategic Flow", fontsize=12, fontweight="bold")

    # ── Panel 2: Local flow geometry ──────────────────────────────────────────
    draw_simplex_clean(ax2, labels=strategy_names)
    vector_field(ax2, A, n_grid=22, alpha=0.25, scale=0.028)
    _, sol_rep, _ = mkt_results[3]          # uniform-start trajectory
    plot_trajectory_clean(ax2, sol_rep, IC_COLORS[3], n_arrows=4, lw=2.2)
    plot_equilibrium(ax2, eq_sol, label=r"$x^*$", offset=(0.10, 0.08))
    ax2.set_title("Local Flow Geometry", fontsize=12, fontweight="bold")

    # ── Panel 3: Basin robustness ─────────────────────────────────────────────
    draw_simplex_clean(ax3, labels=strategy_names)
    cmap = plt.cm.plasma
    for k, x0 in enumerate(grid_ics):
        _, sol = simulate_fn(A, x0, t_end=60, n_steps=6000)
        color = cmap(k / (len(grid_ics) - 1))
        plot_trajectory_clean(ax3, sol, color, n_arrows=2, lw=1.4, alpha=0.7)
    plot_equilibrium(ax3, eq_sol, label=r"$x^*$", offset=(0.10, 0.08))
    ax3.set_title("Basin Robustness", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Market Microstructure — Replicator Dynamics Phase Portrait",
        fontsize=15, fontweight="bold",
    )
    return fig


# ── Sensitivity analysis plot ─────────────────────────────────────────────────

def plot_sensitivity(param_values: np.ndarray,
                     equilibria: np.ndarray,
                     base_case: float = -0.5) -> plt.Figure:
    """
    Two-panel sensitivity figure: line chart + stacked area.

    Parameters
    ----------
    param_values : 1-D array of A[0,0] values.
    equilibria   : (len(param_values), 3) array of equilibrium fractions.
    base_case    : Parameter value to mark with a vertical dashed line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ["Momentum", "Mean-Reversion", "Noise"]

    ax = axes[0]
    for i, lbl in enumerate(labels):
        ax.plot(param_values, equilibria[:, i], color=COLORS[i],
                label=lbl, linewidth=2)
    ax.axvline(base_case, color="k", linestyle="--", linewidth=1.5,
               label=f"Base case A[0,0]={base_case}")
    ax.set_xlabel("Momentum self-payoff A[0,0]")
    ax.set_ylabel("Equilibrium fraction")
    ax.set_title("Equilibrium Composition vs Momentum Self-Payoff")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    ax2 = axes[1]
    ax2.stackplot(param_values,
                  equilibria[:, 0], equilibria[:, 1], equilibria[:, 2],
                  labels=labels,
                  colors=[COLORS[0], COLORS[1], COLORS[2]], alpha=0.7)
    ax2.axvline(base_case, color="k", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Momentum self-payoff A[0,0]")
    ax2.set_ylabel("Equilibrium fraction")
    ax2.set_title("Stacked Equilibrium Composition")
    ax2.legend(loc="upper left")

    fig.suptitle("Sensitivity Analysis: Momentum Self-Payoff",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
