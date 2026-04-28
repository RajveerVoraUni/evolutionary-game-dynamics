"""
main.py
-------
Entry point for the Replicator Dynamics project.

Run:
    python main.py

Reproduces the full experimental pipeline:
  1. Prisoner's Dilemma
  2. Hawk-Dove
  3. Rock-Paper-Scissors (deterministic)
  4. Stability analysis (Jacobian + eigenvalues)
  5. Stochastic RPS (Euler-Maruyama)
  6. Market microstructure (time-series + phase portrait)
  7. Sensitivity analysis

All figures are saved to ./figures/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from games import (
    prisoners_dilemma, hawk_dove, rock_paper_scissors, market_microstructure,
)
from dynamics import simulate
from stochastic import stochastic_replicator
from stability import classify_fixed_point, stability_table
from visualization import (
    save_fig,
    to_cartesian,
    draw_simplex,
    plot_timeseries_2strategy,
    plot_timeseries_3strategy,
    ternary_plot,
    plot_eigenvalues,
    market_phase_portrait,
    plot_sensitivity,
    COLORS, SIGMA_COLORS,
)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit here to change any experimental setting
# ══════════════════════════════════════════════════════════════════════════════

FIGURES_DIR = Path("figures")

# Random seeds
GLOBAL_SEED     = 42
STOCH_SEED      = 42
BASIN_SEED      = 7

# Prisoner's Dilemma
PD_T, PD_R, PD_P, PD_S = 5, 3, 1, 0
PD_ICS = [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
PD_IC_LABELS = ["x_C=0.90", "x_C=0.70", "x_C=0.50", "x_C=0.30", "x_C=0.10"]
PD_T_END, PD_N_STEPS = 30, 3000

# Hawk-Dove
HD_V, HD_C = 4, 6
HD_ICS = [[0.05, 0.95], [0.20, 0.80], [0.50, 0.50], [0.80, 0.20], [0.95, 0.05]]
HD_IC_LABELS = ["x_H=0.05", "x_H=0.20", "x_H=0.50", "x_H=0.80", "x_H=0.95"]
HD_T_END, HD_N_STEPS = 40, 4000

# Rock-Paper-Scissors (deterministic)
RPS_EPS = 0.0
RPS_ICS = [[0.50, 0.30, 0.20], [0.70, 0.20, 0.10],
           [0.10, 0.10, 0.80], [0.35, 0.35, 0.30]]
RPS_IC_LABELS = ["IC1", "IC2", "IC3", "IC4"]
RPS_T_END, RPS_N_STEPS = 80, 8000

# Stochastic RPS
STOCH_X0         = [0.5, 0.3, 0.2]
STOCH_T_END      = 750
STOCH_N_STEPS_TS = 50_000   # time-series (lower resolution for speed)
STOCH_N_STEPS_SX = 75_000   # simplex trajectories (higher resolution)
STOCH_SIGMAS     = [0.02, 0.05, 0.10]

# Market microstructure
MKT_ICS = [
    ([0.60, 0.20, 0.20], "Momentum-heavy"),
    ([0.20, 0.60, 0.20], "MeanRev-heavy"),
    ([0.20, 0.20, 0.60], "Noise-heavy"),
    ([1/3,  1/3,  1/3],  "Uniform start"),
]
MKT_T_END, MKT_N_STEPS = 60, 6000
MKT_STRATEGY_NAMES = ("Momentum", "Mean-Rev", "Noise")
N_BASIN_ICS = 18       # random interior starts for basin-of-attraction panel

# Sensitivity analysis
SENS_PARAM_VALUES = np.linspace(-1.5, 1.0, 40)
SENS_X0 = [1/3, 1/3, 1/3]
SENS_T_END, SENS_N_STEPS = 300, 20_000
SENS_BASE_CASE = -0.5


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _rps_conservation(results: list) -> None:
    print("\nConservation of H(x) = x1·x2·x3:")
    for t, sol, lbl in results:
        H = sol[:, 0] * sol[:, 1] * sol[:, 2]
        drift_pct = abs(H[-1] - H[0]) / (H[0] + 1e-15) * 100
        print(f"  {lbl}: H₀={H[0]:.6f}  H_T={H[-1]:.6f}  drift={drift_pct:.4f}%")


def _stoch_excursion_stats(sol_det: np.ndarray,
                            stoch_results: list) -> None:
    print("\nExcursion statistics (std dev of strategy fractions):")
    header = f"{'Source':<18} {'Rock':>8} {'Paper':>8} {'Scissors':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Deterministic':<18} "
          f"{sol_det[:,0].std():>8.4f} "
          f"{sol_det[:,1].std():>8.4f} "
          f"{sol_det[:,2].std():>10.4f}")
    for _, sol_s, lbl in stoch_results:
        print(f"{lbl:<18} "
              f"{sol_s[:,0].std():>8.4f} "
              f"{sol_s[:,1].std():>8.4f} "
              f"{sol_s[:,2].std():>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Prisoner's Dilemma
# ══════════════════════════════════════════════════════════════════════════════

def run_prisoners_dilemma() -> None:
    _section("Experiment 1 — Prisoner's Dilemma")
    A = prisoners_dilemma(R=PD_R, P=PD_P, T=PD_T, S=PD_S)
    print(f"Payoff matrix:\n{A}")

    results = [
        (*(simulate(A, x0, t_end=PD_T_END, n_steps=PD_N_STEPS)), lbl)
        for x0, lbl in zip(PD_ICS, PD_IC_LABELS)
    ]

    fig = plot_timeseries_2strategy(
        results,
        strategy_names=["Cooperate", "Defect"],
        title="Prisoner's Dilemma — Defection Always Dominates",
        hline=0.0,
        hline_label="x_C = 0",
    )
    save_fig(fig, FIGURES_DIR / "01_prisoners_dilemma.png")

    print(f"\nFinal cooperator shares:")
    for t, sol, lbl in results:
        print(f"  {lbl}: {sol[-1, 0]:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Hawk-Dove
# ══════════════════════════════════════════════════════════════════════════════

def run_hawk_dove() -> None:
    _section("Experiment 2 — Hawk-Dove")
    A = hawk_dove(V=HD_V, C=HD_C)
    x_star = HD_V / HD_C
    print(f"Payoff matrix:\n{A}")
    print(f"Analytical ESS: x*(Hawk) = {HD_V}/{HD_C} = {x_star:.4f}")

    results = [
        (*(simulate(A, x0, t_end=HD_T_END, n_steps=HD_N_STEPS)), lbl)
        for x0, lbl in zip(HD_ICS, HD_IC_LABELS)
    ]

    fig = plot_timeseries_2strategy(
        results,
        strategy_names=["Hawk", "Dove"],
        title="Hawk-Dove — Stable Mixed Equilibrium",
        hline=x_star,
        hline_label=f"ESS x*={x_star:.2f}",
        semilogy_panel=True,
        semilogy_ref=x_star,
    )
    save_fig(fig, FIGURES_DIR / "02_hawk_dove.png")

    print(f"\nFinal Hawk fractions vs ESS ({x_star:.4f}):")
    for t, sol, lbl in results:
        print(f"  {lbl}: {sol[-1,0]:.5f}  |  error = {abs(sol[-1,0]-x_star):.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Rock-Paper-Scissors (deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def run_rps_deterministic() -> None:
    _section("Experiment 3 — Rock-Paper-Scissors (deterministic)")
    A = rock_paper_scissors(eps=RPS_EPS)

    results = [
        (*(simulate(A, x0, t_end=RPS_T_END, n_steps=RPS_N_STEPS)), lbl)
        for x0, lbl in zip(RPS_ICS, RPS_IC_LABELS)
    ]

    # Time series
    fig = plot_timeseries_3strategy(
        results,
        strategy_names=["Rock", "Paper", "Scissors"],
        title="Rock-Paper-Scissors — Time Series (ε=0)",
    )
    save_fig(fig, FIGURES_DIR / "03a_rps_timeseries.png")

    # Simplex orbits
    trajs = [sol for _, sol, _ in results]
    lbls  = [lbl for _, _,  lbl in results]
    fig2, ax = plt.subplots(figsize=(6, 6))
    ternary_plot(trajs, lbls,
                 strategy_names=("Rock", "Paper", "Scissors"),
                 title="RPS — Simplex Trajectories (ε=0)",
                 ax=ax)
    cx_eq, cy_eq = to_cartesian(1/3, 1/3, 1/3)
    ax.plot(cx_eq, cy_eq, "k*", markersize=12, zorder=10, label="Interior NE")
    ax.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    save_fig(fig2, FIGURES_DIR / "03b_rps_simplex.png")

    # H(x) conservation
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    for t, sol, lbl in results:
        ax3.plot(t, sol[:, 0] * sol[:, 1] * sol[:, 2], label=lbl)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("H(x) = x₁x₂x₃")
    ax3.set_title("Conservation of H(x) in RPS (ε=0)")
    ax3.legend()
    fig3.tight_layout()
    save_fig(fig3, FIGURES_DIR / "03c_rps_conservation.png")

    _rps_conservation(results)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Stability analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_stability_analysis() -> None:
    _section("Experiment 4 — Stability Analysis")

    A_pd  = prisoners_dilemma(R=PD_R, P=PD_P, T=PD_T, S=PD_S)
    A_hd  = hawk_dove(V=HD_V, C=HD_C)
    A_rps = rock_paper_scissors(eps=RPS_EPS)

    experiments = [
        ("Prisoner's Dilemma",  A_pd,  [0.0, 1.0]),
        ("Hawk-Dove",           A_hd,  [HD_V/HD_C, 1-HD_V/HD_C]),
        ("Rock-Paper-Scissors", A_rps, [1/3, 1/3, 1/3]),
    ]

    df = stability_table(experiments)
    print("\nStability Summary:")
    print(df.to_string(index=False))

    fig = plot_eigenvalues(experiments, classify_fixed_point)
    save_fig(fig, FIGURES_DIR / "04_eigenvalues.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Stochastic RPS
# ══════════════════════════════════════════════════════════════════════════════

def run_stochastic_rps() -> None:
    _section("Experiment 5 — Stochastic RPS")
    A = rock_paper_scissors(eps=RPS_EPS)
    strategy_names = ["Rock", "Paper", "Scissors"]

    # ── Time-series comparison (coarser grid for speed) ──────────────────────
    t_det, sol_det = simulate(A, STOCH_X0,
                               t_end=STOCH_T_END, n_steps=STOCH_N_STEPS_TS)

    stoch_ts = []
    for sig in STOCH_SIGMAS:
        t_s, sol_s = stochastic_replicator(
            A, STOCH_X0,
            t_end=STOCH_T_END, n_steps=STOCH_N_STEPS_TS,
            sigma=sig, seed=STOCH_SEED,
        )
        stoch_ts.append((t_s, sol_s, f"σ={sig}"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for i, sname in enumerate(strategy_names):
        ax.plot(t_det, sol_det[:, i], color=COLORS[i], label=sname, linewidth=1.5)
    ax.set_title("Deterministic (σ=0)")
    ax.set_xlabel("Time"); ax.set_ylabel("Population share")
    ax.legend(); ax.set_ylim(-0.02, 1.02)

    for idx, (t_s, sol_s, lbl) in enumerate(stoch_ts):
        ax = axes[(idx+1)//2, (idx+1)%2]
        for i, sname in enumerate(strategy_names):
            ax.plot(t_s, sol_s[:, i], color=COLORS[i], alpha=0.8, label=sname, linewidth=0.8)
        ax.set_title(f"Stochastic ({lbl})")
        ax.set_xlabel("Time"); ax.set_ylabel("Population share")
        ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02)

    fig.suptitle(f"RPS: Deterministic vs Stochastic (t=0 to {STOCH_T_END})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "05a_stochastic_timeseries.png")

    # ── Simplex comparison (finer grid) ───────────────────────────────────────
    t_det_sx, sol_det_sx = simulate(A, STOCH_X0,
                                     t_end=STOCH_T_END, n_steps=STOCH_N_STEPS_SX)

    stoch_sx = []
    for sig in STOCH_SIGMAS:
        t_s, sol_s = stochastic_replicator(
            A, STOCH_X0,
            t_end=STOCH_T_END, n_steps=STOCH_N_STEPS_SX,
            sigma=sig, seed=STOCH_SEED,
        )
        stoch_sx.append((t_s, sol_s, f"σ={sig}"))

    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 6))

    def _simplex_panel(ax, sol, color, title):
        draw_simplex(ax, labels=("Rock", "Paper", "Scissors"))
        cx, cy = to_cartesian(sol[:, 0], sol[:, 1], sol[:, 2])
        ax.plot(cx, cy, color=color, linewidth=0.8, alpha=0.8)
        ax.plot(cx[0],  cy[0],  "o", color=color, markersize=8, zorder=5)
        ax.plot(cx[-1], cy[-1], "*", color=color, markersize=14, zorder=5)
        ax.set_title(title, fontsize=10, fontweight="bold")

    _simplex_panel(axes2[0], sol_det_sx, "#2196F3", "Deterministic (σ=0)")
    for idx, ((_, sol_s, lbl), color) in enumerate(zip(stoch_sx, SIGMA_COLORS)):
        _simplex_panel(axes2[idx + 1], sol_s, color, f"Stochastic ({lbl})")

    fig2.suptitle(
        f"RPS Simplex Trajectories — Full Run (t=0 to {STOCH_T_END})\n(○=start  ★=end)",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    save_fig(fig2, FIGURES_DIR / "05b_stochastic_simplex.png")

    _stoch_excursion_stats(sol_det_sx, stoch_sx)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Market microstructure
# ══════════════════════════════════════════════════════════════════════════════

def run_market_microstructure() -> None:
    _section("Experiment 6 — Market Microstructure")
    A = market_microstructure()
    print("Payoff matrix:")
    df_pay = pd.DataFrame(A,
                          index=[f"From {s}" for s in MKT_STRATEGY_NAMES],
                          columns=[f"vs {s}"  for s in MKT_STRATEGY_NAMES])
    print(df_pay.to_string())

    # Simulate canonical initial conditions
    mkt_results = [
        (*(simulate(A, x0, t_end=MKT_T_END, n_steps=MKT_N_STEPS)), lbl)
        for x0, lbl in MKT_ICS
    ]

    eq_sol = mkt_results[0][1][-1]   # converged equilibrium
    print(f"\nEquilibrium: {np.round(eq_sol, 4)}")

    # Time-series 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (t, sol, lbl) in enumerate(mkt_results):
        ax = axes[idx // 2, idx % 2]
        for i, sname in enumerate(MKT_STRATEGY_NAMES):
            ax.plot(t, sol[:, i], color=COLORS[i], label=sname, linewidth=2)
        ax.set_title(f"Initial composition: {lbl}")
        ax.set_xlabel("Time"); ax.set_ylabel("Strategy fraction")
        ax.legend(); ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Market Microstructure: Strategy Evolution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "06a_market_timeseries.png")

    # Random basin initial conditions
    rng = np.random.default_rng(BASIN_SEED)
    grid_ics: list[np.ndarray] = []
    while len(grid_ics) < N_BASIN_ICS:
        v = rng.dirichlet([1, 1, 1])
        if all(v > 0.05):
            grid_ics.append(v)

    # Phase portrait (3 panels)
    fig2 = market_phase_portrait(
        mkt_results, grid_ics, A, eq_sol, MKT_STRATEGY_NAMES, simulate,
    )
    save_fig(fig2, FIGURES_DIR / "06b_market_phase_portrait.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis() -> None:
    _section("Experiment 7 — Sensitivity Analysis")

    equilibria = np.zeros((len(SENS_PARAM_VALUES), 3))

    for k, p in enumerate(SENS_PARAM_VALUES):
        A = market_microstructure()
        A[0, 0] = p
        _, sol = simulate(A, SENS_X0, t_end=SENS_T_END, n_steps=SENS_N_STEPS)
        equilibria[k] = sol[-1]
        print(f"  A[0,0]={p:+.3f}  →  M={sol[-1,0]:.3f}  "
              f"MR={sol[-1,1]:.3f}  N={sol[-1,2]:.3f}")

    fig = plot_sensitivity(SENS_PARAM_VALUES, equilibria, base_case=SENS_BASE_CASE)
    save_fig(fig, FIGURES_DIR / "07_sensitivity.png")

    # Transition threshold
    cross = np.argmax(equilibria[:, 0] > equilibria[:, 1])
    if cross > 0:
        threshold = SENS_PARAM_VALUES[cross]
        print(f"\nRegime transition: Momentum overtakes Mean-Reversion "
              f"at A[0,0] ≈ {threshold:.3f}")
    base_idx = np.argmin(np.abs(SENS_PARAM_VALUES - SENS_BASE_CASE))
    print(f"At base case A[0,0]={SENS_BASE_CASE}:")
    print(f"  Momentum = {equilibria[base_idx, 0]:.4f}")
    print(f"  MeanRev  = {equilibria[base_idx, 1]:.4f}")
    print(f"  Noise    = {equilibria[base_idx, 2]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    np.random.seed(GLOBAL_SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR.resolve()}")

    run_prisoners_dilemma()
    run_hawk_dove()
    run_rps_deterministic()
    run_stability_analysis()
    run_stochastic_rps()
    run_market_microstructure()
    run_sensitivity_analysis()

    print("\n" + "═" * 60)
    print("  Pipeline complete.  All figures saved to ./figures/")
    print("═" * 60)


if __name__ == "__main__":
    main()
