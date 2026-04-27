# Deterministic and Stochastic Replicator Dynamics  
### Stability, Invariants, and Strategic Evolution

This repository contains a computational and theoretical study of **replicator dynamics** in evolutionary game theory, examining how strategic populations evolve under deterministic and stochastic selection.

The project combines:

- analytical stability analysis via Jacobian linearisation,
- stochastic perturbation modelling,
- canonical evolutionary games, and
- a market microstructure application.

---

## Overview

Evolutionary Game Theory (EGT) studies how strategies spread or decline according to relative fitness rather than rational optimization.

The central dynamical model is the **replicator equation**:

$$
\dot{x}_i = x_i\big((Ax)_i - x^\top A x\big)
$$

where:

- $x_i$ is the population share of strategy $i$,
- $A$ is the payoff matrix,
- $(Ax)_i$ is the fitness of strategy $i$,
- $x^\top A x$ is average population fitness.

This project investigates how payoff structure determines:

- equilibrium selection,
- evolutionary stability,
- cyclic coexistence,
- noise-induced boundary drift.

---

## Project Components

### 1. Canonical Games

#### Prisoner’s Dilemma
- Dominance dynamics
- Convergence to defect-only equilibrium
- Stable corner attractor

#### Hawk–Dove
- Mixed-strategy ESS
- Interior stable equilibrium
- Exponential convergence

#### Rock–Paper–Scissors
- Cyclic coexistence
- Neutral center
- Conserved invariant:

$$
H(x)=x_1x_2x_3
$$

---

### 2. Stability Analysis

Fixed points are classified using:

- analytical Jacobian derivation,
- eigenvalue analysis,
- simplex tangent-space reduction.

Stability classes:

- **Stable attractor (ESS)**
- **Unstable equilibrium**
- **Neutral center**

---

### 3. Stochastic Replicator Dynamics

We extend the deterministic model with stochastic perturbations:

$$
dx_i = x_i(f_i-\bar f)\,dt + \sigma x_i(1-x_i)\,dW_t
$$

Key findings:

- deterministic invariants break under noise,
- trajectories diffuse toward simplex boundaries,
- near-extinction states emerge.

---

### 4. Market Microstructure Application

A stylized three-strategy market model:

- **Momentum traders**
- **Mean-reversion traders**
- **Noise traders**

Replicator dynamics are used to study strategic competition and equilibrium selection under crowding costs.

---

### 5. Sensitivity Analysis

We vary the momentum self-payoff parameter:

$$
A_{00}
$$

to study equilibrium transitions between momentum-dominated and mean-reversion-dominated regimes.

---

## Methods

Numerical and analytical methods used:

- ODE integration for deterministic replicator dynamics
- Euler–Maruyama simulation for stochastic dynamics
- Analytical Jacobian evaluation
- Eigenvalue-based local stability classification

Core dependencies:

- NumPy
- SciPy
- Matplotlib
- Pandas

---

## Key Results

- The Prisoner’s Dilemma converges to the defect-only equilibrium.
- Hawk–Dove admits a stable mixed ESS.
- Rock–Paper–Scissors exhibits conserved cyclic dynamics.
- Stochastic perturbations destroy deterministic invariants.
- Market competition exhibits regime-dependent equilibrium selection.

---

## License

This project is licensed under the MIT License.

---

## Author

**Rajveer Vora**  
Undergraduate student of Statistical Data Science at Indian Statistical Institute
