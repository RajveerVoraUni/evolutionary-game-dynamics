# Replicator Dynamics — Python Codebase

Modular, reproducible implementation of deterministic and stochastic
replicator dynamics in evolutionary game theory.

## Structure

```
replicator_project/
├── main.py           # Entry point — runs full pipeline, saves all figures
├── games.py          # Payoff matrix factories + fitness helpers
├── dynamics.py       # Deterministic replicator ODE (scipy.integrate.odeint)
├── stochastic.py     # Stochastic replicator SDE (Euler-Maruyama)
├── stability.py      # Analytical Jacobian + eigenvalue classification
├── visualization.py  # All plotting and simplex utilities
├── figures/          # Output figures (created on first run)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Running `main.py` executes the full pipeline:

1. Prisoner's Dilemma
2. Hawk-Dove
3. Rock-Paper-Scissors (deterministic)
4. Stability analysis (Jacobian eigenvalues)
5. Stochastic RPS (Euler-Maruyama, σ = 0.02 / 0.05 / 0.10)
6. Market microstructure (time-series + phase portrait)
7. Sensitivity analysis (momentum self-payoff sweep)

All figures are saved to `./figures/`.

## Configuration

All experimental parameters are defined at the top of `main.py` as
module-level constants. Edit them there — no changes to other modules needed.

## Module API

```python
from games      import prisoners_dilemma, hawk_dove, rock_paper_scissors, market_microstructure
from dynamics   import simulate
from stochastic import stochastic_replicator
from stability  import classify_fixed_point, stability_table
```
