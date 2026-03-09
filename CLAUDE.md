# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run a KANDy experiment
uv run kandy <system>           # e.g. uv run kandy lorenz
uv run kandy --list             # list available systems

# Run baseline comparisons
uv run kandy-baselines <name>   # e.g. uv run kandy-baselines burgers-fourier
uv run kandy-baselines --list   # list available baselines

# Run research scripts directly
uv run python research_code/burgers_fourier_baselines.py
```

## Algorithm

KANDy (Kolmogorov-Arnold Networks for Dynamics) is a reimplementation of SINDy where
sparse regression is replaced by a KAN and inputs are Koopman-lifted coordinates.

**The model:** `x_dot = A * Psi(phi(x))`
- `phi(x) = theta` -- Koopman lift (user-designed; encodes cross-terms like x*y explicitly)
- `Psi(theta)` -- separable spline map, `psi_i(theta_i)`, learned by a **single-layer** KAN
- `A` -- learned linear mixing matrix (n x m)

**Critical constraints:**
- The KAN is always `width=[m, n]` (one hidden layer). Standard deep KANs cannot
  represent bilinear terms like x*y from raw inputs (proven by zero-set corollary).
- Cross-interaction terms must be in `phi`, not learned from raw states.
- Non-injective observations require the lift to expand the observable space.

**KANDy pipeline:**
1. Design `phi(x)` (include ALL cross-terms for the target system)
2. Apply lift: `theta = phi(x)` for all trajectory snapshots
3. Train: `KAN(width=[m, n], grid, k).fit({train_input: theta, train_label: x_dot}, ...)`
4. Extract `A` (output weights) and `psi_i` (splines) symbolically via `auto_symbolic()`
5. Validate: autoregressive rollout with RK4, compute RMSE/NRMSE

## Project Structure

### Package (`src/kandy/`)
Pip-installable library with modular components:
- `core.py` -- `KANDy` class: fit, predict, rollout, get_formula, score_formula
- `lifts.py` -- Lift classes: PolynomialLift, CustomLift, FourierLift, KANELift, etc.
- `training.py` -- `fit_kan`, `rk4_step`, `make_windows`, `angle_mse`
- `symbolic.py` -- `auto_symbolic_with_costs`, `score_formula`, `formulas_to_latex`
- `numerics.py` -- PDE solvers: `solve_burgers`, `fv_rhs`, Rusanov/Roe/HLLC fluxes
- `plotting.py` -- Publication plots: `plot_all_edges`, `plot_attractor_overlay`, `plot_loss_curves`, `plot_trajectory_error`, `use_pub_style`
- `main.py` -- CLI entry points

### Research code (`research_code/`)
Raw experimental notebooks and scripts:
- `henon.py` -- Henon map (discrete, 2D; lift includes x^2)
- `hopf.ipynb` -- Hopf fibration (S^3 -> S^2, topological)
- `Lorenz (3).ipynb` -- Lorenz ODE (lift must include xy, xz)
- `Inviscid_Burgers (1).ipynb` -- Burgers equation (lift includes u, u_x, u^2)
- `Inviscd-Burgers-fourier-mode-ics.ipynb` -- Burgers with Fourier-mode ICs
- `Kuramoto-Sivashinsky (1).ipynb` -- KS PDE (lift includes u, u_x, u_xx, u_xxxx)
- `Navier-Stokes.ipynb` -- 3D incompressible N-S
- `burgers_fourier_baselines.py` -- SINDy/PDE-FIND baselines for Burgers

### Examples (`examples/`)
Clean example scripts for each dynamical system, plus SINDy/PDEFind baselines.

### Results (`results/`)
All research outputs (plots, reports, data) go here, organized by system name.

### Agent orchestration
Research automation (running experiments, reviewing results, generating reports) is
handled by **Claude Code agents**, not an in-process framework. Claude Code spawns
subagents to run experiments, use multimodal to review plots, and write reports.

## Output conventions
- All research results go to `results/<SystemName>/` (plots as PNG+PDF at 300 dpi)
- Baseline comparisons go to `results/<SystemName>/baselines/`
- Reports include discovered equations in clean symbolic format
- Use `kandy.plotting.use_pub_style()` for publication-quality figures

### Environment
- Python 3.11-3.13, package manager: **uv**
- Core research dependencies: `pykan`, `torch`, `scipy`, `sympy`, `matplotlib`
- Baseline dependencies: `pysindy`, `scikit-learn`
